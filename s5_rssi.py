# -*- coding: utf-8 -*-
"""
s5_rssi.py — A안: 정규화로 지워진 '크기' 정보를 RSSI 로 되살려 본다

동기
---------------------------------------------------------------------------
지금까지의 분석은 전부 frame_norm 정규화를 거쳤다. 정규화는 AGC 이득을 지우지만
'신호가 얼마나 셌는가' 도 같이 지운다. 사람 몸이 전파를 막는 효과는 상당 부분
세기 감소로 나타나므로, 지워진 쪽에 답이 있었을 수 있다.
RSSI 는 정규화로 지워지지 않고 원본에 남아 있는 값이다.

  A-1  RSSI / 노이즈 플로어 / SNR 자체
  A-2  수신률 / 프레임 간격 / 드롭률   (몸이 막으면 오류가 늘 수 있다)
  A-3  정규화된 '모양' x RSSI 로 복원한 '크기' = RSSI 보정 진폭(RSSI-scaled amplitude).
       subcarrier 114개 전수 검정

A-3 이 핵심이다. 기존 subcarrier 분석은 전부 정규화 후 값이라 크기 정보가 없었다.

통계 규칙 (고정)
---------------------------------------------------------------------------
- 표본 단위는 세션 구간. 프레임 단위 검정이나 효과크기 계산은 하지 않는다.
- 같은 참가자의 여러 구간을 독립 표본으로 세지 않는다 -> 참가자 단위 재검정을 병기한다.
- 검정은 정확 순열검정. 정규분포를 가정하지 않는다.
- subcarrier 전수 검정에는 BH FDR 보정을 반드시 건다.
- 주 분석 배치는 CSI_BATCH 로 정한다(미설정이면 B). 다른 배치는 대조로만 병기한다.
"""

from __future__ import annotations

import os
from math import comb
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import csi_core as core
import sessions as S

LINE = "=" * 100
SUB = "-" * 100

# RSSI 해상도 판정 기준 (3-1)
RES_OK = 5          # 서로 다른 값 5개 이상 -> A-1 사용 가능
RES_WEAK = 3        # 3~4개 -> 참고용, 2개 이하 -> 제외

SEP_THRESHOLD = 2.0  # sep 2 미만이면 p 와 무관하게 주장 근거가 없다


# ===========================================================================
# 구간 표본 만들기
# ===========================================================================

def segment_samples(sessions: List[core.Session]) -> List[Dict[str, object]]:
    """
    세션 x 구간마다 표본 1개를 만든다. 이것이 이 분석의 독립 표본 단위다.

    전처리 순서는 기존 스크립트(s2/s3)와 동일하게 맞춘다:
        amplitude -> normalize(frame_norm) -> hampel
    그 다음에만 RSSI 로 복원한 크기를 곱한다. hampel 을 크기 복원 뒤에 걸면
    실제 RSSI 계단 변화까지 임펄스로 보고 지워버리므로 순서를 바꾸면 안 된다.
    """
    out: List[Dict[str, object]] = []
    mask = None
    for s in sessions:
        cfg = s.meta.get("config", {})
        stbc = int(cfg.get("stbc", 0))
        bw40 = str(cfg.get("bandwidth", "1")) == "1"
        fw = str(cfg.get("first_word", "0")) == "1"
        mask = S.analysis_mask(s.n_sub, stbc=stbc, bandwidth_40=bw40,
                               first_word_invalid=fw)

        amp = core.amplitude(s.csi)
        nrm = core.normalize(amp, mask, method=S.NORMALIZE)
        shape, _ = core.hampel(nrm[:, mask], window=S.HAMPEL_WINDOW,
                               n_sigma=S.HAMPEL_SIGMA)

        # RSSI[dBm] -> 선형 진폭 스케일. 전력이 10^(R/10) 이므로 진폭은 10^(R/20).
        scale = np.power(10.0, s.rssi.astype(np.float64) / 20.0)
        absamp = shape * scale[:, None]

        snr = s.rssi.astype(np.float64) - s.noise_floor.astype(np.float64)

        for k, sg in enumerate(s.segments):
            m = sg.mask(s.t, guard=S.SEGMENT_GUARD_SEC)
            n = int(m.sum())
            if n < 50:
                continue
            ti = s.t[m]
            dt = np.diff(ti)
            out.append({
                "session": s.name, "state": sg.label, "subject": s.subject,
                "batch": s.batch, "seg": k, "n": n,
                "t0": sg.t0, "t1": sg.t1,
                # A-1
                "rssi": s.rssi[m].astype(np.float64),
                "rssi_mean": float(s.rssi[m].mean()),
                "nf_mean": float(s.noise_floor[m].mean()),
                "snr_mean": float(snr[m].mean()),
                # A-2
                "rate_hz": float(n / (ti[-1] - ti[0])) if ti[-1] > ti[0] else float("nan"),
                "dt_med": float(np.median(dt)),
                "dt_iqr": float(np.percentile(dt, 75) - np.percentile(dt, 25)),
                "drop_rate": float(s.meta["drop_rate"]),
                "seq_loss": float(s.seq_loss[1]),
                # A-3
                "abs_prof": absamp[m].mean(axis=0),          # (114,)
                "abs_mean": float(absamp[m].mean()),
                "shape_mean": float(shape[m].mean()),
            })
    return out, mask


# ===========================================================================
# 검정
# ===========================================================================

def perm_line(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """정확 순열검정 + sep + 도달 가능한 최소 p 를 한 줄로 묶는다."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 1 or b.size < 1:
        return {"na": a.size, "nb": b.size, "mean_a": float("nan"),
                "mean_b": float("nan"), "diff": float("nan"),
                "sep": float("nan"), "p": float("nan"), "p_min": float("nan")}
    obs, p, pmin = core.exact_perm_test_unpaired(a, b)
    sep = core.sep_ratio(a, b)
    # 퇴화 사례: 양쪽 다 변동이 0 이면 sep_ratio 는 분모 0 때문에 inf 를 돌려준다.
    # 차이까지 0 이면 그건 '완벽한 분리'가 아니라 '잴 것이 없음'이다. inf 로 두면
    # 판정 로직이 이걸 강한 근거로 잘못 읽으므로 여기서 nan 으로 바꾼다.
    if not np.isfinite(sep) and abs(obs) < 1e-12:
        sep = float("nan")
    return {"na": int(a.size), "nb": int(b.size),
            "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "diff": obs, "sep": sep, "p": p, "p_min": pmin}


def by_subject(samples: List[Dict[str, object]], state: str, key: str
               ) -> Tuple[List[str], np.ndarray]:
    """
    참가자 단위로 합친다. 같은 사람의 여러 구간은 중앙값으로 하나로 만든다.
    비점유 세션은 subject 가 "-" 라 참가자가 없으므로 세션 이름을 단위로 쓴다.
    """
    by: Dict[str, List[float]] = {}
    for p in samples:
        if p["state"] != state:
            continue
        unit = p["session"] if p["subject"] in ("", "-") else p["subject"]
        by.setdefault(unit, []).append(p[key])
    names = sorted(by)
    return names, np.array([float(np.median(by[n])) for n in names])


def by_subject_prof(samples: List[Dict[str, object]], state: str
                    ) -> Tuple[List[str], np.ndarray]:
    """참가자 단위 subcarrier 프로파일 (구간 여러 개면 중앙값)."""
    by: Dict[str, List[np.ndarray]] = {}
    for p in samples:
        if p["state"] != state:
            continue
        unit = p["session"] if p["subject"] in ("", "-") else p["subject"]
        by.setdefault(unit, []).append(p["abs_prof"])
    names = sorted(by)
    return names, np.array([np.median(np.stack(by[n]), axis=0) for n in names])


def scalar_table(title: str, samples: List[Dict[str, object]],
                 keys: List[Tuple[str, str]]) -> List[Dict[str, object]]:
    """스칼라 특징들을 still 대 empty 로 검정해 표로 찍는다."""
    print()
    print(LINE)
    print(title)
    print(LINE)
    rows = []
    hdr = "%-14s %8s %8s %9s %7s %8s %8s   %s"
    print(hdr % ("특징", "still", "empty", "차이", "sep", "p", "p_min", "판정"))
    print(SUB)
    for key, label in keys:
        a = [p[key] for p in samples if p["state"] == "still"]
        b = [p[key] for p in samples if p["state"] == "empty"]
        r = perm_line(a, b)
        verdict = _verdict(r)
        print("%-14s %8.3f %8.3f %9.3f %7.2f %8.4f %8.4f   %s"
              % (label, r["mean_a"], r["mean_b"], r["diff"], r["sep"],
                 r["p"], r["p_min"], verdict))
        r["key"] = key
        r["label"] = label
        r["verdict"] = verdict
        rows.append(r)

    # --- 참가자 단위 재검정 ---
    print(SUB)
    print("참가자 단위 재검정 (같은 사람의 여러 구간을 중앙값으로 하나로 합친 뒤 재검정)")
    print(SUB)
    print(hdr % ("특징", "still", "empty", "차이", "sep", "p", "p_min", "판정"))
    for key, label in keys:
        ua, a = by_subject(samples, "still", key)
        ub, b = by_subject(samples, "empty", key)
        r = perm_line(a, b)
        verdict = _verdict(r)
        print("%-14s %8.3f %8.3f %9.3f %7.2f %8.4f %8.4f   %s"
              % (label, r["mean_a"], r["mean_b"], r["diff"], r["sep"],
                 r["p"], r["p_min"], verdict))
        for row in rows:
            if row["key"] == key:
                row["subj"] = r
                row["subj_verdict"] = verdict
    print("  참가자 단위 표본: still %s / empty %s" % (ua, ub))
    return rows


def _verdict(r: Dict[str, float]) -> str:
    """sep 을 1차 기준으로, p 는 보조로 읽는다."""
    if not np.isfinite(r["sep"]):
        return "변동없음(판정불가)"
    if r["sep"] < SEP_THRESHOLD:
        return "근거없음(sep<2)"
    if r["p"] <= r["p_min"] + 1e-12:
        return "sep>=2, p는 최소값(=가장 극단적 배치일 뿐)"
    return "sep>=2, p=%.4f" % r["p"]


def _common_mode_check(pa: np.ndarray, pb: np.ndarray) -> Dict[str, float]:
    """
    '114개 중 N개가 갈렸다' 를 독립 시행 N번으로 읽지 않기 위한 분해.

    각 표본에서 대역 전체 평균(공통 레벨)을 빼고 다시 세면, 남는 것은 '모양' 차이뿐이다.
    두 숫자가 크게 벌어지면 원래 개수는 대부분 하나의 공통 레벨 이동에서 나온 것이다.
    이것은 유의성을 다시 캐는 재시도가 아니라, 이미 나온 개수를 과대해석하지 않기 위한
    분해다. 판정 기준(FDR q)은 바뀌지 않는다.
    """
    def sweep(a, b):
        seps, full = [], []
        for j in range(a.shape[1]):
            seps.append(core.sep_ratio(a[:, j], b[:, j]))
            full.append(bool(a[:, j].min() > b[:, j].max()
                             or a[:, j].max() < b[:, j].min()))
        return np.asarray(seps), np.asarray(full, dtype=bool)

    s0, f0 = sweep(pa, pb)
    s1, f1 = sweep(pa - pa.mean(1, keepdims=True), pb - pb.mean(1, keepdims=True))
    allp = np.vstack([pa, pb])
    C = np.corrcoef(allp.T)
    iu = np.triu_indices(allp.shape[1], 1)
    return {
        "corr_med": float(np.median(C[iu])),
        "n_sep0": int((s0 >= SEP_THRESHOLD).sum()), "n_full0": int(f0.sum()),
        "sep_max0": float(np.nanmax(s0)),
        "n_sep1": int((s1 >= SEP_THRESHOLD).sum()), "n_full1": int(f1.sum()),
        "sep_max1": float(np.nanmax(s1)),
        "sep_level": float(core.sep_ratio(pa.mean(1), pb.mean(1))),
    }


def _finite_ge(v) -> bool:
    """sep 이 유한하고 기준 이상인가. nan/inf 는 근거로 세지 않는다."""
    return v is not None and np.isfinite(v) and v >= SEP_THRESHOLD

# ===========================================================================
# 실행
# ===========================================================================

def main() -> int:
    plt = core.setup_matplotlib()

    all_sessions = S.load_all(verbose=False)
    # 주 분석 배치는 CSI_BATCH 로 정해진다 (미설정이면 "B" = 기존 동작).
    # 0개면 sessions_of_batch() 가 예외를 던진다. 표본 0개로 '성공'하면 안 된다.
    main_sessions = S.sessions_of_batch(all_sessions)
    other_sessions = [s for s in all_sessions if s.batch.upper() != S.BATCH]

    print(LINE)
    print("s5_rssi.py  A안: RSSI 로 복원한 '크기' 정보로 정적 점유를 구분할 수 있는가")
    print(LINE)
    print("  전체 세션 %d개 (배치 %s %d, 그 외 %d)"
          % (len(all_sessions), S.BATCH, len(main_sessions), len(other_sessions)))
    if not other_sessions:
        print("  * 배치 %s 외의 세션이 없어 '타 배치 포함본' 대조를 계산할 수 없다."
              % S.BATCH)
        print("    아래 결과는 전부 배치 %s 단독이며, 이는 원래 주 결과로 삼기로 한 조건이다."
              % S.BATCH)

    samples, mask = segment_samples(main_sessions)
    n_sub_valid = int(mask.sum())
    scs = core.subcarrier_numbers(192, mask)

    still = [p for p in samples if p["state"] == "still"]
    empty = [p for p in samples if p["state"] == "empty"]
    motion = [p for p in samples if p["state"] == "motion"]

    print()
    print("  구간 표본: still %d / empty %d / motion %d  (guard %.0fs 적용)"
          % (len(still), len(empty), len(motion), S.SEGMENT_GUARD_SEC))
    print("  분석 블록 %s, 유효 subcarrier %d개" % (S.ANALYSIS_BLOCK, n_sub_valid))
    print()
    print("  %-14s %-7s %-6s %-6s %7s %9s" % ("구간", "상태", "subj", "batch", "n", "구간[s]"))
    for p in samples:
        print("  %-14s %-7s %-6s %-6s %7d %4.0f-%4.0f"
              % (p["session"], p["state"], p["subject"], p["batch"], p["n"],
                 p["t0"], p["t1"]))

    # =====================================================================
    # 3-1. RSSI 해상도 점검
    # =====================================================================
    print()
    print(LINE)
    print("[RSSI 해상도 점검]  ESP32 RSSI 는 1 dBm 단위로 끊긴다.")
    print("  값 종류가 적으면 평균이 계단처럼 움직여 평균 비교 자체가 무의미해진다.")
    print(LINE)
    print("  %-14s %-7s %6s %8s %8s   %s"
          % ("구간", "상태", "종류", "최소", "최대", "관측된 값"))
    print(SUB)
    n_low = 0
    for p in samples:
        u = np.unique(p["rssi"])
        if u.size < RES_OK:
            n_low += 1
        print("  %-14s %-7s %6d %8.0f %8.0f   %s"
              % (p["session"], p["state"], u.size, u.min(), u.max(),
                 ",".join("%d" % v for v in u[:12]) + ("..." if u.size > 12 else "")))

    pooled_still = np.unique(np.concatenate([p["rssi"] for p in still])) if still else np.array([])
    pooled_empty = np.unique(np.concatenate([p["rssi"] for p in empty])) if empty else np.array([])
    print(SUB)
    print("  풀링: still %d종류 %s / empty %d종류 %s"
          % (pooled_still.size, pooled_still.astype(int).tolist(),
             pooled_empty.size, pooled_empty.astype(int).tolist()))

    min_kinds = min([np.unique(p["rssi"]).size for p in samples]) if samples else 0
    if min_kinds >= RES_OK:
        a1_policy = "사용 가능"
    elif min_kinds >= RES_WEAK:
        a1_policy = "참고용으로만"
    else:
        a1_policy = "결과에서 제외 (A-3만 보고)"
    print()
    print("  판정: 5단계 미만 단위가 %d개였으므로 A-1 은 %s로 처리한다."
          % (n_low, a1_policy))
    print("        (구간별 RSSI 값 종류의 최솟값 = %d)" % min_kinds)

    # =====================================================================
    # 3-2. A-1 / A-2 스칼라
    # =====================================================================
    a1_rows = scalar_table(
        "[A-1] RSSI / 노이즈 플로어 / SNR   (해상도 판정: %s)" % a1_policy,
        samples,
        [("rssi_mean", "RSSI [dBm]"), ("nf_mean", "noise_floor"),
         ("snr_mean", "SNR [dB]")])

    a2_rows = scalar_table(
        "[A-2] 수신률 / 프레임 간격 / 드롭률",
        samples,
        [("rate_hz", "수신률 [Hz]"), ("dt_med", "dt 중앙값 [s]"),
         ("dt_iqr", "dt IQR [s]"), ("drop_rate", "파싱 드롭률"),
         ("seq_loss", "id 손실률")])

    a3_rows = scalar_table(
        "[A-3] RSSI 보정 진폭 (정규화된 모양 x RSSI 복원 크기) — 스칼라 요약",
        samples,
        [("abs_mean", "RSSI보정진폭"), ("shape_mean", "정규화 진폭(대조)")])

    # =====================================================================
    # 3-3. A-3 subcarrier 전수 검정
    # =====================================================================
    def sweep(prof_a: np.ndarray, prof_b: np.ndarray, tag: str) -> Dict[str, object]:
        na, nb = prof_a.shape[0], prof_b.shape[0]
        print()
        print(LINE)
        print("[A-3] subcarrier 전수 검정 — %s   (still n=%d, empty n=%d)" % (tag, na, nb))
        print(LINE)
        if na < 1 or nb < 1:
            print("  표본이 없어 검정할 수 없다.")
            return {}
        ps, seps, diffs, fullsep = [], [], [], []
        for j in range(prof_a.shape[1]):
            a, b = prof_a[:, j], prof_b[:, j]
            obs, p, _ = core.exact_perm_test_unpaired(a, b)
            ps.append(p)
            seps.append(core.sep_ratio(a, b))
            diffs.append(obs)
            fullsep.append(bool(a.min() > b.max() or a.max() < b.min()))
        ps = np.asarray(ps)
        seps = np.asarray(seps)
        diffs = np.asarray(diffs)
        fullsep = np.asarray(fullsep)

        rej, q = core.benjamini_hochberg(ps, q=0.05)
        pmin = core.min_two_sided_p_unpaired(na, nb)
        # 완전분리의 무작위 기댓값: 작은 쪽 nb 개가 양끝에 몰릴 확률 = 2 / C(na+nb, nb)
        exp_full = prof_a.shape[1] * 2.0 / comb(na + nb, nb)

        print("  p 최솟값                : %.4f   (이 표본 수의 하한 p_min = %.4f)"
              % (ps.min(), pmin))
        print("  p<0.05 개수             : %d / %d   <- 참고용" % (int((ps < 0.05).sum()), ps.size))
        print("  ★ FDR q<0.05 개수       : %d / %d   <- 판정 기준"
              % (int(rej.sum()), ps.size))
        print("  q 최솟값                : %.4f" % q.min())
        print("  완전분리 개수           : %d   (무작위 기댓값 %.2f)"
              % (int(fullsep.sum()), exp_full))
        print("  sep 최댓값              : %.2f   (기준 %.1f)" % (np.nanmax(seps), SEP_THRESHOLD))
        print("  sep>=2 개수             : %d" % int((seps >= SEP_THRESHOLD).sum()))
        k = int(np.nanargmax(seps))
        print("  최대 sep subcarrier     : sc %+d  sep=%.2f  p=%.4f  q=%.4f  차이=%+.4f"
              % (scs[k], seps[k], ps[k], q[k], diffs[k]))
        return {"p": ps, "q": q, "sep": seps, "diff": diffs, "fullsep": fullsep,
                "rej": rej, "p_min": pmin, "exp_full": exp_full,
                "na": na, "nb": nb, "tag": tag}

    prof_still = np.array([p["abs_prof"] for p in still])
    prof_empty = np.array([p["abs_prof"] for p in empty])
    sw_sess = sweep(prof_still, prof_empty, "세션 구간 단위")

    us, prof_still_s = by_subject_prof(samples, "still")
    ue, prof_empty_s = by_subject_prof(samples, "empty")
    sw_subj = sweep(prof_still_s, prof_empty_s, "참가자 단위")
    print("  참가자 단위 표본: still %s / empty %s" % (us, ue))

    # =====================================================================
    # 3-4. 시간 드리프트
    # =====================================================================
    print()
    print(LINE)
    print("[시간 드리프트 점검]  RSSI 가 상태가 아니라 수집 순서를 따라 움직이는가")
    print(LINE)
    print("  주의: 로그의 local_timestamp 는 장치 부팅 후 경과 시간이라 세션마다 리셋된다.")
    print("        따라서 실제 수집 시각은 로그에서 복원할 수 없고, 아래 순서는")
    print("        sessions.py 매니페스트 순서를 수집 순서로 '가정'한 것이다.")
    order = [p for p in samples]
    print()
    print("  %-3s %-14s %-7s %10s" % ("#", "구간", "상태", "RSSI평균"))
    for i, p in enumerate(order):
        print("  %-3d %-14s %-7s %10.3f" % (i, p["session"], p["state"], p["rssi_mean"]))
    idx = np.arange(len(order), dtype=np.float64)
    vals = np.array([p["rssi_mean"] for p in order])
    if idx.size >= 3:
        rho = float(np.corrcoef(idx, vals)[0, 1])
        slope = float(np.polyfit(idx, vals, 1)[0])
        print()
        print("  순서 대 RSSI 상관계수   : %+.3f" % rho)
        print("  순서당 기울기           : %+.4f dBm/구간" % slope)
        mono = bool(np.all(np.diff(vals) > 0) or np.all(np.diff(vals) < 0))
        print("  단조 증가/감소인가      : %s" % ("예 -> 드리프트 의심" if mono else "아니오"))
    else:
        rho, slope, mono = float("nan"), float("nan"), False

    # =====================================================================
    # 그림
    # =====================================================================
    _fig_rssi(plt, samples, still, empty, order, idx, vals)
    _fig_abs(plt, scs, prof_still, prof_empty, sw_sess)

    # =====================================================================
    # 보고서
    # =====================================================================
    # 과대해석 방지용 분해: 114개 중 몇 개가 갈린다는 숫자를 독립 시행 수로 읽으면 안 된다.
    # subcarrier 들은 서로 강하게 상관돼 있고, 대역 전체 레벨이 통째로 움직이면
    # 그 하나의 공통 성분 때문에 수십 개가 동시에 갈린 것처럼 보인다.
    decomp = _common_mode_check(prof_still_s, prof_empty_s)
    print()
    print(LINE)
    print("[과대해석 점검] 114개는 독립 시행이 아니다")
    print(LINE)
    print("  subcarrier 쌍 상관 중앙값 : %.3f" % decomp["corr_med"])
    print("  원본            : sep>=2 %3d개, 완전분리 %3d개, sep최대 %.2f"
          % (decomp["n_sep0"], decomp["n_full0"], decomp["sep_max0"]))
    print("  공통레벨 제거 후: sep>=2 %3d개, 완전분리 %3d개, sep최대 %.2f"
          % (decomp["n_sep1"], decomp["n_full1"], decomp["sep_max1"]))
    print("  대역 전체 평균 자체의 sep : %.2f" % decomp["sep_level"])

    _write_report(a1_policy, min_kinds, n_low, samples, still, empty,
                  a1_rows, a2_rows, a3_rows, sw_sess, sw_subj,
                  scs, rho, slope, mono, order, other_sessions, us, ue, decomp)

    # =====================================================================
    # 판정
    # =====================================================================
    print()
    print(LINE)
    ok_sess = bool(sw_sess and sw_sess["rej"].sum() > 0)
    ok_subj = bool(sw_subj and sw_subj["rej"].sum() > 0)
    # A-2(수신률/dt/드롭률/손실률)는 수집 품질 지표이지 상태 특징이 아니다.
    # 이것이 잘 갈린다는 이유로 '점유 구분 근거 있음' 이 나오면 안 되므로 제외한다.
    # A-1 은 RSSI 해상도가 '사용 가능'(서로 다른 값 5종류 이상)일 때만 판정에 넣는다.
    verdict_rows = a3_rows + (a1_rows if a1_policy == "사용 가능" else [])
    scalar_ok = any(_finite_ge(r.get("sep"))
                    and _finite_ge(r.get("subj", {}).get("sep"))
                    for r in verdict_rows)
    if ok_subj and scalar_ok:
        verdict = "A안: 정적 점유 구분 근거 있음"
    elif ok_sess or scalar_ok:
        verdict = "A안: 정적 점유 구분 근거 없음 (세션 단위 신호는 있으나 참가자 단위에서 무너진다)"
    else:
        verdict = "A안: 정적 점유 구분 근거 없음"
    print(verdict)
    print(LINE)
    return 0


def _fig_rssi(plt, samples, still, empty, order, idx, vals) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    for p in order:
        ax[0].plot([order.index(p)], [p["rssi_mean"]], "o", ms=9,
                   color=core.STATE_COLOR.get(p["state"], "#777"))
    ax[0].plot(idx, vals, "-", color="#999", lw=1, zorder=0)
    if idx.size >= 2:
        ax[0].plot(idx, np.polyval(np.polyfit(idx, vals, 1), idx), "--",
                   color="k", lw=1.2, label="선형 추세")
        ax[0].legend(fontsize=8)
    ax[0].set_xticks(idx)
    ax[0].set_xticklabels([p["session"] for p in order], rotation=60,
                          ha="right", fontsize=7)
    ax[0].set_ylabel("구간 평균 RSSI [dBm]")
    ax[0].set_title("수집 순서(가정)에 따른 RSSI\n단조 추세면 상태가 아니라 드리프트다")
    ax[0].grid(alpha=.3)

    bins = np.arange(-60.5, -30.5, 1.0)
    if still:
        ax[1].hist(np.concatenate([p["rssi"] for p in still]), bins=bins,
                   alpha=.55, color=core.STATE_COLOR["still"], label="still", density=True)
    if empty:
        ax[1].hist(np.concatenate([p["rssi"] for p in empty]), bins=bins,
                   alpha=.55, color=core.STATE_COLOR["empty"], label="empty", density=True)
    ax[1].set_xlabel("RSSI [dBm]")
    ax[1].set_ylabel("밀도")
    ax[1].set_title("프레임 RSSI 분포\n포개져 있으면 구분되지 않는다")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3)

    for st, lab in (("still", "still"), ("empty", "empty")):
        v = [p["snr_mean"] for p in samples if p["state"] == st]
        if v:
            ax[2].plot(np.full(len(v), 0 if st == "still" else 1), v, "o", ms=9,
                       color=core.STATE_COLOR[st], label=lab)
    ax[2].set_xticks([0, 1])
    ax[2].set_xticklabels(["still", "empty"])
    ax[2].set_ylabel("구간 평균 SNR [dB]")
    ax[2].set_title("구간 단위 SNR")
    ax[2].grid(alpha=.3)

    fig.suptitle("A-1 / 시간 드리프트 점검 (배치 %s)" % S.BATCH, fontsize=12)
    fig.tight_layout()
    path = os.path.join(S.OUT_DIR, "s5_rssi_%s.png" % S.BATCH_TAG)
    fig.savefig(path)
    plt.close(fig)
    print("  그림 저장: %s" % path)


def _fig_abs(plt, scs, prof_still, prof_empty, sw) -> None:
    fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    o = np.argsort(scs)
    x = scs[o]

    if prof_still.size:
        m = prof_still.mean(axis=0)[o]
        sd = prof_still.std(axis=0, ddof=1)[o] if prof_still.shape[0] > 1 else np.zeros_like(m)
        ax[0].plot(x, m, color=core.STATE_COLOR["still"], lw=1.5, label="still")
        ax[0].fill_between(x, m - sd, m + sd, color=core.STATE_COLOR["still"], alpha=.2)
    if prof_empty.size:
        m = prof_empty.mean(axis=0)[o]
        sd = prof_empty.std(axis=0, ddof=1)[o] if prof_empty.shape[0] > 1 else np.zeros_like(m)
        ax[0].plot(x, m, color=core.STATE_COLOR["empty"], lw=1.5, label="empty")
        ax[0].fill_between(x, m - sd, m + sd, color=core.STATE_COLOR["empty"], alpha=.2)
    ax[0].set_ylabel("RSSI 보정 진폭")
    ax[0].set_title("A-3 RSSI 보정 진폭 프로파일 (음영 = 세션 간 표준편차)")
    ax[0].legend(fontsize=9)
    ax[0].grid(alpha=.3)

    if sw:
        ax[1].plot(x, sw["sep"][o], "k-", lw=1.3)
        ax[1].axhline(SEP_THRESHOLD, color="r", lw=1.5, ls="--",
                      label="sep = %.1f (주장 가능 최소선)" % SEP_THRESHOLD)
        ax[1].legend(fontsize=9)
    ax[1].set_xlabel("subcarrier 번호")
    ax[1].set_ylabel("sep = 차이 / 세션간 SD")
    ax[1].set_title("subcarrier 별 분리도. 검은 선이 빨간 선을 넘는 구간이 있는가")
    ax[1].grid(alpha=.3)

    fig.tight_layout()
    path = os.path.join(S.OUT_DIR, "s5_abs_profile_%s.png" % S.BATCH_TAG)
    fig.savefig(path)
    plt.close(fig)
    print("  그림 저장: %s" % path)


def _write_report(a1_policy, min_kinds, n_low, samples, still, empty,
                  a1_rows, a2_rows, a3_rows, sw_sess, sw_subj,
                  scs, rho, slope, mono, order, other_sessions, us, ue, decomp) -> None:
    L: List[str] = []
    A = L.append
    A("# A안 결과 — RSSI 로 복원한 '크기' 정보")
    A("")
    A("정규화(frame_norm)는 AGC 이득과 함께 '신호 세기'도 지운다. 사람 몸의 차폐 효과는")
    A("상당 부분 세기 감소로 나타나므로, RSSI 로 크기를 되살려 정적 점유(still)와")
    A("비점유(empty)가 갈리는지 확인한 것이 이 분석이다.")
    A("")
    A("표본 단위는 **세션 구간**이다. 프레임은 표본이 아니다.")
    A("검정은 정확 순열검정이며 subcarrier 전수 검정에는 BH FDR 보정을 걸었다.")
    A("")
    if not other_sessions:
        A("> **타 배치 부재**: 현재 `data/` 에 배치 %s 외의 세션이 없어" % S.BATCH)
        A("> '타 배치 포함본' 대조를 계산할 수 없다. 아래는 전부 배치 %s 단독 결과이며," % S.BATCH)
        A("> 이는 원래 주 결과로 삼기로 한 조건이다.")
        A("")
    A("구간 표본: still %d개 / empty %d개" % (len(still), len(empty)))
    A("")

    A("## (1) RSSI 해상도 점검과 A-1 처리 방침")
    A("")
    A("ESP32 RSSI 는 1 dBm 단위로 양자화된다. 값 종류가 적으면 평균이 계단처럼 움직여")
    A("평균 비교 자체가 무의미해진다.")
    A("")
    A("| 구간 | 상태 | RSSI 값 종류 | 최소 | 최대 |")
    A("|---|---|---|---|---|")
    for p in samples:
        u = np.unique(p["rssi"])
        A("| %s | %s | %d | %.0f | %.0f |" % (p["session"], p["state"], u.size,
                                              u.min(), u.max()))
    A("")
    A("**판정: 5단계 미만 단위가 %d개였으므로 A-1 은 %s로 처리한다.**"
      % (n_low, a1_policy))
    A("(구간별 RSSI 값 종류의 최솟값 = %d)" % min_kinds)
    A("")

    def tbl(title, rows, note):
        A("### %s" % title)
        A("")
        A("| 특징 | still | empty | 차이 | sep | p | p_min | 참가자단위 sep | 참가자단위 p |")
        A("|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            sj = r.get("subj", {})
            A("| %s | %.3f | %.3f | %+.3f | %.2f | %.4f | %.4f | %.2f | %.4f |"
              % (r["label"], r["mean_a"], r["mean_b"], r["diff"], r["sep"],
                 r["p"], r["p_min"], sj.get("sep", float("nan")),
                 sj.get("p", float("nan"))))
        A("")
        for r in rows:
            sj = r.get("subj", {})
            if not np.isfinite(r["sep"]):
                claim = ("주장 가능: 없음. 두 클래스 모두 변동이 0 이고 차이도 0 이라 "
                         "분리도를 잴 수 없다.")
            else:
                claim = ("주장 가능: 없음. sep %.2f 는 기준 2.0 미만이라 "
                     "클래스 간 차이가 클래스 내 변동에 묻힌다."
                     % r["sep"]) if r["sep"] < SEP_THRESHOLD else (
                    "sep %.2f 로 기준을 넘지만, 참가자 단위 sep 이 %.2f 라 "
                    "%s." % (r["sep"], sj.get("sep", float("nan")),
                             "같은 사람을 여러 번 센 효과로 보인다"
                             if sj.get("sep", 0) < SEP_THRESHOLD else "참가자 단위에서도 유지된다"))
            A("- **%s** — %s" % (r["label"], claim))
        A("")
        if note:
            A(note)
            A("")

    A("## (2) A-1 / A-2 / A-3 스칼라 표")
    A("")
    tbl("A-1 RSSI / 노이즈 플로어 / SNR", a1_rows,
        "> 해상도 판정에 따라 A-1 은 **%s**." % a1_policy)
    tbl("A-2 수신률 / 프레임 간격 / 드롭률", a2_rows, "")
    tbl("A-3 RSSI 보정 진폭 (스칼라 요약)", a3_rows,
        "> `정규화 진폭(대조)` 는 크기 정보를 뺀 값이다. RSSI 보정 진폭만 갈리고 이쪽은 "
        "안 갈린다면 그 차이는 '크기'에서 온 것이다.\n"
        "> 이름 주의: 장비 보정을 거친 물리적 절대값이 아니라 RSSI 로 스케일만 "
        "되살린 값이다(RSSI-scaled amplitude).")

    A("## (3) A-3 RSSI 보정 진폭의 subcarrier 전수 검정")
    A("")
    A("| 항목 | 세션 구간 단위 | 참가자 단위 |")
    A("|---|---|---|")
    if sw_sess and sw_subj:
        A("| 표본 수 (still / empty) | %d / %d | %d / %d |"
          % (sw_sess["na"], sw_sess["nb"], sw_subj["na"], sw_subj["nb"]))
        A("| p 최솟값 | %.4f | %.4f |" % (sw_sess["p"].min(), sw_subj["p"].min()))
        A("| 도달 가능한 하한 p_min | %.4f | %.4f |" % (sw_sess["p_min"], sw_subj["p_min"]))
        A("| p<0.05 개수 (참고용) | %d / %d | %d / %d |"
          % (int((sw_sess["p"] < .05).sum()), sw_sess["p"].size,
             int((sw_subj["p"] < .05).sum()), sw_subj["p"].size))
        A("| **FDR q<0.05 개수 (판정 기준)** | **%d** | **%d** |"
          % (int(sw_sess["rej"].sum()), int(sw_subj["rej"].sum())))
        A("| 완전분리 개수 | %d | %d |"
          % (int(sw_sess["fullsep"].sum()), int(sw_subj["fullsep"].sum())))
        A("| 완전분리 무작위 기댓값 | %.2f | %.2f |"
          % (sw_sess["exp_full"], sw_subj["exp_full"]))
        A("| sep 최댓값 | %.2f | %.2f |"
          % (np.nanmax(sw_sess["sep"]), np.nanmax(sw_subj["sep"])))
        A("")
        A("- **주장할 수 있는 것** — 114개를 훑으면 완전한 잡음이어도 몇 개는 p<0.05 가 된다. "
          "그래서 판정은 FDR q 로 한다. q<0.05 가 세션 단위 %d개, 참가자 단위 %d개다."
          % (int(sw_sess["rej"].sum()), int(sw_subj["rej"].sum())))
        if sw_subj["p_min"] > 0.05:
            A("- **주장할 수 없는 것** — 참가자 단위 하한 p_min 이 %.4f 이므로, "
              "현재 표본 수로는 참가자 단위에서 p<0.05 가 **원리적으로 불가능하다**. "
              "참가자 단위 p 가 0.05 를 넘는 것은 효과가 없다는 증거가 아니라 "
              "검정력이 없다는 뜻이다." % sw_subj["p_min"])
        else:
            A("- 참가자 단위 하한 p_min 이 %.4f 이므로 p<0.05 자체는 도달 가능하다. "
              "다만 실제 p 최솟값은 %.4f 다."
              % (sw_subj["p_min"], sw_subj["p"].min()))
        A("- 완전분리 %d개는 무작위 기댓값 %.2f 와 비교해서 읽어야 한다."
          % (int(sw_sess["fullsep"].sum()), sw_sess["exp_full"]))
    A("")
    A("### 과대해석 점검 — 114개는 독립 시행이 아니다")
    A("")
    A("참가자 단위에서 완전분리가 %d개(무작위 기댓값 %.2f), sep>=2 가 %d개 나왔다."
      % (decomp["n_full0"], sw_subj["exp_full"] if sw_subj else float("nan"),
         decomp["n_sep0"]))
    A("이 숫자를 '독립적인 발견 %d건' 으로 읽으면 안 된다." % decomp["n_full0"])
    A("")
    A("| 항목 | 값 |")
    A("|---|---|")
    A("| subcarrier 쌍 상관 중앙값 | **%.3f** |" % decomp["corr_med"])
    A("| 원본: sep>=2 / 완전분리 / sep최대 | %d / %d / %.2f "
      % (decomp["n_sep0"], decomp["n_full0"], decomp["sep_max0"]) + "|")
    A("| 공통레벨 제거 후: sep>=2 / 완전분리 / sep최대 | %d / %d / %.2f "
      % (decomp["n_sep1"], decomp["n_full1"], decomp["sep_max1"]) + "|")
    A("| 대역 전체 평균 자체의 sep | %.2f |" % decomp["sep_level"])
    A("")
    A("- subcarrier 끼리 상관이 %.3f 이므로 114개는 사실상 소수의 독립 자유도다."
      % decomp["corr_med"])
    _d0, _d1 = decomp["n_sep0"], decomp["n_sep1"]
    if _d1 < _d0:
        A("- 각 표본에서 대역 전체 평균(공통 레벨)을 빼면 sep>=2 가 %d개 → %d개로 줄어든다. "
          "즉 겉보기 분리의 상당 부분은 subcarrier 별 구조가 아니라 **대역 전체 레벨이 "
          "통째로 이동한 하나의 공통 성분**에서 나온 것이다." % (_d0, _d1))
    elif _d1 > _d0:
        A("- 공통 레벨을 빼면 sep>=2 가 %d개 → %d개로 **늘어난다**. 대역 전체 레벨이 "
          "오히려 subcarrier 별 구조를 가리고 있었다는 뜻이다." % (_d0, _d1))
    else:
        A("- 공통 레벨을 빼도 sep>=2 개수가 %d개로 그대로다. 겉보기 분리가 대역 전체 "
          "레벨에서 온 것인지 이 분해만으로는 가르기 어렵다." % _d0)
    _lv = decomp["sep_level"]
    if np.isfinite(_lv) and _lv >= SEP_THRESHOLD:
        A("- 대역 전체 평균 자체의 sep 은 %.2f 로 기준 %.1f 이상이다. 공통 성분 자체를 "
          "따로 확인할 필요가 있다." % (_lv, SEP_THRESHOLD))
    else:
        A("- 그런데 그 대역 전체 평균 자체의 sep 은 %.2f 로 기준 %.1f 에 못 미친다. "
          "따라서 공통 성분도 단독으로는 주장 근거가 되지 못한다." % (_lv, SEP_THRESHOLD))
    _nq = int(sw_subj["rej"].sum()) if sw_subj else 0
    A("- 이 분해는 유의성을 다시 캐려는 재시도가 아니라 이미 나온 개수를 과대해석하지 "
      "않기 위한 것이다. 판정 기준(FDR q<0.05)은 바뀌지 않았고 참가자 단위 %d개다." % _nq)
    A("")
    A("참가자 단위 표본: still %s / empty %s" % (us, ue))
    A("")

    A("## (4) 시간 드리프트 확인")
    A("")
    A("로그의 `local_timestamp` 는 장치 부팅 후 경과 시간이라 세션마다 리셋된다.")
    A("따라서 실제 수집 시각은 로그에서 복원할 수 없고, 아래 순서는 `sessions.py`")
    A("매니페스트 순서를 수집 순서로 **가정**한 것이다.")
    A("")
    A("| # | 구간 | 상태 | RSSI 평균 |")
    A("|---|---|---|---|")
    for i, p in enumerate(order):
        A("| %d | %s | %s | %.3f |" % (i, p["session"], p["state"], p["rssi_mean"]))
    A("")
    A("- 순서 대 RSSI 상관계수: **%+.3f**" % rho)
    A("- 순서당 기울기: **%+.4f dBm/구간**" % slope)
    A("- 단조 증가/감소: **%s**" % ("예 → 드리프트 의심" if mono else "아니오"))
    A("")
    A("- **주장할 수 있는 것** — %s"
      % ("추세가 단조가 아니므로 RSSI 차이가 순수한 시간 드리프트라고 볼 근거는 약하다."
         if not mono else
         "RSSI 가 수집 순서를 따라 단조 변화하므로, 상태 차이와 드리프트가 교락돼 있다."))
    A("- **주장할 수 없는 것** — 수집 순서 자체가 가정이므로 이 점검은 확정적 배제가 아니다. "
      "드리프트를 제대로 배제하려면 상태를 번갈아 수집한 데이터가 필요하다.")
    A("")

    A("## (5) 그림")
    A("")
    A("- `%s/s5_rssi_%s.png` — 왼쪽: 수집 순서에 따른 RSSI(드리프트 확인), "
      % (os.path.basename(S.OUT_DIR), S.BATCH_TAG) +
      "가운데: still/empty 프레임 RSSI 분포, 오른쪽: 구간 SNR")
    A("- `%s/s5_abs_profile_%s.png` — 위: RSSI 보정 진폭 프로파일, "
      % (os.path.basename(S.OUT_DIR), S.BATCH_TAG) +
      "아래: subcarrier 별 sep 과 기준선 2.0")
    A("")

    A("## 판정")
    A("")
    ok_subj = bool(sw_subj and sw_subj["rej"].sum() > 0)
    # A-2(수신률/dt/드롭률/손실률)는 수집 품질 지표이지 상태 특징이 아니다.
    # 이것이 잘 갈린다는 이유로 '점유 구분 근거 있음' 이 나오면 안 되므로 제외한다.
    # A-1 은 RSSI 해상도가 '사용 가능'(서로 다른 값 5종류 이상)일 때만 판정에 넣는다.
    verdict_rows = a3_rows + (a1_rows if a1_policy == "사용 가능" else [])
    scalar_ok = any(_finite_ge(r.get("sep"))
                    and _finite_ge(r.get("subj", {}).get("sep"))
                    for r in verdict_rows)
    if ok_subj and scalar_ok:
        A("**A안: 정적 점유 구분 근거 있음** — 근거는 참가자 단위에서도 FDR q<0.05 인 "
          "subcarrier 가 %d개 남고, 스칼라 특징의 sep 이 참가자 단위에서도 2.0 이상이기 때문."
          % int(sw_subj["rej"].sum()))
    else:
        why = []
        if sw_sess:
            why.append("세션 단위 FDR q<0.05 가 %d개" % int(sw_sess["rej"].sum()))
        if sw_subj:
            why.append("참가자 단위 FDR q<0.05 가 %d개" % int(sw_subj["rej"].sum()))
        why.append("sep 이 기준 2.0 을 넘고 참가자 단위까지 유지되는 스칼라 특징이 %s"
                   % ("있음" if scalar_ok else "없음"))
        A("**A안: 정적 점유 구분 근거 없음** — 근거는 %s이기 때문이다." % ", ".join(why))
    A("")
    A("판정에 쓴 특징은 A-3(RSSI 보정 진폭)%s 이다. **A-2(수신률·프레임 간격·파싱 "
      "드롭률·ID 손실률)는 수집 품질 QC 지표이므로 점유 판정에서 제외했다.** 표와 "
      "그래프에는 계속 나오지만 위 결론에는 들어가지 않는다."
      % (" 와 A-1(RSSI 계열)" if a1_policy == "사용 가능"
         else " 뿐이다 (A-1 은 RSSI 해상도 판정이 '%s' 라 제외)" % a1_policy))
    A("")
    if ok_subj and scalar_ok:
        A("이 결과는 '현재 파일럿 데이터와 RSSI 기반 특징에서 분리가 관찰되었다'는 뜻이다.")
        A("한 번의 파일럿 결과이므로 재현성이 확인된 것으로 간주하지 않는다.")
    else:
        A("이 결과는 '현재 파일럿 데이터와 RSSI 기반 특징에서는 안정적인 분리를 확인하지")
        A("못했다'는 뜻이며, 다른 특징을 포함한 최종 분류 가능성까지 기각된 것은 아니다.")
    A("")
    _pmin = sw_subj["p_min"] if sw_subj else float("nan")
    if np.isfinite(_pmin) and _pmin > 0.05:
        A("**표본 수 한계(반드시 함께 읽을 것)**: 참가자 단위 정확 순열검정의 도달 가능한")
        A("하한 p 는 %.4f 다. 즉 현재 표본 수로는 참가자 단위에서 p<0.05 가 원리적으로"
          % _pmin)
        A("불가능하다. p 가 안 나온다고 특징이나 파라미터를 바꿔 재시도하는 것은 p-hacking")
        A("이므로 하지 않았다. 이 한계를 해소하려면 표본(특히 비점유 세션과 참가자) 수를")
        A("늘려야 한다.")
    else:
        A("**표본 수**: 참가자 단위 하한 p 는 %.4f 로 p<0.05 가 도달 가능한 범위다."
          % _pmin)
        A("그래도 p 가 안 나온다고 특징이나 파라미터를 바꿔 재시도하는 것은 p-hacking")
        A("이므로 하지 않았다.")

    path = os.path.join(S.OUT_DIR, "A_report.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("  보고서 저장: %s" % path)


if __name__ == "__main__":
    raise SystemExit(main())
