# -*- coding: utf-8 -*-
"""
s3b_structure.py — B안: subcarrier 114개가 '어떻게 묶여 움직이는가'

동기
---------------------------------------------------------------------------
지금까지 본 것은 전부 '값이 얼마나 큰가' 였다 (평균 진폭, 시간 표준편차, 변화량).
아직 안 본 것은 subcarrier 들이 서로 어떻게 묶여 움직이는가 하는 구조다.
사람 몸이 있으면 여러 subcarrier 가 '몸' 이라는 하나의 공통 원인으로 묶여 함께
움직일 수 있다. 진폭 크기는 그대로여도 이 묶임 구조는 달라질 수 있고,
그건 상관행렬로만 보인다.

핵심 지표는 유효 랭크 = exp(고유값 분포의 섀넌 엔트로피) 다.
  - 전부 제각각이면 114 에 가까움
  - 하나의 공통 원인으로 묶이면 값이 내려감
사람이 있으면 still 쪽 값이 더 낮게 나올 것을 기대한다.

통계 규칙 (고정)
---------------------------------------------------------------------------
- 표본 단위는 세션 구간. 프레임 단위 검정은 하지 않는다.
- 같은 참가자의 여러 구간을 독립 표본으로 세지 않는다 -> 참가자 단위 재검정 병기.
- 검정은 정확 순열검정. subcarrier 전수 검정에는 BH FDR 보정 필수.
- 임계값은 비점유(empty) 세션에서만 정한다. 점유 데이터로 정하면 순환논증이다.
- 상관행렬은 수신률에 민감하다. 세션 간 수신률 차이를 반드시 함께 보고한다.
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

SEP_THRESHOLD = 2.0
RATE_WARN_RATIO = 1.20      # 세션 간 수신률 최대/최소 비가 이 값을 넘으면 경고

# 프로토콜에서 '자세를 유지하는' 구간. 나머지 15초짜리는 자세 변경이라 하나로 병합한다.
HOLD_EVENTS = ("baseline_supine", "arm_raised_hold")

# 스크립트 프로토콜의 자세변경 구간 수. turn_left / return_supine_1 /
# turn_right / return_supine_2 로 정확히 4개다.
N_POSTURE_CHANGES = 4


# ===========================================================================
# 표본 만들기
# ===========================================================================

def motion_block_from_events(s: core.Session) -> Optional[core.Segment]:
    """
    events 에서 '자세 변경' 구간들을 하나의 움직임 블록으로 병합한다.
    15초짜리가 여러 개 있어도 사이가 이어져 있으므로 [처음 시작, 마지막 끝] 하나가 된다.
    events 가 없으면 None (그 세션은 segments 의 motion 라벨을 쓴다).
    """
    if not s.events:
        return None
    mv = [e for e in s.events if e.label not in HOLD_EVENTS]
    if not mv:
        return None
    return core.Segment("motion", min(e.t0 for e in mv), max(e.t1 for e in mv))


def build_samples(sessions: List[core.Session]) -> Tuple[List[Dict[str, object]], np.ndarray]:
    """
    세션 x 구간마다 표본 1개. 전처리는 s2/s3 와 동일하게 맞춘다:
        amplitude -> normalize(frame_norm) -> hampel
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
        X, _ = core.hampel(nrm[:, mask], window=S.HAMPEL_WINDOW,
                           n_sigma=S.HAMPEL_SIGMA)

        # 움직임 블록은 events 에서 병합해 만든다. 없으면 segments 를 그대로 쓴다.
        mb = motion_block_from_events(s)
        segs: List[core.Segment] = []
        for sg in s.segments:
            if sg.label == "motion" and mb is not None:
                continue
            segs.append(sg)
        if mb is not None:
            segs.append(mb)
        segs.sort(key=lambda g: g.t0)

        for sg in segs:
            m = sg.mask(s.t, guard=S.SEGMENT_GUARD_SEC)
            n = int(m.sum())
            if n < 200:
                continue
            Xi = X[m]
            ti = s.t[m]

            eig = core.eig_spectrum(Xi)
            k = eig.size
            tot = float(np.nansum(eig))
            C = np.corrcoef(Xi, rowvar=False)
            iu = np.triu_indices(C.shape[0], 1)
            off = C[iu]

            t_mid, d, dt_med = core.diff_matched_dt(ti, Xi)
            mad_series = d.mean(axis=1)

            flat = Xi.reshape(-1)
            from scipy.stats import skew, kurtosis
            out.append({
                "session": s.name, "state": sg.label, "subject": s.subject,
                "batch": s.batch, "n": n, "t0": sg.t0, "t1": sg.t1,
                "rate_hz": float(n / (ti[-1] - ti[0])),
                # 3) 상관 구조
                "eig": eig,
                "eig1": float(eig[0] / tot),
                "eig5": float(eig[:5].sum() / tot),
                "eff_rank": core.effective_rank_entropy(eig),
                "corr_mean": float(off.mean()),
                "corr_sd": float(off.std(ddof=1)),
                "corr_absmean": float(np.abs(off).mean()),
                "corr_mat": C,
                # 2) subcarrier 별 통계량
                "median": np.median(Xi, axis=0),
                "iqr": (np.percentile(Xi, 75, axis=0) - np.percentile(Xi, 25, axis=0)),
                "mean": Xi.mean(axis=0),
                "std": Xi.std(axis=0, ddof=1),
                # 4) 시간 곡선
                "mad_series": mad_series,
                "coh_time": core.coherence_time(Xi, fs=1.0 / dt_med),
                "mad_kurt": float(kurtosis(mad_series, fisher=True, bias=False)),
                # 5) 진폭 분포
                "amp_skew": float(np.median(skew(Xi, axis=0, bias=False))),
                "amp_kurt": core.amp_kurtosis(Xi),
                "q10": float(np.percentile(flat, 10)),
                "q50": float(np.percentile(flat, 50)),
                "q90": float(np.percentile(flat, 90)),
            })
    return out, mask


# ===========================================================================
# 검정 도구 (A안과 동일한 규칙)
# ===========================================================================

def perm_line(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 1 or b.size < 1:
        return {"na": int(a.size), "nb": int(b.size), "mean_a": float("nan"),
                "mean_b": float("nan"), "diff": float("nan"), "sep": float("nan"),
                "p": float("nan"), "p_min": float("nan"), "full": False}
    obs, p, pmin = core.exact_perm_test_unpaired(a, b)
    sep = core.sep_ratio(a, b)
    # 양쪽 다 변동 0 이고 차이도 0 이면 '완벽한 분리'가 아니라 '잴 것이 없음'이다.
    if not np.isfinite(sep) and abs(obs) < 1e-12:
        sep = float("nan")
    full = bool(a.min() > b.max() or a.max() < b.min())
    return {"na": int(a.size), "nb": int(b.size),
            "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "diff": obs, "sep": sep, "p": p, "p_min": pmin, "full": full}


def unit_of(p: Dict[str, object]) -> str:
    """참가자 단위 식별자. 비점유는 참가자가 없으므로 세션 이름을 단위로 쓴다."""
    return p["session"] if p["subject"] in ("", "-") else str(p["subject"])


def collapse(samples, state, key):
    by: Dict[str, List[float]] = {}
    for p in samples:
        if p["state"] != state:
            continue
        by.setdefault(unit_of(p), []).append(p[key])
    names = sorted(by)
    return names, np.array([float(np.median(by[n])) for n in names])


def collapse_prof(samples, state, key):
    by: Dict[str, List[np.ndarray]] = {}
    for p in samples:
        if p["state"] != state:
            continue
        by.setdefault(unit_of(p), []).append(p[key])
    names = sorted(by)
    return names, np.array([np.median(np.stack(by[n]), axis=0) for n in names])


def _verdict(r) -> str:
    if not np.isfinite(r["sep"]):
        return "변동없음(판정불가)"
    if r["sep"] < SEP_THRESHOLD:
        return "근거없음(sep<2)"
    if not r["full"]:
        return "sep>=2 이나 완전분리 아님"
    if r["p"] <= r["p_min"] + 1e-12:
        return "sep>=2, 완전분리, p는 하한값(유의가 아님)"
    return "sep>=2, 완전분리, p=%.4f" % r["p"]


def scalar_table(title: str, samples, keys, a="still", b="empty"):
    print()
    print(LINE)
    print(title)
    print(LINE)
    hdr = "%-16s %10s %10s %10s %7s %6s %8s %8s   %s"
    print(hdr % ("지표", a, b, "차이", "sep", "완전분리", "p", "p_min", "판정"))
    print(SUB)
    rows = []
    for key, label in keys:
        va = [p[key] for p in samples if p["state"] == a]
        vb = [p[key] for p in samples if p["state"] == b]
        r = perm_line(va, vb)
        r["key"], r["label"] = key, label
        r["verdict"] = _verdict(r)
        print(hdr % (label, "%10.4f" % r["mean_a"], "%10.4f" % r["mean_b"],
                     "%+10.4f" % r["diff"], "%7.2f" % r["sep"],
                     str(r["full"]), "%8.4f" % r["p"], "%8.4f" % r["p_min"],
                     r["verdict"]))
        rows.append(r)

    print(SUB)
    print("참가자 단위 재검정 (같은 사람의 여러 구간을 중앙값으로 합친 뒤 재검정)")
    print(SUB)
    print(hdr % ("지표", a, b, "차이", "sep", "완전분리", "p", "p_min", "판정"))
    ua = ub = []
    for key, label in keys:
        ua, va = collapse(samples, a, key)
        ub, vb = collapse(samples, b, key)
        r = perm_line(va, vb)
        r["verdict"] = _verdict(r)
        print(hdr % (label, "%10.4f" % r["mean_a"], "%10.4f" % r["mean_b"],
                     "%+10.4f" % r["diff"], "%7.2f" % r["sep"],
                     str(r["full"]), "%8.4f" % r["p"], "%8.4f" % r["p_min"],
                     r["verdict"]))
        for row in rows:
            if row["key"] == key:
                row["subj"] = r
    print("  참가자 단위 표본: %s %s / %s %s" % (a, ua, b, ub))
    return rows


def sweep(prof_a: np.ndarray, prof_b: np.ndarray, scs: np.ndarray, label: str):
    """subcarrier 전수 검정 한 줄."""
    na, nb = prof_a.shape[0], prof_b.shape[0]
    ps, seps, full = [], [], []
    for j in range(prof_a.shape[1]):
        x, y = prof_a[:, j], prof_b[:, j]
        _, p, _ = core.exact_perm_test_unpaired(x, y)
        ps.append(p)
        seps.append(core.sep_ratio(x, y))
        full.append(bool(x.min() > y.max() or x.max() < y.min()))
    ps = np.asarray(ps); seps = np.asarray(seps); full = np.asarray(full, bool)
    rej, q = core.benjamini_hochberg(ps, q=0.05)
    exp_full = ps.size * 2.0 / comb(na + nb, nb)
    return {"label": label, "p": ps, "q": q, "sep": seps, "full": full,
            "rej": rej, "exp_full": exp_full, "na": na, "nb": nb,
            "p_min": core.min_two_sided_p_unpaired(na, nb),
            "best_sc": int(scs[int(np.nanargmax(seps))])}


def print_sweep_table(title: str, rows: List[Dict[str, object]]):
    print()
    print(LINE)
    print(title)
    print(LINE)
    print("%-10s %7s %10s %10s %14s %12s %12s %8s"
          % ("통계량", "n(a/b)", "p최솟값", "p<0.05", "★FDR q<0.05", "완전분리",
             "무작위기댓값", "sep최대"))
    print(SUB)
    for r in rows:
        print("%-10s %3d/%-3d %10.4f %10d %14d %12d %12.2f %8.2f"
              % (r["label"], r["na"], r["nb"], r["p"].min(),
                 int((r["p"] < .05).sum()), int(r["rej"].sum()),
                 int(r["full"].sum()), r["exp_full"], np.nanmax(r["sep"])))
    print(SUB)
    print("  도달 가능한 하한 p_min = %.4f  (실제 p 가 이 값이면 '가장 극단적 배치'라는 뜻)"
          % rows[0]["p_min"])


# ===========================================================================
# 실행
# ===========================================================================

def main() -> int:
    plt = core.setup_matplotlib()
    allsess = S.load_all(verbose=False)
    # 주 분석 배치는 CSI_BATCH 로 정해진다 (미설정이면 "B" = 기존 동작).
    # 0개면 sessions_of_batch() 가 예외를 던진다. 표본 0개로 '성공'하면 안 된다.
    sessB = S.sessions_of_batch(allsess)
    sessA = [s for s in allsess if s.batch.upper() != S.BATCH]

    print(LINE)
    print("s3b_structure.py  B안: subcarrier 상관 구조로 정적 점유를 구분할 수 있는가")
    print(LINE)
    print("  전체 세션 %d개 (배치 %s %d, 그 외 %d)"
          % (len(allsess), S.BATCH, len(sessB), len(sessA)))

    samples, mask = build_samples(sessB)
    scs = core.subcarrier_numbers(192, mask)
    K = int(mask.sum())

    # =====================================================================
    # 3-1. 표본 구성
    # =====================================================================
    print()
    print(LINE)
    print("[1] 표본 구성")
    print(LINE)
    print("  %-14s %-8s %-6s %8s %11s %9s" % ("세션", "상태", "subj", "n", "구간[s]", "수신률"))
    for p in samples:
        print("  %-14s %-8s %-6s %8d %5.0f-%5.0f %8.3f"
              % (p["session"], p["state"], p["subject"], p["n"],
                 p["t0"], p["t1"], p["rate_hz"]))

    st = {k: [p for p in samples if p["state"] == k] for k in ("still", "empty", "motion")}
    subj_of = {k: sorted({unit_of(p) for p in v}) for k, v in st.items()}
    print(SUB)
    for k in ("still", "motion", "empty"):
        print("  %-7s 구간 %2d개, 단위 %d개 %s"
              % (k, len(st[k]), len(subj_of[k]), subj_of[k]))
    print()
    print("  움직임 블록은 events 에서 자세변경 구간을 병합해 만들었다.")
    # events 를 가진 첫 세션에서 자세변경 라벨을 뽑는다.
    # 예전에는 sessB[3] 로 인덱스를 박아 두어 세션 수가 바뀌면 깨졌다.
    _ev_src = next((x for x in sessB if x.events), None)
    ev = ([e.label for e in _ev_src.events if e.label not in HOLD_EVENTS]
          if _ev_src is not None else [])
    print("  병합 대상 events: %s -> 세션당 1개 블록" % (ev,))
    if len(ev) != N_POSTURE_CHANGES:
        print("  * 주의: 자세변경 event 가 %d개다. 프로토콜상 %d개가 정상이다"
              % (len(ev), N_POSTURE_CHANGES))
        print("    (turn_left / return_supine_1 / turn_right / return_supine_2).")
        print("    baseline_supine 과 arm_raised_hold 는 '유지' 구간이라 still 로 남는다.")
    if len(st["motion"]) != 3:
        print("  * 경고: motion 블록이 %d개다. 3개가 정상이다." % len(st["motion"]))

    # =====================================================================
    # 수신률 점검 (상관 구조 신뢰도)
    # =====================================================================
    rates = np.array([p["rate_hz"] for p in samples])
    ratio = float(rates.max() / rates.min())
    print()
    print(LINE)
    print("[수신률 점검]  상관행렬은 리샘플링과 수신률에 민감하다")
    print(LINE)
    print("  구간 수신률 최소 %.3f Hz / 최대 %.3f Hz / 비 %.4f" % (rates.min(), rates.max(), ratio))
    if ratio >= RATE_WARN_RATIO:
        print("  ** 경고: 세션 간 수신률 차이가 %.1f%% 로 20%% 이상이다." % ((ratio - 1) * 100))
        print("     상관 구조 차이가 상태가 아니라 수신률 차이에서 왔을 수 있다.")
    else:
        print("  차이 %.1f%% 로 20%% 미만이다. 상관 구조 비교의 전제 조건은 충족한다."
              % ((ratio - 1) * 100))

    # =====================================================================
    # 3-3. subcarrier 별 통계량
    # =====================================================================
    sweeps_sess, sweeps_subj = [], []
    for key in ("median", "iqr", "mean", "std"):
        pa = np.array([p[key] for p in st["still"]])
        pb = np.array([p[key] for p in st["empty"]])
        sweeps_sess.append(sweep(pa, pb, scs, key))
        _, qa = collapse_prof(samples, "still", key)
        _, qb = collapse_prof(samples, "empty", key)
        sweeps_subj.append(sweep(qa, qb, scs, key))
    print_sweep_table("[2] subcarrier 별 통계량 전수 검정 (%d개) — 세션 구간 단위" % K,
                      sweeps_sess)
    print_sweep_table("[2] subcarrier 별 통계량 전수 검정 (%d개) — 참가자 단위" % K,
                      sweeps_subj)

    # =====================================================================
    # 3-2. 상관 구조
    # =====================================================================
    corr_rows = scalar_table(
        "[3] 상관 구조 — still 대 empty",
        samples,
        [("eig1", "제1고유값비율"), ("eig5", "상위5비율"),
         ("eff_rank", "유효랭크"), ("corr_mean", "상관 평균"),
         ("corr_sd", "상관 표준편차"), ("corr_absmean", "상관 절대평균")])

    # 대조: motion 은 갈려야 정상이다 (지표가 작동하는지 확인하는 양성 대조)
    mot_rows = scalar_table(
        "[3-대조] 상관 구조 — motion 대 empty  (지표가 작동하는지 보는 양성 대조)",
        samples,
        [("eig1", "제1고유값비율"), ("eig5", "상위5비율"),
         ("eff_rank", "유효랭크"), ("corr_mean", "상관 평균"),
         ("corr_sd", "상관 표준편차"), ("corr_absmean", "상관 절대평균")],
        a="motion", b="empty")

    # =====================================================================
    # 3-4. 시간 곡선 / 진폭 분포
    # =====================================================================
    # 임계값은 비점유 세션에서만 정한다 (순환논증 방지).
    pooled_empty = np.concatenate([p["mad_series"] for p in st["empty"]])
    med = float(np.median(pooled_empty))
    mad = float(np.median(np.abs(pooled_empty - med)))
    thr = med + S.MOTION_MAD_K * 1.4826 * mad
    print()
    print(LINE)
    print("[4] 시간 곡선 — 임계 초과 비율 / 상관 길이 / 첨도")
    print(LINE)
    print("  임계값은 비점유(empty) 세션에서만 산출했다 (점유로 정하면 순환논증).")
    print("  empty 풀링 mad_diff: 중앙값 %.5f, MAD %.5f -> 임계 %.5f (K=%.1f)"
          % (med, mad, thr, S.MOTION_MAD_K))
    for p in samples:
        p["exceed"] = float((p["mad_series"] > thr).mean())

    time_rows = scalar_table(
        "[4] 시간 곡선 — still 대 empty",
        samples,
        [("exceed", "임계초과비율"), ("coh_time", "상관길이[s]"),
         ("mad_kurt", "mad_diff첨도")])

    amp_rows = scalar_table(
        "[5] 진폭 분포 — still 대 empty",
        samples,
        [("amp_skew", "왜도"), ("amp_kurt", "첨도"),
         ("q10", "10분위"), ("q50", "50분위"), ("q90", "90분위")])

    # --- 다중비교 주의 ---
    scal = corr_rows + time_rows + amp_rows
    m = len(scal)
    n_hit = sum(1 for r in scal if np.isfinite(r["p"]) and r["p"] < 0.05)
    n_hit_s = sum(1 for r in scal
                  if np.isfinite(r.get("subj", {}).get("p", np.nan))
                  and r["subj"]["p"] < 0.05)
    p_any = 1.0 - 0.95 ** m
    print()
    print(LINE)
    print("[다중비교 주의]")
    print(LINE)
    print("  스칼라 지표 %d개를 검정했다 (상관구조 %d + 시간곡선 %d + 진폭분포 %d)."
          % (m, len(corr_rows), len(time_rows), len(amp_rows)))
    print("  세션 단위 p<0.05 인 지표 : %d개" % n_hit)
    print("  참가자 단위 p<0.05 인 지표: %d개" % n_hit_s)
    print("  전부 무관해도 %d개 중 하나 이상이 p<0.05 가 될 확률 = 1 - 0.95^%d = %.3f"
          % (m, m, p_any))
    print("  (지표들이 서로 상관돼 있으므로 이 값은 상한이 아니라 참고선이다.)")

    # =====================================================================
    # 3-6. 타 배치 대조
    # =====================================================================
    print()
    print(LINE)
    print("[6] 타 배치 포함 / 제외 대조")
    print(LINE)
    if not sessA:
        print("  배치 %s 외의 세션이 data/ 에 없다." % S.BATCH)
        print("  따라서 '타 배치 포함본'을 계산할 수 없고, 위 결과는 전부 배치 %s 단독이다."
              % S.BATCH)
        print("  대조 자체가 불가능하므로 '배치 차이가 만든 효과'를 배제했다고 말할 수 없다.")
    else:
        samples_all, _ = build_samples(allsess)
        print("  타 배치 포함 표본으로 상관 구조를 재검정한다.")
        scalar_table("[6] 상관 구조 — 타 배치 포함본", samples_all,
                     [("eig1", "제1고유값비율"), ("eff_rank", "유효랭크"),
                      ("corr_absmean", "상관 절대평균")])

    # =====================================================================
    # 그림
    # =====================================================================
    corr_dmax = _fig_corr(plt, st, scs)
    _fig_eigen(plt, st, K)

    # =====================================================================
    # 보고서 + 판정
    # =====================================================================
    ok = _decide(corr_rows, sweeps_sess, sweeps_subj)
    ok["corr_dmax"] = corr_dmax
    print()
    print("  상관행렬 차이(still - empty) 최대 |차이| = %.3f  (상관 범위는 ±1)" % corr_dmax)
    _write_report(samples, st, subj_of, ratio, rates, sweeps_sess, sweeps_subj,
                  corr_rows, mot_rows, time_rows, amp_rows, m, n_hit, n_hit_s,
                  p_any, sessA, thr, med, mad, ok, K, ev)

    print()
    print(LINE)
    print(ok["line"])
    print(LINE)
    return 0


def _decide(corr_rows, sweeps_sess, sweeps_subj) -> Dict[str, object]:
    """상관 구조 지표 중 sep>=2 이고 완전분리이며 참가자 단위까지 버티는 것이 있는가."""
    strong = []
    for r in corr_rows:
        sj = r.get("subj", {})
        if (np.isfinite(r["sep"]) and r["sep"] >= SEP_THRESHOLD and r["full"]
                and np.isfinite(sj.get("sep", np.nan))
                and sj["sep"] >= SEP_THRESHOLD and sj.get("full")):
            strong.append(r["label"])
    q_sess = sum(int(s["rej"].sum()) for s in sweeps_sess)
    q_subj = sum(int(s["rej"].sum()) for s in sweeps_subj)
    if strong and q_subj > 0:
        line = ("B안: 정적 점유 구분 근거 있음 - 근거는 상관 구조 지표 %s 가 세션·참가자 "
                "양쪽에서 sep>=2 이고 완전분리이며, subcarrier 전수 검정에서 FDR q<0.05 가 "
                "%d개 남기 때문" % (", ".join(strong), q_subj))
    else:
        why = []
        why.append("세션 단위 FDR q<0.05 총 %d개" % q_sess)
        why.append("참가자 단위 FDR q<0.05 총 %d개" % q_subj)
        why.append("상관 구조 지표 중 세션·참가자 양쪽에서 sep>=2 이고 완전분리인 것이 %s"
                   % (", ".join(strong) if strong else "없음"))
        line = "B안: 정적 점유 구분 근거 없음 - 근거는 %s이기 때문" % ", ".join(why)
    return {"line": line, "strong": strong, "q_sess": q_sess, "q_subj": q_subj}


def _fig_corr(plt, st, scs) -> float:
    o = np.argsort(scs)
    Cs = np.mean([p["corr_mat"] for p in st["still"]], axis=0)[np.ix_(o, o)]
    Ce = np.mean([p["corr_mat"] for p in st["empty"]], axis=0)[np.ix_(o, o)]
    D = Cs - Ce
    vmax = float(np.abs(D).max())

    fig, ax = plt.subplots(1, 4, figsize=(21, 5.2))
    ext = [scs[o][0], scs[o][-1], scs[o][-1], scs[o][0]]
    for a, M, t in ((ax[0], Cs, "still 평균 상관행렬"), (ax[1], Ce, "empty 평균 상관행렬")):
        im = a.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r", extent=ext)
        a.set_title(t)
        a.set_xlabel("subcarrier"); a.set_ylabel("subcarrier")
        fig.colorbar(im, ax=a, fraction=.046)

    # 판정용 패널은 왼쪽 두 장과 '같은 스케일'(±1) 이어야 한다.
    # 차이만 자동 스케일로 늘리면 0.1 짜리 차이가 ±1 구조처럼 보여 눈속임이 된다.
    im = ax[2].imshow(D, vmin=-1, vmax=1, cmap="RdBu_r", extent=ext)
    ax[2].set_title("★판정 패널: 차이 (still - empty)\n왼쪽 두 장과 같은 ±1 스케일\n"
                    "전면 흰색이면 상관 구조 차이 없음")
    ax[2].set_xlabel("subcarrier"); ax[2].set_ylabel("subcarrier")
    fig.colorbar(im, ax=ax[2], fraction=.046)

    # 숨기지는 않는다. 확대해서 보면 어떤 모양인지 옆에 같이 둔다.
    im = ax[3].imshow(D, vmin=-vmax, vmax=vmax, cmap="RdBu_r", extent=ext)
    ax[3].set_title("같은 차이를 자동 스케일로 확대\n최대 |차이| = %.3f (상관 범위는 ±1)\n"
                    "판정에 쓰면 안 된다" % vmax)
    ax[3].set_xlabel("subcarrier"); ax[3].set_ylabel("subcarrier")
    fig.colorbar(im, ax=ax[3], fraction=.046)

    fig.suptitle("B안 결론 그림: subcarrier 상관 구조 (배치 %s)" % S.BATCH, fontsize=13)
    fig.tight_layout()
    path = os.path.join(S.OUT_DIR, "s3b_corr_diff_%s.png" % S.BATCH_TAG)
    fig.savefig(path); plt.close(fig)
    print("  그림 저장: %s" % path)
    return vmax


def _fig_eigen(plt, st, K) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5))
    x = np.arange(1, K + 1)
    for state in ("still", "empty", "motion"):
        for i, p in enumerate(st[state]):
            e = p["eig"] / p["eig"].sum()
            ax[0].plot(x[:e.size], e, color=core.STATE_COLOR[state], alpha=.55,
                       lw=1.2, label=state if i == 0 else None)
            ax[1].plot(x[:e.size], np.cumsum(e), color=core.STATE_COLOR[state],
                       alpha=.55, lw=1.2, label=state if i == 0 else None)
    ax[0].set_yscale("log")
    core.log_tick_formatter(ax[0].yaxis)
    ax[0].set_xlabel("고유값 순서"); ax[0].set_ylabel("고유값 비율 (log)")
    ax[0].set_title("상관행렬 고유값 스펙트럼")
    ax[1].set_xlabel("고유값 순서"); ax[1].set_ylabel("누적 비율")
    ax[1].set_title("누적 설명 비율")
    for a in ax:
        a.grid(alpha=.3); a.legend(fontsize=9)
    fig.suptitle("고유값 스펙트럼 — 선이 갈리면 묶임 구조가 다르다 (배치 %s)" % S.BATCH,
                 fontsize=12)
    fig.tight_layout()
    path = os.path.join(S.OUT_DIR, "s3b_eigen_%s.png" % S.BATCH_TAG)
    fig.savefig(path); plt.close(fig)
    print("  그림 저장: %s" % path)


def _write_report(samples, st, subj_of, ratio, rates, sw_s, sw_j,
                  corr_rows, mot_rows, time_rows, amp_rows, m, n_hit, n_hit_s,
                  p_any, sessA, thr, med, mad, ok, K, ev) -> None:
    L: List[str] = []
    A = L.append
    A("# B안 결과 — subcarrier 상관 구조")
    A("")
    A("지금까지 본 것은 전부 '값이 얼마나 큰가'였다. 이 분석이 보는 것은 subcarrier %d개가"
      % K)
    A("**서로 어떻게 묶여 움직이는가**다. 사람 몸이라는 하나의 공통 원인이 있으면 여러")
    A("subcarrier 가 함께 움직여 묶임 구조가 달라질 수 있고, 그건 상관행렬로만 보인다.")
    A("")
    A("핵심 지표는 **유효 랭크 = exp(고유값 분포의 섀넌 엔트로피)** 다. %d개가 실질적으로"
      % K)
    A("몇 개의 독립된 방향으로 움직이는지를 하나의 수로 잰 것이다. 사람이 있으면 still 쪽")
    A("값이 더 낮게 나올 것을 기대했다.")
    A("")

    A("## (1) 표본 구성")
    A("")
    A("| 상태 | 구간 수 | 독립 단위 수 | 단위 |")
    A("|---|---|---|---|")
    for k in ("still", "motion", "empty"):
        A("| %s | %d | %d | %s |" % (k, len(st[k]), len(subj_of[k]), ", ".join(subj_of[k])))
    A("")
    A("움직임 블록은 `events` 의 자세변경 구간(%s)을" % ", ".join(ev))
    A("세션당 하나로 병합해 만들었다. motion 블록 %d개가 나왔다." % len(st["motion"]))
    A("")
    A("> 프로토콜의 자세변경 구간은 **%d개**다. `baseline_supine`(120s)과 "
      "`arm_raised_hold`(119s)는 자세를 '유지'하는 구간이라 still 로 남는다."
      % N_POSTURE_CHANGES)
    A("")
    A("- **주장할 수 있는 것** — still 구간 %d개는 참가자 %d명에서 나온 것이다. "
      "구간 수를 표본 수로 읽으면 안 된다."
      % (len(st["still"]), len(subj_of["still"])))
    A("- **주장할 수 없는 것** — empty 는 단위가 %d개뿐이라 어떤 검정도 검정력이 거의 없다."
      % len(subj_of["empty"]))
    A("")

    A("## (6) 세션 간 수신률 차이")
    A("")
    A("상관행렬은 리샘플링과 수신률에 민감하다.")
    A("")
    A("- 구간 수신률: 최소 **%.3f Hz** / 최대 **%.3f Hz** / 비 **%.4f** (차이 %.1f%%)"
      % (rates.min(), rates.max(), ratio, (ratio - 1) * 100))
    if ratio >= RATE_WARN_RATIO:
        A("- **경고: 20% 이상이다.** 상관 구조 차이가 상태가 아니라 수신률 차이에서 "
          "왔을 가능성을 배제할 수 없다.")
    else:
        A("- 20%% 미만이므로 상관 구조 비교의 전제 조건은 충족한다. "
          "수신률이 결과를 만들었다고 볼 근거는 없다.")
    A("")

    def tbl(title, rows, a="still", b="empty"):
        A("### %s" % title)
        A("")
        A("| 지표 | %s | %s | 차이 | sep | 완전분리 | p | p_min | 참가자 sep | 참가자 p |"
          % (a, b))
        A("|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            sj = r.get("subj", {})
            A("| %s | %.4f | %.4f | %+.4f | %.2f | %s | %.4f | %.4f | %.2f | %.4f |"
              % (r["label"], r["mean_a"], r["mean_b"], r["diff"], r["sep"],
                 "O" if r["full"] else "X", r["p"], r["p_min"],
                 sj.get("sep", float("nan")), sj.get("p", float("nan"))))
        A("")

    A("## (2) 상관 구조 6개 지표")
    A("")
    tbl("still 대 empty", corr_rows)
    for r in corr_rows:
        sj = r.get("subj", {})
        if not np.isfinite(r["sep"]) or r["sep"] < SEP_THRESHOLD:
            A("- **%s** — sep %.2f 로 기준 2.0 미만. 이 숫자로는 아무것도 주장할 수 없다."
              % (r["label"], r["sep"]))
        elif not r["full"]:
            A("- **%s** — sep %.2f 로 기준은 넘지만 완전분리가 아니다(두 집단 값이 겹친다). "
              "분리를 주장할 수 없다." % (r["label"], r["sep"]))
        else:
            A("- **%s** — sep %.2f, 완전분리. 다만 참가자 단위 sep %.2f / p %.4f 를 "
              "함께 봐야 한다." % (r["label"], r["sep"], sj.get("sep", float("nan")),
                                sj.get("p", float("nan"))))
    A("")
    A("### 양성 대조: motion 대 empty")
    A("")
    A("지표 자체가 작동하는지 확인하기 위한 대조다. 움직임에서도 안 갈리면 지표가 무딘 것이다.")
    A("")
    tbl("motion 대 empty", mot_rows, a="motion", b="empty")

    A("## (3) subcarrier 별 median / IQR / mean / std 전수 검정")
    A("")
    for tag, sws in (("세션 구간 단위", sw_s), ("참가자 단위", sw_j)):
        A("**%s** (하한 p_min = %.4f)" % (tag, sws[0]["p_min"]))
        A("")
        A("| 통계량 | p 최솟값 | p<0.05 | **FDR q<0.05** | 완전분리 | 무작위 기댓값 | sep 최대 |")
        A("|---|---|---|---|---|---|---|")
        for r in sws:
            A("| %s | %.4f | %d | **%d** | %d | %.2f | %.2f |"
              % (r["label"], r["p"].min(), int((r["p"] < .05).sum()),
                 int(r["rej"].sum()), int(r["full"].sum()), r["exp_full"],
                 np.nanmax(r["sep"])))
        A("")
    A("- **주장할 수 있는 것** — 판정 기준은 FDR q<0.05 다. 세션 단위 합계 %d개, "
      "참가자 단위 합계 %d개." % (ok["q_sess"], ok["q_subj"]))
    A("- **주장할 수 없는 것** — 완전분리 개수는 반드시 무작위 기댓값과 비교해서 읽어야 하고, "
      "subcarrier 들은 서로 강하게 상관돼 있어 %d개를 독립 시행으로 세면 안 된다." % K)
    A("- median 을 추가한 이유는 분포가 치우쳐 있으면 평균보다 중앙값이 더 잘 갈릴 수 "
      "있어서다. 실제로 그런 이득이 있었는지는 위 표의 mean 행과 median 행을 비교하면 된다.")
    A("")

    A("## (4) 시간 곡선 / 진폭 분포와 다중비교 주의")
    A("")
    A("임계값은 **비점유(empty) 세션에서만** 산출했다. 점유 데이터를 보고 정하면 순환논증이다.")
    A("empty 풀링 mad_diff 중앙값 %.5f, MAD %.5f → 임계 %.5f (K=%.1f)"
      % (med, mad, thr, S.MOTION_MAD_K))
    A("")
    tbl("시간 곡선", time_rows)
    tbl("진폭 분포", amp_rows)
    A("**다중비교 주의**")
    A("")
    A("- 스칼라 지표 **%d개**를 검정했다 (상관구조 %d + 시간곡선 %d + 진폭분포 %d)."
      % (m, len(corr_rows), len(time_rows), len(amp_rows)))
    A("- 세션 단위 p<0.05 인 지표 %d개 / 참가자 단위 %d개." % (n_hit, n_hit_s))
    A("- 전부 무관해도 %d개 중 하나 이상이 p<0.05 가 될 확률은 1 − 0.95^%d = **%.3f** 다. "
      "부수 항목에서만 유의가 나오면 다중비교를 먼저 의심해야 한다." % (m, m, p_any))
    A("- 지표들이 서로 상관돼 있어 이 값은 엄밀한 상한이 아니라 참고선이다.")
    A("")

    A("## (5) 타 배치 포함 / 제외 대조")
    A("")
    if not sessA:
        A("현재 `data/` 에 배치 %s 외의 세션이 없어 **타 배치 포함본을 계산할 수 없다.**"
          % S.BATCH)
        A("위 결과는 전부 배치 %s 단독이다." % S.BATCH)
        A("")
        A("- **주장할 수 없는 것** — 대조 자체가 불가능하므로 '이 결과가 배치 차이 때문이 "
          "아니다'라고 말할 수 없다. 다른 배치가 준비되면 이 항목을 다시 채워야 한다.")
    else:
        A("타 배치를 포함한 재검정 결과는 본문 출력의 [6] 절을 참조.")
    A("")

    A("## 그림")
    A("")
    _dm = ok["corr_dmax"]
    _panel = ("사실상 전면 흰색이다" if _dm < 0.2
              else "눈에 띄는 색이 남아 있다")
    A("- `%s/s3b_corr_diff_%s.png` — **이 그림 한 장이 B안의 결론이다.** "
      % (os.path.basename(S.OUT_DIR), S.BATCH_TAG) +
      "세 번째 '판정 패널'은 왼쪽 두 장과 같은 ±1 스케일이며 %s. "
      "네 번째 패널은 같은 차이를 자동 스케일로 확대한 것인데, 최대 |차이|가 "
      "**%.3f** 로 상관 범위 ±1 의 %.0f%% 수준이다. 확대 패널은 스케일이 다르므로 "
      "판정에 쓰면 안 된다." % (_panel, _dm, _dm * 100))
    A("- `%s/s3b_eigen_%s.png` — 고유값 스펙트럼. 파랑(still)과 빨강(empty)이 "
      % (os.path.basename(S.OUT_DIR), S.BATCH_TAG) +
      "갈라지는지, 초록(motion)은 갈리는지.")
    A("")

    A("## 판정")
    A("")
    A("**%s**" % ok["line"])
    A("")
    if ok["strong"] and ok["q_subj"] > 0:
        A("이 결과는 '현재 파일럿 데이터의 상관 구조에서 분리가 관찰되었다'는 뜻이다.")
        A("한 번의 파일럿 결과이므로 재현성이 확인된 것으로 간주하지 않는다.")
    else:
        A("이 결과는 '현재 파일럿 데이터와 상관 구조 특징에서는 안정적인 분리를 확인하지")
        A("못했다'는 뜻이며, 다른 특징을 포함한 최종 분류 가능성까지 기각된 것은 아니다.")
    A("")
    _pmin = sw_j[0]["p_min"]
    if np.isfinite(_pmin) and _pmin > 0.05:
        A("**표본 수 한계**: 참가자 단위 정확 순열검정의 도달 가능한 하한 p 는 %.4f 다."
          % _pmin)
        A("즉 현재 표본 수로는 참가자 단위에서 p<0.05 가 원리적으로 불가능하다. p 가 안")
        A("나온다고 특징이나 파라미터를 바꿔 재시도하는 것은 p-hacking 이므로 하지 않았다.")
    else:
        A("**표본 수**: 참가자 단위 하한 p 는 %.4f 로 p<0.05 가 도달 가능한 범위다."
          % _pmin)
        A("그래도 특징이나 파라미터를 바꿔가며 재시도하지 않았다 (p-hacking 금지).")

    path = os.path.join(S.OUT_DIR, "B_report.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("  보고서 저장: %s" % path)


if __name__ == "__main__":
    raise SystemExit(main())
