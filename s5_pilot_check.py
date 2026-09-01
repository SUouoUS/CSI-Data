# -*- coding: utf-8 -*-
"""
s5_pilot_check.py — 파일럿 진단

이 스크립트가 판정하는 것: "본수집에 들어가기 전에 무엇을 확정해야 하는가".
숫자를 좋게 만들기 위한 파라미터 조정은 하지 않는다. 모든 조건은 s2 와 동일하게 고정하고,
비교가 필요한 곳에서는 모든 세션에 똑같은 조건을 적용한다.

네 부분으로 나뉜다.
  A. P02/P03 의 호흡 피크가 물리적으로 호흡이라고 볼 만한가
     (한 세션 결과이므로 재현성 확인으로 간주하지 않는다)
  B. P01 만 낮은 원인을 데이터로 분해
  C. 호흡 말고 물리적 근거가 있는 다른 특징으로 empty/stable 이 갈리는가
  D. 다음 파일럿에서 무엇을 몇 분 모아야 하는가
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np

import csi_core as core
import sessions as S

plt = core.setup_matplotlib()

RULE = "=" * 104
SUB = "-" * 104

FS = S.RESAMPLE_FS
BAND = S.BREATH_BAND
NOISE = (0.6, 2.0)
NPERSEG_LONG = 2048     # s2 와 동일
WIN_LONG = 280.0        # s2 와 동일

# B 절 전용: 5분 세션과 움직임 세션의 정지 구간을 같은 조건으로 비교하기 위한 창.
# 움직임 세션의 정지 구간이 약 110초이므로 여기에 맞춘다.
# 이 창으로 낸 SNR 값은 위의 280초 값과 절대 비교할 수 없다. 서로 다른 척도다.
WIN_SHORT = 105.0
NPERSEG_SHORT = 1024


def prep(sess: core.Session):
    cfg = sess.meta.get("config", {})
    stbc = int(cfg.get("stbc", 0))
    bw40 = str(cfg.get("bandwidth", "1")) == "1"
    fw = str(cfg.get("first_word", "0")) == "1"
    mask = S.analysis_mask(sess.n_sub, stbc=stbc, bandwidth_40=bw40,
                           first_word_invalid=fw)
    amp = core.amplitude(sess.csi)
    nrm = core.normalize(amp, mask, method=S.NORMALIZE)
    X, _ = core.hampel(nrm[:, mask], window=S.HAMPEL_WINDOW, n_sigma=S.HAMPEL_SIGMA)
    return sess.t, X, mask


def snr_and_freq(t, X, t0, t1, win, nperseg):
    """[t0,t1) 안의 첫 창에서 subcarrier별 (SNR, 피크주파수). 창이 안 나오면 None."""
    m = (t >= t0) & (t < t0 + win)
    if m.sum() < nperseg // 2 or t0 + win > t1:
        return None, None
    _, Y = core.resample_uniform(t[m], X[m], FS)
    if Y.shape[0] < nperseg:
        return None, None
    res = [core.breath_snr(Y[:, j], FS, BAND, NOISE, nperseg)
           for j in range(Y.shape[1])]
    return (np.array([r[0] for r in res]), np.array([r[1] for r in res]))


def noise_floor_level(t, X, t0, t1, win, nperseg):
    """0.6-2Hz PSD 중앙값 = 잡음 바닥의 절대 수준. subcarrier 중앙값."""
    from scipy.signal import welch, detrend
    m = (t >= t0) & (t < t0 + win)
    if m.sum() < nperseg // 2:
        return float("nan"), float("nan")
    _, Y = core.resample_uniform(t[m], X[m], FS)
    if Y.shape[0] < nperseg:
        return float("nan"), float("nan")
    f, p = welch(detrend(Y, axis=0), fs=FS, nperseg=nperseg, axis=0)
    n = (f >= NOISE[0]) & (f < NOISE[1])
    b = (f >= BAND[0]) & (f < BAND[1])
    return (float(np.median(np.median(p[n], axis=0))),      # 잡음 바닥
            float(np.median(p[b].max(axis=0))))             # 호흡대역 절대 피크


# ===========================================================================
def section_A(sessions, static_rows):
    print()
    print(RULE)
    print("A. 호흡 피크가 물리적으로 '호흡' 이라고 볼 만한가")
    print(RULE)
    print("검사 논리: 호흡은 가슴 변위라는 하나의 물리적 원인이 여러 subcarrier 를")
    print("동시에 같은 주기로 변조한다. 따라서 진짜 호흡이면 상위 subcarrier 들의")
    print("피크 주파수가 한 값에 모인다. 잡음 피크는 subcarrier 마다 제각각이다.")
    print()
    print("%-11s %-6s %8s %9s %9s %8s %10s"
          % ("session", "state", "최고SNR", "최빈주파수", "호흡수", "일치율", "전후반일치"))
    print(SUB)
    for r in static_rows:
        snr, fr = r["snr"], r["freq"]
        if snr is None:
            continue
        top = np.argsort(-snr)[:10]
        f_top = fr[top]
        # 최빈 주파수: 상위 10개의 중앙값. 일치율은 그 값 ±0.02Hz 안에 든 비율.
        f_mode = float(np.median(f_top))
        agree = float(np.mean(np.abs(f_top - f_mode) <= 0.02))
        # 전후반 안정성: 구간을 반으로 나눠 각각의 최빈 주파수가 맞는지
        half = r["dur"] / 2.0
        s1, f1 = snr_and_freq(r["t"], r["X"], r["t0"], r["t0"] + half,
                              min(half - 1, 130.0), NPERSEG_SHORT)
        s2_, f2 = snr_and_freq(r["t"], r["X"], r["t0"] + half, r["t1"],
                               min(half - 1, 130.0), NPERSEG_SHORT)
        if s1 is None or s2_ is None:
            stab = "-"
        else:
            m1 = float(np.median(f1[np.argsort(-s1)[:10]]))
            m2 = float(np.median(f2[np.argsort(-s2_)[:10]]))
            stab = "%.3f/%.3f %s" % (m1, m2, "일치" if abs(m1 - m2) <= 0.03 else "불일치")
        print("%-11s %-6s %8.2f %8.3fHz %7.1f회/분 %7.0f%% %10s"
              % (r["name"], r["state"], snr.max(), f_mode, f_mode * 60,
                 100 * agree, stab))
    print()
    print("해석: 일치율이 높고 호흡수가 9-30회/분 범위이며 전후반이 일치하면")
    print("      '호흡으로 설명 가능한 신호' 다. 다만 세션 1회 결과이므로")
    print("      재현성이 확인된 것은 아니다. 반복 측정이 필요하다.")


# ===========================================================================
def section_B(sessions, static_rows):
    print()
    print(RULE)
    print("B. P01 만 breath_snr_max 가 낮은 원인 분해")
    print(RULE)

    print("B-1. 잡음이 큰 것인가, 신호가 약한 것인가")
    print("     SNR = 호흡대역 피크 / 잡음바닥 이므로 분자와 분모를 따로 본다.")
    print()
    print("%-11s %-6s %14s %14s %10s %9s"
          % ("session", "state", "잡음바닥", "호흡대역 피크", "비(=SNR)", "RSSI"))
    print(SUB)
    for r in static_rows:
        nf, pk = noise_floor_level(r["t"], r["X"], r["t0"], r["t1"],
                                   WIN_LONG, NPERSEG_LONG)
        print("%-11s %-6s %14.3e %14.3e %10.2f %9.2f"
              % (r["name"], r["state"], nf, pk, pk / nf if nf else float("nan"),
                 r["rssi"]))
    print()
    print("     분모(잡음바닥)가 비슷한데 분자(피크)만 작으면 '신호가 약한 것' 이고,")
    print("     분자는 비슷한데 분모가 크면 '잡음이 큰 것' 이다.")

    print()
    print("B-2. 특정 subcarrier 선택 문제인가")
    print("     상위 subcarrier 만 좋은 것이 아니라, 몇 개의 subcarrier 가")
    print("     어느 수준 이상인지를 본다. 선택 문제라면 상위 몇 개는 높아야 한다.")
    print()
    print("%-11s %-6s %9s %9s %9s %9s %9s"
          % ("session", "state", "최고", "상위5중앙", "상위20중앙", "전체중앙", "SNR>5개수"))
    print(SUB)
    for r in static_rows:
        snr = r["snr"]
        if snr is None:
            continue
        o = np.sort(snr)[::-1]
        print("%-11s %-6s %9.2f %9.2f %9.2f %9.2f %9d"
              % (r["name"], r["state"], o[0], np.median(o[:5]),
                 np.median(o[:20]), np.median(o), int((snr > 5).sum())))

    print()
    print("B-3. 참가자 차이인가 세션 차이인가")
    print("     같은 사람의 stable 세션과 motion 세션의 정지 구간을 비교한다.")
    print("     두 세션은 다른 녹화이므로, 같은 사람인데 값이 갈리면 '세션 차이',")
    print("     같이 낮으면 '참가자 또는 배치 차이' 다.")
    print("     주의: 아래는 창 %.0f초 / nperseg %d 로 통일해 잰 값이라"
          % (WIN_SHORT, NPERSEG_SHORT))
    print("           앞의 280초 기준 SNR 값과 직접 비교할 수 없다.")
    print()
    print("%-8s %14s %14s %14s %12s"
          % ("피험자", "stable 세션", "motion 전반정지", "motion 후반정지", "판정"))
    print(SUB)
    for subj in ("P01", "P02", "P03"):
        vals = []
        for s in sessions:
            if s.subject != subj:
                continue
            t, X, _ = prep(s)
            for sg in s.segments:
                if sg.label != "still":
                    continue
                snr, _ = snr_and_freq(t, X, sg.t0 + S.SEGMENT_GUARD_SEC,
                                      sg.t1 - S.SEGMENT_GUARD_SEC,
                                      WIN_SHORT, NPERSEG_SHORT)
                vals.append(np.nan if snr is None else float(snr.max()))
        while len(vals) < 3:
            vals.append(np.nan)
        spread = np.nanmax(vals) - np.nanmin(vals)
        verdict = "세션 간 변동 큼" if spread > 5 else "일관"
        print("%-8s %14.2f %14.2f %14.2f %12s"
              % (subj, vals[0], vals[1], vals[2], verdict))

    print()
    print("B-4. 채널 자체가 다른가 (배치/자세 영향)")
    print("     정규화 진폭 프로파일의 코사인 유사도. 1에 가까울수록 채널이 닮았다.")
    print()
    names = [r["name"] for r in static_rows]
    P = np.stack([r["prof"] for r in static_rows])
    Pn = P / np.linalg.norm(P, axis=1, keepdims=True)
    Cm = Pn @ Pn.T
    print("      " + " ".join("%10s" % n[:10] for n in names))
    for i, n in enumerate(names):
        print("%-11s" % n[:11] + " ".join("%10.4f" % Cm[i, j]
                                          for j in range(len(names))))
    print()
    print("     같은 상태끼리 유사도가 높고 다른 상태와 낮으면 채널이 상태를 반영한다.")
    print("     상태와 무관하게 뒤섞이면 배치나 세션 조건이 더 지배적이라는 뜻이다.")


# ===========================================================================
ALT_FEATURES = [
    ("coherence_time", "채널 자기상관 감쇠시간 [초]",
     "사람이 있으면 미세 움직임으로 채널이 더 빨리 decorrelate 된다"),
    ("effective_rank", "공분산 실효 자유도",
     "사람은 산란체다. 다중경로 성분이 늘면 자유도가 올라간다"),
    ("amp_kurtosis", "진폭 분포 초과첨도",
     "직시경로 지배(Rician) 대 산란 지배(Rayleigh) 의 분포 차이"),
    ("lowfreq_ratio", "0.01-0.1Hz 전력비",
     "가만히 있어도 자세는 수십 초 규모로 바뀐다. 빈 방에는 없는 성분"),
    ("amp_std_time", "시간 표준편차 (기존)", "참고용"),
    ("mad_diff", "프레임간 변화량 (기존)", "참고용"),
    ("pc1_var_ratio", "제1주성분 분산비 (기존)", "참고용"),
    ("breath_snr_max", "호흡 SNR 최댓값 (기존)", "참고용"),
]


def section_C(static_rows):
    print()
    print(RULE)
    print("C. 호흡 말고 물리적 근거가 있는 다른 특징")
    print(RULE)
    print("[주의] 아래는 특징 %d개를 한꺼번에 시험한 결과다. 여러 개를 시험하면"
          % len(ALT_FEATURES))
    print("       그중 하나가 우연히 좋아 보일 확률이 올라간다. 따라서 잘 나온 것만")
    print("       골라 쓰면 안 되고, 전부를 보고한 뒤 새 데이터로 확증해야 한다.")
    print()

    a_rows = [r for r in static_rows if r["state"] == "still"]
    b_rows = [r for r in static_rows if r["state"] == "empty"]
    na, nb = len(a_rows), len(b_rows)
    minp = core.min_two_sided_p_unpaired(na, nb)
    print("  재실 %d세션 vs 비재실 %d세션, 도달 가능한 최소 양측 p = %.4f  %s"
          % (na, nb, minp,
             "" if minp <= 0.05 else "<== p<0.05 는 원리적으로 불가능"))
    print()
    res = []
    for key, title, _ in ALT_FEATURES:
        a = [r["alt"][key] for r in a_rows if np.isfinite(r["alt"][key])]
        b = [r["alt"][key] for r in b_rows if np.isfinite(r["alt"][key])]
        if len(a) < 2 or len(b) < 2:
            continue
        _, p, _ = core.exact_perm_test_unpaired(a, b)
        res.append((key, np.median(a), np.median(b), p, core.sep_ratio(a, b),
                    (min(a) > max(b)) or (min(b) > max(a))))
    _, padj = core.benjamini_hochberg([r[3] for r in res], q=0.05)

    print("  %-16s %10s %10s %9s %9s %8s %-9s %s"
          % ("feature", "재실중앙", "비재실중앙", "p", "BH보정p", "sep", "완전분리", "판정"))
    print("  " + SUB)
    for (key, ma, mb, p, sep, full), pa in zip(res, padj):
        ok = (sep >= 2.0 and full and pa <= 0.05)
        print("  %-16s %10.4f %10.4f %9.4f %9.4f %8.2f %-9s %s"
              % (key, ma, mb, p, pa, sep, "예" if full else "아니오",
                 "구분 근거 있음" if ok else "근거 부족"))
    print()
    print("  BH보정p 는 특징 %d개를 동시에 시험한 것을 보정한 값이다." % len(res))
    print("  보정 전 p 만 보고 특징을 고르면 그것이 곧 다중비교 오류다.")
    print()
    print("  세션별 값")
    print("  %-11s %-6s " % ("session", "state")
          + " ".join("%14s" % k for k, _, _ in ALT_FEATURES[:4]))
    print("  " + SUB)
    for r in static_rows:
        print("  %-11s %-6s " % (r["name"], r["state"])
              + " ".join("%14.4f" % r["alt"][k] for k, _, _ in ALT_FEATURES[:4]))
    print()
    for key, title, why in ALT_FEATURES[:4]:
        print("  %-16s %s" % (key, why))


# ===========================================================================
def main() -> int:
    print(RULE)
    print("s5_pilot_check.py — 파일럿 진단 (튜닝 없음, 조건은 s2 와 동일하게 고정)")
    print(RULE)
    sessions = S.load_all(verbose=False)

    # 5분짜리 정적 세션만 (still 전용 / empty). 조건이 같아야 비교가 성립한다.
    static_rows = []
    for s in sessions:
        if len(s.segments) != 1 or s.segments[0].label not in ("still", "empty"):
            continue
        sg = s.segments[0]
        t, X, mask = prep(s)
        t0 = sg.t0 + S.SEGMENT_GUARD_SEC
        t1 = sg.t1 - S.SEGMENT_GUARD_SEC
        snr, fr = snr_and_freq(t, X, t0, t1, WIN_LONG, NPERSEG_LONG)
        m = (t >= t0) & (t < t1)
        Xi = X[m]
        ti = t[m]
        _, dd, _ = core.diff_matched_dt(ti, Xi)
        static_rows.append({
            "name": s.name, "state": sg.label, "subject": s.subject,
            "batch": s.batch, "t": t, "X": X, "t0": t0, "t1": t1,
            "dur": t1 - t0, "snr": snr, "freq": fr,
            "prof": Xi.mean(axis=0),
            "rssi": float(s.rssi.mean()),
            "alt": {
                "coherence_time": core.coherence_time(Xi, s.rate_hz),
                "effective_rank": core.effective_rank(Xi),
                "amp_kurtosis": core.amp_kurtosis(Xi),
                "lowfreq_ratio": core.lowfreq_power_ratio(ti, Xi, FS,
                                                          nperseg=NPERSEG_LONG),
                "amp_std_time": float(Xi.std(axis=0, ddof=1).mean()),
                "mad_diff": float(dd.mean()),
                "pc1_var_ratio": core.pc1(Xi)[1],
                "breath_snr_max": float(snr.max()) if snr is not None else np.nan,
            },
        })
    static_rows.sort(key=lambda r: (r["state"], r["name"]))
    print("정적 5분 세션 %d개: %s"
          % (len(static_rows), ", ".join("%s(%s)" % (r["name"], r["state"])
                                         for r in static_rows)))

    section_A(sessions, static_rows)
    section_B(sessions, static_rows)
    section_C(static_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
