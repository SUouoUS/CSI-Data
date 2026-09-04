# -*- coding: utf-8 -*-
"""
s2_session_stats.py — 세션 단위 비교

이 스크립트가 판정하는 것: "특징값이 클래스(재실/비재실/움직임)에 따라 갈리는가,
그리고 그 차이를 표본 수가 뒷받침하는가".

독립 표본의 단위는 프레임이 아니라 '세션' 이다. (결함 D1)
한 세션의 수천 프레임은 자기상관 때문에 독립 표본 1개다.
따라서 세션마다 비중첩 윈도우로 특징을 뽑은 뒤, 그 중앙값 1개를 세션 대표값으로 삼고
세션 대표값끼리만 검정한다. 윈도우 분포는 그림에서 참고로만 보여준다.

검정은 정확 순열검정(전수)이다. 표본 수가 작아 근사 분포를 쓸 수 없기 때문이다.
현재 표본 수로 도달 가능한 최소 p 를 항상 함께 출력한다.
그 값이 0.05 보다 크면 어떤 특징을 써도 p<0.05 는 원리적으로 불가능하다.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np

import csi_core as core
import sessions as S

plt = core.setup_matplotlib()

RULE = "=" * 100
SUB = "-" * 100

FEATURES = [
    ("amp_mean", "정규화 평균진폭 (스펙트럼 평탄도)"),
    ("amp_std_time", "subcarrier별 시간 표준편차의 평균"),
    ("mad_diff", "프레임간 절대변화량 평균"),
    ("mad_diff_p95", "프레임간 절대변화량 p95"),
    ("pc1_var_ratio", "제1주성분 분산비"),
    ("breath_ratio", "호흡대역 전력비 (PC1 기반, 참고용)"),
    ("breath_snr_max", "호흡 SNR 최댓값 (subcarrier별)"),
    ("breath_n_sub", "호흡 SNR이 임계 초과인 subcarrier 수"),
]

# 호흡 SNR 분석 조건. 모든 구간에서 동일하게 적용해야 비교가 공정하다.
#
# 창 길이를 280초로 크게 잡은 이유:
#   SNR 은 '호흡 대역 최대 피크 / 잡음 바닥' 이다. welch 평균 횟수가 적으면
#   잡음 피크 자체가 크게 튀어 비재실에서도 높은 값이 나온다.
#   실제로 창 100초 / nperseg 1024 로 재면 비재실의 최대 SNR 이 12.0 까지 올라가
#   재실과 겹쳐 버린다. 창 280초 / nperseg 2048 로 늘리면 평균 횟수가 늘어
#   잡음 바닥이 안정되고 비재실 최대 SNR 이 6~7 수준으로 내려간다.
#
# 대가: 280초보다 짧은 구간은 이 특징을 계산할 수 없다(NaN).
#   움직임 세션의 정지 구간(약 110초)이 여기 해당한다.
#   즉 호흡 특징은 5분짜리 정적 세션끼리만 비교된다. 이는 공정성을 위한 것이며,
#   짧은 구간에 억지로 값을 채우면 긴 구간이 유리해져 비교가 무효가 된다.
BREATH_SNR_WIN = 280.0
BREATH_SNR_NPERSEG = 2048  # 20Hz 기준 102.4초 -> 분해능 0.0098 Hz
BREATH_NOISE_BAND = (0.6, 2.0)

# --- 탐색적 특징에 대한 경고 -------------------------------------------------
# breath_snr_max / breath_n_sub 는 기존 breath_ratio 가 실패한 원인을 진단하다가
# 같은 데이터에서 발견한 특징이다. 따라서 이 데이터로 낸 p값은 확증이 아니라
# 탐색적 결과다. 새로 수집한 데이터에서 재현되어야 주장할 수 있다.
# 이 사실은 출력에도 매번 명시한다.


def prep(sess: core.Session) -> Tuple[np.ndarray, np.ndarray]:
    """정규화 -> hampel. 리샘플은 스펙트럼 특징에서만 한다."""
    cfg = sess.meta.get("config", {})
    stbc = int(cfg.get("stbc", 0))
    bw40 = str(cfg.get("bandwidth", "1")) == "1"
    fw = str(cfg.get("first_word", "0")) == "1"
    mask = S.analysis_mask(sess.n_sub, stbc=stbc, bandwidth_40=bw40,
                           first_word_invalid=fw)
    amp = core.amplitude(sess.csi)
    nrm = core.normalize(amp, mask, method=S.NORMALIZE)
    X, _ = core.hampel(nrm[:, mask], window=S.HAMPEL_WINDOW,
                       n_sigma=S.HAMPEL_SIGMA)
    return sess.t, X


def window_features(t: np.ndarray, X: np.ndarray, t0: float, t1: float
                    ) -> Dict[str, List[float]]:
    """
    [t0, t1) 구간을 비중첩 윈도우로 잘라 윈도우별 특징을 만든다.
    중첩을 주면 표본 수를 인공적으로 늘리는 것이므로 금지한다.
    """
    out: Dict[str, List[float]] = {k: [] for k, _ in FEATURES}
    w = S.WINDOW_SEC
    a = t0
    while a + w <= t1:
        m = (t >= a) & (t < a + w)
        if m.sum() >= 20:
            Xi = X[m]
            ti = t[m]
            out["amp_mean"].append(float(Xi.mean()))
            out["amp_std_time"].append(float(Xi.std(axis=0, ddof=1).mean()))
            _, d, _ = core.diff_matched_dt(ti, Xi)
            dm = d.mean(axis=1)
            out["mad_diff"].append(float(dm.mean()))
            out["mad_diff_p95"].append(float(np.percentile(dm, 95)))
            _, vr = core.pc1(Xi)
            out["pc1_var_ratio"].append(float(vr))
        a += S.WINDOW_STEP_SEC

    # 호흡은 창이 길어야 한다. welch 분해능이 fs/nperseg 라 10초 창으로는
    # 0.15-0.5 Hz 를 나눌 수 없다. 그래서 이 특징만 60초 비중첩 창을 쓴다.
    a = t0
    while a + S.BREATH_WINDOW_SEC <= t1:
        m = (t >= a) & (t < a + S.BREATH_WINDOW_SEC)
        if m.sum() >= 200:
            g, Y = core.resample_uniform(t[m], X[m], S.RESAMPLE_FS)
            p1, _ = core.pc1(Y)
            out["breath_ratio"].append(
                core.band_power_ratio(p1, S.RESAMPLE_FS, S.BREATH_BAND, S.BREATH_FULL))
        a += S.BREATH_WINDOW_SEC

    # subcarrier 별 호흡 SNR. 창 100초 비중첩, nperseg 고정으로 조건을 통일한다.
    m_all = (t >= t0) & (t < t1)
    profiles = core.breath_snr_profile(
        t[m_all], X[m_all], S.RESAMPLE_FS,
        win_sec=BREATH_SNR_WIN, nperseg=BREATH_SNR_NPERSEG,
        band=S.BREATH_BAND, noise=BREATH_NOISE_BAND) if m_all.sum() > 0 else []
    out["_snr_profiles"] = profiles
    out["breath_snr_max"] = [float(np.nanmax(p)) if p.size else float("nan")
                             for p in profiles]
    out["breath_n_sub"] = []      # 임계값이 정해진 뒤 2단계에서 채운다
    return out


def collect(sessions: List[core.Session]) -> List[Dict[str, object]]:
    """(세션, 구간) 단위로 윈도우 특징과 대표값(중앙값)을 만든다."""
    rows = []
    for s in sessions:
        t, X = prep(s)
        for k, sg in enumerate(s.segments):
            wf = window_features(t, X, sg.t0 + S.SEGMENT_GUARD_SEC,
                                 sg.t1 - S.SEGMENT_GUARD_SEC)
            rep = {}
            for f, _ in FEATURES:
                v = np.asarray(wf[f], dtype=float)
                v = v[np.isfinite(v)]
                rep[f] = float(np.median(v)) if v.size else float("nan")
            rows.append({
                "session": s.name, "subject": s.subject, "batch": s.batch,
                "scenario": s.scenario, "state": sg.label, "seg_index": k,
                "t0": sg.t0, "t1": sg.t1,
                "n_win": len(wf["amp_mean"]), "windows": wf, "rep": rep,
            })
    return rows


def apply_breath_threshold(rows) -> Dict[str, float]:
    """
    '호흡 SNR 이 임계를 넘는 subcarrier 수' 특징의 임계값을 정하고 채운다.

    임계값은 비재실(empty) 세션에서만 뽑는다. 재실 데이터를 보고 정하면 순환논증이다.
    다만 비재실 세션 자신을 평가할 때 자기 값으로 정한 임계를 쓰면 그것도 자기참조이므로,
    비재실 세션에는 leave-one-out 을 적용한다(자기를 뺀 나머지 비재실로 임계 결정).
    """
    empty_by_sess: Dict[str, List[np.ndarray]] = {}
    for r in rows:
        if r["state"] != "empty":
            continue
        empty_by_sess.setdefault(r["session"], []).extend(r["windows"]["_snr_profiles"])

    def thr_excluding(sess_name: str) -> float:
        vals = [p.max() for s, ps in empty_by_sess.items() if s != sess_name
                for p in ps if p.size]
        return float(max(vals)) if vals else float("nan")

    thr_map = {}
    for r in rows:
        thr = thr_excluding(r["session"]) if r["state"] == "empty" \
            else thr_excluding("")           # 재실은 전체 비재실 사용
        thr_map[r["session"] + "/" + str(r["seg_index"])] = thr
        counts = ([int((p > thr).sum()) for p in r["windows"]["_snr_profiles"]]
                  if np.isfinite(thr) else [])
        r["windows"]["breath_n_sub"] = counts
        r["rep"]["breath_n_sub"] = float(np.median(counts)) if counts else float("nan")
    return {"global": thr_excluding(""),
            "per_empty": {s: thr_excluding(s) for s in empty_by_sess}}


def session_values(rows, state: str, feature: str, sessions_filter=None
                   ) -> Tuple[List[str], List[float]]:
    """
    세션 대표값을 뽑는다. 한 세션에 같은 상태 구간이 여러 개면
    그 세션 안에서 다시 중앙값을 내어 '세션당 값 1개' 를 유지한다.
    """
    by_sess: Dict[str, List[float]] = {}
    for r in rows:
        if r["state"] != state:
            continue
        if sessions_filter is not None and r["session"] not in sessions_filter:
            continue
        v = r["rep"][feature]
        if np.isfinite(v):
            by_sess.setdefault(r["session"], []).append(v)
    names = sorted(by_sess)
    return names, [float(np.median(by_sess[n])) for n in names]


def report_unpaired(title, rows, state_a, state_b, filt_a=None, filt_b=None):
    print()
    print(RULE)
    print(title)
    print(RULE)
    na_names, _ = session_values(rows, state_a, FEATURES[0][0], filt_a)
    nb_names, _ = session_values(rows, state_b, FEATURES[0][0], filt_b)
    na, nb = len(na_names), len(nb_names)
    print("  A(%s) 세션 %d개: %s" % (state_a, na, ", ".join(na_names)))
    print("  B(%s) 세션 %d개: %s" % (state_b, nb, ", ".join(nb_names)))
    if na < 1 or nb < 1:
        print("  비교할 세션이 없다.")
        return
    minp = core.min_two_sided_p_unpaired(na, nb)
    print("  도달 가능한 최소 양측 p = 2 / C(%d,%d) = %.4f  %s"
          % (na + nb, na, minp,
             "" if minp <= 0.05 else "<== 이 표본 수로는 p<0.05 가 원리적으로 불가능하다"))
    print()
    print("  %-15s %10s %10s %10s %9s %9s  %s"
          % ("feature", "A중앙", "B중앙", "차이", "p", "sep", "판정"))
    print("  " + SUB)
    for f, _ in FEATURES:
        _, a = session_values(rows, state_a, f, filt_a)
        _, b = session_values(rows, state_b, f, filt_b)
        if len(a) < 1 or len(b) < 1:
            continue
        obs, p, _ = core.exact_perm_test_unpaired(a, b)
        sep = core.sep_ratio(a, b)
        verdict = "구분 근거 있음" if sep >= 2.0 else "근거 부족"
        print("  %-15s %10.4f %10.4f %+10.4f %9.4f %9.2f  %s"
              % (f, np.median(a), np.median(b), obs, p, sep, verdict))
    print()
    print("  sep = |클래스간 차이| / 클래스내 세션 표준편차. 2 미만이면 구분 주장 불가.")


def report_paired(title, rows, state_a, state_b):
    print()
    print(RULE)
    print(title)
    print(RULE)
    # 두 상태를 모두 가진 세션만
    per: Dict[str, Dict[str, float]] = {}
    for r in rows:
        per.setdefault(r["session"], {})
        if r["state"] in (state_a, state_b):
            per[r["session"]].setdefault(r["state"], []).append(r["rep"])
    names = [s for s, d in per.items() if state_a in d and state_b in d]
    names.sort()
    print("  세션 내 %s / %s 대조가 가능한 세션 %d개: %s"
          % (state_a, state_b, len(names), ", ".join(names)))
    if not names:
        print("  해당 세션이 없다.")
        return
    minp = core.min_two_sided_p_paired(len(names))
    print("  도달 가능한 최소 양측 p = 2 / 2^%d = %.4f  %s"
          % (len(names), minp,
             "" if minp <= 0.05 else "<== 이 표본 수로는 p<0.05 가 원리적으로 불가능하다"))
    print()
    print("  %-15s %10s %10s %10s %9s  %s"
          % ("feature", "A중앙", "B중앙", "차이중앙", "p", "세션별 차이"))
    print("  " + SUB)
    for f, _ in FEATURES:
        diffs, detail = [], []
        for n in names:
            va = np.median([x[f] for x in per[n][state_a] if np.isfinite(x[f])])
            vb = np.median([x[f] for x in per[n][state_b] if np.isfinite(x[f])])
            if np.isfinite(va) and np.isfinite(vb):
                diffs.append(va - vb)
                detail.append("%+.4f" % (va - vb))
        if len(diffs) < 2:
            continue
        obs, p, _ = core.exact_perm_test_paired(diffs)
        aa = [np.median([x[f] for x in per[n][state_a]]) for n in names]
        bb = [np.median([x[f] for x in per[n][state_b]]) for n in names]
        same_sign = all(d > 0 for d in diffs) or all(d < 0 for d in diffs)
        print("  %-15s %10.4f %10.4f %+10.4f %9.4f  %s %s"
              % (f, np.median(aa), np.median(bb), obs, p, " ".join(detail),
                 "(부호 일치)" if same_sign else ""))
    print()
    print("  세션 내 대조는 환경 드리프트가 상쇄되므로 세션 간 비교보다 훨씬 강한 증거다.")
    print("  '부호 일치' 는 모든 세션에서 같은 방향으로 움직였다는 뜻이다.")


def plot_windows(rows, path):
    """세션×구간별 윈도우 분포 박스플롯. 세션 대표값은 별도 마커로 표시."""
    labels, data, colors = [], [], []
    for r in rows:
        for f_i, (f, _) in enumerate(FEATURES):
            pass
    n_f = len(FEATURES)
    fig, axes = plt.subplots(n_f, 1, figsize=(max(10, 0.55 * len(rows) + 4), 2.6 * n_f),
                             sharex=True)
    xs = np.arange(len(rows))
    xticklab = ["%s\n%s" % (r["session"], r["state"]) for r in rows]
    for ax, (f, title) in zip(axes, FEATURES):
        vals = [np.asarray(r["windows"][f], dtype=float) for r in rows]
        vals = [v[np.isfinite(v)] for v in vals]
        bp = ax.boxplot([v if v.size else [np.nan] for v in vals],
                        positions=xs, widths=0.6, patch_artist=True,
                        showfliers=False, medianprops=dict(color="k", lw=1.2))
        for patch, r in zip(bp["boxes"], rows):
            patch.set_facecolor(core.STATE_COLOR.get(r["state"], "#999999"))
            patch.set_alpha(0.55)
        reps = [r["rep"][f] for r in rows]
        ax.plot(xs, reps, "kD", ms=4, label="세션 대표값(중앙값)")
        ax.set_ylabel(f, fontsize=9)
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(alpha=0.25, axis="y")
    axes[0].legend(fontsize=8, loc="best")
    axes[-1].set_xticks(xs)
    axes[-1].set_xticklabels(xticklab, rotation=90, fontsize=7)
    fig.suptitle("세션·구간별 윈도우 분포 (박스=%d초 비중첩 윈도우, 마름모=세션 대표값)"
                 % int(S.WINDOW_SEC), y=0.995)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("  그림 저장: %s" % path)


def plot_session_points(rows, path):
    """세션 대표값만 클래스별로 찍은 그림. 이게 실제 검정에 쓰인 표본이다."""
    states = ["still", "motion", "empty"]
    n_f = len(FEATURES)
    ncol = 3
    nrow = int(np.ceil(n_f / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (f, title) in zip(axes, FEATURES):
        for xi, st in enumerate(states):
            names, vals = session_values(rows, st, f)
            if not vals:
                continue
            jitter = (np.random.RandomState(0).rand(len(vals)) - 0.5) * 0.18
            ax.scatter(np.full(len(vals), xi) + jitter, vals, s=48,
                       color=core.STATE_COLOR[st], edgecolor="k", linewidth=0.5,
                       zorder=3)
            for nm, v, j in zip(names, vals, jitter):
                ax.annotate(nm, (xi + j, v), fontsize=6, ha="left", va="bottom",
                            xytext=(3, 2), textcoords="offset points")
            ax.hlines(np.median(vals), xi - 0.28, xi + 0.28,
                      color=core.STATE_COLOR[st], lw=2)
        ax.set_xticks(range(len(states)))
        ax.set_xticklabels(states)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.25, axis="y")
    for ax in axes[n_f:]:
        ax.axis("off")
    fig.suptitle("세션 대표값 (점 1개 = 세션 단위 관측값 1개). 검정은 이 점들로만 한다.", y=1.0)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("  그림 저장: %s" % path)


def main() -> int:
    print(RULE)
    print("s2_session_stats.py — 세션 단위 비교")
    print(RULE)
    print("윈도우 %.0fs (step %.0fs, 중첩 없음) / 호흡창 %.0fs / 정규화 %s / 분석블록 %s"
          % (S.WINDOW_SEC, S.WINDOW_STEP_SEC, S.BREATH_WINDOW_SEC,
             S.NORMALIZE, S.ANALYSIS_BLOCK))
    print()

    sessions = S.load_all(verbose=True)
    rows = collect(sessions)
    thr_info = apply_breath_threshold(rows)

    print()
    print(RULE)
    print("세션·구간별 윈도우 수")
    print(RULE)
    for r in rows:
        print("  %-11s %-6s %6.1f-%6.1f s   윈도우 %3d   호흡창 %d   SNR창 %d"
              % (r["session"], r["state"], r["t0"], r["t1"], r["n_win"],
                 len(r["windows"]["breath_ratio"]),
                 len(r["windows"]["_snr_profiles"])))

    print()
    print(RULE)
    print("호흡 SNR 임계값 (비재실 세션에서만 결정)")
    print(RULE)
    print("  재실 세션 평가에 쓰는 임계값 : %.2f  (전체 비재실 세션의 SNR 최댓값)"
          % thr_info["global"])
    for s, v in sorted(thr_info["per_empty"].items()):
        print("  비재실 %-8s 평가용 임계값 : %.2f  (자기 제외, leave-one-out)" % (s, v))
    print("  * 재실 데이터를 보고 임계를 정하면 순환논증이므로 비재실에서만 뽑았다.")
    print()
    print("  [주의] breath_snr_max / breath_n_sub 는 기존 breath_ratio 가 실패한 원인을")
    print("         진단하다가 '이 데이터에서' 발견한 특징이다. 따라서 아래 p값은")
    print("         확증이 아니라 탐색적 결과이며, 새로 수집한 데이터에서 재현되어야")
    print("         비로소 주장할 수 있다.")

    still_only = {s.name for s in sessions
                  if [sg.label for sg in s.segments] == ["still"]}
    all_occ = {s.name for s in sessions
               if any(sg.label in ("still", "motion") for sg in s.segments)}

    report_unpaired("비교 1) 재실(정지 전용 세션) vs 비재실",
                    rows, "still", "empty", filt_a=still_only)
    report_unpaired("비교 2) 재실 전체(움직임 세션의 정지 구간 포함) vs 비재실",
                    rows, "still", "empty", filt_a=all_occ)
    report_paired("비교 3) 세션 내 대조 — 정지 vs 움직임 (같은 세션, 같은 사람)",
                  rows, "motion", "still")
    report_paired(
        "비교 4) 세션 내 대조 — 정지 재실 vs 비재실 (같은 세션, 같은 사람)",
        rows,
        "still",
        "empty",
    )

    plot_windows(rows, os.path.join(S.OUT_DIR, "s2_window_boxplot.png"))
    plot_session_points(rows, os.path.join(S.OUT_DIR, "s2_session_points.png"))

    # CSV
    csv_path = os.path.join(S.OUT_DIR, "s2_session_features.csv")
    with open(csv_path, "w", encoding="utf-8-sig") as fh:
        fh.write("session,subject,batch,scenario,state,seg_index,t0,t1,n_win,"
                 + ",".join(f for f, _ in FEATURES) + "\n")
        for r in rows:
            fh.write("%s,%s,%s,%s,%s,%d,%.1f,%.1f,%d," %
                     (r["session"], r["subject"], r["batch"], r["scenario"],
                      r["state"], r["seg_index"], r["t0"], r["t1"], r["n_win"]))
            fh.write(",".join("%.6f" % r["rep"][f] for f, _ in FEATURES) + "\n")
    print("  CSV 저장: %s" % csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
