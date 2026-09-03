# -*- coding: utf-8 -*-
"""
s6_visualize.py — 세션 타임라인 시각화

배치 C 의 14분 프로토콜(empty 3분 -> still 5분 -> empty 3분, 전환마다 대기 1분)처럼
한 세션 안에서 상태가 바뀌는 데이터를 눈으로 확인하기 위한 스크립트다.
검정은 하지 않는다. 검정은 s2(세션 단위) / s3(subcarrier 단위) 가 한다.

이 스크립트가 답하는 질문
---------------------------------------------------------------------------
  1) 라벨 경계에서 신호가 실제로 바뀌는가. 바뀐다면 t_offset 이 맞다는 뜻이다.
     라벨보다 이르거나 늦게 바뀌면 sessions.py 의 t_offset 을 고쳐야 한다.
  2) 같은 라벨 구간 안에서 값이 안정적인가, 아니면 한 방향으로 계속 흐르는가.
  3) 세션 내 empty 두 구간(전/후)이 서로 닮았는가. 닮지 않았다면 그 세션의
     empty 를 하나의 상태로 묶어 쓰는 것 자체를 의심해야 한다.

주의: 그림은 가설을 만드는 도구다. 여기서 눈으로 고른 구간·대역·특징으로
      같은 데이터를 다시 검정하면 p값은 의미를 잃는다.

산출물 (out/)
---------------------------------------------------------------------------
  s6_timeline_<세션>.png   시간축 4단 패널 (변화량 / PC1 / 윈도우 특징 / RSSI)
  s6_heatmap_<세션>.png    subcarrier x 시간 진폭 편차 히트맵
  s6_paired_states.png     세션 내 상태별 대표값 (같은 세션을 선으로 연결)
  s6_psd.png               구간별 PC1 스펙트럼 (호흡 대역 표시)
  s6_windows.csv           윈도우별 특징값 원자료
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

# 타임라인용 윈도우. s2 의 검정용 윈도우(비중첩)와 목적이 다르다.
# 여기서는 '시간에 따라 어떻게 변하는가' 를 보려는 것이므로 촘촘히 본다.
# 중첩 윈도우이므로 이 값은 그림 전용이며 검정에 쓰지 않는다.
TL_WINDOW_SEC = 10.0
TL_STEP_SEC = 2.0

# 대표값(구간 요약)은 검정과 같은 조건을 쓴다. 비중첩 + guard.
REP_WINDOW_SEC = S.WINDOW_SEC

HEATMAP_BIN_SEC = 2.0     # 히트맵 시간축 묶음. 15000 프레임을 그대로 그릴 수 없다.
PSD_NPERSEG = 2048        # 20 Hz 기준 102.4초 -> 분해능 0.0098 Hz

FEATURES = [
    ("mad_diff", "프레임간 변화량"),
    ("amp_std_time", "시간 표준편차"),
    ("amp_mean", "정규화 평균진폭"),
    ("pc1_var_ratio", "PC1 분산비"),
]


# ===========================================================================
# 전처리 / 특징
# ===========================================================================

def prep(sess: core.Session) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """정규화 -> hampel. s2/s3 와 같은 전처리여야 그림과 검정이 같은 것을 본다."""
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
    return sess.t, X, mask


def motion_trace(t: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    프레임간 절대변화량 시계열 + 이동평균.
    리샘플하지 않고 diff_matched_dt 로 뽑는다. 보간은 변화량을 과소평가한다.
    """
    t_mid, d, dt = core.diff_matched_dt(t, X)
    y = d.mean(axis=1)
    w = max(1, int(round(S.MOTION_SMOOTH_SEC / dt)))
    if w <= 1:
        return t_mid, y
    y_s = np.convolve(y, np.ones(w) / w, mode="same")
    # 양끝은 커널이 다 채워지지 않아 값이 눌린다. 그대로 두면 경계 인공물이 된다.
    y_s[:w] = np.nan
    y_s[-w:] = np.nan
    return t_mid, y_s


def sliding_features(t: np.ndarray, X: np.ndarray, win: float, step: float
                     ) -> Dict[str, np.ndarray]:
    """
    슬라이딩 윈도우 특징. win == step 이면 비중첩이 되어 s2 와 같은 조건이 된다.
    중첩(step < win)은 표본 수를 인공적으로 늘리므로 그림에만 쓴다.
    """
    tc: List[float] = []
    out: Dict[str, List[float]] = {k: [] for k, _ in FEATURES}
    if t.size < 2:
        res = {k: np.zeros(0) for k in out}
        res["t"] = np.zeros(0)
        return res
    a = float(t[0])
    t_end = float(t[-1])
    while a + win <= t_end:
        m = (t >= a) & (t < a + win)
        if m.sum() >= 20:
            Xi, ti = X[m], t[m]
            tc.append(a + win / 2.0)
            out["amp_mean"].append(float(Xi.mean()))
            out["amp_std_time"].append(float(Xi.std(axis=0, ddof=1).mean()))
            _, d, _ = core.diff_matched_dt(ti, Xi)
            out["mad_diff"].append(float(d.mean()))
            _, vr = core.pc1(Xi)
            out["pc1_var_ratio"].append(float(vr))
        a += step
    res = {k: np.asarray(v, dtype=float) for k, v in out.items()}
    res["t"] = np.asarray(tc, dtype=float)
    return res


def segment_windows(t: np.ndarray, X: np.ndarray, t0: float, t1: float
                    ) -> Dict[str, np.ndarray]:
    """구간 안 비중첩 윈도우 특징. 구간 대표값 계산용."""
    m = (t >= t0) & (t < t1)
    return sliding_features(t[m], X[m], REP_WINDOW_SEC, REP_WINDOW_SEC)


# ===========================================================================
# 그림 공통
# ===========================================================================

def shade_segments(ax, sess: core.Session, label_text: bool = False):
    """
    상태 구간을 색으로 칠하고, 라벨 없는 '대기' 구간은 회색 사선으로 표시한다.
    guard 로 잘려 분석에서 빠지는 양끝은 더 옅게 칠해 구분한다.
    어디까지가 실제로 쓰인 구간인지 그림에서 바로 보이게 하기 위한 것이다.
    """
    y0, y1 = ax.get_ylim()
    g = S.SEGMENT_GUARD_SEC
    for sg in sess.segments:
        c = core.STATE_COLOR.get(sg.label, "#7f7f7f")
        ax.axvspan(sg.t0 + g, max(sg.t1 - g, sg.t0 + g), color=c, alpha=0.10,
                   lw=0, zorder=0)
        ax.axvspan(sg.t0, min(sg.t0 + g, sg.t1), color=c, alpha=0.04, lw=0, zorder=0)
        ax.axvspan(max(sg.t1 - g, sg.t0), sg.t1, color=c, alpha=0.04, lw=0, zorder=0)
        ax.axvline(sg.t0, color=c, lw=0.8, alpha=0.6, zorder=1)
        ax.axvline(sg.t1, color=c, lw=0.8, alpha=0.6, zorder=1)
        if label_text:
            ax.text((sg.t0 + sg.t1) / 2.0, y1, sg.label, ha="center", va="bottom",
                    fontsize=9, color=c)

    # 라벨이 없는 시간(대기). 구간 사이의 빈 곳을 사선으로 칠한다.
    bounds = sorted((sg.t0, sg.t1) for sg in sess.segments)
    gaps = [(float(sess.t[0]), bounds[0][0])] if bounds else []
    for (a0, a1), (b0, _) in zip(bounds, bounds[1:]):
        gaps.append((a1, b0))
    if bounds:
        gaps.append((bounds[-1][1], float(sess.t[-1])))
    for a, b in gaps:
        if b - a > 1.0:
            ax.axvspan(a, b, facecolor="none", edgecolor="#999999", hatch="///",
                       alpha=0.30, lw=0, zorder=0)
    ax.set_ylim(y0, y1)


def mark_events(ax, sess: core.Session):
    for _, at in sess.events:
        ax.axvline(at, color="k", lw=0.9, ls="--", alpha=0.7, zorder=2)


# ===========================================================================
# 1. 세션 타임라인
# ===========================================================================

def plot_timeline(sess: core.Session, t: np.ndarray, X: np.ndarray, path: str):
    tl = sliding_features(t, X, TL_WINDOW_SEC, TL_STEP_SEC)
    tm, ms = motion_trace(t, X)
    p1, vr = core.pc1(X)

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)

    ax = axes[0]
    ax.plot(tm, ms, color="#333333", lw=0.7)
    ax.set_ylabel("프레임간 변화량\n(%.0fs 이동평균)" % S.MOTION_SMOOTH_SEC)
    ax.set_title("%s  (%s, subj=%s, batch=%s, %.2f Hz, id손실 %.1f%%)"
                 % (sess.name, sess.scenario, sess.subject, sess.batch,
                    sess.rate_hz, 100.0 * sess.seq_loss[1]), fontsize=11)

    ax = axes[1]
    ax.plot(t, p1, color="#1f77b4", lw=0.5)
    ax.set_ylabel("PC1 (분산비 %.2f)" % vr)

    ax = axes[2]
    for f, title in FEATURES[:2]:
        v = tl[f]
        if v.size and np.isfinite(np.nanmedian(v)) and np.nanmedian(v) != 0:
            ax.plot(tl["t"], v / np.nanmedian(v), lw=1.2, label=title)
    ax.set_ylabel("윈도우 특징 (중앙값=1)\n%.0fs 창 / %.0fs 이동"
                  % (TL_WINDOW_SEC, TL_STEP_SEC))
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[3]
    ax.plot(sess.t, sess.rssi, color="#8c564b", lw=0.5, label="RSSI")
    ax.plot(sess.t, sess.noise_floor, color="#7f7f7f", lw=0.5, label="noise floor")
    ax.set_ylabel("dBm")
    ax.set_xlabel("시간 [s]")
    ax.legend(fontsize=8, loc="upper right")

    for k, ax in enumerate(axes):
        shade_segments(ax, sess, label_text=(k == 0))
        mark_events(ax, sess)
        ax.grid(alpha=0.25)
    axes[0].set_xlim(float(sess.t[0]), float(sess.t[-1]))

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("  그림 저장: %s" % path)


# ===========================================================================
# 2. subcarrier x 시간 히트맵
# ===========================================================================

def plot_heatmap(sess: core.Session, t: np.ndarray, X: np.ndarray,
                 mask: np.ndarray, path: str):
    """
    subcarrier 별 세션 중앙값을 뺀 편차를 그린다.
    subcarrier 마다 절대 진폭이 크게 달라, 원값을 그리면 subcarrier 간 차이만
    보이고 시간 변화가 묻힌다.
    """
    _, scs_all = core.block_subcarriers(sess.n_sub, 0)
    sc = scs_all[mask]
    order = np.argsort(sc)
    sc_sorted = sc[order]

    edges = np.arange(t[0], t[-1] + HEATMAP_BIN_SEC, HEATMAP_BIN_SEC)
    idx = np.digitize(t, edges) - 1
    nb = len(edges) - 1
    Z = np.full((nb, X.shape[1]), np.nan)
    for b in range(nb):
        m = idx == b
        if m.sum() >= 5:
            Z[b] = X[m].mean(axis=0)
    Z = Z - np.nanmedian(Z, axis=0, keepdims=True)
    Z = Z[:, order]

    v = float(np.nanpercentile(np.abs(Z), 98))
    fig, ax = plt.subplots(figsize=(13, 5))
    im = ax.imshow(Z.T, aspect="auto", origin="lower", cmap="RdBu_r",
                   vmin=-v, vmax=v, interpolation="nearest",
                   extent=(float(edges[0]), float(edges[-1]),
                           float(sc_sorted[0]), float(sc_sorted[-1])))
    fig.colorbar(im, ax=ax, label="정규화 진폭 편차 (subcarrier별 세션 중앙값 기준)")
    for sg in sess.segments:
        ax.axvline(sg.t0, color="k", lw=1.0)
        ax.axvline(sg.t1, color="k", lw=1.0)
        ax.text((sg.t0 + sg.t1) / 2.0, float(sc_sorted[-1]), sg.label,
                ha="center", va="bottom", fontsize=9,
                color=core.STATE_COLOR.get(sg.label, "k"))
    ax.set_xlabel("시간 [s]  (%.0fs 평균)" % HEATMAP_BIN_SEC)
    ax.set_ylabel("subcarrier")
    ax.set_title("%s — subcarrier x 시간 (%s %d개)"
                 % (sess.name, S.ANALYSIS_BLOCK, X.shape[1]), fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("  그림 저장: %s" % path)


# ===========================================================================
# 3. 세션 내 상태 대표값 (대응 비교 그림)
# ===========================================================================

def plot_paired(reps: List[Dict[str, object]], path: str):
    """
    세션마다 상태별 대표값을 찍고 같은 세션을 선으로 잇는다.
    선의 기울기 방향이 세션마다 일치하는지가 이 그림의 요점이다.
    p값은 s2 가 낸다. 여기서는 방향과 크기만 본다.
    """
    states = [st for st in ("empty", "still", "motion")
              if any(st in r["by_state"] for r in reps)]
    if len(states) < 2:
        print("  [skip] 상태가 하나뿐이라 대응 그림을 그리지 않는다.")
        return
    n_f = len(FEATURES)
    ncol = min(4, n_f)
    nrow = int(np.ceil(n_f / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.7 * ncol, 3.5 * nrow),
                             squeeze=False)
    axes = axes.ravel()
    for ax, (f, title) in zip(axes, FEATURES):
        for r in reps:
            xs, ys = [], []
            for xi, st in enumerate(states):
                v = r["by_state"].get(st, {}).get(f, float("nan"))
                if np.isfinite(v):
                    xs.append(xi)
                    ys.append(v)
            if len(xs) >= 2:
                ax.plot(xs, ys, color="#999999", lw=1.0, zorder=2)
            for xi, y in zip(xs, ys):
                ax.scatter([xi], [y], s=46, zorder=3,
                           color=core.STATE_COLOR.get(states[xi], "k"),
                           edgecolor="k", linewidth=0.5)
            if xs:
                ax.annotate(str(r["session"]), (xs[-1], ys[-1]), fontsize=6,
                            xytext=(4, 2), textcoords="offset points")
        ax.set_xticks(range(len(states)))
        ax.set_xticklabels(states)
        ax.set_xlim(-0.4, len(states) - 0.6)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25, axis="y")
    for ax in axes[n_f:]:
        ax.axis("off")
    fig.suptitle("세션 내 상태 대표값 — 선은 같은 세션이다. "
                 "기울기 방향이 모든 세션에서 같은지가 핵심이다.", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("  그림 저장: %s" % path)


# ===========================================================================
# 4. 구간별 PC1 스펙트럼
# ===========================================================================

def _plain_log_labels(ax):
    """
    로그축 눈금을 mathtext 없이 평문으로 적는다.
    한글 폰트(Malgun Gothic 등)에 상단첨자 마이너스(U+207B) 글리프가 없어서
    기본 로그 눈금('10^-1')이 네모로 깨진다. 그래서 직접 적는다.
    """
    from matplotlib.ticker import FixedFormatter, FixedLocator, LogLocator

    xt = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    ax.xaxis.set_major_locator(FixedLocator(xt))
    ax.xaxis.set_major_formatter(FixedFormatter(["%g" % v for v in xt]))
    ax.xaxis.set_minor_locator(LogLocator(subs="all", numticks=100))
    ax.xaxis.set_minor_formatter(FixedFormatter([]))

    lo, hi = ax.get_ylim()
    e0, e1 = int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi)))
    yt = [10.0 ** e for e in range(e0, e1 + 1)]
    ax.yaxis.set_major_locator(FixedLocator(yt))
    ax.yaxis.set_major_formatter(FixedFormatter(["1e%d" % e
                                                 for e in range(e0, e1 + 1)]))
    ax.set_ylim(lo, hi)


def plot_psd(psds: List[Dict[str, object]], path: str):
    """
    구간별 PC1 의 welch 스펙트럼. 호흡 대역을 띠로 표시한다.
    구간 길이가 다르면 welch 평균 횟수가 달라 잡음 바닥이 달라지므로
    nperseg 를 고정하고, 그 길이를 못 채우는 구간은 아예 그리지 않는다.
    """
    if not psds:
        print("  [skip] nperseg 를 채우는 구간이 없다.")
        return
    names = sorted({str(p["session"]) for p in psds})
    ncol = min(2, len(names))
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.4 * ncol, 3.7 * nrow),
                             squeeze=False)
    axes = axes.ravel()
    for ax, nm in zip(axes, names):
        for p in [q for q in psds if str(q["session"]) == nm]:
            ax.loglog(p["f"], p["p"], lw=1.0, alpha=0.85,
                      color=core.STATE_COLOR.get(str(p["state"]), "k"),
                      label="%s@%d" % (p["state"], p["seg"]))
        ax.axvspan(S.BREATH_BAND[0], S.BREATH_BAND[1], color="#ffcc00",
                   alpha=0.18, lw=0)
        ax.set_xlim(0.02, 2.0)
        ax.set_title(nm, fontsize=10)
        ax.set_xlabel("주파수 [Hz]")
        ax.set_ylabel("PC1 PSD")
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=7)
        _plain_log_labels(ax)
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.suptitle("구간별 PC1 스펙트럼 (노란 띠 = 호흡 대역 %.2f-%.2f Hz). "
                 "그림에서 고른 대역으로 같은 데이터를 다시 검정하면 순환논증이다."
                 % S.BREATH_BAND, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("  그림 저장: %s" % path)


# ===========================================================================
# main
# ===========================================================================

def main() -> int:
    from scipy.signal import welch

    print(RULE)
    print("s6_visualize - 세션 타임라인 시각화")
    print(RULE)
    sessions = S.load_all()

    reps: List[Dict[str, object]] = []
    psds: List[Dict[str, object]] = []
    csv_rows: List[str] = []

    for sess in sessions:
        print()
        print("[%s] %s  구간 %d개" % (sess.name, sess.scenario, len(sess.segments)))
        t, X, mask = prep(sess)

        plot_timeline(sess, t, X,
                      os.path.join(S.OUT_DIR, "s6_timeline_%s.png" % sess.name))
        plot_heatmap(sess, t, X, mask,
                     os.path.join(S.OUT_DIR, "s6_heatmap_%s.png" % sess.name))

        acc: Dict[str, Dict[str, List[float]]] = {}
        for k, sg in enumerate(sess.segments):
            a = sg.t0 + S.SEGMENT_GUARD_SEC
            b = sg.t1 - S.SEGMENT_GUARD_SEC
            wf = segment_windows(t, X, a, b)
            n_win = int(wf["t"].size)
            cells = []
            for f, _ in FEATURES:
                v = wf[f][np.isfinite(wf[f])]
                med = float(np.median(v)) if v.size else float("nan")
                acc.setdefault(sg.label, {}).setdefault(f, []).append(med)
                cells.append("%s=%.4f" % (f, med))
            print("  %-6s %6.1f-%6.1f  창 %2d개  %s"
                  % (sg.label, sg.t0, sg.t1, n_win, "  ".join(cells)))
            for i in range(n_win):
                csv_rows.append("%s,%s,%s,%s,%d,%.1f,%s\n" % (
                    sess.name, sess.subject, sess.batch, sg.label, k, wf["t"][i],
                    ",".join("%.6f" % wf[f][i] for f, _ in FEATURES)))

            # PC1 스펙트럼. 스펙트럼 계열이므로 균일 격자에 올린 뒤 계산한다.
            m = (t >= a) & (t < b)
            if m.sum() > 200:
                _, Y = core.resample_uniform(t[m], X[m], S.RESAMPLE_FS)
                if Y.shape[0] >= PSD_NPERSEG:
                    p1, _ = core.pc1(Y)
                    f_, pxx = welch(p1, fs=S.RESAMPLE_FS, nperseg=PSD_NPERSEG)
                    psds.append({"session": sess.name, "state": sg.label,
                                 "seg": k, "f": f_, "p": pxx})
                else:
                    print("     [note] %s@%d 은 %.0f초라 nperseg %d 을 못 채운다"
                          " -> 스펙트럼 제외 (조건 통일)"
                          % (sg.label, k, b - a, PSD_NPERSEG))

        # 같은 라벨 구간이 여러 개면 그 중앙값을 세션 대표값으로 쓴다 (s2 와 같은 규칙).
        by_state = {st: {f: float(np.nanmedian(d[f])) for f, _ in FEATURES}
                    for st, d in acc.items()}
        reps.append({"session": sess.name, "subject": sess.subject,
                     "batch": sess.batch, "by_state": by_state})

    print()
    print(RULE)
    plot_paired(reps, os.path.join(S.OUT_DIR, "s6_paired_states.png"))
    plot_psd(psds, os.path.join(S.OUT_DIR, "s6_psd.png"))

    csv_path = os.path.join(S.OUT_DIR, "s6_windows.csv")
    with open(csv_path, "w", encoding="utf-8-sig") as fh:
        fh.write("session,subject,batch,state,seg_index,t_center,%s\n"
                 % ",".join(f for f, _ in FEATURES))
        fh.writelines(csv_rows)
    print("  CSV 저장: %s  (%d행)" % (csv_path, len(csv_rows)))

    print()
    print("읽는 법")
    print(SUB)
    print("  1) 값이 바뀌는 시각이 라벨 경계와 어긋나면 sessions.py 의 t_offset 을")
    print("     그 차이만큼 고친다. 그림을 보고 라벨을 옮기는 것이 아니라, 기록된")
    print("     프로토콜과 로그 시각의 정렬을 맞추는 작업이다.")
    print("  2) 같은 라벨 구간 안에서 값이 한 방향으로 계속 흐르면 환경 드리프트다.")
    print("     세션 내 대응 비교가 이 문제에 대한 정공법이다.")
    print("  3) empty 두 구간(전/후)이 서로 다르면 그 세션의 empty 를 하나의 상태로")
    print("     묶어 쓰는 것 자체를 의심해야 한다.")
    print("  4) 이 그림들은 가설 생성용이다. 여기서 눈으로 고른 구간·대역·특징으로")
    print("     같은 데이터를 다시 검정하면 p값은 의미를 잃는다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
