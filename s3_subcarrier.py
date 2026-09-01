# -*- coding: utf-8 -*-
"""
s3_subcarrier.py — subcarrier 단위 분석

이 스크립트가 판정하는 것: "특정 subcarrier 가 클래스를 가르는가, 아니면
그렇게 보이는 것이 우연인가".

핵심 규칙 두 가지.
  1) 효과크기의 분모는 '세션 간' pooled SD 다. (결함 D5)
     세션 내 시간 표준편차로 나누면 자기상관 때문에 Cohen's d 가 아니다.
  2) subcarrier 를 수백 개 스크리닝하면 우연히 갈리는 것이 반드시 나온다.
     그래서 '완전 분리되는 subcarrier 개수' 를 '무작위여도 기대되는 개수' 와
     나란히 출력한다. 두 값이 비슷하면 잡음을 고른 것이다.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np

import csi_core as core
import sessions as S

plt = core.setup_matplotlib()

RULE = "=" * 104
SUB = "-" * 104


def session_profiles(sessions):
    """세션×구간마다 subcarrier 프로파일(mean/std/mad)을 만든다."""
    out = []
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
        for k, sg in enumerate(s.segments):
            m = sg.mask(s.t, guard=S.SEGMENT_GUARD_SEC)
            if m.sum() < 50:
                continue
            Xi = X[m]
            ti = s.t[m]
            _, d, _ = core.diff_matched_dt(ti, Xi)
            out.append({
                "session": s.name, "state": sg.label, "subject": s.subject,
                "batch": s.batch, "seg": k,
                "mean": Xi.mean(axis=0),
                "std": Xi.std(axis=0, ddof=1),
                "mad": d.mean(axis=0),
            })
    return out, mask


def per_session(profs, state, key, filt=None):
    """세션당 프로파일 1개로 합친다 (같은 상태 구간이 여러 개면 중앙값)."""
    by: Dict[str, List[np.ndarray]] = {}
    for p in profs:
        if p["state"] != state:
            continue
        if filt is not None and p["session"] not in filt:
            continue
        by.setdefault(p["session"], []).append(p[key])
    names = sorted(by)
    return names, np.array([np.median(np.stack(by[n]), axis=0) for n in names])


def screen(A, B, names_a, names_b, sc, label):
    """subcarrier 스크리닝. 효과크기 / 완전분리 / 우연 기대치."""
    na, nb = A.shape[0], B.shape[0]
    minp = core.min_two_sided_p_unpaired(na, nb)
    n_valid = A.shape[1]

    print()
    print(RULE)
    print("subcarrier 스크리닝 — %s   (A %d세션 vs B %d세션, 유효 %d개)"
          % (label, na, nb, n_valid))
    print(RULE)
    print("  도달 가능한 최소 양측 p = %.4f" % minp)

    d = np.empty(n_valid)
    ps = np.empty(n_valid)
    fullsep = np.zeros(n_valid, dtype=bool)
    for j in range(n_valid):
        a, b = A[:, j], B[:, j]
        d[j] = core.cohens_d_sessions(a, b)
        _, ps[j], _ = core.exact_perm_test_unpaired(a, b)
        fullsep[j] = (a.min() > b.max()) or (b.min() > a.max())

    rej, padj = core.benjamini_hochberg(ps, q=0.05)
    n_sep = int(fullsep.sum())
    expect = n_valid * minp          # 무작위여도 우연히 완전분리될 개수의 기댓값

    print("  완전 분리되는 subcarrier : %d 개" % n_sep)
    print("  무작위여도 기대되는 개수  : %.1f 개  (유효수 %d × 최소p %.4f)"
          % (expect, n_valid, minp))
    if expect > 0 and n_sep <= expect * 1.5:
        print("  -> 두 값이 비슷하다. 이 분리는 잡음을 고른 것으로 봐야 한다.")
    elif n_sep == 0:
        print("  -> 완전 분리되는 subcarrier 가 하나도 없다.")
    else:
        print("  -> 기대치보다 %.1f 배 많다. 다만 이것만으로 인과를 주장할 수는 없다."
              % (n_sep / expect if expect else float("inf")))
    print("  BH(FDR 0.05) 통과 subcarrier : %d 개" % int(rej.sum()))

    order = np.argsort(-np.abs(d))[:20]
    print()
    print("  |효과크기| 상위 20 (효과크기 분모 = 세션 간 pooled SD)")
    print("  %-5s %-8s %9s %9s %9s %9s %-9s" %
          ("rank", "sc", "A중앙", "B중앙", "d", "p", "완전분리"))
    print("  " + SUB)
    for r, j in enumerate(order, 1):
        print("  %-5d %-8d %9.4f %9.4f %+9.2f %9.4f %-9s"
              % (r, sc[j], np.median(A[:, j]), np.median(B[:, j]),
                 d[j], ps[j], "예" if fullsep[j] else "아니오"))
    return d, ps, fullsep


def plot_profiles(profs, mask, sc, path):
    """세션마다 한 선. mean / std / mad 3단. 상태별 색."""
    # HT-LTF 의 index 순서는 +2..+58, -58..-2 이므로 그대로 그리면 선이 되감긴다.
    o = np.argsort(sc)
    scs = sc[o]
    keys = [("mean", "정규화 평균진폭"), ("std", "시간 표준편차"),
            ("mad", "프레임간 절대변화량")]
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for ax, (k, title) in zip(axes, keys):
        for p in profs:
            ax.plot(scs, p[k][o], lw=1.0, alpha=0.8,
                    color=core.STATE_COLOR.get(p["state"], "#888"))
        ax.set_ylabel(title, fontsize=9)
        ax.grid(alpha=0.25)
        ax.axvline(0, color="k", lw=0.6, ls=":")
    handles = [plt.Line2D([], [], color=c, lw=2, label=s)
               for s, c in core.STATE_COLOR.items() if s != "mixed"]
    axes[0].legend(handles=handles, fontsize=9, ncol=3, loc="lower center")
    axes[-1].set_xlabel("subcarrier 번호 (HT-LTF, 40MHz)")
    fig.suptitle("세션·구간별 subcarrier 프로파일 (선 1개 = 세션 구간 1개)", y=0.995)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("  그림 저장: %s" % path)


def plot_class_mean(profs, sc, path):
    """클래스 평균 ± 세션 간 범위. 겹치면 구분 불가라는 게 한눈에 보인다."""
    o = np.argsort(sc)
    scs = sc[o]
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for ax, key, title in [(axes[0], "mean", "정규화 평균진폭"),
                           (axes[1], "std", "시간 표준편차")]:
        for st in ("still", "motion", "empty"):
            names, M = per_session(profs, st, key)
            if M.size == 0:
                continue
            M = M[:, o]
            c = core.STATE_COLOR[st]
            ax.plot(scs, M.mean(axis=0), color=c, lw=1.8,
                    label="%s (n=%d세션)" % (st, M.shape[0]))
            ax.fill_between(scs, M.min(axis=0), M.max(axis=0), color=c, alpha=0.18)
        ax.set_ylabel(title, fontsize=9)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    axes[-1].set_xlabel("subcarrier 번호 (HT-LTF, 40MHz)")
    fig.suptitle("클래스 평균과 세션 간 최소~최대 범위. 띠가 겹치면 세션 단위로 구분되지 않는다.",
                 y=0.98)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("  그림 저장: %s" % path)


def plot_heatmap(profs, sc, path):
    """세션×subcarrier 히트맵. 같은 클래스끼리 무늬가 닮았는지 본다."""
    o = np.argsort(sc)
    scs = sc[o]
    rows = [p for p in profs if p["state"] in ("still", "empty", "motion")]
    rows = sorted(rows, key=lambda p: (p["state"], p["session"], p["seg"]))
    M = np.stack([p["mean"][o] for p in rows])
    labels = ["%s / %s" % (p["session"], p["state"]) for p in rows]
    fig, ax = plt.subplots(figsize=(13, 0.34 * len(rows) + 2.4))
    im = ax.imshow(M, aspect="auto", cmap="viridis",
                   extent=[scs[0], scs[-1], len(rows) - 0.5, -0.5])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("subcarrier 번호")
    for i, p in enumerate(rows):
        ax.add_patch(plt.Rectangle((scs[0] - 3.5, i - 0.5), 2.5, 1.0,
                                   color=core.STATE_COLOR[p["state"]], clip_on=False))
    fig.colorbar(im, ax=ax, label="정규화 평균진폭")
    ax.set_title("세션×subcarrier 진폭 히트맵 (왼쪽 색막대 = 상태)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("  그림 저장: %s" % path)


def main() -> int:
    print(RULE)
    print("s3_subcarrier.py — subcarrier 단위 분석")
    print(RULE)
    sessions = S.load_all(verbose=False)
    profs, mask = session_profiles(sessions)
    names_all, scs_all = core.block_subcarriers(192, 0)
    sc = scs_all[mask]
    print("유효 subcarrier %d개 (%s), 프로파일 %d개"
          % (mask.sum(), S.ANALYSIS_BLOCK, len(profs)))

    still_only = {s.name for s in sessions
                  if [sg.label for sg in s.segments] == ["still"]}

    na, A = per_session(profs, "still", "mean", filt=still_only)
    nb, B = per_session(profs, "empty", "mean")
    print("\n[비교 1] 정지 전용 세션 %s  vs  비재실 %s" % (na, nb))
    d1, p1, f1 = screen(A, B, na, nb, sc, "재실(정지) vs 비재실 — 평균진폭")

    na2, A2 = per_session(profs, "still", "mean")
    print("\n[비교 2] 재실 전체 %s  vs  비재실 %s" % (na2, nb))
    d2, p2, f2 = screen(A2, B, na2, nb, sc, "재실 전체 vs 비재실 — 평균진폭")

    # 세션 내 대조: 움직임 vs 정지 (같은 세션)
    print()
    print(RULE)
    print("세션 내 대조 — 움직임 구간 vs 정지 구간 (같은 세션, 시간 표준편차)")
    print(RULE)
    per: Dict[str, Dict[str, List[np.ndarray]]] = {}
    for p in profs:
        per.setdefault(p["session"], {}).setdefault(p["state"], []).append(p["std"])
    pair_names = [s for s, dd in per.items() if "motion" in dd and "still" in dd]
    pair_names.sort()
    if pair_names:
        D = np.stack([np.median(np.stack(per[s]["motion"]), axis=0)
                      - np.median(np.stack(per[s]["still"]), axis=0)
                      for s in pair_names])
        minp = core.min_two_sided_p_paired(len(pair_names))
        same_sign = np.all(D > 0, axis=0) | np.all(D < 0, axis=0)
        print("  대조 가능 세션 %d개: %s" % (len(pair_names), ", ".join(pair_names)))
        print("  도달 가능한 최소 양측 p = %.4f" % minp)
        print("  모든 세션에서 같은 방향으로 움직인 subcarrier: %d / %d (%.1f%%)"
              % (int(same_sign.sum()), D.shape[1],
                 100.0 * same_sign.sum() / D.shape[1]))
        print("  무작위여도 기대되는 비율: %.1f%%  (2 / 2^%d)"
              % (100.0 * 2 / 2 ** len(pair_names), len(pair_names)))
        print("  차이 중앙값: %+.5f  (양수면 움직임 구간의 변동이 더 크다)"
              % float(np.median(D)))

    plot_profiles(profs, mask, sc, os.path.join(S.OUT_DIR, "s3_profiles.png"))
    plot_class_mean(profs, sc, os.path.join(S.OUT_DIR, "s3_class_mean.png"))
    plot_heatmap(profs, sc, os.path.join(S.OUT_DIR, "s3_heatmap.png"))

    csv_path = os.path.join(S.OUT_DIR, "s3_subcarrier_stats.csv")
    with open(csv_path, "w", encoding="utf-8-sig") as fh:
        fh.write("subcarrier,still_mean,empty_mean,d_still_vs_empty,p_perm,full_sep\n")
        for j in range(len(sc)):
            fh.write("%d,%.6f,%.6f,%.4f,%.4f,%d\n"
                     % (sc[j], np.median(A2[:, j]), np.median(B[:, j]),
                        d2[j], p2[j], int(f2[j])))
    print("  CSV 저장: %s" % csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
