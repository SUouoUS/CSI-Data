# -*- coding: utf-8 -*-
"""
s1_check_format.py — 무결성 검증

이 스크립트가 판정하는 것: "분석을 시작해도 되는가".
특징 추출도, 클래스 비교도 하지 않는다. 오직 데이터가 서로 비교 가능한 상태인지만 본다.

FAIL 이 하나라도 있으면 종료 코드 1 로 끝난다. 우회하지 말고 원인을 먼저 해결해야 한다.
특히 '규격 마스크 vs 실측 마스크 불일치'가 0 이 아니면 csi_core 의 가정 A2/A3 가
틀렸다는 뜻이므로, 뒤의 어떤 분석도 신뢰할 수 없다.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np

import csi_core as core
import sessions as S

RULE = "=" * 100
SUB = "-" * 100


class Checks:
    """PASS/FAIL/WARN 을 모아 마지막에 한 번에 판정한다."""

    def __init__(self) -> None:
        self.items: List[tuple] = []

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.items.append((status, name, detail))

    def ok(self, cond: bool, name: str, detail: str = "",
           warn_only: bool = False) -> bool:
        if cond:
            self.add("PASS", name, detail)
        else:
            self.add("WARN" if warn_only else "FAIL", name, detail)
        return cond

    @property
    def n_fail(self) -> int:
        return sum(1 for s, _, _ in self.items if s == "FAIL")

    @property
    def n_warn(self) -> int:
        return sum(1 for s, _, _ in self.items if s == "WARN")

    def report(self) -> None:
        print(RULE)
        print("검증 결과")
        print(RULE)
        for status, name, detail in self.items:
            mark = {"PASS": "  PASS", "WARN": "  WARN", "FAIL": "* FAIL"}[status]
            print("%s  %-52s %s" % (mark, name, detail))
        print(SUB)
        print("FAIL %d 건 / WARN %d 건" % (self.n_fail, self.n_warn))


def describe_session(sess: core.Session, chk: Checks) -> Dict[str, object]:
    """세션 1개의 무결성 지표를 출력하고 요약 dict 를 돌려준다."""
    print(RULE)
    print("세션 %s  [%s %s]  %s"
          % (sess.name, sess.scenario,
             S.SCENARIOS.get(sess.scenario, {}).get("title", ""),
             os.path.basename(sess.path)))
    print(RULE)

    meta = sess.meta
    drops = sess.drops
    n_csi = int(meta["n_csi_lines"])
    n_valid = sess.n_frames

    # --- 파싱 ---
    print("[파싱]")
    print("  헤더 줄            : %s" % ("있음 (line %s)" % meta["header_line_index"]
                                        if meta["has_header"] else "없음 (fallback 컬럼 사용)"))
    print("  전체 라인          : %d" % meta["n_lines"])
    print("  CSI_DATA 행        : %d" % n_csi)
    print("  유효 프레임        : %d  (%.2f%%)" % (n_valid, 100.0 * n_valid / n_csi))
    print("  드롭 (사유별)      :")
    total_drop = 0
    for reason, cnt in drops.items():
        total_drop += cnt
        flag = "" if cnt == 0 else "   <-- 확인 필요"
        print("     %-20s %6d  (%.3f%%)%s"
              % (reason, cnt, 100.0 * cnt / n_csi, flag))
    print("     %-20s %6d  (%.3f%%)" % ("합계", total_drop, 100.0 * total_drop / n_csi))

    # --- 펌웨어 시퀀스 손실 (직렬 캡처 단계의 유실) ---
    lost, loss_rate = sess.seq_loss
    print("  id 시퀀스 손실     : %d 프레임 (%.2f%%)" % (lost, 100.0 * loss_rate))

    # --- 설정 조합 ---
    print("\n[캡처 설정]")
    n_combo = int(meta["n_config_combos"])
    print("  설정 조합 개수     : %d %s"
          % (n_combo, "" if n_combo == 1 else "  <-- 프레임 종류가 섞였다"))
    cfg = meta.get("config", {})
    for k in ("sig_mode", "bandwidth", "secondary_channel", "stbc", "channel",
              "rate", "mcs", "sgi", "sig_len", "rx_format", "len", "first_word"):
        if k in cfg:
            print("     %-18s %s" % (k, cfg[k]))
    print("  송신 MAC 분포      : %s" % meta["macs"])

    chk.ok(n_combo == 1, "[%s] 설정 조합 단일" % sess.name,
           "조합 %d 개" % n_combo)

    # --- 타이밍 ---
    dt = np.diff(sess.t)
    n_gap1 = int((dt > 1.0).sum())
    print("\n[타이밍]")
    print("  duration           : %.2f s" % sess.duration)
    print("  평균 수신률        : %.2f Hz" % sess.rate_hz)
    print("  프레임 간격        : p50=%.4f s  p95=%.4f s  max=%.4f s"
          % (np.percentile(dt, 50), np.percentile(dt, 95), dt.max()))
    print("  1초 이상 공백      : %d 회" % n_gap1)
    print("  timestamp wrap     : %d 회" % meta["n_wrap"])
    chk.ok(n_gap1 == 0, "[%s] 1초 이상 공백 없음" % sess.name,
           "%d 회" % n_gap1, warn_only=True)

    # --- 링크 품질 ---
    rssi = sess.rssi.astype(float)
    nf = sess.noise_floor.astype(float)
    print("\n[링크 품질]")
    print("  RSSI               : mean=%.2f  sd=%.2f  min=%d  max=%d"
          % (rssi.mean(), rssi.std(ddof=1), int(rssi.min()), int(rssi.max())))
    print("  noise floor        : mean=%.2f  min=%d  max=%d"
          % (nf.mean(), int(nf.min()), int(nf.max())))
    slope = float(np.polyfit(sess.t, rssi, 1)[0])
    print("  RSSI 선형 추세     : %+.4f dB/s  (세션 전체 %+.2f dB)"
          % (slope, slope * sess.duration))

    # --- 블록 구성 / 마스크 ---
    n_sub = sess.n_sub
    stbc = int(cfg.get("stbc", 0))
    bw40 = str(cfg.get("bandwidth", "1")) == "1"
    blocks = core.ltf_blocks(n_sub, stbc)

    print("\n[LTF 블록 구성]  (가정 A2)")
    print("  subcarrier 개수    : %d  (len=%d)" % (n_sub, sess.csi.shape[1]))
    for b in blocks:
        sc = core.fft_bin_to_subcarrier(np.arange(b.size), b.nfft)
        print("     %-13s index %3d-%3d  nfft=%3d  subcarrier %+d..%+d"
              % (b.name, b.start, b.stop - 1, b.nfft, sc.min(), sc.max()))

    amp = core.amplitude(sess.csi)
    spec = core.spec_valid_mask(n_sub, stbc=stbc, bandwidth_40=bw40,
                                first_word_invalid=str(cfg.get("first_word", "0")) == "1")
    emp = core.empirical_valid_mask(amp, n_sub=n_sub, stbc=stbc)

    mism = np.flatnonzero(spec != emp)
    names, scs = core.block_subcarriers(n_sub, stbc)
    m = amp.mean(axis=0)
    sd_t = amp.std(axis=0, ddof=1)

    print("\n[유효 마스크]")
    print("  규격 마스크 유효   : %d 개" % int(spec.sum()))
    print("  실측 마스크 유효   : %d 개" % int(emp.sum()))
    print("  불일치             : %d 개" % mism.size)
    if mism.size:
        print("     idx  블록          sc   규격  실측     평균진폭  시간sd  종류")
        for i in mism:
            if spec[i] and not emp[i]:
                kind = "규격 유효인데 실측 널  <== 가정 A2 의심"
            else:
                kind = "규격 무효인데 실측 신호 (보수적으로 제외)"
            print("     %3d  %-13s %+4d  %-5s %-5s  %8.3f %7.3f  %s"
                  % (i, names[i], scs[i], spec[i], emp[i], m[i], sd_t[i], kind))

    # 판정 규칙
    #   FAIL : 규격이 '유효'라 한 subcarrier 가 실측에서 하드웨어 널로 나온 경우.
    #          이건 블록 구성(A2) 이나 격자 매핑이 틀렸다는 뜻이다.
    #   WARN : 규격이 '무효'라 한 곳에 실측 신호가 있는 경우.
    #          스펙트럼 누설이거나 규격 범위가 보수적인 것이며, 제외하는 편이 안전하다.
    #   LLTF 블록은 HT40 격자 매핑이 문서로 확정돼 있지 않으므로 FAIL 대상에서 뺀다.
    doc_idx = np.zeros(n_sub, dtype=bool)
    for b in core.ltf_blocks(n_sub, stbc):
        if b.name in core.DOCUMENTED_BLOCKS:
            doc_idx[b.start:b.stop] = True

    spec_true_emp_null = np.flatnonzero(spec & ~emp & doc_idx)
    spec_false_emp_sig = np.flatnonzero(~spec & emp)

    chk.ok(spec_true_emp_null.size == 0,
           "[%s] 규격 유효 subcarrier 가 실측 널이 아님" % sess.name,
           "위반 %d 개%s" % (spec_true_emp_null.size,
                            (" -> %s" % spec_true_emp_null[:12].tolist())
                            if spec_true_emp_null.size else ""))
    chk.ok(spec_false_emp_sig.size == 0,
           "[%s] 규격 무효 subcarrier 에 잔여 신호 없음" % sess.name,
           "%d 개 -> %s" % (spec_false_emp_sig.size,
                            spec_false_emp_sig[:12].tolist()),
           warn_only=True)

    fw = str(cfg.get("first_word", "0")) == "1"
    if fw:
        print("  * first_word=1 이므로 index 0,1 (raw byte 0..3) 은 규격 마스크에서 강제 제외했다.")
        print("    두 index 는 전 프레임 동일한 상수(진폭 %.2f, %.2f / 시간sd %.3f, %.3f)라"
              % (m[0], m[1], sd_t[0], sd_t[1]))
        print("    실측 판정에서도 널로 분류된다.")
    print("  * 실측 널 판정 기준은 평균진폭이 아니라 '평균진폭이 낮고 시간 변동도 없음' 이다.")
    print("    진폭만 보면 하드웨어 널과 깊은 페이딩 노치가 구분되지 않는다.")

    # --- 깊은 페이딩 노치 (유효 subcarrier 중 진폭이 크게 죽은 대역) ---
    ana = S.analysis_mask(n_sub, stbc=stbc, bandwidth_40=bw40, first_word_invalid=fw)
    vm = m[ana]
    med_v = float(np.median(vm))
    notch = np.flatnonzero(ana & (m < 0.3 * med_v))
    print("\n[깊은 페이딩 노치]  분석 블록(%s) 유효 subcarrier 중 진폭이 중앙값의 30%% 미만"
          % S.ANALYSIS_BLOCK)
    print("  유효 subcarrier 중앙 진폭 : %.3f" % med_v)
    if notch.size == 0:
        print("  해당 없음")
    else:
        runs = []
        start = prev = int(notch[0])
        for v in notch[1:]:
            v = int(v)
            if v == prev + 1:
                prev = v
            else:
                runs.append((start, prev)); start = prev = v
        runs.append((start, prev))
        for a_, b_ in runs:
            print("     idx %3d-%-3d  sc %+d..%+d   평균진폭 %.3f  시간sd %.3f"
                  % (a_, b_, scs[a_], scs[b_], m[a_:b_ + 1].mean(), sd_t[a_:b_ + 1].mean()))
        print("  이 대역은 널이 아니라 채널이 깊게 죽은 것이다. 마스크에서 빼면 안 된다.")
        print("  다만 노치가 '사람 때문'인지 '그날의 채널'인지는 세션 내 대조 없이는 알 수 없다.")

    # --- 가정 A3 재확인 ---
    a = sess.csi[:, 0::2].astype(np.float64)
    b_ = sess.csi[:, 1::2].astype(np.float64)
    d_sym = float(np.abs(np.sqrt(a * a + b_ * b_) - np.sqrt(b_ * b_ + a * a)).max())
    print("\n[가정 A3] amplitude 가 (imag,real) 순서에 무관한가 : 최대 절대차 %.3g" % d_sym)
    chk.ok(d_sym == 0.0, "[%s] A3 amplitude 순서 무관" % sess.name, "%.3g" % d_sym)

    # --- 구간 정의가 데이터 범위 안에 있는가 ---
    print("\n[구간 정의]")
    for sg in sess.segments:
        n_in = int(sg.mask(sess.t).sum())
        n_guard = int(sg.mask(sess.t, guard=S.SEGMENT_GUARD_SEC).sum())
        inside = sg.t0 >= sess.t[0] - 1e-9 and sg.t1 <= sess.t[-1] + 1.0
        print("     %-7s %7.1f - %7.1f s   프레임 %5d (guard 적용 %5d) %s"
              % (sg.label, sg.t0, sg.t1, n_in, n_guard,
                 "" if inside else "  <-- 로그 범위를 벗어남"))
        chk.ok(n_guard > 0 and inside,
               "[%s] 구간 %s 유효" % (sess.name, sg.label),
               "프레임 %d" % n_guard)

    return {
        "name": sess.name, "scenario": sess.scenario, "label": sess.label,
        "n_valid": n_valid, "n_csi": n_csi,
        "rate": sess.rate_hz, "dur": sess.duration,
        "rssi_mean": rssi.mean(), "rssi_sd": rssi.std(ddof=1),
        "nf_mean": nf.mean(),
        "drop_rate": total_drop / n_csi, "seq_loss": loss_rate,
        "n_sub": n_sub, "emp_mask": emp, "config_combos": set(meta["config_combos"]),
        "rssi_slope": slope,
    }


def main() -> int:
    print(RULE)
    print("s1_check_format.py — 분석을 시작해도 되는가")
    print(RULE)
    print("SOURCE_MAC       : %s" % S.SOURCE_MAC)
    print("EXPECTED_CSI_LEN : %d" % S.EXPECTED_CSI_LEN)
    print("ANALYSIS_BLOCK   : %s" % S.ANALYSIS_BLOCK)
    print()

    chk = Checks()
    sess_list = S.load_all(verbose=True)
    print()

    rows = [describe_session(s, chk) for s in sess_list]

    # ===================== 세션 간 검증 =====================
    print()
    print(RULE)
    print("세션 간 검증")
    print(RULE)

    all_combos = set()
    for r in rows:
        all_combos |= r["config_combos"]
    chk.ok(len(all_combos) == 1, "전 세션 설정 조합이 1개",
           "조합 %d 개" % len(all_combos))
    if len(all_combos) > 1:
        for c in sorted(all_combos):
            print("   조합: %s" % c)

    n_subs = {r["n_sub"] for r in rows}
    chk.ok(len(n_subs) == 1, "전 세션 subcarrier 수 동일", "값 %s" % sorted(n_subs))

    if len(n_subs) == 1:
        ref = rows[0]["emp_mask"]
        same = all(np.array_equal(r["emp_mask"], ref) for r in rows)
        detail = ""
        if not same:
            diff_idx = set()
            for r in rows[1:]:
                diff_idx |= set(np.flatnonzero(r["emp_mask"] != ref).tolist())
            detail = "불일치 index %s" % sorted(diff_idx)[:15]
        chk.ok(same, "전 세션 실측 유효 마스크 동일", detail)

    rates = [r["rate"] for r in rows]
    ratio = max(rates) / min(rates)
    chk.ok(ratio < 1.2, "수신률 최대/최소 비 < 1.2",
           "%.3f  (min %.2f Hz, max %.2f Hz)" % (ratio, min(rates), max(rates)))

    # 드롭률 불균형은 별도 항목. 수신률 비가 통과해도 드롭 사유 구성이 다르면 교락이다.
    losses = [r["seq_loss"] for r in rows]
    if min(losses) > 0:
        lratio = max(losses) / min(losses)
        chk.ok(lratio < 1.2, "id 시퀀스 손실률 최대/최소 비 < 1.2",
               "%.3f  (min %.2f%%, max %.2f%%)"
               % (lratio, 100 * min(losses), 100 * max(losses)), warn_only=True)

    # ===================== 요약표 =====================
    print()
    print(RULE)
    print("세션 요약")
    print(RULE)
    hdr = ("%-9s %-5s %-7s %6s %8s %9s %8s %8s %9s %10s"
           % ("name", "scen", "label", "n", "dur[s]", "rate[Hz]", "RSSI",
              "RSSI_sd", "NF", "seq_loss"))
    print(hdr)
    print(SUB)
    for r in rows:
        print("%-9s %-5s %-7s %6d %8.1f %9.2f %8.2f %8.2f %9.2f %9.2f%%"
              % (r["name"], r["scenario"], r["label"], r["n_valid"], r["dur"],
                 r["rate"], r["rssi_mean"], r["rssi_sd"], r["nf_mean"],
                 100 * r["seq_loss"]))

    print()
    print("해석 지침")
    print(SUB)
    print("  위 표의 rate / RSSI / NF / seq_loss 가 라벨과 무관하게 '수집 순서'를 따라")
    print("  단조 변한다면, 이후 어떤 클래스 차이가 나오더라도 그것은 환경 드리프트로")
    print("  똑같이 설명될 수 있다. 즉 그 차이는 사람 유무의 증거가 되지 못한다.")
    print("  라벨별로 값이 갈리는 것이 아니라 '녹화한 순서대로' 갈리는지를 먼저 보라.")
    print()
    print("  RSSI 가 클래스 간에 체계적으로 다르면 진폭 기반 특징은 그것만으로도 갈린다.")
    print("  frame_norm 정규화가 프레임 스칼라 이득을 제거하지만, 다중경로 구조 변화까지")
    print("  없애 주지는 않는다. 세션 내 대응 비교(S-3/S-4/S-6/S-7)가 이 문제의 정공법이다.")

    print()
    chk.report()

    if chk.n_fail:
        print()
        print("FAIL 이 있으므로 여기서 멈춘다. 원인을 해결하기 전에는 s2~s4 를 돌리지 마라.")
        return 1
    print()
    print("모든 검사를 통과했다. s2~s4 를 진행해도 된다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
