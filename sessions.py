# -*- coding: utf-8 -*-
"""
sessions.py — 데이터 매니페스트 겸 전역 설정

모든 스크립트는 여기서만 '어떤 파일이 어떤 상태인가'를 읽는다.
새 녹화를 추가할 때 손대는 파일도 여기 하나뿐이다.

실험 시나리오 (S-1 ~ S-7)
---------------------------------------------------------------------------
  S-1  정상 수면        still(2분)
  S-2  이탈             empty(5분)
  S-3  장기 이탈        still(1분) -> empty(3~5분)
  S-4  복귀             empty(1분) -> still(2분)
  S-5  수면 중 뒤척임   still(1분) -> motion(1분) -> still(2분)
  S-6  실제 이탈        still(2분) -> motion(이탈) -> empty(2분)
  S-7  단기 이탈 후 복귀 still(1분) -> empty(30초) -> still(2분)

상태 라벨은 still(재실·정지) / empty(비재실) / motion(움직임) 세 가지다.

이 설계의 통계적 의미
---------------------------------------------------------------------------
S-3, S-4, S-6, S-7 은 한 세션 안에 still 과 empty 가 모두 있다.
따라서 still 대 empty 를 '세션 간'이 아니라 '세션 내 대응(paired)'으로 비교할 수 있고,
세션 간 환경 드리프트라는 교락 요인이 구조적으로 상쇄된다.
이는 세션 간 비교보다 훨씬 강한 설계다.

다만 대응 부호뒤집기 순열검정의 최소 양측 p 는 2 / 2^n 이므로
  n=4 -> 0.125 / n=5 -> 0.0625 / n=6 -> 0.03125
대조를 제공하는 세션이 6개 이상이어야 p<0.05 가 원리적으로 가능해진다.
즉 시나리오당 반복 녹화가 필요하다. 이 값은 s2 가 매 실행마다 다시 출력한다.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

import csi_core as core

# ===========================================================================
# 경로
# ===========================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# 결과 경로. run_analysis.py 가 분석마다 다른 폴더를 지정한다.
# 환경변수를 쓰는 이유: 각 스크립트의 계산식을 건드리지 않고 출력 위치만 바꾸기 위해서다.
DEFAULT_OUT_DIR = os.path.join(BASE_DIR, "out")
OUT_DIR = os.environ.get("CSI_OUT_DIR", DEFAULT_OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# 분석 대상 배치. 미설정이면 "B" 이므로 기존 동작과 완전히 같다.
# 이 값 하나가 세션 선택과 출력 파일명 접미사를 동시에 정한다.
BATCH = (os.environ.get("CSI_BATCH", "").strip() or "B").upper()
BATCH_TAG = "batch" + BATCH        # 파일명 접미사: batchB / batchC

# ===========================================================================
# 캡처 설정
# ===========================================================================

EXPECTED_CSI_LEN = 384          # 40MHz HT non-STBC -> 192 complex

# 송신 노드 MAC. None 이면 MAC 필터를 하지 않는다.
# 반드시 실제 송신기 MAC 으로 채워야 한다. 비워 두면 주변 AP 의 CSI 가 섞여 들어와
# 모든 비교가 오염된다. (현재 값은 두 로그에서 관측된 유일한 송신 MAC)
SOURCE_MAC: Optional[str] = "1a:00:00:00:00:00"

# ===========================================================================
# 전처리 / 특징 파라미터
# ===========================================================================

NORMALIZE = "frame_norm"        # AGC 이득 제거. "none" 으로 두면 비정규화 비교.

WINDOW_SEC = 10.0               # 특징 추출 창 길이
WINDOW_STEP_SEC = 10.0          # 창 이동량. WINDOW_SEC 과 같아야 한다(중첩 금지).

RESAMPLE_FS = 20.0              # 스펙트럼 계열 특징용 균일 격자 [Hz]
#   실제 수신률은 17-18 Hz 라 20 Hz 는 소폭 업샘플링이다.
#   보간은 인접 표본을 서로 닮게 만들어 frame-to-frame 변화량을 과소평가하므로,
#   변화량 계열 특징(mad_diff, motion indicator)은 리샘플하지 않고
#   csi_core.diff_matched_dt() 로 원시 시간축에서 'dt 가 같은 인접쌍'만 골라 계산한다.
#   이렇게 하면 20 Hz 격자를 유지하면서도 변화량 특징이 보간에 오염되지 않는다.

BREATH_WINDOW_SEC = 60.0        # 호흡 대역 특징 전용 창 (비중첩)
#   welch 의 주파수 분해능은 fs/nperseg 다. 10 초 창은 0.1 Hz 수준이라
#   호흡 대역 0.15-0.5 Hz 를 4구간으로밖에 못 나눈다. 그래서 호흡 특징만 창을 길게 쓴다.

HAMPEL_WINDOW = 11
HAMPEL_SIGMA = 3.0

BREATH_BAND = (0.15, 0.5)       # 9-30 회/분
BREATH_FULL = (0.05, 2.0)

SEGMENT_GUARD_SEC = 5.0         # 구간 경계에서 잘라낼 시간. 전환 직후 잔여 움직임 제외.

MOTION_SMOOTH_SEC = 1.0         # motion indicator 이동평균 길이
MOTION_MAD_K = 3.0              # 임계값 = median + K * 1.4826 * MAD (empty 세션에서만 산출)

# 주 분석 블록. 40MHz HT-LTF 는 ±2..±58 (114개) 로 40MHz 전 대역을 균일하게 덮고
# 규격과 실측이 정확히 일치한다. LLTF 는 20MHz 레거시 프리앰블이라 진폭 스케일이
# 절반 수준이고 HT40 격자에서의 저장 방식이 문서로 확정되지 않아 주 분석에서 제외한다.
# (s1 은 두 블록 모두 검사한다. "both" 로 바꾸면 전 블록을 쓴다.)
ANALYSIS_BLOCK = "HT-LTF"

# ===========================================================================
# 라벨
# ===========================================================================

STATE_LABELS = ("still", "empty", "motion")

# 이진 분류(재실 / 비재실)로 묶을 때의 매핑. 뒤척임은 재실이다.
BINARY_MAP: Dict[str, str] = {
    "still": "occupied",
    "motion": "occupied",
    "empty": "empty",
}

# ===========================================================================
# 시나리오 정의 (공칭 구간. 실제 녹화 길이는 SESSIONS 에서 개별 지정한다)
# ===========================================================================

SCENARIOS: Dict[str, Dict[str, object]] = {
    "S-1": {"title": "정상 수면",
            "purpose": "재실 상태의 CSI 특성 및 baseline 확보",
            "plan": [("still", 120.0)]},
    "S-2": {"title": "이탈",
            "purpose": "비재실 상태의 CSI 특성 및 이탈 baseline 확보",
            "plan": [("empty", 300.0)]},
    "S-3": {"title": "장기 이탈",
            "purpose": "지속적 이탈 상태를 안정적으로 검출하는지 검증",
            "plan": [("still", 60.0), ("empty", 240.0)]},
    "S-4": {"title": "복귀",
            "purpose": "사람이 돌아왔을 때 재실 상태로 정상 복귀하는지 검증",
            "plan": [("empty", 60.0), ("still", 120.0)]},
    "S-5": {"title": "수면 중 뒤척임",
            "purpose": "뒤척임을 이탈로 오판하지 않는지 검증",
            "plan": [("still", 60.0), ("motion", 60.0), ("still", 120.0)]},
    "S-6": {"title": "실제 이탈",
            "purpose": "실제 침대 이탈을 정상 검출하는지 검증",
            "plan": [("still", 120.0), ("motion", 15.0), ("empty", 120.0)]},
    "S-7": {"title": "단기 이탈 후 복귀",
            "purpose": "tau 임계값에 따른 단기 이탈 오검출 여부 검증",
            "plan": [("still", 60.0), ("empty", 30.0), ("still", 120.0)]},
    "S-8": {"title": "빈방-수면-빈방",
            "purpose": "한 세션 안에서 empty/still 을 대응 비교 (배치 C 14분 프로토콜)",
            "plan": [("empty", 180.0), ("still", 300.0), ("empty", 180.0)]},
}


def plan_to_segments(scenario: str, t0: float = 0.0
                     ) -> List[Tuple[str, float, float]]:
    """시나리오 공칭 계획을 (label, t_start, t_end) 리스트로 편다."""
    segs = []
    t = t0
    for label, dur in SCENARIOS[scenario]["plan"]:
        segs.append((label, t, t + dur))
        t += dur
    return segs


# ===========================================================================
# 실제 녹화 매니페스트
# ===========================================================================
#
# 각 항목:
#   name      : 고유 이름 (그림/표에 쓰인다)
#   scenario  : S-1 ~ S-7
#   file      : data/ 기준 파일명
#   segments  : [(label, t_start, t_end), ...]  로그 t=0 기준 초.
#               생략하면 시나리오 공칭 계획을 그대로 쓴다.
#   t_offset  : 로그 t=0 과 프로토콜 t=0 의 차이 [초].
#               s4_temporal.py --scan-offset 으로 추정해 여기에 적는다.
#               segments 의 모든 경계에 더해진다.
#
# 주의: CSI 로그의 t=0 과 타이머 t=0 이 같다는 보장이 없다.
#       전환이 있는 시나리오(S-3~S-7)는 반드시 오프셋을 확정하고 넣어야 한다.

#   subject   : 피험자 식별자. 같은 사람의 stable/motion 은 완전 독립 표본이 아니다.
#              비점유 세션은 "-".
#   batch     : 수집 배치. 배치가 다르면 장비 위치나 환경이 달랐을 수 있다.
#               "B" = 9/1 수집분, "A" = 선행 수집분.
#   actions   : motion 구간 안에서 정해진 동작을 수행한 시각들 [초].
#   events    : 타이머로 통제된 스크립트 세션의 프로토콜 구간 (SCRIPT_EVENTS).
#               스크립트 세션에만 넣는다. t_offset 이 모든 경계에 더해진다.

MOTION_STILL_1 = (0.0, 120.0)      # 2분 정자세
MOTION_ACTIVE = (120.0, 180.0)     # 1분 움직임 (15초 간격 동작)
MOTION_STILL_2 = (180.0, 299.0)    # 2분 정자세
MOTION_ACTIONS = [120.0, 135.0, 150.0, 165.0]   # 15초 간격 = 4회

# 스크립트 세션 프로토콜 (타이머로 통제됨).
# segments 는 이 6구간을 still / motion / still 세 덩어리로 묶은 것이다.
#   - 0-120 과 180-299 는 자세를 '유지'하는 구간이므로 still 로 묶인다.
#     다만 180-299 는 팔을 올린 자세라 0-120 과 자세가 다르다. 두 구간을
#     같은 still 로 뭉뚱그리면 자세 차이가 상태 차이로 새어 들어갈 수 있다.
#   - 120-180 은 자세 전환 4회가 들어 있어 motion 으로 묶인다.
#     actions 의 4개 시각이 각각 turn_left / return_supine_1 / turn_right /
#     return_supine_2 의 시작과 정확히 일치한다.
# ---------------------------------------------------------------------------
# 14분 프로토콜 (배치 C, 2026-09-02 수집)
# ---------------------------------------------------------------------------
#    0- 1분  대기       라벨 없음 (녹화 시작 / 퇴실 여유)
#    1- 4분  빈방 유지  -> empty
#    4- 5분  대기       라벨 없음 (입실해서 눕는 시간)
#    5-10분  수면       -> still
#   10-11분  대기       라벨 없음 (일어나서 퇴실하는 시간)
#   11-14분  빈방       -> empty  (종료)
#
# 대기 구간은 still 도 empty 도 아니므로 segments 에 넣지 않는다. 그 덕에 라벨
# 구간마다 앞뒤로 60초 완충이 생기며, SEGMENT_GUARD_SEC(5초)보다 훨씬 넉넉하다.
#
# 마지막 경계를 840.0 이 아니라 839.0 으로 둔 이유: 4개 파일의 실측 길이가
# 839.36-839.52 초다. 840 으로 두면 존재하지 않는 시각을 구간 끝으로 가리킨다.
#
# 배치 C 는 한 세션 안에 empty 와 still 이 모두 있어 '세션 내 대응' 비교가
# 가능하다. 대응 세션이 4개이므로 도달 가능한 최소 양측 p 는 2/2^4 = 0.125 다.
P14_EMPTY_1 = (60.0, 240.0)      # 3분 빈방
P14_STILL = (300.0, 600.0)       # 5분 수면
P14_EMPTY_2 = (660.0, 839.0)     # 3분 빈방 (파일 끝)

P14_SEGMENTS: List[Tuple[str, float, float]] = [
    ("empty",) + P14_EMPTY_1,
    ("still",) + P14_STILL,
    ("empty",) + P14_EMPTY_2,
]

SCRIPT_EVENTS: List[Tuple[str, float, float]] = [
    ("baseline_supine",   0.0, 120.0),   # 정자세 유지
    ("turn_left",       120.0, 135.0),   # 좌측와위
    ("return_supine_1", 135.0, 150.0),   # 정자세 복귀
    ("turn_right",      150.0, 165.0),   # 우측와위
    ("return_supine_2", 165.0, 180.0),   # 정자세 복귀
    ("arm_raised_hold", 180.0, 299.0),   # 팔 올린 자세로 유지
]

SESSIONS: List[Dict[str, object]] = [
    # =======================================================================
    # 배치 B : 피험자 3명 + 비점유 2회. 수신률 19.1-19.4 Hz, id 손실 1.1-2.7%.
    # =======================================================================
    {"name": "P01_stable", "scenario": "S-1", "file": "P01_stable.txt",
     "subject": "P01", "batch": "B", "t_offset": 0.0,
     "segments": [("still", 0.0, 299.0)]},
    {"name": "P02_stable", "scenario": "S-1", "file": "P02_stable.txt",
     "subject": "P02", "batch": "B", "t_offset": 0.0,
     "segments": [("still", 0.0, 299.0)]},
    {"name": "P03_stable", "scenario": "S-1", "file": "P03_stable.txt",
     "subject": "P03", "batch": "B", "t_offset": 0.0,
     "segments": [("still", 0.0, 299.0)]},

    # 스크립트 세션. 2분 정자세 -> 자세 전환 4회 -> 팔 올린 자세 유지.
    # t_offset 은 s4_temporal.py --scan-offset 으로 확정한 뒤 채운다.
    # 지금은 0.0 이므로 events 경계는 '공칭값'이고 실측으로 확정된 것이 아니다.
    {"name": "P01_motion", "scenario": "S-5", "file": "P01_motion.txt",
     "subject": "P01", "batch": "B", "t_offset": 0.0,
     "segments": [("still",) + MOTION_STILL_1, ("motion",) + MOTION_ACTIVE,
                  ("still",) + MOTION_STILL_2],
     "actions": MOTION_ACTIONS, "events": SCRIPT_EVENTS,
     "static_pairs": [("still@0", "still@1")]},
    {"name": "P02_motion", "scenario": "S-5", "file": "P02_motion.txt",
     "subject": "P02", "batch": "B", "t_offset": 0.0,
     "segments": [("still",) + MOTION_STILL_1, ("motion",) + MOTION_ACTIVE,
                  ("still",) + MOTION_STILL_2],
     "actions": MOTION_ACTIONS, "events": SCRIPT_EVENTS,
     "static_pairs": [("still@0", "still@1")]},
    {"name": "P03_motion", "scenario": "S-5", "file": "P03_motion.txt",
     "subject": "P03", "batch": "B", "t_offset": 0.0,
     "segments": [("still",) + MOTION_STILL_1, ("motion",) + MOTION_ACTIVE,
                  ("still",) + MOTION_STILL_2],
     "actions": MOTION_ACTIONS, "events": SCRIPT_EVENTS,
     "static_pairs": [("still@0", "still@1")]},

    {"name": "empty1", "scenario": "S-2", "file": "empty1.txt",
     "subject": "-", "batch": "B", "t_offset": 0.0,
     "segments": [("empty", 0.0, 299.0)]},
    # 파일 확장자가 .txtk 로 되어 있다(오타로 보인다). 원본을 건드리지 않고 그대로 읽는다.
    # 이름을 empty2.txt 로 고치면 아래 file 값도 함께 바꿔야 한다.
    {"name": "empty2", "scenario": "S-2", "file": "empty2.txtk",
     "subject": "-", "batch": "B", "t_offset": 0.0,
     "segments": [("empty", 0.0, 299.0)]},

    # =======================================================================
    # 배치 A : 먼저 수집한 stable_5min.txt / empty_5min.txt 2개가 있었으나
    #          data/ 에서 삭제되어 항목을 제거했다 (파일 없는 항목은 넣지 않는다).
    #          수신률 17.5-18.1 Hz, id 손실 7.7-10.7% 로 배치 B 와 캡처 조건이
    #          뚜렷이 달랐으므로, 파일을 되살릴 때도 배치 B 와 섞어 쓰면 안 된다.
    #          복원하려면 data/ 에 파일을 되돌리고 아래 형태로 다시 넣는다.
    #   {"name": "A_stable", "scenario": "S-1", "file": "stable_5min.txt",
    #    "subject": "A?", "batch": "A", "t_offset": 0.0,
    #    "segments": [("still", 0.0, 299.0)]},
    #   {"name": "A_empty", "scenario": "S-2", "file": "empty_5min.txt",
    #    "subject": "-", "batch": "A", "t_offset": 0.0,
    #    "segments": [("empty", 0.0, 299.0)]},
    # =======================================================================

    # =======================================================================
    # 배치 C : 14분 프로토콜(S-8), 피험자 4명. 2026-09-02 수집.
    #
    # ★ 아직 SESSIONS 에 넣지 않았다. 두 조건이 모두 충족되기 전에는 넣지 않는다.
    #     (1) 참가자 ID 매핑 확정
    #         전달받은 원본은 subject 가 실명 약자였다. 그 4명이 배치 B 의
    #         P01-P03 과 동일 인물인지 확인되지 않았다. 같은 사람인데 새 ID 를
    #         주면 참가자 단위 통계가 틀어지므로 임의로 P04.. 를 붙이지 않는다.
    #     (2) 익명화된 raw 파일이 data/ 에 준비될 것 (0902_<ID>.txt 형식)
    #
    #   가짜 ID 로 자리표시자 세션을 넣으면 load_all() 과 sessions_of_batch()
    #   가 그것을 실제 분석 대상으로 골라 버린다. 그래서 주석으로만 둔다.
    #
    #   두 조건이 충족되면 아래 4줄의 <ID> 를 확정 ID 로 바꿔 위 리스트 안으로
    #   옮기면 된다. segments 와 static_pairs 는 그대로 쓰면 된다.
    #
    #   {"name": "C_<ID>", "scenario": "S-8", "file": "0902_<ID>.txt",
    #    "subject": "<ID>", "batch": "C", "t_offset": 0.0,
    #    "segments": P14_SEGMENTS,
    #    "static_pairs": [("empty@0", "empty@1")]},
    #
    #   t_offset = 0 근거: 전체 길이가 839.4-839.5 초로 공칭 840 초와 0.5 초
    #   안에서 맞으므로 녹화 시작과 타이머 시작이 사실상 같았다고 볼 근거는
    #   있다. 다만 독립적으로 확정한 값은 아니다. 경계가 어긋난 정황이 보이면
    #   s6_visualize.py 의 타임라인으로 확인한 뒤 고친다.
    # =======================================================================
]


# ===========================================================================
# 로딩
# ===========================================================================

def analysis_mask(n_sub: int, stbc: int = 0, bandwidth_40: bool = True,
                  first_word_invalid: bool = True,
                  block: Optional[str] = None) -> np.ndarray:
    """
    특징 추출에 쓸 subcarrier 마스크.
    규격 유효 마스크에 ANALYSIS_BLOCK 제한을 추가로 건다.
    """
    mask = core.spec_valid_mask(n_sub, stbc=stbc, bandwidth_40=bandwidth_40,
                                first_word_invalid=first_word_invalid)
    blk = ANALYSIS_BLOCK if block is None else block
    if blk == "both":
        return mask
    sel = np.zeros(n_sub, dtype=bool)
    found = False
    for b in core.ltf_blocks(n_sub, stbc):
        if b.name == blk:
            sel[b.start:b.stop] = True
            found = True
    if not found:
        raise ValueError("ANALYSIS_BLOCK=%r 에 해당하는 블록이 없다 (n_sub=%d)"
                         % (blk, n_sub))
    return mask & sel


def _session_label(segments: List[Tuple[str, float, float]]) -> str:
    """세션 대표 라벨. 구간이 하나면 그 상태, 여러 개면 mixed."""
    labels = {s[0] for s in segments}
    if len(labels) == 1:
        return next(iter(labels))
    return "mixed"


def load_all(verbose: bool = True) -> List[core.Session]:
    """
    SESSIONS 매니페스트를 모두 읽는다.
    파일이 없으면 건너뛰고 안내를 출력한다. 그 외의 오류는 그대로 올린다.
    """
    out: List[core.Session] = []
    for spec in SESSIONS:
        name = str(spec["name"])
        scenario = str(spec.get("scenario", ""))
        path = os.path.join(DATA_DIR, str(spec["file"]))

        if not os.path.exists(path):
            if verbose:
                print("[skip] %s : 파일 없음 -> %s" % (name, path))
            continue

        parsed = core.parse_file(path, expected_len=EXPECTED_CSI_LEN,
                                 source_mac=SOURCE_MAC)

        raw_segs = spec.get("segments")
        if raw_segs is None:
            if not scenario:
                raise ValueError("%s: segments 도 scenario 도 없다." % name)
            raw_segs = plan_to_segments(scenario)
        raw_segs = [(str(a), float(b), float(c)) for a, b, c in raw_segs]

        off = float(spec.get("t_offset", 0.0))
        segs = [core.Segment(a, b + off, c + off) for a, b, c in raw_segs]

        for sg in segs:
            if sg.label not in STATE_LABELS:
                raise ValueError("%s: 알 수 없는 상태 라벨 %r" % (name, sg.label))

        # 전환 시각 = 각 구간의 시작(첫 구간 제외)
        markers = [("%s->%s" % (segs[i - 1].label, segs[i].label), segs[i].t0)
                   for i in range(1, len(segs))]
        # 지정된 동작 수행 시각 (motion 구간 안의 15초 간격 동작 등)
        for j, at in enumerate(spec.get("actions", [])):
            markers.append(("action%d" % (j + 1), float(at) + off))
        markers.sort(key=lambda e: e[1])

        # 스크립트 세션의 프로토콜 구간. t_offset 을 똑같이 적용한다.
        # 여기서 검증하지 않고 넘기면 이름이 틀린 구간을 조용히 쓰게 되므로 즉시 멈춘다.
        evs = []
        for a, b, c in spec.get("events", []):
            t0, t1 = float(b) + off, float(c) + off
            if t1 <= t0:
                raise ValueError("%s: events 구간 %r 의 끝이 시작보다 앞이다 (%.1f, %.1f)"
                                 % (name, a, t0, t1))
            evs.append(core.Segment(str(a), t0, t1))
        if evs:
            # 프로토콜 구간이 상태 구간(segments) 전체를 덮는지 확인한다.
            # 어긋나면 둘 중 하나가 잘못 적힌 것이므로 분석 전에 잡아야 한다.
            if abs(evs[0].t0 - segs[0].t0) > 1e-6 or abs(evs[-1].t1 - segs[-1].t1) > 1e-6:
                raise ValueError(
                    "%s: events 범위 [%.1f, %.1f] 가 segments 범위 [%.1f, %.1f] 와 다르다"
                    % (name, evs[0].t0, evs[-1].t1, segs[0].t0, segs[-1].t1))
            for i in range(1, len(evs)):
                if abs(evs[i].t0 - evs[i - 1].t1) > 1e-6:
                    raise ValueError("%s: events 구간 %r 과 %r 사이가 이어지지 않는다"
                                     % (name, evs[i - 1].label, evs[i].label))

        sess = core.Session(
            name=name,
            label=str(spec.get("label", _session_label(raw_segs))),
            path=path,
            csi=parsed["csi"], rssi=parsed["rssi"],
            noise_floor=parsed["noise_floor"], t=parsed["t"], seq=parsed["seq"],
            meta=parsed["meta"], drops=parsed["drops"],
            scenario=scenario,
            subject=str(spec.get("subject", "")),
            batch=str(spec.get("batch", "")),
            segments=segs,
            events=evs,
            markers=markers,
            static_pairs=[tuple(p) for p in spec.get("static_pairs", [])],
        )
        if verbose:
            print("[load] %-11s %-4s %-6s subj=%-4s batch=%-2s n=%5d  %.1fs  %.2f Hz"
                  % (name, scenario, sess.label, sess.subject, sess.batch,
                     sess.n_frames, sess.duration, sess.rate_hz))
        out.append(sess)

    if not out:
        raise RuntimeError(
            "읽을 수 있는 세션이 하나도 없다. data/ 에 파일이 있는지, "
            "sessions.py 의 SESSIONS 매니페스트가 맞는지 확인해야 한다.")
    return out


def sessions_of_batch(sessions: List[core.Session],
                      batch: Optional[str] = None) -> List[core.Session]:
    """
    해당 배치의 세션만 고른다. 기본값은 CSI_BATCH 로 정해진 BATCH 다.

    0개면 조용히 빈 리스트를 돌려주지 않고 예외를 던진다. 배치를 잘못 지정한 채
    분석이 '표본 0개로 성공' 하는 것이 가장 위험하기 때문이다.
    매니페스트에 그 배치가 아예 없는 경우와, 있지만 파일이 없어 못 읽은 경우를
    구분해서 알린다. 원인이 다르면 조치도 다르기 때문이다.
    """
    want = (batch or BATCH).upper()
    got = [s for s in sessions if s.batch.upper() == want]
    if got:
        return got

    declared = sorted({str(sp.get("batch", "")).upper() for sp in SESSIONS})
    loaded = sorted({s.batch.upper() for s in sessions})
    if want not in declared:
        raise ValueError(
            "배치 %r 이 sessions.py 의 SESSIONS 매니페스트에 없다.\n"
            "  매니페스트에 선언된 배치: %s\n"
            "  배치 C 는 참가자 ID 매핑과 익명화된 raw 파일이 준비되기 전까지"
            " 주석으로만 두었다." % (want, declared or ["(없음)"]))
    raise ValueError(
        "배치 %r 이 매니페스트에는 있으나 읽어들인 세션이 0개다.\n"
        "  data/ 에 해당 배치의 파일이 있는지 확인해야 한다.\n"
        "  실제로 읽힌 배치: %s" % (want, loaded or ["(없음)"]))


def sessions_by_state(sessions: List[core.Session], state: str
                      ) -> List[core.Session]:
    """해당 상태 구간을 하나라도 가진 세션."""
    return [s for s in sessions if any(sg.label == state for sg in s.segments)]


def paired_sessions(sessions: List[core.Session], a: str = "still",
                    b: str = "empty") -> List[core.Session]:
    """세션 내에 두 상태가 모두 있는 세션 = 대응 비교가 가능한 세션."""
    return [s for s in sessions
            if any(sg.label == a for sg in s.segments)
            and any(sg.label == b for sg in s.segments)]
