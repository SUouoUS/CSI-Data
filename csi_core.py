# -*- coding: utf-8 -*-
"""
csi_core.py — ESP32 CSI 공통 모듈 (파싱 / 전처리 / 특징)

이 프로젝트의 모든 스크립트는 원시 로그를 여기서만 읽는다.
개별 스크립트에 파싱이나 전처리를 다시 구현하면 그림마다 전처리가 갈리므로 금지한다.

===========================================================================
검증해야 할 가정 (s1_check_format.py 가 매 실행마다 재확인한다)
===========================================================================

[A1] 컬럼 순서
    esp-csi 표준 25열은 마지막에서 네 번째가 rx_state 이지만,
    이 프로젝트의 펌웨어 로그는 그 자리가 rx_format 이고 값이 전 프레임 1 로 상수다.
    따라서 'rx_state == 0' 을 필터로 쓰면 전 프레임이 드롭된다.
    -> rx_format 은 필터가 아니라 '설정 일관성 검사' 항목으로만 쓴다.
    파일에 헤더 줄이 있으면 언제나 헤더를 우선하고, 없을 때만 FALLBACK_COLUMNS 를 쓴다.

[A2] len=384 의 블록 구성
    384 byte = 192 complex 는 두 가지 조합에서 나온다.
      (a) 20MHz + HT + STBC      -> LLTF 64 + HT-LTF 64 + STBC-HT-LTF 64
      (b) 40MHz + HT + non-STBC  -> LLTF 64 + HT-LTF 128
    이 프로젝트 데이터의 메타는 bandwidth=1(40MHz), stbc=0, secondary_channel=2 이고,
    실측 널 패턴도 (b) 와 정확히 일치한다. 근거는 다음과 같다.

      index 64..191 을 128-bin FFT (j=idx-64, sc = j if j<64 else j-128) 로 풀면
        sc  0        (idx 64)            평균진폭 1.6   -> DC null
        sc ±1        (idx 65, 191)       평균진폭 5.2   -> DC 인접
        sc ±2..±58   (idx 66..122,
                          134..190)      평균진폭 30-43 -> 유효 (114개)
        sc ±59       (idx 123, 133)      평균진폭 5-7   -> 전이
        sc ±60..±64  (idx 124..132)      평균진폭 0.6-1.4 -> guard
      ±2..±58 (114개) 은 802.11n HT40 의 규격 부반송파 집합 그대로다.
      (a) 가설이라면 64..127 과 128..191 각각에 LLTF 형 널(로컬 0, 27..37)이 있어야
      하는데 하나도 없다. 블록간 진폭 프로파일 상관도 -0.14 ~ 0.05 로 반복 신호가 아니다.

    따라서 ltf_blocks() 는 stbc 값을 보고 (a)/(b) 를 구분한다.

[A3] 저장 순서 (imag, real)
    ESP-IDF 문서 기준 CSI 버퍼의 저장 순서는 (imag, real) 이다.
    amplitude 만 쓰면 순서와 무관하지만(제곱합이라 대칭), phase 를 쓰는 순간 부호가
    통째로 뒤집히므로 to_complex() 에 imag_first 플래그로 노출한다.
    이 프로젝트는 amplitude 만 쓰므로 A3 가 결과를 바꾸지 않는다.
    -> 실측 확인: |sqrt(a^2+b^2) - sqrt(b^2+a^2)| 의 최댓값이 0.0 임을 s1 이 재확인한다.

[A4] first_word_invalid
    메타의 first_word 가 1 이면 CSI 버퍼의 첫 4바이트가 하드웨어 버그로 무효다.
    실측에서 raw index 0..3 (= complex pair 0, 1) 이 전 프레임 상수 (47, -16, 2, 0) 였다.
    -> first_word=1 인 프레임은 pair 0,1 을 마스크에서 강제로 제외한다.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import welch

# ===========================================================================
# 1. 파싱
# ===========================================================================

# 헤더 줄이 없는 파일을 위한 fallback. 이 프로젝트 펌웨어 기준(rx_state 가 아니라 rx_format).
FALLBACK_COLUMNS: Tuple[str, ...] = (
    "type", "id", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth",
    "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding", "sgi",
    "noise_floor", "ampdu_cnt", "channel", "secondary_channel",
    "local_timestamp", "ant", "sig_len", "rx_format", "len", "first_word",
    "data",
)

# 프레임마다 값이 같아야 하는(= 캡처 설정을 규정하는) 필드.
CONFIG_FIELDS: Tuple[str, ...] = (
    "rate", "sig_mode", "mcs", "bandwidth", "smoothing", "not_sounding",
    "aggregation", "stbc", "fec_coding", "sgi", "channel", "secondary_channel",
    "ant", "sig_len", "rx_format", "len", "first_word",
)

_INT_RE = re.compile(r"-?\d+")
_TS_WRAP = 1 << 32  # local_timestamp 는 uint32 (us) 이므로 2^32 에서 랩어라운드

CACHE_VERSION = 3  # 캐시 포맷이 바뀌면 올린다. 옛 캐시는 자동 무효화된다.


class CSIParseError(RuntimeError):
    """파싱을 계속할 수 없는 상태. 조용히 넘기지 않고 즉시 멈추기 위한 예외."""


def _find_header(lines: Sequence[str]) -> Tuple[Optional[int], Optional[List[str]]]:
    """헤더 줄(type,...,data)을 찾는다. 없으면 (None, None)."""
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("type,") and s.endswith("data"):
            return i, s.split(",")
    return None, None


def _cache_path(path: str) -> str:
    base, _ = os.path.splitext(path)
    return base + ".cache.npz"


def _cache_is_fresh(cache: str, src: str) -> bool:
    if not os.path.exists(cache):
        return False
    try:
        with np.load(cache, allow_pickle=False) as z:
            if int(z["cache_version"]) != CACHE_VERSION:
                return False
            return (float(z["src_mtime"]) >= os.path.getmtime(src) - 1e-6
                    and int(z["src_size"]) == os.path.getsize(src))
    except Exception:
        return False


def parse_file(
    path: str,
    expected_len: int = 384,
    source_mac: Optional[str] = None,
    use_cache: bool = True,
) -> Dict[str, object]:
    """
    CSI 로그 1개를 파싱한다.

    data 필드는 "[1,-2,3,...]" 형태이고 내부에 쉼표가 있어서 csv.reader 로는 컬럼이 밀린다.
    또 파일에 따라 큰따옴표가 있을 수도 없을 수도 있다.
    따라서 행에서 '[' 위치를 찾아 헤더부/데이터부로 자른 뒤,
    헤더부 꼬리의 ' , " ' 를 제거하고 쉼표로 나눈다.

    드롭된 행은 사유별로 센다. 클래스마다 드롭률이 다르면 그 자체가 교락 요인이므로
    조용히 버려서는 안 된다.
    """
    if not os.path.exists(path):
        raise CSIParseError("파일이 없다: %s" % path)

    cache = _cache_path(path)
    if use_cache and _cache_is_fresh(cache, path):
        with np.load(cache, allow_pickle=False) as z:
            return {
                "csi": z["csi"],
                "rssi": z["rssi"],
                "noise_floor": z["noise_floor"],
                "t": z["t"],
                "seq": z["seq"],
                "meta": json.loads(str(z["meta_json"])),
                "drops": json.loads(str(z["drops_json"])),
                "columns": json.loads(str(z["columns_json"])),
                "from_cache": True,
            }

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    hdr_idx, columns = _find_header(lines)
    has_header = columns is not None
    if columns is None:
        columns = list(FALLBACK_COLUMNS)
    col = {name: i for i, name in enumerate(columns)}
    n_head_expected = len(columns) - 1  # 마지막 'data' 는 데이터부로 분리된다

    for need in ("mac", "len", "local_timestamp", "rssi", "noise_floor", "id"):
        if need not in col:
            raise CSIParseError("필수 컬럼 '%s' 이 헤더에 없다: %s" % (need, path))

    drops: Dict[str, int] = {
        "no_bracket": 0,          # '[' 가 없어 데이터부를 찾지 못함
        "head_field_count": 0,    # 헤더부 필드 수가 컬럼 수와 다름
        "head_parse_error": 0,    # 정수 변환 실패
        "meta_len_mismatch": 0,   # 메타의 len 이 expected_len 과 다름
        "data_len_mismatch": 0,   # data 정수 개수가 expected_len 과 다름
        "mac_mismatch": 0,        # 지정한 송신 MAC 이 아님
    }

    csi_rows: List[np.ndarray] = []
    rssi_l: List[int] = []
    nf_l: List[int] = []
    ts_l: List[int] = []
    seq_l: List[int] = []
    config_counter: Dict[Tuple, int] = {}
    macs: Dict[str, int] = {}
    n_csi_lines = 0

    cfg_idx = [(f, col[f]) for f in CONFIG_FIELDS if f in col]

    for ln in lines:
        s = ln.strip()
        if not s.startswith("CSI_DATA"):
            continue
        n_csi_lines += 1

        b = s.find("[")
        if b < 0:
            drops["no_bracket"] += 1
            continue

        head = s[:b].rstrip(' ,"').split(",")
        if len(head) != n_head_expected:
            drops["head_field_count"] += 1
            continue

        mac = head[col["mac"]]
        macs[mac] = macs.get(mac, 0) + 1
        if source_mac is not None and mac != source_mac:
            drops["mac_mismatch"] += 1
            continue

        try:
            meta_len = int(head[col["len"]])
            rssi_v = int(head[col["rssi"]])
            nf_v = int(head[col["noise_floor"]])
            ts_v = int(head[col["local_timestamp"]])
            seq_v = int(head[col["id"]])
        except ValueError:
            drops["head_parse_error"] += 1
            continue

        if meta_len != expected_len:
            drops["meta_len_mismatch"] += 1
            continue

        vals = _INT_RE.findall(s[b:])
        if len(vals) != expected_len:
            drops["data_len_mismatch"] += 1
            continue

        csi_rows.append(np.fromiter((int(v) for v in vals), dtype=np.int16,
                                    count=expected_len))
        rssi_l.append(rssi_v)
        nf_l.append(nf_v)
        ts_l.append(ts_v)
        seq_l.append(seq_v)

        key = tuple(head[i] for _, i in cfg_idx)
        config_counter[key] = config_counter.get(key, 0) + 1

    if not csi_rows:
        raise CSIParseError(
            "유효 프레임이 0개다: %s\n  CSI_DATA 행=%d, 드롭 내역=%s"
            % (path, n_csi_lines, drops))

    csi = np.stack(csi_rows)
    rssi = np.asarray(rssi_l, dtype=np.int16)
    nf = np.asarray(nf_l, dtype=np.int16)
    seq = np.asarray(seq_l, dtype=np.int64)

    # --- local_timestamp 랩어라운드 보정 (다중 wrap 대응) ---
    ts = np.asarray(ts_l, dtype=np.int64)
    d = np.diff(ts)
    n_wrap = np.cumsum(np.concatenate([[0], (d < 0).astype(np.int64)]))
    t = (ts + n_wrap * _TS_WRAP - ts[0]).astype(np.float64) / 1e6

    if not np.all(np.diff(t) > 0):
        n_bad = int((np.diff(t) <= 0).sum())
        raise CSIParseError(
            "wrap 보정 후에도 시간이 단조증가하지 않는다 (%d 지점): %s" % (n_bad, path))

    meta: Dict[str, object] = {
        "has_header": has_header,
        "header_line_index": hdr_idx,
        "n_lines": len(lines),
        "n_csi_lines": n_csi_lines,
        "n_valid": int(csi.shape[0]),
        "expected_len": int(expected_len),
        "macs": macs,
        "n_wrap": int(n_wrap[-1]),
        "config_fields": [f for f, _ in cfg_idx],
        "config_combos": {"|".join(k): v for k, v in config_counter.items()},
        "n_config_combos": len(config_counter),
    }
    if len(config_counter) >= 1:
        # 가장 흔한 조합을 대표 설정으로 노출한다 (블록 구성 판정에 쓰인다).
        top = max(config_counter.items(), key=lambda kv: kv[1])[0]
        meta["config"] = {f: v for (f, _), v in zip(cfg_idx, top)}

    if use_cache:
        try:
            np.savez_compressed(
                cache,
                cache_version=np.int64(CACHE_VERSION),
                src_mtime=np.float64(os.path.getmtime(path)),
                src_size=np.int64(os.path.getsize(path)),
                csi=csi, rssi=rssi, noise_floor=nf, t=t, seq=seq,
                meta_json=np.str_(json.dumps(meta, ensure_ascii=False)),
                drops_json=np.str_(json.dumps(drops, ensure_ascii=False)),
                columns_json=np.str_(json.dumps(columns, ensure_ascii=False)),
            )
        except OSError as exc:  # 캐시 실패는 분석을 막을 이유가 아니지만 침묵하지도 않는다
            print("[warn] 캐시 저장 실패 (%s): %s" % (cache, exc))

    return {"csi": csi, "rssi": rssi, "noise_floor": nf, "t": t, "seq": seq,
            "meta": meta, "drops": drops, "columns": columns,
            "from_cache": False}


# ===========================================================================
# 2. Session
# ===========================================================================

@dataclass
class Segment:
    """세션 안의 상태 구간. label 은 still / empty / motion."""
    label: str
    t0: float
    t1: float

    def mask(self, t: np.ndarray, guard: float = 0.0) -> np.ndarray:
        """구간에 속한 프레임 마스크. guard 초만큼 양끝을 잘라 전이 잔여를 제외한다."""
        return (t >= self.t0 + guard) & (t < self.t1 - guard)


@dataclass
class Session:
    name: str
    label: str                      # 세션 대표 라벨 (occupied / empty / mixed ...)
    path: str
    csi: np.ndarray                 # (n_frames, 2K) int16
    rssi: np.ndarray
    noise_floor: np.ndarray
    t: np.ndarray                   # 초, 첫 프레임 기준 0
    seq: np.ndarray                 # 펌웨어 id (프레임 카운터)
    meta: Dict[str, object]
    drops: Dict[str, int]
    scenario: str = ""
    subject: str = ""               # 피험자 식별자. 같은 사람의 세션은 완전 독립이 아니다.
    batch: str = ""                 # 수집 배치. 배치가 다르면 환경 조건이 다를 수 있다.
    segments: List[Segment] = field(default_factory=list)
    events: List[Tuple[str, float]] = field(default_factory=list)  # (이름, 시각)
    static_pairs: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return int(self.csi.shape[0])

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if self.n_frames > 1 else 0.0

    @property
    def rate_hz(self) -> float:
        return self.n_frames / self.duration if self.duration > 0 else float("nan")

    @property
    def n_sub(self) -> int:
        return int(self.csi.shape[1] // 2)

    @property
    def seq_loss(self) -> Tuple[int, float]:
        """펌웨어 id 시퀀스로 본 (손실 프레임 수, 손실률). 직렬 캡처 유실을 잡아낸다."""
        if self.n_frames < 2:
            return 0, 0.0
        expect = int(self.seq[-1] - self.seq[0]) + 1
        if expect <= 0:
            return 0, 0.0
        lost = expect - self.n_frames
        return lost, lost / expect

    def segments_by_label(self, label: str) -> List[Segment]:
        return [sg for sg in self.segments if sg.label == label]

    def segment(self, ref: str) -> Optional[Segment]:
        """
        구간 참조. 같은 라벨이 여러 번 나오는 시나리오가 있으므로
        'still@1' 처럼 라벨 뒤에 몇 번째인지 붙일 수 있다. 생략하면 첫 번째다.
        """
        if "@" in ref:
            label, _, k = ref.partition("@")
            items = self.segments_by_label(label)
            idx = int(k)
            return items[idx] if 0 <= idx < len(items) else None
        for sg in self.segments:
            if sg.label == ref:
                return sg
        return None


# ===========================================================================
# 3. 전처리
# ===========================================================================

def to_complex(csi: np.ndarray, imag_first: bool = True) -> np.ndarray:
    """
    (n, 2K) int -> (n, K) complex.

    ESP-IDF 문서 기준 저장 순서는 (imag, real) 이므로 기본값은 imag_first=True.
    amplitude 만 쓰면 순서와 무관하지만 phase 를 쓰는 순간 켤레가 뒤집히므로
    플래그로 명시적으로 노출한다. (가정 A3)
    """
    a = csi[:, 0::2].astype(np.float64)
    b = csi[:, 1::2].astype(np.float64)
    if imag_first:
        return b + 1j * a          # b=real, a=imag
    return a + 1j * b


def amplitude(csi: np.ndarray) -> np.ndarray:
    """(n, 2K) -> (n, K) 진폭. 저장 순서와 무관하다."""
    a = csi[:, 0::2].astype(np.float64)
    b = csi[:, 1::2].astype(np.float64)
    return np.sqrt(a * a + b * b)


@dataclass(frozen=True)
class LTFBlock:
    name: str
    start: int      # complex index (포함)
    stop: int       # complex index (미포함)
    nfft: int       # 이 블록의 FFT bin 수

    @property
    def size(self) -> int:
        return self.stop - self.start


def ltf_blocks(n_sub: int, stbc: int = 0) -> List[LTFBlock]:
    """
    complex subcarrier 개수 -> LTF 블록 구성. (가정 A2)

    192 는 두 가지 해석이 가능하므로 stbc 로 구분한다.
      stbc=0 -> 40MHz HT non-STBC : LLTF 64 + HT-LTF 128   <- 이 프로젝트
      stbc=1 -> 20MHz HT STBC     : LLTF 64 + HT-LTF 64 + STBC-HT-LTF 64
    """
    if n_sub == 192:
        if stbc:
            return [LTFBlock("LLTF", 0, 64, 64),
                    LTFBlock("HT-LTF", 64, 128, 64),
                    LTFBlock("STBC-HT-LTF", 128, 192, 64)]
        return [LTFBlock("LLTF", 0, 64, 64),
                LTFBlock("HT-LTF", 64, 192, 128)]
    if n_sub == 128:
        return [LTFBlock("LLTF", 0, 64, 64), LTFBlock("HT-LTF", 64, 128, 64)]
    if n_sub == 64:
        return [LTFBlock("LLTF", 0, 64, 64)]
    raise CSIParseError("알 수 없는 subcarrier 개수: %d" % n_sub)


def fft_bin_to_subcarrier(k: int | np.ndarray, nfft: int = 64):
    """FFT bin 번호 -> 부호 있는 subcarrier 번호. k < nfft/2 이면 k, 아니면 k-nfft."""
    k = np.asarray(k)
    return np.where(k < nfft // 2, k, k - nfft)


def block_subcarriers(n_sub: int, stbc: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    전체 index 배열에 대해 (블록 이름 배열, subcarrier 번호 배열)을 만든다.
    그림 축과 CSV 라벨에 쓴다.
    """
    names = np.empty(n_sub, dtype=object)
    scs = np.zeros(n_sub, dtype=np.int32)
    for blk in ltf_blocks(n_sub, stbc):
        local = np.arange(blk.size)
        names[blk.start:blk.stop] = blk.name
        scs[blk.start:blk.stop] = fft_bin_to_subcarrier(local, blk.nfft)
    return names, scs


# 유효 subcarrier 의 |sc| 범위. (하한, 상한) 포함.
#
#   HT-LTF / 128bin (HT40)  : ±2..±58 -> 114개.
#       802.11n HT40 규격 그대로다. 실측 널 패턴과 정확히 일치한다.
#       실측에서는 sc ±1 과 ±59 에도 진폭 4~8.5 가 남는데(스펙트럼 누설),
#       규격을 그대로 지키는 쪽이 보수적이므로 유효로 넣지 않는다.
#   HT-LTF / 64bin  (HT20)  : ±1..±28 -> 56개.
#   LLTF   / 64bin  (HT20)  : ±1..±26 -> 52개.
#   LLTF   / 64bin  (HT40)  : ±5..±32 -> 55개.   <-- 규격이 아니라 실측 역산값
#       HT40 캡처에서 LLTF 블록이 어떤 격자로 저장되는지는 ESP-IDF 문서에 규정돼 있지 않다.
#       그래서 이 범위만은 '규격'이 아니라 실측에서 역산한 것이다.
#       근거: idx 0..4 와 60..63 은 전 프레임 진폭이 정확히 0 이거나(2-4, 60-63)
#             전 프레임 동일한 상수 쓰레기값이고(0, 1 = first_word_invalid),
#             idx 5..59 는 모두 시간에 따라 변동하는 실제 신호다.
#       개수 55 는 L-LTF 규격의 52 와 다르므로 이 매핑은 확정된 것이 아니다.
#       이 불확실성 때문에 이 프로젝트의 주 분석 블록은 LLTF 가 아니라 HT-LTF 다.
_SPEC_RANGE = {
    ("HT-LTF", 128): (2, 58),
    ("STBC-HT-LTF", 128): (2, 58),
    ("HT-LTF", 64): (1, 28),
    ("STBC-HT-LTF", 64): (1, 28),
    ("LLTF", 64, 20): (1, 26),
    ("LLTF", 64, 40): (5, 32),
}

# 규격이 문서로 확정된 블록. s1 은 이 블록에서만 규격/실측 불일치를 FAIL 로 판정한다.
DOCUMENTED_BLOCKS = ("HT-LTF", "STBC-HT-LTF")


def spec_valid_mask(n_sub: int, stbc: int = 0, bandwidth_40: bool = True,
                    first_word_invalid: bool = True) -> np.ndarray:
    """
    규격(및 문서화되지 않은 부분은 실측)에 근거한 유효 subcarrier 마스크.
    DC / guard 의 null subcarrier 를 평균에 포함시키면 신호가 희석되므로 반드시 쓴다.
    first_word_invalid=True 이면 complex pair 0,1 (raw byte 0..3) 도 제외한다. (가정 A4)
    """
    mask = np.zeros(n_sub, dtype=bool)
    for blk in ltf_blocks(n_sub, stbc):
        if blk.name == "LLTF":
            key = ("LLTF", blk.nfft, 40 if bandwidth_40 else 20)
        else:
            key = (blk.name, blk.nfft)
        if key not in _SPEC_RANGE:
            raise CSIParseError("규격 마스크 정의가 없는 블록: %s" % (key,))
        lo, hi = _SPEC_RANGE[key]
        sc = np.abs(fft_bin_to_subcarrier(np.arange(blk.size), blk.nfft))
        mask[blk.start:blk.stop] = (sc >= lo) & (sc <= hi)
    if first_word_invalid:
        mask[0:2] = False
    return mask


def empirical_valid_mask(amp: np.ndarray, n_sub: Optional[int] = None,
                         stbc: int = 0, mean_rel: float = 0.10,
                         sd_rel: float = 0.25) -> np.ndarray:
    """
    실측으로 '하드웨어 널(DC/guard)' 을 찾아 유효 subcarrier 마스크를 만든다.
    규격 마스크와의 교차검증용이다.

    판정 기준이 평균진폭이 아니라 '평균진폭 AND 시간 표준편차' 인 이유:
      진폭만 보면 하드웨어 널과 깊은 페이딩 노치를 구분할 수 없다.
      실제로 이 프로젝트의 stable 세션은 subcarrier -52..-44 의 평균진폭이 4.7 로
      guard(0.9) 와 크게 다르지 않다. 하지만 성격이 전혀 다르다.

        하드웨어 널 : 신호가 아예 없어 ADC 양자화 잡음만 있고 시간에 따라 변하지 않는다.
                      실측 시간 표준편차 <= 0.80
        페이딩 노치 : 진폭이 낮을 뿐 실제 채널이라 시간에 따라 변동한다.
                      실측 시간 표준편차 >= 1.02

      노치를 널로 오판해 마스크에서 빼면 가장 강한 신호 후보를 버리게 된다.
      그래서 '진폭도 낮고 시간 변동도 없을 때만' 널로 판정한다.
      시간 표준편차가 정확히 0 이면 상수이므로 무조건 널이다.
    """
    m = amp.mean(axis=0)
    sd = amp.std(axis=0, ddof=1) if amp.shape[0] > 1 else np.zeros_like(m)
    if n_sub is None:
        n_sub = m.size
    mask = np.zeros(n_sub, dtype=bool)
    for blk in ltf_blocks(n_sub, stbc):
        mm = m[blk.start:blk.stop]
        ss = sd[blk.start:blk.stop]
        med_m = np.median(mm)
        med_s = np.median(ss)
        is_null = (ss <= 0) | ((mm < mean_rel * med_m) & (ss < sd_rel * med_s))
        mask[blk.start:blk.stop] = ~is_null
    return mask


def normalize(amp: np.ndarray, mask: np.ndarray,
              method: str = "frame_norm") -> np.ndarray:
    """
    AGC 이득 제거. ESP32 CSI amplitude 는 AGC 이득이 곱해진 상대값이라
    정규화 없이 세션 간 평균을 비교하면 재현 불가능한 숫자가 나온다. (결함 D2)

    frame_norm : 프레임마다 유효 subcarrier 의 L2 노름으로 나눈다.
                 프레임 전체에 곱해진 스칼라 이득이 정확히 소거된다.
    none       : 정규화 없음. 비교용.
    """
    if method == "none":
        return amp.astype(np.float64, copy=True)
    if method != "frame_norm":
        raise ValueError("알 수 없는 정규화 방식: %s" % method)

    n = np.linalg.norm(amp[:, mask], axis=1)
    bad = n <= 0
    if np.any(bad):
        raise CSIParseError("유효 subcarrier 노름이 0 인 프레임이 %d 개다. "
                            "마스크나 데이터를 확인해야 한다." % int(bad.sum()))
    # 노름이 1 이 되면 값이 너무 작아 읽기 불편하므로 유효 subcarrier 수의 sqrt 를 곱해
    # '평균 진폭 1 근처' 스케일로 맞춘다. 세션 간 비교에는 영향이 없다.
    scale = np.sqrt(int(mask.sum()))
    return amp / n[:, None] * scale


def hampel(X: np.ndarray, window: int = 11, n_sigma: float = 3.0
           ) -> Tuple[np.ndarray, int]:
    """
    시간축 임펄스 잡음 제거. (n_frames, n_sub) 의 axis=0 을 따라 동작한다.
    반환: (필터된 배열, 치환된 표본 수)
    """
    if X.ndim == 1:
        X = X[:, None]
        squeeze = True
    else:
        squeeze = False
    if window % 2 == 0:
        window += 1
    med = median_filter(X, size=(window, 1), mode="nearest")
    dev = np.abs(X - med)
    mad = median_filter(dev, size=(window, 1), mode="nearest")
    sigma = 1.4826 * mad
    bad = (sigma > 0) & (dev > n_sigma * sigma)
    out = X.copy()
    out[bad] = med[bad]
    if squeeze:
        out = out[:, 0]
    return out, int(bad.sum())


def resample_uniform(t: np.ndarray, X: np.ndarray, fs: float
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    비균일 샘플링을 균일 격자에 올린다. (결함 D6)
    수신률이 세션마다 다르면 그 차이가 변화량 특징 차이로 새어 들어가므로
    스펙트럼 계열 특징은 반드시 리샘플 후에 계산한다.

    주의: 실제 수신률(약 17-18 Hz)보다 fs 가 높으면 업샘플링(보간)이 되어
    frame-to-frame 변화량이 희석된다. 그래서 변화량 계열 특징은 리샘플이 아니라
    diff_matched_dt() 로 원시 시간축에서 계산한다.
    """
    if X.ndim == 1:
        X = X[:, None]
        squeeze = True
    else:
        squeeze = False
    grid = np.arange(t[0], t[-1] + 1e-9, 1.0 / fs)
    Y = np.empty((grid.size, X.shape[1]), dtype=np.float64)
    for j in range(X.shape[1]):
        Y[:, j] = np.interp(grid, t, X[:, j])
    if squeeze:
        Y = Y[:, 0]
    return grid, Y


def diff_matched_dt(t: np.ndarray, X: np.ndarray, tol: float = 0.2
                    ) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    frame-to-frame 절대변화량을 '동일한 dt 를 가진 인접쌍'에서만 계산한다.

    왜 리샘플링 대신 이 방식인가:
      - 20 Hz 리샘플은 실제 수신률(17-18 Hz)보다 높아 보간이 되고,
        보간은 인접 표본을 서로 닮게 만들어 변화량을 체계적으로 과소평가한다.
      - 반면 dt 를 맞춰 원시 표본만 쓰면 보간이 전혀 없고,
        세션 간 수신률 차이가 변화량에 새어드는 문제(D6)도 동시에 해결된다.
        모든 세션이 '같은 dt 에서 잰 변화량'을 비교하게 되기 때문이다.

    반환: (쌍의 중간 시각, 쌍별 subcarrier 평균 절대변화량, 사용한 dt 중앙값)
    """
    dt = np.diff(t)
    dt_med = float(np.median(dt))
    keep = np.abs(dt - dt_med) <= tol * dt_med
    if keep.sum() < 2:
        raise CSIParseError("dt 가 일정한 인접쌍이 거의 없다 (dt 중앙값 %.4f s)" % dt_med)
    d = np.abs(np.diff(X, axis=0))[keep]
    t_mid = (t[:-1] + t[1:])[keep] / 2.0
    return t_mid, d, dt_med


def band_power_ratio(x: np.ndarray, fs: float,
                     band: Tuple[float, float] = (0.15, 0.5),
                     full: Tuple[float, float] = (0.05, 2.0),
                     nperseg: Optional[int] = None) -> float:
    """
    welch PSD 기준 호흡 대역(band) 전력이 관심 대역(full) 전체에서 차지하는 비율.

    주의: 주파수 분해능은 fs/nperseg 다. 호흡 대역 0.15-0.5 Hz 를 구분하려면
    최소 수십 초의 창이 필요하다. 10 초 창에서는 분해능이 0.1 Hz 수준이라
    이 값을 호흡수의 근거로 쓸 수 없다. 호출 측에서 충분히 긴 창을 줘야 한다.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    if x.size < 8:
        return float("nan")
    if nperseg is None:
        nperseg = min(x.size, 1 << int(np.floor(np.log2(max(x.size // 2, 8)))) * 2)
        nperseg = min(nperseg, x.size)
    f, p = welch(x, fs=fs, nperseg=int(nperseg))
    num = p[(f >= band[0]) & (f < band[1])].sum()
    den = p[(f >= full[0]) & (f < full[1])].sum()
    if den <= 0:
        return float("nan")
    return float(num / den)


def breath_snr(x: np.ndarray, fs: float,
               band: Tuple[float, float] = (0.15, 0.5),
               noise: Tuple[float, float] = (0.6, 2.0),
               nperseg: int = 1024) -> Tuple[float, float]:
    """
    한 시계열에서 호흡 대역 최대 피크가 잡음 바닥 대비 몇 배인지.
    반환: (SNR, 피크 주파수)

    band_power_ratio 와의 차이:
      band_power_ratio 는 '대역 전력의 비율' 이라 넓은 잡음에 묻힌다.
      여기서는 '뾰족한 피크가 있는가' 를 본다. 호흡은 준주기 신호라 피크로 나타난다.
      잡음 바닥은 호흡보다 빠른 대역(0.6-2Hz)의 중앙값으로 추정한다.
    """
    from scipy.signal import detrend
    x = np.asarray(x, dtype=np.float64)
    if x.size < nperseg:
        return float("nan"), float("nan")
    x = detrend(x)
    f, p = welch(x, fs=fs, nperseg=int(nperseg))
    b = (f >= band[0]) & (f < band[1])
    n = (f >= noise[0]) & (f < noise[1])
    if not b.any() or not n.any():
        return float("nan"), float("nan")
    floor = float(np.median(p[n]))
    if floor <= 0:
        return float("nan"), float("nan")
    i = int(np.argmax(p[b]))
    return float(p[b][i] / floor), float(f[b][i])


def breath_snr_profile(t: np.ndarray, X: np.ndarray, fs: float,
                       win_sec: float = 100.0, nperseg: int = 1024,
                       band: Tuple[float, float] = (0.15, 0.5),
                       noise: Tuple[float, float] = (0.6, 2.0)
                       ) -> List[np.ndarray]:
    """
    구간을 고정 길이 비중첩 창으로 잘라, 창마다 subcarrier 별 호흡 SNR 프로파일을 만든다.

    창 길이와 nperseg 를 고정하는 이유:
      SNR 은 주파수 분해능과 평균 횟수에 따라 달라진다. 구간 길이가 제각각인 채로
      전체를 쓰면 긴 구간이 유리해져 세션 간 비교가 불공정해진다.
      그래서 모든 구간을 같은 길이 창으로 잘라 같은 조건에서 잰다.

    호흡 성분은 114개 subcarrier 중 소수에만 나타난다. 전체를 PC1 로 뭉개면
    나머지 subcarrier 의 잡음에 묻히므로, subcarrier 별로 따로 재야 한다.
    """
    out: List[np.ndarray] = []
    a = float(t[0])
    while a + win_sec <= t[-1]:
        m = (t >= a) & (t < a + win_sec)
        if m.sum() >= nperseg // 2:
            _, Y = resample_uniform(t[m], X[m], fs)
            if Y.shape[0] >= nperseg:
                out.append(np.array([breath_snr(Y[:, j], fs, band, noise, nperseg)[0]
                                     for j in range(Y.shape[1])]))
        a += win_sec
    return out


def coherence_time(X: np.ndarray, fs: float, thresh: float = 1.0 / np.e,
                   max_lag_sec: float = 20.0) -> float:
    """
    채널 자기상관이 thresh 로 떨어지는 시간 [초]. subcarrier 별로 구해 중앙값을 낸다.

    물리적 근거: 방 안에 사람이 있으면 미세 움직임과 호흡으로 산란 환경이 계속
    조금씩 바뀌므로 채널이 더 빨리 decorrelate 된다. 빈 방은 오래 그대로 있다.
    진폭 크기가 아니라 '변화의 시간 구조' 를 보므로 AGC 나 배치 차이에 덜 민감하다.
    """
    n, k = X.shape
    max_lag = min(int(max_lag_sec * fs), n - 2)
    if max_lag < 2:
        return float("nan")
    Xc = X - X.mean(axis=0, keepdims=True)
    var = (Xc ** 2).mean(axis=0)
    good = var > 0
    if not np.any(good):
        return float("nan")
    Xc = Xc[:, good]
    var = var[good]
    out = np.full(Xc.shape[1], np.nan)
    prev = np.ones(Xc.shape[1])
    for lag in range(1, max_lag + 1):
        r = (Xc[:-lag] * Xc[lag:]).mean(axis=0) / var
        newly = np.isnan(out) & (r < thresh)
        if np.any(newly):
            # 선형 보간으로 교차 시점을 추정한다
            frac = (prev[newly] - thresh) / np.maximum(prev[newly] - r[newly], 1e-12)
            out[newly] = (lag - 1 + frac) / fs
        prev = r
        if not np.any(np.isnan(out)):
            break
    return float(np.nanmedian(out))


def effective_rank(X: np.ndarray) -> float:
    """
    subcarrier 공분산 고유값의 참여비 (sum L)^2 / sum L^2.
    채널이 몇 개의 독립 자유도로 설명되는지를 나타낸다.

    물리적 근거: 사람은 산란체다. 방 안에 사람이 있으면 다중경로 성분이 늘어
    채널의 실효 자유도가 올라간다. 빈 방은 소수의 강한 경로가 지배한다.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    if Xc.shape[0] < 2:
        return float("nan")
    C = np.cov(Xc, rowvar=False)
    w = np.linalg.eigvalsh(C)
    w = w[w > 0]
    if w.size == 0:
        return float("nan")
    return float(w.sum() ** 2 / (w ** 2).sum())


def amp_kurtosis(X: np.ndarray) -> float:
    """
    subcarrier 별 진폭 분포의 초과첨도 중앙값.

    물리적 근거: 직시경로가 지배하면 Rician(첨도 낮음), 산란이 지배하면
    Rayleigh 에 가까워진다(첨도 높음). 사람이 들어오면 분포 모양이 바뀐다.
    """
    from scipy.stats import kurtosis
    return float(np.median(kurtosis(X, axis=0, fisher=True, bias=False)))


def lowfreq_power_ratio(t: np.ndarray, X: np.ndarray, fs: float,
                        band: Tuple[float, float] = (0.01, 0.1),
                        full: Tuple[float, float] = (0.01, 2.0),
                        nperseg: int = 2048) -> float:
    """
    아주 느린 대역(기본 0.01-0.1Hz)의 전력 비율. subcarrier 중앙값.

    물리적 근거: 사람이 '가만히' 있어도 자세는 수십 초 규모로 조금씩 바뀐다.
    빈 방에는 그런 성분이 없다. 호흡보다 느린 대역이라 호흡 특징과 독립적이다.
    """
    _, Y = resample_uniform(t, X, fs)
    if Y.shape[0] < nperseg:
        return float("nan")
    f, p = welch(Y, fs=fs, nperseg=nperseg, axis=0)
    b = (f >= band[0]) & (f < band[1])
    a = (f >= full[0]) & (f < full[1])
    if not b.any() or not a.any():
        return float("nan")
    num = p[b].sum(axis=0)
    den = p[a].sum(axis=0)
    ok = den > 0
    if not np.any(ok):
        return float("nan")
    return float(np.median(num[ok] / den[ok]))


def pc1(X: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    (n, K) 의 제1주성분 시계열과 그 분산비.
    subcarrier 를 하나로 요약해 호흡/움직임 성분을 보기 위한 것이다.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    # 표본 수보다 subcarrier 가 적으므로 economy SVD 로 충분하다
    u, s, _ = np.linalg.svd(Xc, full_matrices=False)
    if s.size == 0 or s.sum() == 0:
        return np.zeros(X.shape[0]), float("nan")
    var = s ** 2
    return u[:, 0] * s[0], float(var[0] / var.sum())


# ===========================================================================
# 4. 통계 (세션 단위)
# ===========================================================================

def min_two_sided_p_unpaired(na: int, nb: int) -> float:
    """
    비대응 정확 순열검정에서 도달 가능한 최소 양측 p.

    총 C(na+nb, na) 개 배정 중 관측만큼 극단인 것의 비율이 p 다.
      na == nb : A/B 를 맞바꾼 배정이 통계량의 부호만 뒤집어 |통계량| 이 같으므로
                 극단값이 항상 짝으로 나온다 -> 최소 2 / C.
      na != nb : 맞바꾼 배정은 애초에 유효한 배정이 아니라(A 의 크기가 달라진다)
                 관측 배정 하나만 극단일 수 있다 -> 최소 1 / C.
    이 구분을 놓치면 표본 수가 다를 때 검정 가능성을 실제보다 비관적으로 보게 된다.
    """
    from math import comb
    if na < 1 or nb < 1:
        return float("nan")
    c = comb(na + nb, na)
    return (2.0 / c) if na == nb else (1.0 / c)


def min_two_sided_p_paired(n: int) -> float:
    """대응(부호뒤집기) 정확 순열검정에서 도달 가능한 최소 양측 p = 2 / 2^n."""
    if n < 1:
        return float("nan")
    return 2.0 / (2 ** n)


def exact_perm_test_unpaired(a: Sequence[float], b: Sequence[float]
                             ) -> Tuple[float, float, float]:
    """
    세션 단위 정확 순열검정 (전수 조합). 통계량은 평균 차이.
    반환: (관측 차이, 양측 p, 도달 가능한 최소 양측 p)
    """
    from itertools import combinations
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = a.size, b.size
    obs = float(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    idx = np.arange(na + nb)
    count = 0
    total = 0
    for c in combinations(idx, na):
        c = list(c)
        rest = np.setdiff1d(idx, c, assume_unique=True)
        stat = pool[c].mean() - pool[rest].mean()
        total += 1
        if abs(stat) >= abs(obs) - 1e-12:
            count += 1
    return obs, count / total, min_two_sided_p_unpaired(na, nb)


def exact_perm_test_paired(d: Sequence[float]) -> Tuple[float, float, float]:
    """
    대응 표본 정확 순열검정 (부호 전수 뒤집기). d 는 쌍별 차이.
    세션 내 still/empty 대조처럼 세션 간 드리프트가 상쇄되는 설계에 쓴다.
    반환: (관측 평균차, 양측 p, 도달 가능한 최소 양측 p)
    """
    from itertools import product
    d = np.asarray(d, dtype=np.float64)
    n = d.size
    obs = float(d.mean())
    count = 0
    total = 0
    for signs in product((1.0, -1.0), repeat=n):
        stat = float(np.mean(d * np.asarray(signs)))
        total += 1
        if abs(stat) >= abs(obs) - 1e-12:
            count += 1
    return obs, count / total, min_two_sided_p_paired(n)


def sep_ratio(a: Sequence[float], b: Sequence[float]) -> float:
    """
    |클래스 간 차이| / 클래스 내 세션 표준편차(pooled).
    2 미만이면 구분한다고 주장할 근거가 없다.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = a.size, b.size
    if na < 2 and nb < 2:
        return float("nan")
    ss = 0.0
    df = 0
    if na >= 2:
        ss += ((a - a.mean()) ** 2).sum(); df += na - 1
    if nb >= 2:
        ss += ((b - b.mean()) ** 2).sum(); df += nb - 1
    sd = np.sqrt(ss / df) if df > 0 else 0.0
    if sd == 0:
        return float("inf")
    return float(abs(a.mean() - b.mean()) / sd)


def cohens_d_sessions(a: Sequence[float], b: Sequence[float]) -> float:
    """
    세션 단위 pooled SD 로 나눈 효과크기. (결함 D5)
    분모에 '세션 내 시간 표준편차'를 쓰면 자기상관 때문에 Cohen's d 가 아니다.
    """
    return sep_ratio(a, b) * np.sign(np.mean(a) - np.mean(b))


def setup_matplotlib():
    """
    그림 공통 설정. Agg 백엔드를 강제하고 한글 폰트를 잡는다.
    plt.show() 는 쓰지 않는다. 결과물은 전부 파일로 저장한다.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    avail = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim"):
        if cand in avail:
            plt.rcParams["font.family"] = cand
            break
    else:
        print("[warn] 한글 폰트를 찾지 못했다. 그림의 한글이 깨질 수 있다.")
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["savefig.bbox"] = "tight"
    return plt


# 상태별 고정 색. 모든 그림에서 같은 색을 쓴다.
STATE_COLOR = {
    "still": "#1f77b4",     # 재실·정지
    "empty": "#d62728",     # 비재실
    "motion": "#2ca02c",    # 움직임
    "mixed": "#7f7f7f",
}


def benjamini_hochberg(p: Sequence[float], q: float = 0.05
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    BH FDR 보정. subcarrier 수백 개를 스크리닝할 때 다중비교 보정 없이
    상위 20개를 뽑으면 잡음을 고른 것이다. (결함 D5)
    반환: (기각 여부 배열, 보정 p 배열)
    """
    p = np.asarray(p, dtype=np.float64)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    return out <= q, out
