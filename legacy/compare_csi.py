import csv
import ast
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

STABLE_FILE = DATA_DIR / "stable_5min.txt"
EMPTY_FILE = DATA_DIR / "empty_5min.txt"

EXPECTED_CSI_LEN = 384


# ============================================================
# CSI RAW 파일 읽기
# ============================================================

def load_csi(filepath):
    frames = []
    rssi_values = []
    timestamps = []

    total_csi_rows = 0
    invalid_rows = 0

    with open(
        filepath,
        "r",
        encoding="utf-8",
        errors="ignore",
        newline=""
    ) as f:

        reader = csv.reader(f)

        for row in reader:
            if not row:
                continue

            if row[0] != "CSI_DATA":
                continue

            total_csi_rows += 1

            try:
                rssi = int(row[3])
                timestamp = int(row[18])

                # 실제 CSI_DATA CSV 구조 기준
                csi_len = int(row[22])
                csi = ast.literal_eval(row[24])

                if csi_len != EXPECTED_CSI_LEN:
                    invalid_rows += 1
                    continue

                if len(csi) != EXPECTED_CSI_LEN:
                    invalid_rows += 1
                    continue

                frames.append(csi)
                rssi_values.append(rssi)
                timestamps.append(timestamp)

            except (
                ValueError,
                SyntaxError,
                IndexError,
                TypeError
            ):
                invalid_rows += 1

    if not frames:
        raise RuntimeError(
            f"정상 CSI 데이터를 찾지 못했습니다: {filepath}"
        )

    return {
        "csi": np.asarray(frames, dtype=np.float64),
        "rssi": np.asarray(rssi_values, dtype=np.float64),
        "timestamp": np.asarray(timestamps, dtype=np.float64),
        "total_rows": total_csi_rows,
        "invalid_rows": invalid_rows,
    }


# ============================================================
# I/Q → Amplitude
# ============================================================

def calculate_amplitude(csi):

    i = csi[:, 0::2]
    q = csi[:, 1::2]

    return np.sqrt(i ** 2 + q ** 2)


# ============================================================
# Timestamp → 초
# ============================================================

def timestamp_to_seconds(timestamps):

    relative = timestamps - timestamps[0]

    relative = np.where(
        relative < 0,
        relative + (2 ** 32),
        relative
    )

    return relative / 1_000_000.0


# ============================================================
# 상태 하나 분석
# ============================================================

def analyze(name, filepath):

    data = load_csi(filepath)

    csi = data["csi"]
    rssi = data["rssi"]
    timestamps = data["timestamp"]

    amplitude = calculate_amplitude(csi)

    # 각 프레임의 전체 평균 amplitude
    frame_amp_mean = np.mean(
        amplitude,
        axis=1
    )

    # 연속 프레임 사이 amplitude 변화량
    if len(amplitude) >= 2:
        frame_diff = np.mean(
            np.abs(
                np.diff(
                    amplitude,
                    axis=0
                )
            ),
            axis=1
        )
    else:
        frame_diff = np.array(
            [],
            dtype=np.float64
        )

    time_sec = timestamp_to_seconds(
        timestamps
    )

    if (
        len(time_sec) >= 2
        and time_sec[-1] > 0
    ):
        receive_rate = (
            len(time_sec) - 1
        ) / time_sec[-1]
    else:
        receive_rate = float("nan")

    return {
        "name": name,
        "filepath": filepath,

        "frames": len(csi),
        "invalid": data["invalid_rows"],

        "receive_rate": receive_rate,

        "rssi_mean": np.mean(rssi),
        "rssi_std": np.std(rssi),
        "rssi_min": np.min(rssi),
        "rssi_max": np.max(rssi),

        "amp_mean": np.mean(frame_amp_mean),
        "amp_std": np.std(frame_amp_mean),

        "diff_mean": (
            np.mean(frame_diff)
            if len(frame_diff)
            else float("nan")
        ),

        "diff_std": (
            np.std(frame_diff)
            if len(frame_diff)
            else float("nan")
        ),

        "time": time_sec,
        "frame_amp_mean": frame_amp_mean,
        "frame_diff": frame_diff,
    }


# ============================================================
# 결과 출력
# ============================================================

def print_result(result):

    print()
    print("=" * 55)
    print(result["name"])
    print("=" * 55)

    print(f"파일                 : {result['filepath']}")
    print(f"정상 CSI 프레임      : {result['frames']}")
    print(f"비정상 CSI 프레임    : {result['invalid']}")
    print(f"평균 수신률          : {result['receive_rate']:.2f} Hz")

    print()
    print("[RSSI]")

    print(f"평균                 : {result['rssi_mean']:.3f} dBm")
    print(f"표준편차             : {result['rssi_std']:.3f} dB")

    print(
        f"범위                 : "
        f"{result['rssi_min']:.0f} ~ "
        f"{result['rssi_max']:.0f} dBm"
    )

    print()
    print("[CSI Amplitude]")

    print(f"평균                 : {result['amp_mean']:.3f}")
    print(f"시간적 표준편차      : {result['amp_std']:.3f}")

    print()
    print("[Frame-to-frame variation]")

    print(f"평균 변화량          : {result['diff_mean']:.3f}")
    print(f"변화량 표준편차      : {result['diff_std']:.3f}")


# ============================================================
# 평균 Amplitude 그래프
# ============================================================

def plot_amplitude(stable, empty):

    plt.figure(figsize=(13, 6))

    plt.plot(
        stable["time"],
        stable["frame_amp_mean"],
        label="Stable"
    )

    plt.plot(
        empty["time"],
        empty["frame_amp_mean"],
        label="Empty"
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Mean CSI amplitude")
    plt.title("CSI Mean Amplitude: Stable vs Empty")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output = (
        BASE_DIR
        / "stable_vs_empty_amplitude.png"
    )

    plt.savefig(
        output,
        dpi=150
    )

    print(
        f"\nAmplitude 그래프 저장: {output}"
    )

    plt.show()


# ============================================================
# Frame 변화량 그래프
# ============================================================

def plot_variation(stable, empty):

    plt.figure(figsize=(13, 6))

    plt.plot(
        stable["time"][1:],
        stable["frame_diff"],
        label="Stable"
    )

    plt.plot(
        empty["time"][1:],
        empty["frame_diff"],
        label="Empty"
    )

    plt.xlabel("Time (seconds)")

    plt.ylabel(
        "Mean absolute CSI amplitude difference"
    )

    plt.title(
        "CSI Frame-to-Frame Variation: Stable vs Empty"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output = (
        BASE_DIR
        / "stable_vs_empty_variation.png"
    )

    plt.savefig(
        output,
        dpi=150
    )

    print(
        f"Variation 그래프 저장: {output}"
    )

    plt.show()


# ============================================================
# 실행
# ============================================================

def main():

    print("CSI 파일을 읽는 중...")

    stable = analyze(
        "STABLE (person lying still)",
        STABLE_FILE
    )

    empty = analyze(
        "EMPTY (no person)",
        EMPTY_FILE
    )

    print_result(stable)
    print_result(empty)

    print()
    print("=" * 55)
    print("STABLE vs EMPTY 비교")
    print("=" * 55)

    print(
        f"RSSI 평균 차이        : "
        f"{stable['rssi_mean'] - empty['rssi_mean']:.3f} dB"
    )

    print(
        f"Amplitude 평균 차이   : "
        f"{stable['amp_mean'] - empty['amp_mean']:.3f}"
    )

    print(
        f"Amplitude 변동성 차이 : "
        f"{stable['amp_std'] - empty['amp_std']:.3f}"
    )

    print(
        f"Frame 변화량 차이     : "
        f"{stable['diff_mean'] - empty['diff_mean']:.3f}"
    )

    plot_amplitude(
        stable,
        empty
    )

    plot_variation(
        stable,
        empty
    )


if __name__ == "__main__":
    main()