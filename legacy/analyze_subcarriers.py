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

    total_rows = 0
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

            total_rows += 1

            try:
                rssi = int(row[3])
                timestamp = int(row[18])

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
        "csi": np.asarray(
            frames,
            dtype=np.float64
        ),

        "rssi": np.asarray(
            rssi_values,
            dtype=np.float64
        ),

        "timestamp": np.asarray(
            timestamps,
            dtype=np.float64
        ),

        "total_rows": total_rows,
        "invalid_rows": invalid_rows,
    }


# ============================================================
# I/Q → Amplitude
# ============================================================

def calculate_amplitude(csi):

    # 현재 파일럿:
    # 384 raw values
    # → 192개의 I/Q pair

    i = csi[:, 0::2]
    q = csi[:, 1::2]

    return np.sqrt(
        i ** 2 + q ** 2
    )


# ============================================================
# CSI index별 통계
# ============================================================

def calculate_statistics(amplitude):

    mean = np.mean(
        amplitude,
        axis=0
    )

    std = np.std(
        amplitude,
        axis=0
    )

    median = np.median(
        amplitude,
        axis=0
    )

    q25 = np.percentile(
        amplitude,
        25,
        axis=0
    )

    q75 = np.percentile(
        amplitude,
        75,
        axis=0
    )

    iqr = q75 - q25

    return {
        "mean": mean,
        "std": std,
        "median": median,
        "iqr": iqr,
    }


# ============================================================
# Frame-to-frame 변화량
# ============================================================

def calculate_frame_difference(amplitude):

    if len(amplitude) < 2:

        return np.zeros(
            amplitude.shape[1],
            dtype=np.float64
        )

    diff = np.abs(
        np.diff(
            amplitude,
            axis=0
        )
    )

    return np.mean(
        diff,
        axis=0
    )


# ============================================================
# CSI index → LTF 구간 표시
# 현재 384 raw = 192 I/Q pair 조건의 탐색용 매핑
# ============================================================

def map_csi_index(index):
    """
    현재 실험의 len=384 조건에서 사용하는 탐색용 매핑.

    0~63   : LLTF 영역으로 분류
    64~191 : HT-LTF 영역으로 분류

    주의:
    논문 최종 분석 전에 ESP-IDF CSI 문서/구현과
    정확한 subcarrier 번호 대응을 다시 검증할 것.
    """

    if 0 <= index <= 63:
        return "LLTF", index

    elif 64 <= index <= 191:
        return "HT-LTF", index - 64

    else:
        return "UNKNOWN", -1


# ============================================================
# 분석
# ============================================================

def analyze():

    print()
    print("CSI 데이터를 읽는 중...")

    stable_data = load_csi(
        STABLE_FILE
    )

    empty_data = load_csi(
        EMPTY_FILE
    )

    stable_amp = calculate_amplitude(
        stable_data["csi"]
    )

    empty_amp = calculate_amplitude(
        empty_data["csi"]
    )

    stable_stats = calculate_statistics(
        stable_amp
    )

    empty_stats = calculate_statistics(
        empty_amp
    )

    stable_diff = calculate_frame_difference(
        stable_amp
    )

    empty_diff = calculate_frame_difference(
        empty_amp
    )

    mean_difference = np.abs(
        stable_stats["mean"]
        - empty_stats["mean"]
    )

    std_difference = np.abs(
        stable_stats["std"]
        - empty_stats["std"]
    )

    diff_difference = np.abs(
        stable_diff
        - empty_diff
    )

    # 탐색용 차이 점수
    pooled_std = np.sqrt(
        (
            stable_stats["std"] ** 2
            + empty_stats["std"] ** 2
        ) / 2
    )

    effect_score = np.divide(
        mean_difference,
        pooled_std,
        out=np.zeros_like(
            mean_difference
        ),
        where=pooled_std > 0
    )

    return {
        "stable_data": stable_data,
        "empty_data": empty_data,

        "stable_amp": stable_amp,
        "empty_amp": empty_amp,

        "stable_stats": stable_stats,
        "empty_stats": empty_stats,

        "stable_diff": stable_diff,
        "empty_diff": empty_diff,

        "mean_difference": mean_difference,
        "std_difference": std_difference,
        "diff_difference": diff_difference,

        "effect_score": effect_score,
    }


# ============================================================
# 요약
# ============================================================

def print_summary(result):

    print()
    print("=" * 65)
    print("CSI INDEX ANALYSIS SUMMARY")
    print("=" * 65)

    print(
        "Stable 정상 프레임 :",
        len(
            result["stable_data"]["csi"]
        )
    )

    print(
        "Stable 비정상     :",
        result[
            "stable_data"
        ]["invalid_rows"]
    )

    print(
        "Empty 정상 프레임  :",
        len(
            result["empty_data"]["csi"]
        )
    )

    print(
        "Empty 비정상      :",
        result[
            "empty_data"
        ]["invalid_rows"]
    )

    print(
        "CSI raw length     :",
        EXPECTED_CSI_LEN
    )

    print(
        "I/Q pair 개수      :",
        result[
            "stable_amp"
        ].shape[1]
    )


# ============================================================
# 차이가 큰 index TOP 20
# ============================================================

def print_top_indices(
    result,
    top_n=20
):

    mean_difference = (
        result["mean_difference"]
    )

    effect_score = (
        result["effect_score"]
    )

    stable_mean = (
        result["stable_stats"]["mean"]
    )

    empty_mean = (
        result["empty_stats"]["mean"]
    )

    stable_diff = (
        result["stable_diff"]
    )

    empty_diff = (
        result["empty_diff"]
    )

    indices = np.argsort(
        effect_score
    )[::-1]

    print()
    print("=" * 115)

    print(
        f"상태 차이가 큰 CSI index TOP {top_n}"
    )

    print("=" * 115)

    print(
        f"{'Rank':<6}"
        f"{'Index':<8}"
        f"{'LTF':<10}"
        f"{'LocalIdx':<10}"
        f"{'StableMean':<14}"
        f"{'EmptyMean':<14}"
        f"{'AbsDiff':<12}"
        f"{'Effect':<12}"
        f"{'StableVar':<14}"
        f"{'EmptyVar':<14}"
    )

    for rank, idx in enumerate(
        indices[:top_n],
        start=1
    ):

        ltf_name, local_idx = map_csi_index(idx)

        print(
            f"{rank:<6}"
            f"{idx:<8}"
            f"{ltf_name:<10}"
            f"{local_idx:<10}"
            f"{stable_mean[idx]:<14.3f}"
            f"{empty_mean[idx]:<14.3f}"
            f"{mean_difference[idx]:<12.3f}"
            f"{effect_score[idx]:<12.3f}"
            f"{stable_diff[idx]:<14.3f}"
            f"{empty_diff[idx]:<14.3f}"
        )


# ============================================================
# CSV 저장
# ============================================================

def save_csv(result):

    output = (
        BASE_DIR
        / "subcarrier_analysis.csv"
    )

    stable_stats = (
        result["stable_stats"]
    )

    empty_stats = (
        result["empty_stats"]
    )

    with open(
        output,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "csi_index",
            "ltf_block",
            "ltf_local_index",

            "stable_mean",
            "empty_mean",
            "mean_abs_difference",

            "stable_std",
            "empty_std",

            "stable_median",
            "empty_median",

            "stable_iqr",
            "empty_iqr",

            "stable_frame_diff",
            "empty_frame_diff",

            "effect_score"
        ])

        stable_mean = result["stable_stats"]["mean"]
        empty_mean = result["empty_stats"]["mean"]

        stable_std = result["stable_stats"]["std"]
        empty_std = result["empty_stats"]["std"]

        stable_median = result["stable_stats"]["median"]
        empty_median = result["empty_stats"]["median"]

        stable_iqr = result["stable_stats"]["iqr"]
        empty_iqr = result["empty_stats"]["iqr"]

        stable_diff = result["stable_diff"]
        empty_diff = result["empty_diff"]

        mean_difference = result["mean_difference"]
        effect_score = result["effect_score"]

        for idx in range(len(stable_mean)):

            ltf_name, local_idx = map_csi_index(idx)

            writer.writerow([
                idx,
                ltf_name,
                local_idx,

                stable_mean[idx],
                empty_mean[idx],
                mean_difference[idx],

                stable_std[idx],
                empty_std[idx],

                stable_median[idx],
                empty_median[idx],

                stable_iqr[idx],
                empty_iqr[idx],

                stable_diff[idx],
                empty_diff[idx],

                effect_score[idx]
            ])

    print()
    print("CSV 저장 완료:", output)


# ============================================================
# 그래프 1
# CSI index별 평균 amplitude
# ============================================================

def plot_mean_amplitude(result):

    stable_mean = result["stable_stats"]["mean"]
    empty_mean = result["empty_stats"]["mean"]

    indices = np.arange(len(stable_mean))

    plt.figure(figsize=(15, 6))

    plt.plot(
        indices,
        stable_mean,
        label="Stable"
    )

    plt.plot(
        indices,
        empty_mean,
        label="Empty"
    )

    plt.xlabel("CSI I/Q pair index")
    plt.ylabel("Mean amplitude")

    plt.title(
        "Mean CSI Amplitude by CSI Index"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output = (
        BASE_DIR
        / "subcarrier_mean_amplitude.png"
    )

    plt.savefig(
        output,
        dpi=150
    )

    print(
        "그래프 저장:",
        output
    )

    plt.show()


# ============================================================
# 그래프 2
# Stable / Empty 평균 amplitude 차이
# ============================================================

def plot_mean_difference(result):

    difference = result[
        "mean_difference"
    ]

    indices = np.arange(
        len(difference)
    )

    plt.figure(figsize=(15, 6))

    plt.bar(
        indices,
        difference
    )

    plt.xlabel(
        "CSI I/Q pair index"
    )

    plt.ylabel(
        "Absolute mean amplitude difference"
    )

    plt.title(
        "Stable vs Empty Amplitude Difference by CSI Index"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    output = (
        BASE_DIR
        / "subcarrier_amplitude_difference.png"
    )

    plt.savefig(
        output,
        dpi=150
    )

    print(
        "그래프 저장:",
        output
    )

    plt.show()


# ============================================================
# 그래프 3
# CSI index별 Frame-to-frame 변화량
# ============================================================

def plot_frame_difference(result):

    stable_diff = result[
        "stable_diff"
    ]

    empty_diff = result[
        "empty_diff"
    ]

    indices = np.arange(
        len(stable_diff)
    )

    plt.figure(figsize=(15, 6))

    plt.plot(
        indices,
        stable_diff,
        label="Stable"
    )

    plt.plot(
        indices,
        empty_diff,
        label="Empty"
    )

    plt.xlabel(
        "CSI I/Q pair index"
    )

    plt.ylabel(
        "Mean frame-to-frame amplitude difference"
    )

    plt.title(
        "CSI Temporal Variation by CSI Index"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output = (
        BASE_DIR
        / "subcarrier_frame_variation.png"
    )

    plt.savefig(
        output,
        dpi=150
    )

    print(
        "그래프 저장:",
        output
    )

    plt.show()


# ============================================================
# 그래프 4
# 탐색용 Effect Score
# ============================================================

def plot_effect_score(result):

    effect_score = result[
        "effect_score"
    ]

    indices = np.arange(
        len(effect_score)
    )

    plt.figure(figsize=(15, 6))

    plt.bar(
        indices,
        effect_score
    )

    plt.xlabel(
        "CSI I/Q pair index"
    )

    plt.ylabel(
        "Exploratory effect score"
    )

    plt.title(
        "Stable vs Empty Difference Score by CSI Index"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    output = (
        BASE_DIR
        / "subcarrier_effect_score.png"
    )

    plt.savefig(
        output,
        dpi=150
    )

    print(
        "그래프 저장:",
        output
    )

    plt.show()


# ============================================================
# main
# ============================================================

def main():

    result = analyze()

    print_summary(
        result
    )

    print_top_indices(
        result,
        top_n=20
    )

    save_csv(
        result
    )

    plot_mean_amplitude(
        result
    )

    plot_mean_difference(
        result
    )

    plot_frame_difference(
        result
    )

    plot_effect_score(
        result
    )


if __name__ == "__main__":
    main()