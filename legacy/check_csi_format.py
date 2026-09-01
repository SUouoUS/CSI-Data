import csv
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

FILES = {
    "STABLE": DATA_DIR / "stable_5min.txt",
    "EMPTY": DATA_DIR / "empty_5min.txt",
}


def inspect_file(name, filepath):
    counter = Counter()
    total = 0

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

            total += 1

            try:
                sig_mode = int(row[5])
                bandwidth = int(row[7])
                stbc = int(row[11])
                secondary_channel = int(row[17])
                csi_len = int(row[22])

                key = (
                    sig_mode,
                    bandwidth,
                    stbc,
                    secondary_channel,
                    csi_len,
                )

                counter[key] += 1

            except (
                ValueError,
                IndexError
            ):
                continue

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print("총 CSI 프레임:", total)

    for key, count in counter.items():
        print()
        print("sig_mode         :", key[0])
        print("bandwidth        :", key[1])
        print("stbc             :", key[2])
        print("secondary_channel:", key[3])
        print("CSI len          :", key[4])
        print("프레임 수        :", count)


def main():
    for name, filepath in FILES.items():
        inspect_file(
            name,
            filepath
        )


if __name__ == "__main__":
    main()