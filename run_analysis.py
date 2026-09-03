# -*- coding: utf-8 -*-
"""
run_analysis.py — 분석 실행기

여러 스크립트를 정해진 순서로 돌리고, 결과를 배치 x 분석방법 폴더로 분리한다.
분석 로직은 전혀 들어 있지 않다. 계산은 전부 기존 스크립트가 그대로 한다.
이 파일이 하는 일은 다음뿐이다.

  - 분석 방법 선택과 실행 순서 관리
  - 출력 폴더 지정 (CSI_OUT_DIR 환경변수로 각 스크립트에 전달)
  - 배치 선택 (CSI_BATCH 환경변수)
  - 터미널 출력 + console.log 동시 기록
  - 성공 / 실패 / 미지원 / QC실패로인한중단 구분
  - run_metadata.json 기록

사용법
---------------------------------------------------------------------------
  python run_analysis.py <mode> [--batch batch_b|batch_c] [--dry-run] [--keep-going]

주의: 이 저장소는 CSILEEP 파일럿 분석용 로컬 저장소다. sessions.py 의 구간 정의는
      파일럿 분석용 임시 매니페스트이며 메인 csileep 저장소의 공식 데이터 계약이
      아니다. 자세한 내용은 docs/RESULTS_GUIDE.md 를 보라.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent
OUT_ROOT = BASE_DIR / "out"

# 배치 식별자 -> (결과 폴더 이름, sessions.py 의 batch 값, 설명)
BATCHES: Dict[str, Dict[str, str]] = {
    "batch_b": {"dir": "batch_b_2026-09-01", "code": "B",
                "desc": "피험자 3명 + 비점유 2회, 5분 세션 (2026-09-01 수집)"},
    "batch_c": {"dir": "batch_c_2026-09-02", "code": "C",
                "desc": "피험자 4명, 14분 empty->still->empty (2026-09-02 수집)"},
}
DEFAULT_BATCH = "batch_b"

# 분석 방법 정의. scripts 는 실행 순서 그대로다.
METHODS: Dict[str, Dict[str, object]] = {
    "qc": {
        "dir": "qc",
        "title": "데이터 품질 검사",
        "desc": "raw CSI 형식·길이·송신 MAC·RSSI·수신률·손실률을 분석 전에 확인한다",
        "scripts": ["s1_check_format.py"],
    },
    "baseline": {
        "dir": "baseline",
        "title": "기존 정규화 기반 분석",
        "desc": "정규화된 CSI 에서 시간 변화와 subcarrier 패턴을 분석한다 (ML V1 메인)",
        "scripts": ["s2_session_stats.py", "s3_subcarrier.py"],
    },
    "pilot": {
        "dir": "pilot",
        "title": "추가 파일럿 진단",
        "desc": "호흡 관련 주기 성분과 대체 특징을 탐색한다 (최종 ML 파이프라인 아님)",
        "scripts": ["s5_pilot_check.py"],
    },
    "a": {
        "dir": "method_a_rssi",
        "title": "A안 RSSI·신호 크기",
        "desc": "정규화로 지워진 신호 크기를 RSSI 로 되살려 점유 구분에 쓸 수 있는지 본다",
        "scripts": ["s5_rssi.py"],
    },
    "b": {
        "dir": "method_b_structure",
        "title": "B안 subcarrier 상관 구조",
        "desc": "subcarrier 들이 서로 어떻게 묶여 움직이는지(상관행렬·고유값·유효랭크) 본다",
        "scripts": ["s3b_structure.py"],
    },
    "transition": {
        "dir": "transition",
        "title": "상태 전환 시각화",
        "desc": "empty->still->empty 세션의 시간 흐름·heatmap·PSD 를 그린다 (검정 안 함)",
        "scripts": ["s6_visualize.py"],
    },
}

# all 모드의 실행 순서. QC 가 맨 앞이어야 한다.
ALL_ORDER = ["qc", "baseline", "pilot", "a", "b", "transition"]

# 배치별 지원 매트릭스.
#
# batch_c 에서 a / b 를 뺀 이유 (코드 근거):
#   - 배치 C 는 한 세션 안에 empty 와 still 이 모두 있는 '세션 내 대응' 설계다.
#     그런데 A안·B안은 exact_perm_test_unpaired 만 쓴다. 참가자 단위로 합치면
#     still 쪽 단위와 empty 쪽 단위가 같은 사람이 되어, 대응 설계를 독립 표본으로
#     검정하는 꼴이 된다. 하한 p 도 4대4 비대응 0.0286 대 대응 0.125 로 어긋난다.
#   - B안은 배치 B 프로토콜(HOLD_EVENTS, events 기반 motion 블록, motion 대 empty
#     양성 대조)을 구조적으로 전제한다. 배치 C 에는 motion 구간이 아예 없다.
#   대응 검정으로 바꾸는 것은 분석 방법 변경이므로 별도 검토와 승인이 필요하다.
#   그때까지 억지로 실행하지 않고 incompatible 로 표시한다.
BATCH_SUPPORT: Dict[str, Dict[str, object]] = {
    "batch_b": {
        "supported": {"qc", "baseline", "pilot", "a", "b"},
        "reasons": {
            "transition": "배치 B 에는 empty->still->empty 전환 프로토콜이 없다",
        },
    },
    "batch_c": {
        "supported": {"qc", "baseline", "pilot", "transition"},
        "reasons": {
            "a": "배치 C 는 세션 내 대응 설계인데 A안은 비대응 순열검정만 쓴다 "
                 "(대응 검정 적용은 별도 승인 필요)",
            "b": "배치 C 는 세션 내 대응 설계이고 motion 구간이 없는데 B안은 "
                 "비대응 검정과 배치 B 프로토콜(events/motion)을 전제한다",
        },
    },
}

STATUS_OK = "success"
STATUS_FAIL = "failed"
STATUS_INCOMPAT = "incompatible"
STATUS_SKIP_QC = "skipped_due_to_qc_failure"

RULE = "=" * 78


# ===========================================================================
# 도우미
# ===========================================================================

def usage() -> str:
    lines = [
        "사용법: python run_analysis.py [mode] [옵션]",
        "",
        "mode:",
        "  qc          데이터 품질 검사",
        "  baseline    기존 정규화 기반 분석",
        "  pilot       추가 파일럿 진단",
        "  a           A안 RSSI·신호 크기 분석",
        "  b           B안 subcarrier 상관 구조 분석",
        "  transition  상태 전환 시각화 (empty->still->empty)",
        "  all         전체 분석 순서대로 실행",
        "",
        "옵션:",
        "  --batch batch_b|batch_c   분석할 수집 배치 (기본 %s)" % DEFAULT_BATCH,
        "  --dry-run                 실행 계획만 출력. 파일도 폴더도 만들지 않는다",
        "  --keep-going              QC 가 실패해도 후속 분석을 계속한다",
        "",
        "배치:",
    ]
    for key, b in BATCHES.items():
        lines.append("  %-9s %s -> out/%s/" % (key, b["desc"], b["dir"]))
    return "\n".join(lines)


def git_commit() -> Optional[str]:
    """현재 커밋 해시. git 저장소가 아니거나 실패하면 None (예외를 내지 않는다)."""
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(BASE_DIR),
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            return r.stdout.strip() or None
    except Exception:
        pass
    return None


def check_batch(batch_key: str) -> Optional[str]:
    """
    배치를 실제로 쓸 수 있는지 미리 본다. 문제가 없으면 None, 있으면 사유 문자열.

    필터가 0개를 고르거나 파일이 전부 없는 채로 '성공' 하는 것이 가장 위험하므로
    실행 전에 잡는다.
    """
    code = BATCHES[batch_key]["code"]
    try:
        env_backup = os.environ.get("CSI_BATCH")
        os.environ["CSI_BATCH"] = code
        sys.path.insert(0, str(BASE_DIR))
        import importlib
        import sessions as S
        importlib.reload(S)
    except Exception as exc:
        return "sessions.py 를 읽지 못했다: %s" % exc
    finally:
        if env_backup is None:
            os.environ.pop("CSI_BATCH", None)
        else:
            os.environ["CSI_BATCH"] = env_backup

    specs = [sp for sp in S.SESSIONS if str(sp.get("batch", "")).upper() == code]
    if not specs:
        declared = sorted({str(sp.get("batch", "")).upper() for sp in S.SESSIONS})
        return ("배치 %s 가 sessions.py 의 SESSIONS 매니페스트에 없다 "
                "(선언된 배치: %s)" % (code, declared or ["(없음)"]))

    missing = [str(sp["file"]) for sp in specs
               if not (BASE_DIR / "data" / str(sp["file"])).exists()]
    if len(missing) == len(specs):
        return ("배치 %s 의 raw 파일이 data/ 에 하나도 없다 (%d개 전부 없음: %s)"
                % (code, len(missing), ", ".join(missing[:4])))
    return None


def ensure_readme(outdir: Path, batch_key: str, mode: str) -> None:
    """
    결과 폴더 README. **파일이 없을 때만** 만든다.
    팀원이 손으로 적은 메모를 재실행이 지우면 안 되기 때문이다.
    팀 공용 설명은 docs/RESULTS_GUIDE.md 에 있다 (out/ 는 git 에서 제외되므로).
    """
    path = outdir / "README.md"
    if path.exists():
        return
    m = METHODS[mode]
    b = BATCHES[batch_key]
    txt = [
        "# %s — %s" % (b["dir"], m["title"]),
        "",
        "이 폴더는 `run_analysis.py` 가 만든 **가장 최근 실행 결과**다.",
        "과거 결과는 `archive/` 에서 날짜별로 볼 수 있다.",
        "",
        "## 실행 명령",
        "",
        "```bash",
        "python run_analysis.py %s --batch %s" % (mode, batch_key),
        "```",
        "",
        "## 목적",
        "",
        str(m["desc"]),
        "",
        "## 사용 코드",
        "",
    ]
    txt += ["- `%s`" % s for s in m["scripts"]]
    txt += [
        "",
        "## 대상 배치",
        "",
        "%s (%s)" % (b["dir"], b["desc"]),
        "",
        "---",
        "",
        "분석 방법 간 차이와 결과 파일 설명은 `docs/RESULTS_GUIDE.md` 를 보라.",
        "이 파일은 재실행해도 덮어쓰이지 않으므로 메모를 적어 두어도 된다.",
        "",
    ]
    path.write_text("\n".join(txt), encoding="utf-8")


def run_script(script: str, outdir: Path, batch_key: str, log) -> int:
    """
    스크립트 하나를 돌린다. 출력은 터미널과 log 에 동시에 쓴다.
    반환은 종료코드.
    """
    env = os.environ.copy()
    env["CSI_OUT_DIR"] = str(outdir)
    env["CSI_BATCH"] = str(BATCHES[batch_key]["code"])
    # 한글 출력이 cp949 콘솔에서 UnicodeEncodeError 로 죽는 것을 막는다.
    env["PYTHONIOENCODING"] = "utf-8"

    header = "$ %s %s   (CSI_OUT_DIR=%s, CSI_BATCH=%s)" % (
        Path(sys.executable).name, script, outdir, env["CSI_BATCH"])
    print(header)
    log.write(header + "\n")

    proc = subprocess.Popen(
        [sys.executable, script],
        cwd=str(BASE_DIR), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        print(line)
        log.write(line + "\n")
    proc.stdout.close()
    return proc.wait()


def write_metadata(outdir: Path, batch_key: str, mode: str, status: str,
                   started: datetime, duration: float, argv: List[str],
                   note: Optional[str] = None) -> None:
    m = METHODS[mode]
    skip = {"README.md", "run_metadata.json", "console.log"}
    produced = sorted(p.name for p in outdir.iterdir()
                      if p.is_file() and p.name not in skip) if outdir.exists() else []
    meta = {
        "method": mode,
        "title": m["title"],
        "description": m["desc"],
        "batch": batch_key,
        "batch_directory": BATCHES[batch_key]["dir"],
        "executed_at": started.isoformat(timespec="seconds"),
        "command": "python " + " ".join(argv),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "scripts": list(m["scripts"]),
        "output_directory": str(outdir.relative_to(BASE_DIR)).replace("\\", "/"),
        "duration_sec": round(duration, 2),
        "status": status,
        "generated_files": produced,
        "git_commit": git_commit(),
    }
    if note:
        meta["note"] = note
    # 개인정보와 raw CSI 내용은 넣지 않는다. 참가자 ID 도 넣지 않는다.
    (outdir / "run_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ===========================================================================
# 실행
# ===========================================================================

def do_dry_run(modes: List[str], batch_key: str) -> int:
    b = BATCHES[batch_key]
    sup = BATCH_SUPPORT[batch_key]
    print(RULE)
    print("실행 계획 (--dry-run: 아무것도 만들지 않는다)")
    print(RULE)
    print("배치     : %s — %s" % (batch_key, b["desc"]))
    print("결과 루트: out/%s/" % b["dir"])
    reason = check_batch(batch_key)
    print("사전 점검: %s" % (reason if reason else "이상 없음"))
    print()
    for mode in modes:
        m = METHODS[mode]
        ok = mode in sup["supported"]
        print("[%s] %s" % (m["title"], "" if ok else "-- 미지원 --"))
        if not ok:
            print("  건너뜀: %s" % sup["reasons"].get(mode, "이 배치에서 지원하지 않는다"))
            print()
            continue
        print("  목적      : %s" % m["desc"])
        if len(m["scripts"]) == 1:
            print("  실행 파일 : %s" % m["scripts"][0])
        else:
            print("  실행 파일 :")
            for i, sc in enumerate(m["scripts"], 1):
                print("    %d. %s" % (i, sc))
        print("  결과 폴더 : out/%s/%s/" % (b["dir"], m["dir"]))
        print()
    print("QC 게이트: all 모드에서 QC 실패 시 후속 분석 중단 "
          "(--keep-going 이면 계속)")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("mode", nargs="?")
    ap.add_argument("--batch", default=DEFAULT_BATCH)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("-h", "--help", action="store_true")
    try:
        args, unknown = ap.parse_known_args(argv)
    except SystemExit:
        print(usage())
        return 2
    if unknown:
        print("알 수 없는 인자: %s\n" % " ".join(unknown))
        print(usage())
        return 2
    if args.help or not args.mode:
        print(usage())
        return 0 if args.help else 2
    if args.mode not in METHODS and args.mode != "all":
        print("알 수 없는 mode: %r\n" % args.mode)
        print(usage())
        return 2
    if args.batch not in BATCHES:
        print("알 수 없는 batch: %r\n" % args.batch)
        print(usage())
        return 2

    modes = ALL_ORDER if args.mode == "all" else [args.mode]
    batch_key = args.batch

    if args.dry_run:
        return do_dry_run(modes, batch_key)

    batch_dir = OUT_ROOT / BATCHES[batch_key]["dir"]
    sup = BATCH_SUPPORT[batch_key]
    started_all = time.time()

    print(RULE)
    print("run_analysis.py  mode=%s  batch=%s" % (args.mode, batch_key))
    print(RULE)

    # --- 배치 사전 점검. 여기서 걸리면 어떤 분석도 성공으로 표시하지 않는다. ---
    bad = check_batch(batch_key)
    if bad:
        print("[중단] 배치 점검 실패: %s" % bad)
        print()
        print("어떤 분석도 실행하지 않았다. 성공한 단계는 없다.")
        return 1

    results: List[Dict[str, str]] = []
    qc_failed = False

    for mode in modes:
        m = METHODS[mode]
        outdir = batch_dir / str(m["dir"])

        # --- 배치 지원 여부 ---
        if mode not in sup["supported"]:
            reason = sup["reasons"].get(mode, "이 배치에서 지원하지 않는다")
            print()
            print("[미지원] %s — %s" % (m["title"], reason))
            results.append({"mode": mode, "status": STATUS_INCOMPAT,
                            "reason": reason})
            continue

        # --- QC 게이트 ---
        if qc_failed and not args.keep_going:
            # 실행하지 않았으므로 결과 폴더를 만들지도, 기존 파일을 건드리지도 않는다.
            # 폴더에 이전 성공 결과가 남아 있다면 그 결과와 metadata 는 그대로 둔다.
            # 여기서 run_metadata.json 만 'skipped' 로 덮으면, 실제로는 성공했던 옛
            # 결과가 이번 실행의 실패 산물처럼 보인다. 중단 기록은 아래 실행 요약에만
            # 남긴다.
            print()
            print("[중단] %s — QC 실패로 실행하지 않음 (--keep-going 으로 무시 가능)"
                  % m["title"])
            if (outdir / "run_metadata.json").exists():
                print("       기존 결과 폴더는 그대로 둔다: out/%s/%s/"
                      % (BATCHES[batch_key]["dir"], m["dir"]))
            results.append({"mode": mode, "status": STATUS_SKIP_QC,
                            "reason": "QC 실패로 실행하지 않음"})
            continue

        outdir.mkdir(parents=True, exist_ok=True)
        ensure_readme(outdir, batch_key, mode)

        print()
        print(RULE)
        print("[%s]  -> out/%s/%s/" % (m["title"], BATCHES[batch_key]["dir"], m["dir"]))
        print(RULE)

        started = datetime.now()
        t0 = time.time()
        rc = 0
        failed_script = None
        with (outdir / "console.log").open("w", encoding="utf-8") as log:
            log.write("# %s / %s / %s\n" % (batch_key, mode, started.isoformat(timespec="seconds")))
            for script in m["scripts"]:
                rc = run_script(script, outdir, batch_key, log)
                if rc != 0:
                    failed_script = script
                    msg = "[실패] %s 종료코드 %d" % (script, rc)
                    print(msg)
                    log.write(msg + "\n")
                    break
        dur = time.time() - t0

        status = STATUS_OK if rc == 0 else STATUS_FAIL
        write_metadata(outdir, batch_key, mode, status, started, dur,
                       ["run_analysis.py"] + argv,
                       note=None if rc == 0 else "실패 스크립트: %s (종료코드 %d)"
                            % (failed_script, rc))
        results.append({"mode": mode, "status": status,
                        "reason": "" if rc == 0 else "%s 종료코드 %d" % (failed_script, rc)})
        if mode == "qc" and rc != 0:
            qc_failed = True
            if not args.keep_going:
                print()
                print("** QC 가 실패했다. 잘못된 데이터로 후속 분석을 돌리지 않는다. **")
                print("   무시하고 계속하려면 --keep-going 을 주면 된다.")

    # --- 요약 ---
    label = {STATUS_OK: "성공", STATUS_FAIL: "실패",
             STATUS_INCOMPAT: "미지원", STATUS_SKIP_QC: "중단"}
    print()
    print(RULE)
    print("실행 요약  (batch=%s, %.1f초)" % (batch_key, time.time() - started_all))
    print(RULE)
    for r in results:
        line = "[%s] %s" % (label[r["status"]], METHODS[r["mode"]]["title"])
        if r.get("reason"):
            line += " — %s" % r["reason"]
        print(line)

    ok = [r for r in results if r["status"] == STATUS_OK]
    bad_ = [r for r in results if r["status"] == STATUS_FAIL]
    inc = [r for r in results if r["status"] == STATUS_INCOMPAT]
    skp = [r for r in results if r["status"] == STATUS_SKIP_QC]

    if ok:
        print()
        print("성공한 결과:")
        for r in ok:
            print("- out/%s/%s/" % (BATCHES[batch_key]["dir"], METHODS[r["mode"]]["dir"]))
    if bad_:
        print()
        print("실패한 단계:")
        for r in bad_:
            for sc in METHODS[r["mode"]]["scripts"]:
                print("- %s" % sc)
    if skp:
        print()
        print("QC 실패로 실행하지 않은 단계: %s"
              % ", ".join(METHODS[r["mode"]]["title"] for r in skp))
        print("  이 단계들의 결과 폴더는 건드리지 않았다. 폴더에 파일이 남아 있다면")
        print("  그것은 이전 실행의 결과이며, run_metadata.json 이 그 실행을 가리킨다.")
    if inc:
        print()
        print("이 배치에서 지원하지 않는 분석: %s"
              % ", ".join(METHODS[r["mode"]]["title"] for r in inc))

    return 0 if not bad_ and not skp else 1


if __name__ == "__main__":
    raise SystemExit(main())
