# CSI 분석 결과 안내

이 문서는 `C:\CSI_Project` 파일럿 분석 저장소의 실행 방법과 결과 폴더를 설명한다.
처음 보는 팀원이 이 문서만 읽고 결과를 찾을 수 있도록 쓴 것이다.

> [!IMPORTANT]
> **팀 결정 전, 파일럿용 임시 구성이다.**
> 아래 세 가지는 아직 팀 합의가 끝나지 않았고, 이 저장소의 구성은 그 결정을
> 앞당기거나 대신하지 않는다.
>
> 1. **혼합 세션의 구간 라벨을 메인 저장소 어디에 저장할지** —
>    `sessions.py` 의 `segments` 정의는 **파일럿 분석용 임시 매니페스트**이며
>    메인 csileep 저장소의 공식 데이터 계약이 아니다.
> 2. **`empty` 의 participant_id 와 LOPO 처리 방법** — 현재 비점유 세션은
>    `subject = "-"` 로 두고 참가자 단위 집계에서 세션 이름을 단위로 쓴다.
>    이것은 파일럿 통계를 돌리기 위한 임시 규칙이지 공식 정책이 아니다.
> 3. **ESP32 raw 로그를 메인 schema 형식으로 변환할지** — 이 저장소는 raw 로그를
>    그대로 읽는다. 변환 코드는 넣지 않았다.
>
> 따라서 이 저장소의 산출물을 메인 저장소의 feature CSV 나 LOPO 입력으로 그대로
> 쓰면 안 된다.

---

## 실행

```bash
python run_analysis.py all                 # 전체 (기본 batch_b)
python run_analysis.py qc                  # 데이터 품질 검사
python run_analysis.py baseline            # 기존 정규화 기반 분석
python run_analysis.py pilot               # 추가 파일럿 진단
python run_analysis.py a                   # A안 RSSI·신호 크기
python run_analysis.py b                   # B안 상관 구조
python run_analysis.py transition          # 상태 전환 시각화

python run_analysis.py all --batch batch_c # 배치 지정
python run_analysis.py all --dry-run       # 실행 계획만 (아무것도 만들지 않음)
python run_analysis.py all --keep-going    # QC 가 실패해도 계속
```

인자 없이 실행하면 사용법이 나온다.

**QC 게이트**: `all` 모드에서 QC(`s1_check_format.py`)가 실패하면 그 배치의 후속
분석을 중단한다. 형식이 깨진 데이터로 돌린 결과가 정상 결과처럼 남는 것이 더
위험하기 때문이다. 무시하려면 `--keep-going` 을 명시해야 한다.

---

## 폴더 구조

```
out/                      항상 가장 최근 실행 결과만 들어 있다
  batch_b_2026-09-01/
    qc/  baseline/  pilot/  method_a_rssi/  method_b_structure/
  batch_c_2026-09-02/
    qc/  baseline/  pilot/  transition/

archive/                  과거 결과 보관 (자동으로 지우지 않는다)
  2026-09-01/                    9월 1일 분석 결과
  2026-09-02_before_refactor/    폴더 구조 개편 직전의 최신 결과
  old_outputs/                   초기 탐색 스크립트 결과
```

각 결과 폴더에는 다음 3개가 항상 있다.

| 파일 | 내용 |
|---|---|
| `README.md` | 그 폴더가 무엇인지. **재실행해도 덮어쓰지 않으므로 메모를 적어도 된다** |
| `run_metadata.json` | 실행 시각·Python·스크립트·소요 시간·상태·생성 파일·git commit |
| `console.log` | 그 실행의 터미널 출력 전체 (오류 포함) |

`out/` 은 `.gitignore` 에 있어 GitHub 에 올라가지 않는다. 그래서 팀 공용 설명은
`out/` 안이 아니라 **이 문서**에 둔다.

---

## 배치

| 배치 | 구성 | 프로토콜 |
|---|---|---|
| `batch_b` (2026-09-01) | 피험자 3명 + 비점유 2회, 세션당 5분 | `still` 5분 / `still`→`motion`→`still` / `empty` 5분 |
| `batch_c` (2026-09-02) | 피험자 4명, 세션당 약 14분 | `empty` 3분 → `still` 5분 → `empty` 3분 (전환 전후 1분 대기는 라벨에서 제외) |

> [!WARNING]
> **배치 C 는 아직 실행할 수 없다.** 참가자 ID 매핑이 확정되지 않았고 익명화된
> raw 파일이 `data/` 에 준비되지 않았다. 그래서 `sessions.py` 의 실행 가능한
> `SESSIONS` 목록에 배치 C 세션을 넣지 않았고, 프로토콜 구간 상수
> (`P14_SEGMENTS`)와 주석 TODO 만 두었다. 두 조건이 충족되면 주석을 풀면 된다.

---

## 분석 방법 — 무엇이 어떻게 다른가

### `qc` — 데이터 품질 검사 (`s1_check_format.py`)

분석 전에 raw CSI 가 정상인지 본다. CSI 길이, 송신 MAC, `first_word`, RSSI,
noise floor, 수신률, 패킷 손실, 시간축 단조성을 확인한다.
**결과 파일을 만들지 않는다.** 판정은 `console.log` 에 남는다.

### `baseline` — 기존 정규화 기반 분석 (`s2_session_stats.py` → `s3_subcarrier.py`)

**현재 ML V1 의 기본 분석 방법이다.** 정규화된 CSI 에서 시간 변화와 subcarrier
패턴을 본다. 주요 특징 후보는 `amp_std_time`, `mad_diff`, `mad_diff_p95`,
`pc1_var_ratio` 다.

| 파일 | 내용 |
|---|---|
| `s2_session_features.csv` | 세션·상태 구간별 특징 대표값. **파일럿 비교용이며 최종 ML 학습용 10초 window CSV 가 아니다** |
| `s2_window_boxplot.png` | 세션 안의 10초 window 특징 분포 |
| `s2_session_points.png` | 세션 단위 특징 대표값 비교 (점 1개 = 세션 단위 관측값 1개) |
| `s3_class_mean.png` | 상태별 subcarrier 평균 |
| `s3_heatmap.png` | 세션별 subcarrier 진폭 패턴 |
| `s3_profiles.png` | 세션·구간별 subcarrier 프로파일 |
| `s3_subcarrier_stats.csv` | `still` 과 `empty` 의 subcarrier 별 비교 결과 |

window 개수가 299초 구간에서 28개, 120초에서 11개, 60초에서 5개, 119초에서
10개인 것은 **정상이다.** 구간 양끝에서 `SEGMENT_GUARD_SEC = 5.0` 초씩 잘라내기
때문이며 off-by-one 오류가 아니다.

### `pilot` — 추가 파일럿 진단 (`s5_pilot_check.py`)

호흡 가능성이 있는 주기 성분과 대체 특징을 탐색하고 본수집 전 확인 사항을
점검한다. **최종 ML 파이프라인이 아니다.** 호흡 관련 특징은 현재 ML V1 에서
제외한다. 결과 파일을 만들지 않으며 판정은 `console.log` 에 남는다.

### `a` — A안: RSSI·신호 크기 (`s5_rssi.py`)

기존 frame normalization 에서 사라진 전체 신호 크기가 `still`/`empty` 구분에
도움이 되는지 본다. RSSI, noise floor, SNR, 그리고 **RSSI 보정 진폭**
(RSSI-scaled amplitude) 을 분석한다.

> **명칭 주의**: `RSSI 보정 진폭` 은 장비 보정을 거친 물리적 절대 진폭이 아니라
> 정규화된 모양에 RSSI 로 스케일만 되살린 값이다.

> **QC 지표 주의**: 수신률·프레임 간격·파싱 드롭률·ID 손실률(A-2)은 ML 특징이
> 아니라 **데이터 수집 품질 지표**다. 표와 그래프에는 나오지만
> **최종 점유 판정에서는 제외한다.** A-1(RSSI 계열)은 RSSI 해상도가 서로 다른 값
> 5종류 이상일 때만 판정에 포함하고, 3~4종류면 참고 표시만, 2종류 이하면 제외한다.

A안은 기존 방법을 대체하는 분석이 아니라 **추가 후보 분석**이다.
결과: `A_report.md`, `s5_rssi_<batch>.png`, `s5_abs_profile_<batch>.png`.

### `b` — B안: subcarrier 상관 구조 (`s3b_structure.py`)

114개 subcarrier 가 서로 함께 움직이는 구조가 `still`/`empty` 에서 달라지는지
본다. 상관행렬, 제1고유값 비율, 상위 5개 고유값 비율, 유효 랭크
(= exp(고유값 분포의 섀넌 엔트로피)), 상관계수 평균·표준편차를 본다.

B안은 현재 최종 ML 특징이 **아니라 탐색용 분석**이다. 기존 `pc1_var_ratio` 와
일부 정보가 겹칠 수 있다.
결과: `B_report.md`, `s3b_corr_diff_<batch>.png`, `s3b_eigen_<batch>.png`.

### `transition` — 상태 전환 시각화 (`s6_visualize.py`)

`empty → still → empty` 처럼 한 세션 안에서 상태가 바뀌는 데이터를 눈으로
확인한다. **검정은 하지 않는다.** 라벨 경계에서 신호가 실제로 바뀌는지(=
`t_offset` 이 맞는지), 같은 라벨 구간이 안정적인지, 세션 내 `empty` 두 구간이
서로 닮았는지를 본다.
결과: `s6_timeline_*.png`, `s6_heatmap_*.png`, `s6_paired_states.png`,
`s6_psd.png`, `s6_windows.csv`.

`s6` 는 A안도 B안도 아니다. A안은 RSSI·진폭 크기, B안은 상관 구조,
`s6` 는 시간 전환 시각화다.

---

## 배치별 지원 매트릭스

| 분석 | batch_b | batch_c | 미지원 사유 |
|---|---|---|---|
| `qc` | O | O | |
| `baseline` | O | O | |
| `pilot` | O | O | |
| `a` (A안) | O | **미지원** | 배치 C 는 세션 내 대응 설계인데 A안은 비대응 순열검정만 쓴다 |
| `b` (B안) | O | **미지원** | 위와 같고, 추가로 B안은 배치 B 프로토콜(`events` 기반 motion 블록, motion 대 empty 양성 대조)을 전제한다. 배치 C 에는 motion 구간이 없다 |
| `transition` | 미지원 | O | 배치 B 에는 `empty→still→empty` 전환 프로토콜이 없다 |

**배치 C 에서 A안·B안을 막아 둔 이유를 조금 더 설명한다.** 배치 C 는 한 세션
안에 `empty` 와 `still` 이 모두 있어 같은 참가자 안에서 비교하는 **대응(paired)
설계**다. 그런데 A안·B안은 `exact_perm_test_unpaired` 만 쓴다. 참가자 단위로
합치면 `still` 쪽 단위와 `empty` 쪽 단위가 **같은 사람**이 되어, 대응 자료를
독립 표본처럼 검정하게 된다. 도달 가능한 하한 p 도 4 대 4 비대응이 0.0286,
올바른 대응 검정이 0.125 로 어긋나서 **p 를 실제보다 낙관적으로 보고**하게 된다.

대응 검정으로 바꾸는 것은 분석 방법 변경이므로 별도 검토와 승인이 필요하다.
그때까지 `run_analysis.py` 는 억지로 실행하지 않고 `incompatible` 로 표시한다.

---

## 결과를 읽을 때

- **독립 표본의 단위는 프레임이 아니라 세션 구간이다.** 한 세션의 수천 프레임은
  자기상관 때문에 독립 표본 1개다.
- **같은 참가자의 여러 구간을 독립 표본처럼 세지 않는다.** 보고서의
  `참가자 단위 재검정` 줄을 세션 단위 결과와 반드시 함께 읽어야 한다.
- **`sep` 가 2 미만이면 p 와 무관하게 주장 근거가 없다.** `sep` 는
  클래스 간 차이를 클래스 내 세션 표준편차로 나눈 값이다.
- **실제 p 가 `p_min` 과 같다는 것은 "가장 극단적인 배치"라는 뜻이지 "유의하다"는
  뜻이 아니다.** 표본 수로 정해지는 하한이므로 항상 함께 봐야 한다.
- **subcarrier 전수 검정의 판정 기준은 FDR 보정 후 `q` 다.** 114개를 훑으면
  완전한 잡음에서도 몇 개는 `p<0.05` 가 된다. subcarrier 끼리 상관이 크므로
  "114개 중 N개가 갈렸다"를 독립 시행 N건으로 읽으면 안 된다.
- **검출 임계값은 비점유(`empty`) 세션에서만 정한다.** 점유 데이터를 보고 정하면
  순환논증이다.
- 파일럿 결과를 확정적으로 쓰지 않는다. 한 번의 실행에서 나온 결과는 재현성이
  확인된 것으로 간주하지 않는다.

---

## 개인정보

참가자는 익명 ID 로만 기록한다. 결과 파일명, 그래프 제목·범례, CSV, Markdown
보고서, `console.log`, `run_metadata.json` 어디에도 실명이나 이니셜이 남으면
안 된다. `run_metadata.json` 에는 참가자 ID 자체를 넣지 않는다.

`data/`, `out/`, `archive/`, `incoming_batch_c/` 는 `.gitignore` 에 있어
GitHub 에 올라가지 않는다.
