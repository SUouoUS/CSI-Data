# CSI-Data

> **CSILEEP / 도전학기 국제학술대회 연구** — 파일럿 데이터 분석 저장소

ESP32 Wi-Fi CSI로 **`empty` / `still` / `motion` 3상태 분류**와 **bed-exit event detection**(침대 이탈 검출)을 목표로 한다. 현재는 본수집 전 **배치·프로토콜을 확정하는 파일럿 단계**다.

> [!NOTE]
> 호흡 분석은 목표가 아니다. `empty`와 `still`의 분리가 약해서 **추가로 탐색한 특징 후보 중 하나**이며,
> reference sensor로 검증한 적이 없으므로 "호흡 검출"이라고 부르지 않는다.

---

## 측정 환경 — 기술 검증 완료

ESP32 2대(TX–RX), 802.11n **40 MHz**, channel 11, 약 **19 Hz**, 세션당 5분.

`len = 384` byte = 192 complex이며, 실측 널 패턴으로 **LLTF(64) + HT-LTF(128)** 구성임을 확정했다.
분석에는 HT-LTF의 유효 subcarrier(채널을 구성하는 세부 주파수 성분) **114개**(±2 … ±58)만 쓴다.
`first_word = 1`이므로 버퍼 **첫 4바이트는 하드웨어 버그로 무효**이며 마스크에서 제외한다.

<details>
<summary>블록 구성 판정 근거</summary>

`len=384`는 `20 MHz + STBC`(64×3)로도, `40 MHz + non-STBC`(64+128)로도 해석된다.
메타데이터가 `bandwidth=1`, `stbc=0`이고 index 64–191을 128-bin FFT로 풀었을 때
널 패턴이 802.11n HT40 규격과 일치하므로 후자로 확정했다.

```
sc  0        (idx 64)              1.6      → DC null
sc ±2 … ±58  (idx 66–122,134–190)  30–43    → 유효 (114개)
sc ±60… ±64  (idx 124–132)         0.6–1.4  → guard
```
`first_word` 무효 구간은 전 프레임 상수값 `47, -16, 2, 0`으로 나타난다.
</details>

---

## 파일럿 데이터

**배치 B — 현재 `data/`에 있는 8세션.** 분석은 전부 이 배치로 돌린다.

| 파일 | 구성 | 비고 |
|---|---|---|
| `P01_stable` `P02_stable` `P03_stable` | still 5분 | 피험자 3명 |
| `P01_motion` `P02_motion` `P03_motion` | still 2분 → **motion 1분** → still 2분 | 15초 간격 자세변경 4회 |
| `empty1` `empty2` | empty 5분 | |

**배치 A — `stable_5min` / `empty_5min` 2세션. 현재 `data/`에 없다.**
아래 「결과」절의 `A_stable` `A_empty` 수치는 이 파일들이 있던 때의 실행 결과다.
배치 A와 배치 B는 캡처 조건이 달랐다(수신률 17.5–18.1 vs 19.1–19.4 Hz, 프레임 손실
7.7–10.7 % vs 1.1–2.7 %). 파일을 되살리더라도 섞어 쓰면 배치 효과가 상태 효과로
오인될 수 있으므로 `sessions.py`의 `batch`로 층화해서 봐야 한다.

**배치 C — 피험자 4명, 세션당 약 14분 (`empty` 3분 → `still` 5분 → `empty` 3분,
전환 전후 1분 대기는 라벨에서 제외).**

> [!WARNING]
> **배치 C는 아직 실행할 수 없다.** 참가자 ID 매핑이 확정되지 않았고 익명화된 raw
> 파일이 `data/`에 준비되지 않았다. `sessions.py`에는 프로토콜 구간 상수
> (`P14_SEGMENTS`)와 주석 TODO만 두었고, **실행 가능한 `SESSIONS` 목록에는 넣지
> 않았다.** 미확정 세션이 분석 대상으로 선택되면 안 되기 때문이다.
> 배치 C에서는 A안·B안도 지원하지 않는다 — 이유는
> [`docs/RESULTS_GUIDE.md`](docs/RESULTS_GUIDE.md)의 배치별 지원 매트릭스를 보라.

---

## 결과

### 확인된 것

- CSI 구조·유효 mask·metadata를 실측으로 확정했고, 10세션 모두 무결성 검사(`s1`)를 통과했다.
- **`motion` vs `still`** — 세션 내 대조에서 **114 / 114 subcarrier**가 피험자 3명 전원 같은 방향으로 반응했고(우연이면 25 % 기대), 시간 표준편차가 **+36 %** 증가했다. 다만 대조 세션이 3개뿐이라 최소 도달 가능 p가 0.25이므로, **관찰은 일관하나 통계적 확증에는 반복이 더 필요하다.**

### 아직 검증되지 않은 것

**`empty` vs `still`** — **현재 파일럿 데이터와 지금까지 시도한 특징에서는 안정적인 분리를 확인하지 못했다.** 다른 특징이나 ML 분류기를 포함한 최종 분류 가능성까지 기각된 것은 아니다. 진폭 프로파일은 상태와 무관하게 세션 간 유사도 0.994–0.9995였고, 시도한 특징 8개 중 7개가 sep(두 상태의 분리 정도) ≤ 1.33이었다.

**호흡 가능성이 있는 주기 성분** — 일부 세션에서 관찰되었다. `breath_snr_max`(신호가 잡음보다 얼마나 뚜렷한지) 값은 다음과 같다.

| still | | empty | |
|---|---|---|---|
| `P03_stable` | 14.70 | **`empty1`** | **8.24** |
| `P02_stable` | 10.31 | `A_empty` | 5.44 |
| `A_stable` | 9.71 | `empty2` | 4.55 |
| `P01_stable` | 5.88 | | |

재실 4세션 중 3세션이 모든 비재실보다 높았으나 **`P01_stable`(5.88)이 `empty1`(8.24)보다 낮다.**
기존 PC1 기반 특징보다 분리도는 개선되었지만 **참가자·세션 전체에서 안정적인 점유 판별 특징이라고 결론내릴 수 없다.**

- `P02_stable`의 0.273 Hz(약 16.4회/분) 피크는 세션 전후반에 유지되었다. **호흡 후보**다.
- `P03_stable`은 SNR이 가장 높으나 피크가 0.156 → 0.293 Hz로 변동해 **현재 분석에서는** 호흡으로 판단하기 어렵다. P03의 CSI에 호흡 정보가 없다는 뜻은 아니다.

**`lowfreq_ratio`**(0.01–0.1 Hz 전력비) — 완전 분리, p = 0.0286이었으나 **BH(FDR) 보정 후 p = 0.229**로 유의하지 않다. 여러 특징을 탐색한 결과 중 하나이므로 **재현이 필요한 탐색적 결과**로만 둔다.

### 현재 한계

조건별 반복이 1–2회뿐이라 참가자 차이와 세션 변동을 분리할 수 없다. `breath_snr_max`는 분석 창을 280초 → 105초로 바꾸면 피험자 순위가 뒤집힐 만큼 측정 조건에 민감하다. 배치 A·B가 섞여 있고, 호흡 해석을 검증할 reference sensor가 없다.

---

## 분석 원칙

- **독립 표본의 단위는 프레임이 아니라 session**(한 번의 연속된 측정)이다. 정확 순열검정을 쓰고 표본 수로 정해지는 최소 p를 항상 병기한다.
- 임계값은 **`empty` 세션에서만** 정한다. 점유 데이터로 정하면 순환논증이다.
- 효과크기 분모는 세션 간 pooled SD, subcarrier 스크리닝에는 BH(FDR) 보정을 적용한다. 윈도우 중첩 금지.

> [!IMPORTANT]
> **`p < 0.05` 달성 자체를 목표로 하지 않는다.** p값과 sep는 탐색 단계의 참고 지표이며 최종 결론이 아니다.
> 이미 분석한 데이터에서 filter · window length · frequency band를 반복 변경해 좋은 p값을 찾지 않는다.
> 파일럿에서 나온 가설은 **새로운 반복 데이터로 검증한다.**

---

## 다음 단계 — 파일럿 2 (12세션, 60분)

sep나 p값을 높이는 실험이 **아니라** 배치·프로토콜을 확정하기 위한 실험이다. 확인할 질문은 다음과 같다.

- `P02`의 호흡 후보 특징이 **반복 세션에서 재현되는가?**
- `P01`의 낮은 값이 **다시 나타나는가?**
- **참가자 차이와 세션 변동 중 어느 영향이 큰가?**
- `empty` 세션의 특징은 **반복 측정에서도 안정적인가?**
- 새 배치를 쓴다면 기존 배치보다 **`empty`/`still` 구분의 재현성이 실제로 좋아지는가?**

피험자당 3회 + `empty` 3회. 무작위 완전블록으로 순서를 배치해 조건과 시간의 교락을 차단한다.

```
1~4    P02   empty  P01   P03
5~8    empty P03    P02   P01
9~12   P03   P01    empty P02
```

TX/RX를 잇는 경로 근처에 가슴이 오도록 두면 미세 움직임이 CSI에 더 잘 반영될 **가능성**이 있다. 검증되지 않은 **후보 배치**이며 필수 조건이 아니다.

> [!CAUTION]
> 새 배치를 쓴다면 기존 batch와 섞지 않는다. `sessions.py`의 `batch`에 새 라벨을 주어 별도 수집하고,
> 재현성을 비교한 뒤 **배치 1개 · 프로토콜 1개**를 확정해 본수집으로 넘어간다.

---

## 실행

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install numpy scipy matplotlib

python run_analysis.py all         # 전체 분석 (QC → baseline → pilot → A안 → B안)
python run_analysis.py qc          # 데이터 품질 검사
python run_analysis.py baseline    # 기존 정규화 기반 분석 (ML V1 메인)
python run_analysis.py a           # A안: RSSI·신호 크기
python run_analysis.py b           # B안: subcarrier 상관 구조
python run_analysis.py transition  # 상태 전환 시각화 (배치 C 전용)

python run_analysis.py all --batch batch_c   # 배치 지정
python run_analysis.py all --dry-run         # 실행 계획만 출력
```

인자 없이 실행하면 사용법이 나온다. `all` 모드는 QC가 실패하면 후속 분석을
중단한다(`--keep-going`으로 무시 가능).

**결과 구조**

```
out/batch_b_2026-09-01/{qc,baseline,pilot,method_a_rssi,method_b_structure}/
out/batch_c_2026-09-02/{qc,baseline,pilot,transition}/
archive/{2026-09-01, 2026-09-02_before_refactor, old_outputs}/
```

`out/`에는 **가장 최근 실행 결과만** 있고 과거 결과는 `archive/`에 날짜별로 남는다.
각 결과 폴더에 `README.md`·`run_metadata.json`·`console.log`가 함께 생성된다.

> [!IMPORTANT]
> **팀 결정 전, 파일럿용 임시 구성이다.** 다음 세 가지는 아직 합의가 끝나지 않았다.
> ① 혼합 세션의 구간 라벨을 메인 저장소 어디에 저장할지 — `sessions.py`의 `segments`
> 정의는 **파일럿 분석용 임시 매니페스트**이며 메인 csileep의 공식 데이터 계약이 아니다.
> ② `empty`의 participant_id와 LOPO 처리 방법. ③ ESP32 raw 로그를 메인 schema 형식으로
> 변환할지. 이 저장소의 산출물을 메인 저장소의 feature CSV나 LOPO 입력으로 그대로 쓰면 안 된다.

분석 방법별 차이와 결과 파일 설명은 **[`docs/RESULTS_GUIDE.md`](docs/RESULTS_GUIDE.md)** 에 있다.

파싱·전처리·특징은 전부 `csi_core.py`에만 둔다.
새 세션은 `data/`에 넣고 `sessions.py`의 `SESSIONS`에 한 줄 추가하면 된다.

```python
{"name": "b2_01_P02", "scenario": "S-1", "file": "b2_01_P02_stable.txt",
 "subject": "P02", "batch": "B2", "segments": [("still", 0.0, 299.0)]},
```

<details>
<summary>문제 해결용 — 개별 스크립트 직접 실행</summary>

`run_analysis.py`를 거치지 않고 스크립트 하나만 돌려보고 싶을 때 쓴다.
출력 위치와 대상 배치는 환경변수로 지정한다. 지정하지 않으면 결과가 `out/` 루트에
바로 떨어지고 배치는 `B`가 된다.

```bash
# Windows PowerShell
$env:CSI_OUT_DIR = "C:\CSI_Project\out\batch_b_2026-09-01\baseline"
$env:CSI_BATCH   = "B"
$env:PYTHONIOENCODING = "utf-8"     # cp949 콘솔에서 한글 출력이 깨지는 것을 막는다
python s2_session_stats.py
```

| 스크립트 | 역할 | run_analysis.py 모드 |
|---|---|---|
| `s1_check_format.py` | 형식·무결성 검사 (FAIL이면 종료코드 1) | `qc` |
| `s2_session_stats.py` | 세션 단위 비교 | `baseline` |
| `s3_subcarrier.py` | subcarrier 스크리닝 | `baseline` |
| `s5_pilot_check.py` | 파일럿 진단 | `pilot` |
| `s5_rssi.py` | A안 RSSI·신호 크기 | `a` |
| `s3b_structure.py` | B안 상관 구조 | `b` |
| `s6_visualize.py` | 상태 전환 시각화 | `transition` |

</details>

---

참가자 측정 데이터를 포함하므로 **비공개 저장소로 유지한다.** 피험자는 `P01`–`P03`으로 익명화되어 있다.
