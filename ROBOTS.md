# 어느 오리인가 — 42cm 와 25cm 구분

> **헷갈리면 여기부터 읽는다.**
> 이 폴더의 이름은 `Microduck` 이지만, **실제로 만드는 것은 Open Duck Mini v2(42cm)** 다.

---

## 한 줄 구분

| | 🟢 **만드는 것** | ⚪ **참고만 하는 것** |
|---|---|---|
| 이름 | **Open Duck Mini v2** | Microduck |
| 키 | **42 cm** | 25 cm |
| 만든 곳 | apirrone (개인) | Pollen Robotics (회사) |
| 구하는 법 | **직접 만든다** | $399 완제품 예약판매 |
| 이 폴더에서 | **여기 있는 거의 전부** | `microduck_ref/` 안에만 |

---

## 사양 비교 — 헷갈리는 부품이 여기 다 있다

| | Open Duck Mini v2 (42cm) | Microduck (25cm) |
|---|---|---|
| **두뇌** | **Raspberry Pi Zero 2W** (리눅스, 512MB) | Radxa Zero 3W (RK3566, NPU 0.8 TOPS) |
| **서보** | **Feetech STS3215 ×14** (7.4V, 19kg·cm) | Dynamixel XL330 ×15 |
| 배터리 | 18650 ×2 (2S) | Sony NP-F970 |
| 발 | TPU 발바닥 | 롤러블레이드 바퀴 |
| 소프트웨어 | Python (`Open_Duck_Mini_Runtime`) | Rust 데몬 (`robotd` 외) |
| 학습 | MuJoCo Playground + PPO | mjlab + PPO |
| 출력 부품 | **36종 / 51개** | 43종 (시뮬용 메시) |

---

## 왜 42cm 를 만드나 — 한 표로

| | Open Duck Mini v2 | Microduck |
|---|---|---|
| STL | ✅ 프린트용, 실측 완료 | ⚠️ 있지만 시뮬용 메시 |
| **BOM** | ✅ | ❌ **없음** |
| **조립도** | ✅ | ❌ **없음** |
| **배선도** | ✅ | ❌ **없음** |
| **커스텀 PCB** | ✅ 없음 (필요 없음) | ❌ **HAT 회로도·거버 없음** |
| 서보 값 | ~$210 | ~$360 |

**막는 건 CAD 가 아니라 커스텀 HAT 한 장과 조립 문서다.** 자세한 근거는
[`microduck_ref/README.md`](microduck_ref/README.md).

---

## ⚠️ 자주 섞이는 것 4가지

**1. Pi Zero 2W ≠ Pico 2 W**

| | Raspberry Pi **Zero 2 W** | Raspberry Pi **Pico 2 W** |
|---|---|---|
| 정체 | **리눅스 컴퓨터** | 마이크로컨트롤러 |
| RAM | **512 MB** | 520 KB |
| 우리 로봇 | ✅ **이것** | ❌ 정책 ONNX(884KB)도 안 들어감 |

**2. 폴더 이름이 `Microduck` 인 것** — 처음에 Microduck 을 목표로 잡았던 흔적이다.
경로 참조가 많아 **바꾸지 않는다.** 이름은 무시하고 내용을 본다.

**3. STS3215 ≠ ST3215** — `STS3215` 가 7.4V / 19kg·cm 로 우리 것.
Waveshare 의 `ST3215` 는 12V / 30kg·cm 로 **다른 물건**이다 (`HARDWARE_PREP.md` §3).

**4. `HLS2909M` / `HL-2909-C001` ≠ `STS3215`** — 알리바바 Feite 스토어에
**"오픈덕 마이크로덕 ... 유사 XL330M288T"** 로 팔리는 서보가 있다(₩30,723). `XL330` 은
**Microduck(25cm)** 의 다이나믹셀이므로 **이건 우리 것이 아니다.**

| | HLS2909M | **STS3215 (우리 것)** |
|---|---|---|
| 전압 | **12V** ← 2S→3S 로 전원 계통 전체가 뒤집힌다 | **7.4V** |
| 토크 · 속도 | 9~14kg·cm · 79RPM | **19kg·cm · 45RPM** |
| 시뮬 학습값 | — | `1.86 N·m` · `4.71 rad/s` (1:345) |
| 14개 값 | ₩430,122 | **₩286,403** |

**5. 608ZZ ≠ 626ZZ** — 업스트림 BOM 이 규격을 안 적었고, STL 포켓 실측 결과
**626ZZ(6×19×6)** 다 (`HARDWARE_PREP.md` §4).

---

## 이 폴더에서 어디를 보나

| 파일 | 내용 |
|---|---|
| `README.md` | 전체 지도 (영어) |
| `NEXT_STEPS.md` | 계획·결정 로그·하드웨어 스펙·클러스터 접속 |
| `HARDWARE_PREP.md` | **발주·BOM·조립·비전 계획.** 부품 얘기는 전부 여기 |
| `BOM.md` | **발주 체크리스트 한 장.** 살 때 손에 들고 보는 것. 근거는 `HARDWARE_PREP.md` |
| `SIM_NOTES.md` | 시뮬 상세 — 기립, 머리추종 문제, 토크 측정 |
| `print/` | STL 36종 + `PRINT_CHECKLIST.md` |
| `ubai/` | UBAI SLURM 스크립트 |
| **`microduck_ref/`** | **Microduck(25cm) 조사 기록. 만들지 않는 로봇.** |
