# Open Duck Mini v2 — 출력 체크리스트

출처: `apirrone/Open_Duck_Mini` **v2 브랜치** `print/` 디렉토리 (2026-09-01 기준)  
원문 지침: `print_guide_original.md`

## 소재

| 소재 | 대상 | 인필 |
|---|---|---|
| **PLA** | 34종 / 49개 (나머지 전부) | 15% |
| **TPU** | `foot_bottom_tpu.stl` x2 (발바닥) | 40% |

> 원문: *"All the parts are printed in standard PLA with 15% infill, except for `foot_bottom_tpu.stl`, which is to be printed in TPU at 40% infill."*

레이어 높이 · 서포트 · 온도는 원문에 지정 없음 → 0.20mm Standard + 서포트 자동으로 충분.

## 필요량 (STL 실측 기반 추정)

- PLA **약 990 g** → 1 kg 한 롤로는 빠듯. **1.5 kg 이상 준비**
- TPU **약 34 g** → 최소 단위 한 롤이면 충분 (AMS 말고 **외부 스풀**로)
- 최대 부품 `head.stl` **199.8 × 197.3 mm** → 베드 200×200 이상 필수
- 예상 출력시간: X1C **25~35 h / 6~7판**, H2D **20~28 h / 4판**

## 부품 36종 / **51개**

### 다리 · 발

| ☐ | 파일 | 수량 | 크기 X×Y×Z (mm) | 소재 |
|---|---|---|---|---|
| ☐ | `left_cache.stl` | **x1** | 90.0 × 40.0 × 152.6 | PLA |
| ☐ | `right_cache.stl` | **x1** | 90.0 × 40.0 × 152.6 | PLA |
| ☐ | `foot_top.stl` | **x2** | 103.5 × 34.9 × 46.0 | PLA |
| ☐ | `foot_side.stl` | **x2** | 103.5 × 12.0 × 46.0 | PLA |
| ☐ | `foot_bottom_tpu.stl` | **x2** | 101.9 × 40.7 × 8.0 | **TPU** |
| ☐ | `foot_bottom_pla.stl` | **x2** | 93.0 × 27.7 × 8.0 | PLA |
| ☐ | `left_roll_to_pitch.stl` | **x1** | 46.9 × 86.1 × 30.7 | PLA |
| ☐ | `right_roll_to_pitch.stl` | **x1** | 46.9 × 86.1 × 30.7 | PLA |
| ☐ | `knee_to_ankle_left_sheet.stl` | **x4** | 30.7 × 8.4 × 70.6 | PLA |
| ☐ | `knee_to_ankle_right_sheet.stl` | **x4** | 30.7 × 7.2 × 70.6 | PLA |
| ☐ | `roll_motor_top.stl` | **x2** | 38.0 × 28.7 × 20.0 | PLA |
| ☐ | `leg_spacer.stl` | **x4** | 28.1 × 36.9 × 10.0 | PLA |
| ☐ | `roll_motor_bottom.stl` | **x2** | 32.0 × 30.7 × 31.4 | PLA |

### 몸통 · 배터리

| ☐ | 파일 | 수량 | 크기 X×Y×Z (mm) | 소재 |
|---|---|---|---|---|
| ☐ | `body_middle_bottom.stl` | **x1** | 150.0 × 110.0 × 86.2 | PLA |
| ☐ | `body_middle_top.stl` | **x1** | 150.0 × 110.0 × 56.8 | PLA |
| ☐ | `body_front.stl` | **x1** | 10.0 × 110.0 × 143.0 | PLA |
| ☐ | `trunk_top.stl` | **x1** | 125.5 × 100.7 × 41.0 | PLA |
| ☐ | `body_back.stl` | **x1** | 40.0 × 110.0 × 125.0 | PLA |
| ☐ | `trunk_bottom.stl` | **x1** | 54.4 × 108.0 × 73.7 | PLA |
| ☐ | `battery_pack_lid.stl` | **x1** | 7.5 × 51.4 × 86.4 | PLA |

### 목 · 머리 · 안테나

| ☐ | 파일 | 수량 | 크기 X×Y×Z (mm) | 소재 |
|---|---|---|---|---|
| ☐ | `head.stl` | **x1** | 199.8 × 197.3 × 59.4 | PLA |
| ☐ | `head_bot_sheet.stl` | **x1** | 193.8 × 191.6 × 3.0 | PLA |
| ☐ | `head_yaw_to_roll.stl` | **x2** ⚠️ | 88.0 × 80.0 × 42.2 | PLA |
| ☐ | `head_pitch_to_yaw.stl` | **x2** ⚠️ | 32.0 × 44.9 × 72.0 | PLA |
| ☐ | `head_roll_mount.stl` | **x1** | 40.0 × 64.7 × 15.1 | PLA |
| ☐ | `neck_left_sheet.stl` | **x1** | 30.7 × 7.1 × 58.0 | PLA |
| ☐ | `neck_right_sheet.stl` | **x1** | 30.7 × 5.8 × 58.0 | PLA |
| ☐ | `left_antenna_holder.stl` | **x1** | 26.9 × 20.0 × 41.7 | PLA |
| ☐ | `right_antenna_holder.stl` | **x1** | 26.9 × 20.0 × 41.7 | PLA |
| ☐ | `right_eye.stl` | **x1** | 8.4 × 40.0 × 40.0 | PLA |
| ☐ | `left_eye.stl` | **x1** | 4.3 × 35.0 × 35.0 | PLA |

### 전장 · 장식

| ☐ | 파일 | 수량 | 크기 X×Y×Z (mm) | 소재 |
|---|---|---|---|---|
| ☐ | `speaker_stand.stl` | **x1** | 51.4 × 46.0 × 47.4 | PLA |
| ☐ | `flash_light_module.stl` | **x1** | 28.0 × 31.5 × 41.0 | PLA |
| ☐ | `speaker_interface.stl` | **x1** | 38.6 × 20.8 × 36.9 | PLA |
| ☐ | `bulb.stl` | **x1** | 2.0 × 35.0 × 35.0 | PLA |
| ☐ | `flash_reflector_interface.stl` | **x1** | 12.7 × 23.0 × 23.0 | PLA |

## 슬라이싱 절차 (Bambu Studio)

1. 이 폴더의 STL 36개 전체 임포트
2. **수량 반영** — 아래만 1개가 아님:
   - `foot_bottom_pla.stl` **x2**
   - `foot_bottom_tpu.stl` **x2**
   - `foot_side.stl` **x2**
   - `foot_top.stl` **x2**
   - `knee_to_ankle_left_sheet.stl` **x4**
   - `knee_to_ankle_right_sheet.stl` **x4**
   - `leg_spacer.stl` **x4**
   - `roll_motor_bottom.stl` **x2**
   - `roll_motor_top.stl` **x2**
3. 0.20mm Standard / 인필 15% / 서포트 자동
4. `foot_bottom_tpu.stl` x2 는 **따로 한 판**, TPU 40%, 저속, 외부 스풀
5. Auto arrange → 판 나누기 → 슬라이스하면 정확한 시간이 나옴

## 주의

- ⚠️ **`head_yaw_to_roll` · `head_pitch_to_yaw` 는 여분 1개씩 더 뽑는다 (x2).**
  실제 제작자 제보(2026-09-14, 임규원): *"넘어지면서 목 부분 출력물이 파손됐다. 둘 중 하나였다."*
  **금속 베어링이 아니라 베어링이 물리는 출력물** 쪽이다. 정책 브링업은 수십~수백 번 넘어지는 과정이라
  파손을 전제로 잡는다. 추가 부피는 **+43 cm³**(29.4 + 13.6), FDM 주문에 몇 천 원 수준이고
  **재출력은 2~4주**다. ⛔ SLS 나일론으로 옮기지 말 것 — 인필 없이 꽉 차서 **머리에 ~19g** 이 붙고
  sim 질량(2.107 kg)이 어긋난다 (`HARDWARE_PREP.md` §8)
- **판마다 단색으로.** AMS 색 교체 퍼지 낭비가 본 물량에 맞먹을 수 있음
- **PLA 장시간 출력 시 상단/도어 개방** (밀폐 챔버 열크립)
- **안테나(귀)는 이 목록에 없음** — 시뮬 메시 `antenna.stl`(길이 146mm, 뿌리 Ø1.8mm)은 FDM으로 못 뽑는다. **Ø1.5~2mm 피아노선/카본로드 150mm ×2 + 비즈로 대체**하고 홀더만 출력한다 (`HARDWARE_PREP.md` §4)
- M3 열삽입 인서트 · **626ZZ(6×19×6)** 베어링 ×3 은 별도 구매. ⚠️ **608ZZ 아님** — STL 포켓 실측 Ø19.8~19.9 (`HARDWARE_PREP.md` §4)
- **JLCPCB 외주 시**: 위 부품 중 16종 28개가 FDM 최소 크기 30×30×10mm 에 걸린다 → SLS 나일론으로 분리 발주 (`HARDWARE_PREP.md` §8)