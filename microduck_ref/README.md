# Microduck (25cm) — 조사 기록

> ⚠️ **이 폴더는 "만들지 않는 로봇" 의 기록이다.**
> 실제 제작 대상은 Open Duck Mini v2(42cm) — [`../ROBOTS.md`](../ROBOTS.md) 참조.
> 여기 있는 내용은 **결정의 근거**로만 남긴다.

조사일 2026-09-10. 방법: GitHub API 로 실제 디렉터리·파일을 받아 확인.

---

## 1. 무엇인가

Pollen Robotics 의 25cm / 800g 2족 보행 오리 로봇. **$399 완제품 예약판매.**
RK3566 위에서 Rust 데몬들이 50Hz 제어 루프로 서보 15개를 구동한다.

| 레포 | 내용 | 라이선스 |
|---|---|---|
| [`pollen-robotics/microduck`](https://github.com/pollen-robotics/microduck) | 로봇 소프트웨어 (Rust) | Apache-2.0 |
| [`pollen-robotics/microduck_rl`](https://github.com/pollen-robotics/microduck_rl) | mjlab RL 학습 환경 + **로봇 에셋** | Apache-2.0 |

---

## 2. ✅ 공개돼 있는 것 — 이전 기록 정정

**"Microduck 은 하드웨어가 전부 비공개" 라고 적어뒀던 것은 틀렸다.**

[`microduck_rl/src/mjlab_microduck/robot/microduck/assets/`](https://github.com/pollen-robotics/microduck_rl/tree/develop/src/mjlab_microduck/robot/microduck/assets)
에 **86개 파일 = STL 43 + `.part` 43** 이 있다. (기본 브랜치는 `main` 이 아니라 `develop`)

주요 부품: `xl330` · `np_f970` · `tire`/`rim`/`roller_blade` · `jaw`/`jaw_soft`/`soft_mouth_top` ·
`left/right_shell` · `top/bottom_head_shell` · `m12_lens_holder` · `elec_rpi_robot_hat_pcb` ·
`trunk_base` · `upper_leg_left/right` · `sole_left/right` · `seeed_bearing`

같은 디렉터리 상위에 **MJCF 풀셋**도 있다:
`robot_walk.xml` · `robot_groundcontact_rollers.xml` · `*_backlash.xml` ·
`joints_properties.xml` · 도메인 랜덤화 config 7종.

### 다만 STL 은 프린트용이 아니라 **시뮬용 메시**다

8개 파일이 **바이트 단위로 동일한 1,048,584 B = 정확히 20,970 삼각형**이다
(`bottom_head_shell`, `hip_l`, `left_shell`, `np_f970`, `pcb__raspberry_pi_zero_2_w`,
`right_shell`, `seeed_bearing` ×2). 1 MiB 캡에 걸려 **데시메이트된 것** — 원본 정밀도가 아니다.

### `.part` 는 형상이 아니라 Onshape 메타데이터다

377~432 바이트짜리 JSON. 내용:

```json
{ "documentId": "804927696f06d877f3f1803e",
  "elementId": "d6fcdccc8b25aaa256e7e213",
  "partId": "RiED", "name": "left_shell <1>", "type": "Part" }
```

onshape-to-robot 이 뽑은 것이고, **원본 CAD 는 저 Onshape 문서 안에 있다.**
비로그인으로 열면 "Sign in" 페이지(1.5KB)가 나온다 — **파라메트릭 원본은 잠겨 있다.**

---

## 3. ✅ 컴퓨트 보드는 커스텀이 아니다 — 시판 Radxa Zero 3W

`docs/robot/install-dev.md` 에 그대로 적혀 있다:

> Use the [Armbian imager](https://www.armbian.com/radxa-zero-3/). Pick **Radxa Zero 3**,
> then **Armbian 26.2.1 Minimal**.

ssh 유저도 `radxa@...` 이고, BLE 항목에 **aic8800 라디오**(WiFi 6 / BT 5.4) 얘기가 나온다
→ **Radxa ZERO 3W** 확정.

| Rockchip RK3566 | |
|---|---|
| CPU | Cortex-A55 ×4 (Radxa 보드에서 1.6GHz) |
| GPU | Mali-G52 2EE |
| NPU | INT8 0.8 TOPS |
| VPU | **H.264/H.265 인코딩 1080p@60**, 디코딩 4K@60 |

Radxa ZERO 3W: **65 × 30 mm — 라즈베리파이 Zero 와 같은 폼팩터**, LPDDR4 1~8GB,
온보드 eMMC 옵션, 40핀 GPIO 헤더, MIPI CSI. 대략 $15~50 의 시판품.

> CAD 에 `pcb__raspberry_pi_zero_2_w.stl` 이 있는 이유가 이것 — 실제 부품이 Pi Zero 라서가 아니라
> **Zero 3W 가 Pi Zero 와 외형이 같아서** 자리표시자로 넣어둔 것이다.

왜 Pi Zero 2W 가 아니라 이것을 골랐나: 512MB 로는 WebRTC 스택 + 물체인식(`duck-detect`,
`pet-detect`)을 못 돌린다. 하드웨어 H.264 인코더는 Pi Zero 2W 에도 있다(VideoCore IV, 1080p30).

---

## 4. ❌ 없는 것 — 자작을 막는 것

| 필요한 것 | 상태 |
|---|---|
| **커스텀 HAT PCB** | ❌ `elec_rpi_robot_hat_pcb.stl` 은 **겉모양 메시일 뿐.** schematic·gerber 없음 |
| **BOM** | ❌ 없음 |
| **배선도** | ❌ 없음 |
| **조립도** | ❌ 없음 |
| 서보 15개 (XL330급) | ⚠️ ~$360 — **완제품 $399 와 비슷해진다** |
| 소프트 파츠 | ❌ `jaw_soft` · `soft_mouth_top` · `tire` 재질/경도 스펙 없음 |

`docs/` 는 전부 소프트웨어다 — 데몬, 업데이트, 게임패드 페어링, WebRTC.
**하드웨어 조립 문서가 0개다.**

그리고 README 가 명시한다:

> Everything you need to **run** a Microduck is here. **If you want one, get yours here.**

**이미 산 사람용 레포지 만드는 사람용이 아니다.**

---

## 5. 결론

**만들 수 없다.** 단 이유를 정확히 해둔다:

- ~~"CAD 가 비공개라서"~~ → **틀렸다.** STL 과 MJCF 는 Apache-2.0 으로 공개돼 있다
- ~~"커스텀 메인보드라서"~~ → **틀렸다.** 컴퓨트 보드는 시판 Radxa Zero 3W 다
- ✅ **"커스텀 HAT 한 장의 회로도가 없고, BOM·배선도·조립도가 전혀 없어서"**

HAT 은 Dynamixel TTL 버스 15채널 + 전원 + IMU + ToF + 오디오를 처리하는 보드다.
겉모양 메시만 보고 역설계하는 것은 몇 달짜리 전자공학 프로젝트이고,
Rust 스택 전체가 그 보드를 전제로 돈다.

---

## 6. 그래도 가져다 쓸 수 있는 것

Apache-2.0 이므로 아래는 Open Duck Mini v2 작업에 참고 가능하다:

- **MJCF 백래시 모델링** — `robot_*_backlash.xml`, `add_backlash.py`
- **roller standup 정책** — `docs/roller_standup_policy_summary.md`
- **도메인 랜덤화 config** — `config_mjcf_*.json` 7종
- `joints_properties.xml` 의 서보 클래스 정의 방식

우리 sim2real 튜닝(2~6주)에서 막힐 때 비교 대상으로 볼 것.
