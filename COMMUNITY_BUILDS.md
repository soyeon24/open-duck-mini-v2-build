# 실제로 만든 사람들 — Open Duck Mini v2 커뮤니티 조사

조사일: 2026-09-12. 목적은 부품 주문 전에 **실제로 걷게 만든 사람들의 함정 목록**을 확보하는 것.
업스트림([apirrone/Open_Duck_Mini](https://github.com/apirrone/Open_Duck_Mini), 4.0k★, 포크 283개)
자체는 제외하고, 남이 만든 결과물만 정리한다.

---

## 0. 시간 없으면 이 셋만 봐라

| 순위 | 누구 | 왜 |
|---|---|---|
| 1 | **이건이 (igeoni) + 임규원** — 한국인 2인팀 | 유일하게 찾은 한국 빌드. 걷는 영상 있음. **2026-09-13 본인 답장으로 함정 두 개 확보**: 배터리 구동 시 보행 중 모터 반복 셧다운(미해결), 목 베어링 파손 |
| 2 | **Masato Kobayashi (오사카대 조교수)** | 이 프로젝트로 **논문(EmoLo)을 냈다**. 유학 지원 포트폴리오 관점에서 정확히 같은 경로 |
| 3 | **Frank Fu** | RDK X5 설치 가이드 7.2만 조회 + 블로그에 **실전 함정 4개**가 구체적으로 적혀 있음 (§5) |

---

## 1. 한국 빌드 — 이건이 · 임규원

- 영상: [Open Duck Mini Walking](https://www.youtube.com/watch?v=Gzo25EPKiA8) (2026-07-26, 2,027 조회)
- 채널: [@GeonI-Lee](https://www.youtube.com/@GeonI-Lee) / 블로그 [igeoni.github.io](https://igeoni.github.io)
- 빌드 로그: [igeoni/Open_Duck_Mini_Runtime](https://github.com/igeoni/Open_Duck_Mini_Runtime) (README에 사진 + 영상)
- 역할 분담: 이건이 = 3D 프레임 최적화·배선·기구 조립 / 임규원([gwlim3012](https://github.com/gwlim3012)) = 런타임 환경·IMU·모터 캘리브레이션
- README에 "직접 Sim2Real 정책을 처음부터 학습시킬 계획"이라고 적혀 있음 → 우리와 같은 방향

### 연락처

| | 이건이 (Geoni Lee) | 임규원 (Gyuwon Lim) |
|---|---|---|
| 소속 | 한양대 ERICA 로봇공학과 B.S. / AeiRobot 자율주행팀 SW엔지니어 (2024.09–2026.01) | 서울과기대 인공지능응용학과 |
| 관심사 | Visual/LiDAR SLAM, Robot Perception, **Humanoid State Estimation** | 런타임·임베디드 |
| 이메일 | 공개 안 함 | **gwlim3012@gmail.com** (본인이 GitHub 프로필에 공개) |
| LinkedIn | [건이-이-7228b4309](https://www.linkedin.com/in/%EA%B1%B4%EC%9D%B4-%EC%9D%B4-7228b4309/) | — |
| GitHub | [igeoni](https://github.com/igeoni) | [gwlim3012](https://github.com/gwlim3012) |

- **GitHub Issue 불가** — 해당 저장소는 포크라 Issues 탭이 꺼져 있음 (`has_issues=false`).
- 저장소 README 커밋은 전부 임규원이 썼고 마지막이 2026-03-22.
- 이건이는 2026-09-09에도 GitHub 활동 중 (legged robotics / vision-language navigation 저장소 star) → 살아있고 방향도 계속 legged 쪽.
- **권고: 임규원에게 메일 + 이건이에게 LinkedIn, 둘 다.** 같은 팀이라 중복 무해.

**BOM — 업스트림과 동일** (2026-09-13 임규원 본인 확인)

| 항목 | 업스트림 v2 | 이 팀 |
|---|---|---|
| 서보 | Feetech STS3215 (7.4 V) ×14 | **동일** |
| IMU | BNO055 | BNO085 |
| 컴퓨트 | Pi Zero 2W | 동일 |
| 조이패드 | — | PS5 DualSense |

> ⚠️ **정정.** 저장소 README에 "Dynamixel XL-330 ×14"로 적혀 있으나 **표기 오류**다.
> 문의 결과 실제로는 Feetech STS3215 7.4 V를 썼다고 본인이 확인해 주었다 (2026-09-13).
> 따라서 "0.52 N·m로 걸었다"는 앞선 추정은 성립하지 않고, **우리 forcerange 결론을
> 다시 볼 이유는 없다.** 서보는 공식 BOM대로 STS3215로 간다.

### 답장에서 얻은 것 (2026-09-13, 임규원)

**일정 — 학기 중 병행 가능**

- 2인 진행. 동료(이건이)가 로봇 경험 있는 졸업생이라 기구 조립은 수월한 편이었음.
- 부품 준비·3D 프린팅 시간 제외, **주 1~2일씩 약 2개월**만에 첫 보행.
- 방학을 통째로 비울 규모는 아님. 단 하드웨어가 처음이면 **배선과 sim2real에서 예상보다 시간이 든다.**

**함정 1 — 배터리 구동 시 모터 반복 셧다운 (미해결)**

- 배터리로 구동하면 **보행 중 모터가 반복적으로 꺼짐.**
- 전압 강하/순간 전류 부족을 의심해 더 여유 있는 배터리로 교체 → **증상 그대로.**
- 배선·납땜 접촉 불량도 의심했으나 원인 특정 실패.
- **데모 영상은 외부 전원에 연결한 상태로 촬영하고 프로젝트를 마무리했다.**
  → 즉 공개된 그 보행 영상은 **테더드**다. 배터리 자율 보행은 달성 못 함.

> **우리 가설.** 배터리 용량 문제가 아닐 수 있다. MuJoCo 기본 `forcerange`가 ±3.23 N·m인데
> STS3215 실제 스톨은 1.86이고, 우리 측정에서 **무릎이 1.86 천장에 보행 시간의 14~16% 동안
> 붙어 있었다.** 스톨 근처에서 전류가 급증하므로 14개가 동시에 그 구간에 들어가면 전압이
> 내려앉아 서보가 리셋된다. 이 경우 **배터리를 키워도 안 낫는 게 당연하다** — 용량이 아니라
> 정책이 없는 토크를 요구하는 문제이므로. `fr186`(1.86으로 학습)은 토크를 덜 쓰도록 배웠으니
> 전류 피크도 낮을 가능성이 있다. **`ubai/current_probe.py`로 실기에서 검증할 1순위 항목.**

**함정 2 — 목 베어링이 약하다**

- 보행 중 넘어지면 **쉽게 파손**. 그 팀도 한 번 파손 후 재조립.
- → 넘어짐 실험(일어서기 태스크!)을 할 거면 **여분을 같이 주문해야 한다.**

**추가 연락처**

- 부품 선정·3D 프린팅·조립은 **이건이**가 담당. 그쪽이 훨씬 자세히 안다.
- 이건이 이메일: **egune41@gmail.com** (임규원이 안내)
- 임규원 본인은 "실제 제작 중 문제 생기면 편하게 연락 달라"고 함.

이건이 본인 이력이 legged RL / SLAM (FAST-LIO2, GTSAM, `unitree_rl_lab`, `cassie_rl_walking`,
`legged_state_estimator`)이라 연구 방향도 우리와 겹친다. **연락해 볼 1순위.**

---

## 2. 논문까지 간 사례 — Masato Kobayashi (오사카대)

- GitHub [mertcookimg](https://github.com/mertcookimg) / 개인 페이지 [mertcookimg.github.io](https://mertcookimg.github.io/en/)
- **논문: EmoLo — Emotion-Inspired Expressive Locomotion via Single-Policy RL on Low-Cost Bipedal Robots**
  - [프로젝트 페이지](https://mertcookimg.github.io/emolo/) · [engrXiv (DOI 10.31224/6741)](https://engrxiv.org/preprint/view/6741/version/8768)
  - Open Duck Mini V2에서 Happy/Neutral/Sad 3개 스타일을 **단일 정책**으로 style-conditioned RL.
    locomotion 항은 업스트림 baseline 그대로 두고, head-pitch 자세 + 모션 활동량을 bounded reward와
    command-dependent gate로 변조. 센서 노이즈·지연·외력·모터 한계를 sim2real로 넣고 ONNX로 온보드 추론.
  - → **업스트림 베이스라인 위에 축 하나를 더해서 논문이 된다**는 실증. 우리 standup task와 같은 전략.
- 도구: [Open_Duck_Mini_Viewer](https://github.com/mertcookimg/Open_Duck_Mini_Viewer) (67★) — 브라우저에서 실기 조종
- 유튜브: [きまぐれロボ | Kimagure Robo](https://www.youtube.com/@KimagureRobo) — 재생목록 [Open Duck Mini Robot](https://www.youtube.com/playlist?list=PLaBoRESpdcoitlhH2syP1BevPEo56shl4) 19편

빌드 진행 순서가 영상으로 그대로 남아 있다 (참고할 마일스톤):
`Failed Case` → [`First !`](https://www.youtube.com/watch?v=ckbeoQmlCY4) → `Toddling!` →
[`Walking!`](https://www.youtube.com/watch?v=vUepZ1z1wNY) → [`Outside!`](https://www.youtube.com/watch?v=wUj4VV_aGyI) →
`Gray` → [`NLP2026 데모`](https://www.youtube.com/watch?v=DUxftuiY_3U) → `EmoLo` →
[`60초 빌드 영상`](https://www.youtube.com/watch?v=DajdOeYJZdk) (4.3만 조회)

---

## 3. 유튜브 — 완성한 사람들

| 채널 | 영상 | 조회 | 건진 것 |
|---|---|---|---|
| [Frank Fu](https://www.youtube.com/watch?v=RAM2A3u7v5A) | OpenDuck Mini Part2 설치 가이드 – RDK X5 | **7.2만** | Pi Zero 2W 대신 **D-Robotics RDK X5**. 설치·IMU 캘리브·서보 설정·모션 생성·배포 전 과정 |
| [Back to Engineering (Iulia Feroli)](https://www.youtube.com/watch?v=tt-g_fi-eGU) | 4 Days of Soldering, Assembling, and Electrical Exorcisms (13:57, 챕터 13개) | 1.3만 | **납땜 초보가 겪은 실패가 챕터 단위로 정리됨.** 열삽입너트, 부품 누락, 배선도 혼란, 서보 디버깅 |
| [かばやん](https://www.youtube.com/watch?v=-DNNGdwmtdY) | [BDX Droid] 두 발로 걷는다!! 미니 BDX 만들었다 (18:38) | 8.6천 | 일본어 풀 빌드 |
| [Ibrahim Mohammad](https://www.youtube.com/watch?v=5-WnzBsH1KU) | BDX Droid running on **Jetson Orin Nano Super** | 9.1천 | 온보드 컴퓨트 업그레이드 사례 |
| [Joel Griffin Dodd](https://www.youtube.com/watch?v=3HR9vKmnp2o) | **Vision now working** for the Open Duck mini | 180 | ⭐ 우리 "따라다니기" 목표와 같은 축. [주조 고무 발바닥 테스트](https://www.youtube.com/watch?v=bNbs7PuKjuQ)도 있음 |
| Theo Moore-Calters | Duck Mini Pro Headless | 1.2천 | 액추에이터 10개 · **Pi 5** · 순수 RL · ROS 2 · 머리 없음 |
| [Luis Cruz](https://www.youtube.com/watch?v=jdxxMn5aJY8) | Mujoco ROS2 Control \| Duck Mini | 3.1천 | ONNX 컨트롤러 + ROS2 |
| [Kevin Jeffries](https://www.youtube.com/shorts/yyuLw7NWTbc) | Tnkr 키트 개봉 | 3.4만 | 키트 구성품 확인용 |
| [Google for Developers](https://www.youtube.com/watch?v=pLwB_63yUBY) | Gemma Playground: Robot Duck | 5.4만 | Xavier Plantaz가 Pirrone이 만든 실기 2대로 온디바이스 Gemma 데모 |
| [Cereboto](https://www.youtube.com/watch?v=bRxUWu2gBXc) | DIY Open Duck Mini V2 소개 | 1.7천 | 완제품 판매사 |
| [Association Caliban](https://www.youtube.com/watch?v=uZwMqvsVOQE) | Apérobot 0xA4 - Open Duck Mini V2 | — | 프랑스 로봇 모임 발표 |

---

## 4. GitHub — 실기를 굴리는 포크 (Runtime, 업스트림 `v2` 대비 커밋 수)

| 커밋 | 저장소 | 누구 | 한 일 |
|---:|---|---|---|
| 66 | [zhiquanyeo/Open_Duck_Mini_Runtime](https://github.com/zhiquanyeo/Open_Duck_Mini_Runtime/tree/feature/neopixels-zq) | Zhiquan Yeo (NY, 전 FRC/WPILib 개발자) | **가장 앞서감.** NeoPixel 눈, 원격 게임패드, 텔레메트리 서버, 웹 UI, 자세계, 사운드, 부팅/재시작 스크립트, IMU 트림·거버너·전압 측정 |
| 32 | [ioloizou/Open_Duck_Mini_Runtime](https://github.com/ioloizou/Open_Duck_Mini_Runtime) | Ioannis Loizou (INRIA Nancy 박사과정, 휴머노이드 로코-매니퓰레이션) | IMU 캘리브레이션 데이터, 속도 정책, **감정 표현(intrigued/sad/happy) 오실레이터** |
| 21 | [agentkaerf/Open_Duck_Mini_Runtime](https://github.com/agentkaerf/Open_Duck_Mini_Runtime) | — | **BNO055 → BNO085 포팅** (축 리맵 포함), I2C 400 kHz, 서보 **전류 읽기**, 포즈 캡처/재생, UART 눈, rustypot 1.5.0 대응 |
| 7 | [isamejima/Open_Duck_Mini_Runtime](https://github.com/isamejima/Open_Duck_Mini_Runtime/tree/v2_dynamixel) | Ippei Samejima (JP) | **Dynamixel XM430-W350-R 포팅** |
| 7 | [Yeq6X/Open_Duck_Mini_Runtime](https://github.com/Yeq6X/Open_Duck_Mini_Runtime) | Yeq6X (JP) | **real2sim 관절 뷰어 + 오프셋 튜너 GUI**, IMU 틸트 스트리밍 |
| 5 | [igeoni/Open_Duck_Mini_Runtime](https://github.com/igeoni/Open_Duck_Mini_Runtime) | 이건이 | 한국 빌드 로그 (§1) |
| 4 | [EAOZONE/Open_Duck_Mini_Runtime](https://github.com/EAOZONE/Open_Duck_Mini_Runtime) | — | — |
| 2 | [lookc4/Open_Duck_Mini_Runtime_RDK_X5](https://github.com/lookc4/Open_Duck_Mini_Runtime_RDK_X5) | — | RDK X5 포팅 |
| 1 | [TommyZihao/Open_Duck_Mini_Runtime](https://github.com/TommyZihao/Open_Duck_Mini_Runtime) | 同济子豪兄 (중국 로보틱스 교육자) | 중국어권 확산 |

> `agentkaerf`의 **서보 전류 읽기**는 우리 forcerange 실험을 실기에서 검증할 때 그대로 쓸 수 있다.
> `Yeq6X`의 **real2sim 뷰어**는 sim-to-real 튜닝 단계에서 제일 먼저 붙일 도구.

---

## 5. Frank Fu 블로그가 정리한 실전 함정

[frankfu.blog — Understanding RL through OpenDuck](https://frankfu.blog/openai/understanding-reinforcement-learning-through-openduck/)

1. **`imu_upside_down` 미설정** → 걷는 중 이상 진동, 균형 못 잡음 (조립 방향에 따라 반드시 확인).
2. **캘리브레이션 중 중력이 수평으로 읽힘** → 서보 22번 또는 12번이 수평으로 안 박혔다는 뜻.
   나사 풀고 zeroing 스크립트로 센터 복귀.
3. **Wi-Fi 연결 문제** — 공식 문서와 실제가 다름.
4. **32 GB SD카드가 7 GB로만 보임** → `raspi-config`로 파일시스템 확장.

---

## 6. GitHub — 학습 쪽 포크 (Open_Duck_Playground, 우리와 같은 레이어)

업스트림 187★ / 포크 90개. `main` 대비 앞선 커밋 수:

| 커밋 | 저장소 | 누구 | 한 일 |
|---:|---|---|---|
| 140 | [Rhoban/Sigmaban_playground](https://github.com/Rhoban/Sigmaban_playground) | Rhoban 연구실 (보르도, 공동제작자 Grégoire Passault 소속) | Playground를 **RoboCup 휴머노이드 Sigmaban**에 이식 |
| 82 | [leeygang/Open_Duck_Playground](https://github.com/leeygang/Open_Duck_Playground) | Yonggang Li | `wildrobot_dev` — 자체 로봇용 joystick 환경 이식, ONNX 내보내기 정리 |
| 44 | [lorenhsu1128/Open_Duck_Playground](https://github.com/lorenhsu1128/Open_Duck_Playground) | Loren Hsu (TW) | **발바닥 높이 기반 balance reward 재작성**, 발 들기 보상 + **9억 스텝** 학습, 복합 지형, `--home_pos` |
| 31 | [fabc-why/Open_Duck_Playground](https://github.com/fabc-why/Open_Duck_Playground) | Tetsuya Yamada (JP) | 머리·몸통 충돌판정 추가, 벽/문/공이 있는 환경, roslibpy + OpenCV 연동 |
| 29 | [dfriishecht/Open_Duck_Playground](https://github.com/dfriishecht/Open_Duck_Playground) | Dexter Friis-Hecht (로보틱스/비전 엔지니어) | **standing jog** 태스크, imitation reward 축소 + push 크기 증가, README에 정책 결과 정리 |
| 26 | [aviadarn/Open_Duck_Playground](https://github.com/aviadarn/Open_Duck_Playground) | — | **jump 태스크** (액추에이터 4배), **vast.ai 학습 대시보드 + Prometheus/Grafana/Loki** |
| 15 | [nickoenig37/Open_Duck_Playground_Waddle](https://github.com/nickoenig37/Open_Duck_Playground_Waddle) | Nick Koenig | **백래시 모델링 + 12 V 서보 설정** XML |
| 4 | [ja773/Open_Duck_Playground_IIB_Project](https://github.com/ja773/Open_Duck_Playground_IIB_Project) | Jiyaad Ali | 케임브리지 **IIB 학부 졸업 프로젝트** (로그북 포함) |
| — | [GiulioRomualdi/isaaclab.open_duck_mini](https://github.com/GiulioRomualdi/isaaclab.open_duck_mini) | Giulio Romualdi (IIT) | **IsaacLab 이식** |
| — | [jayjayhust/open-duck-mini_diy_flow](https://github.com/jayjayhust/open-duck-mini_diy_flow) (7★) | 중국 | 手搓 전 과정 기록 + Isaac Sim/Lab 이슈 로그 (미해결 4건) |
| — | [openbiped/Automagerie](https://github.com/openbiped/Automagerie) | — | OpenDuck 시뮬레이션 모델·모션 생성 모음 |

> `nickoenig37`의 **백래시 XML**과 `aviadarn`의 **jump 태스크**가 우리 standup task와 가장 가깝다.
> `lorenhsu1128`은 balance reward를 발바닥 높이로 다시 쓴 사례라 참고 가치 있음.

---

## 7. 조립 중인 사람 (우리와 같은 단계)

- [seanmayer/open-duck-mini-build](https://github.com/seanmayer/open-duck-mini-build) — 2026-09-06 기준
  **전체 58%, 프린트 49/49 완료**, 하드웨어 소싱 75%, 기구 조립 5%, 캘리브레이션 0%.
  BOM CSV·프린트 진척표·사진·빌드 로그를 다 공개. [블로그 글](https://scalablehuman.com/2026/08/14/building-open-duck-mini-v2-from-software-project-to-physical-robot/)
  - 그가 강조한 것: **모터 ID 설정은 조립 전에 해라** (안 그러면 분해해야 함),
    UBEC의 5 V 출력을 STS3215 전원으로 쓰지 말 것.
- 대만 연구실 ([@dingcheng.h Threads](https://www.threads.com/@dingcheng.h/post/DW_VJrrD-9p/)) — 40 cm, STS3215 ×14,
  Pi Zero 2W, BNO055, 전 3D 프린트, BOM 2만 NTD 미만. **안정 보행 + 외란 내성 검증 완료**,
  다음 단계로 OpenClaw multi-agent 연결. 코드: [Laiting20021202/Open_Duck_Mini_Runtime_HRC_version](https://github.com/Laiting20021202/Open_Duck_Mini_Runtime_HRC_version)

---

## 8. 안 만들고 사는 길 (부품 주문 전 참고)

| 판매처 | 형태 | 비고 |
|---|---|---|
| [Tnkr](https://tnkr.ai/open-duck-mini/open-duck-mini-v2/kit) | 조립 키트 (부품+하드웨어+전자) | Back to Engineering, Kevin Jeffries가 쓴 그 키트. [빌드 가이드](https://tnkr.ai/explore/docs/open-duck-mini/open-duck-mini-v2) 공개 |
| [Cereboto](https://cereboto.com/product/open-duck-mini-v2/) | 완제품 | — |
| [Tindie / opmake](https://www.tindie.com/products/wsk/open-duck-mini-v2-integrated-robot-kit-rk3576d/) | 완제품, **RK3576D + NPU** | 온보드 RL 추론 여유 있음 |
| [Pollen Robotics / HF **Microduck**](https://pollen-robotics.com/microduck/) | $399 완제품, 25 cm | 같은 계보의 상용판. 15 모터, 카메라, LiDAR, IMU 2개, RK3566. 2026-08-27 예약 시작, 1만 대 이상 판매. 배송 목표 2026 크리스마스 전 |

> Microduck은 **우리 v2와 다른 로봇**이다 (25 cm vs 42 cm, 15모터 vs 14모터, RK3566 vs Pi Zero 2W).
> 정책이 그대로 안 넘어간다. 이 폴더 이름이 `Microduck`인 것과는 무관.
> 다만 커뮤니티 관심이 그쪽으로 쏠리는 중이라 [joeynyc/awesome-microduck](https://github.com/joeynyc/awesome-microduck) (97★)에
> 시뮬레이터·정책·복제 프로젝트가 모이고 있다.

---

## 9. 우리 위치

- **한국 빌드는 이건이 팀 하나뿐.** 두 번째가 된다.
- 우리처럼 **forcerange를 실측 스톨 토크로 내려서 재학습한 비교 실험을 한 포크는 못 찾았다.**
  `nickoenig37`이 백래시를, `lorenhsu1128`이 balance reward를 건드렸지만 액추에이터 한계 자체를
  변수로 다룬 사례는 없음.
- 업스트림 버그 4개 수정도 중복 없음 — 위 포크 어디에서도 `get_gravity` 센서 주소 버그나
  `head_pos` 미등록 버그를 고친 커밋이 안 보인다.
- **EmoLo가 증명한 경로**: 업스트림 베이스라인 + 축 하나 = 논문.

## 10. 다음 행동

1. 이건이(igeoni)에게 연락 — XL-330 선택 이유와 실제 보행 품질. 부품 주문 전에 답 받으면 제일 이득.
2. `agentkaerf` 포크의 **서보 전류 읽기** 코드 확보 → 실기에서 1.86 N·m 가정 검증용.
3. `Yeq6X`의 **real2sim 뷰어** 북마크 → 조립 직후 첫 도구.
4. Frank Fu 함정 4개를 [`HARDWARE_PREP.md`](HARDWARE_PREP.md)에 체크리스트로 옮기기.
5. 비전(따라다니기)은 Joel Griffin Dodd가 유일한 선례 — 영상 보고 카메라 위치 참고.
