# Open Duck Mini v2 — 진행 메모

## 결정 사항
- **목표**: Microduck 대신 **Open Duck Mini v2**를 직접 제작 (Microduck은 HW 비공개 = 자작 불가, $399 완제품 예약판매만 가능)
- 두 프로젝트 모두 방법론 동일: MuJoCo에서 PPO 학습 → ONNX 내보내기 → 실기 배포 (sim2real)

## 내 환경 제약
- 로컬 GPU 없음 (Intel Iris Xe)
- **학습은 UBAI(서울시립대 도시과학빅데이터·AI연구원) 슈퍼컴 사용** ← 접속 확인 완료 (2026-08-26)
  - Gate1 `<GATE1_IP>` / Gate2 `<GATE2_IP>`, 계정 `<YOUR_ID>`, 키 `<KEY_DIR>\ubai-<YOUR_ID>.pem`
  - ⚠️ **사설 IP라 학교 네트워크에서만 접속됨. VPN 미제공** → 학교에서 `sbatch` 던져놓고 집에서 방치하는 방식
  - 파티션: gpu1(RTX3090x4, 14노드) / gpu2·gpu6(A10) / gpu3(A6000 Ada) / gpu4·gpu5(A6000) / cpu1·cpu2
  - 게이트 노드 인터넷 O, GPU X, 시스템 python 3.6.8, 홈 `/home1/<YOUR_ID>` (GPFS 244T 여유)
  - ⚠️ 실적자료에 **사사 표기 의무**:
    "본 논문은 서울시립대학교 도시과학빅데이터·AI연구원의 슈퍼컴퓨팅 자원을 지원 받아 수행되었습니다."
  - **실행 스크립트 일체: `ubai/` 폴더. 절차는 `ubai/README.md` 참고**
    - `00_recon.sh` 정찰 / `10_setup.sh` 환경설치 / `20_verify_gpu.sh` GPU확인
    - `30_train.sbatch` 학습(walltime 초과 시 체크포인트 자동 이어받기 체인) / `40_status.sh` 상태확인
  - 아직 미확인: **파티션 최대 walltime**, 컴퓨트 노드 NVIDIA 드라이버 버전(cuda12 wheel은 ≥525 필요), 컴퓨트 노드 인터넷

### 역할 분담
| 어디서 | 무엇을 |
|---|---|
| UBAI 클러스터 | PPO 학습 (`runner.py`). GPU 필요, 헤드리스 OK |
| 내 노트북 | 정책 시각화/추론 (`mujoco_infer.py`). MuJoCo 뷰어라 GUI 필요, CPU로 충분 |

### 학습 스택 결정 (2026-08-31)
- **conda 대신 uv** 사용 — 레포 pyproject가 uv 기준이고, conda 26.5의 ToS 이슈를 피함
- **`module load cuda` 하지 않음** — `jax[cuda12]`가 nvidia-*-cu12 wheel로 자기 CUDA를 들고 오므로
  module의 cuda/11.6.2를 로드하면 LD_LIBRARY_PATH가 충돌해서 오히려 깨짐. 중요한 건 드라이버 버전뿐
- ~~`runner.py`에 `--restore_checkpoint_path` 있음 → walltime 체인 재개 가능~~
  ⚠️ **2026-09-04 정정: 실제로 걸어보니 깨진다.** `policy_params_fn` 이 `ocp.PyTreeCheckpointer().save()`
  로 직접 저장하는데 brax `checkpoint.load()` 는 자기 포맷을 기대해서 못 읽는다
  (`ValueError: Expected list, got RestoreArgs(...)`). 300M 걷기가 한 잡에 끝나서 체인이
  한 번도 실제로 안 걸려 여태 안 드러났다. **48시간 넘는 학습 전에 먼저 고칠 것.** 상세는 `SIM_NOTES.md` 발견 4
- 체크포인트 저장 시 **ONNX도 자동 export**됨 (`common/runner.py:policy_params_fn`) → 중간에 끊겨도 정책 회수 가능
- ⚠️ brax 체크포인트는 파라미터만 저장, Adam 상태는 미저장 → 세그먼트를 잘게 쪼갤수록 손해
- ⚠️ **`playground==0.0.5` 로 핀 고정 필수.** pyproject 가 `playground>=0.0.3` 이라 상한이 없어서
  uv 가 최신 0.2.0 을 끌어오는데, 0.1.0 부터 `mujoco_playground._src.collision` 이 사라져
  `joystick.py` import 가 깨진다. 레포 마지막 커밋(2025-08-05) 시점 버전은 0.0.5 (2025-06-23)
- **잡 1개당 최대 48시간** (`MaxWall=2-00:00:00`, 계정 association 한도). 파티션 MaxTime 은 UNLIMITED 지만
  association 이 먼저 걸린다. `--time` 을 넘겨 잡으면 `PENDING (AssocMaxWallDurationPerJobLimit)` 로 영영 안 돎
- 계정 한도: 동시 실행 10잡 / 제출 20잡 / GPU 12장. 드라이버 595.45.04, 계산 노드 인터넷 O

- WSL2 Ubuntu 설치됨 (jax[cuda12]가 Linux 전용이라 필요시 사용)
- Python 3.12.2 / git 있음 / uv 없음
- 납땜 가능 ✅ / 3D 프린터 접근 가능 ✅ — **Bambu Lab H2D 또는 X1 Carbon** (2026-09-01 확인)
  - ~~TPU 출력 가능한지 확인 필요~~ → **해결**. 둘 다 다이렉트 드라이브라 TPU 가능.
    단 **AMS 말고 외부 스풀 홀더**로 저속 출력할 것 (발바닥 2개, 합 34g뿐)
  - 베드도 문제 없음: 최대 부품 head 199.8x197.3mm < X1C 256x256 / H2D 325x320

## 하드웨어 스펙 (Open Duck Mini v2)
- 42cm, 3D 프린팅 부품 35개 이상 (PLA + TPU)
- Feetech **STS3215** 시리얼 버스 서보 **14개** (다리당 5개 + 목/머리 4개) + SG90 마이크로서보 2개 (귀)
- Raspberry Pi Zero 2W / BNO055 9-DOF IMU / 발 접촉 스위치 4개
- 전원: 18650 2S + BMS + 5V UBEC + XT30
- M3 열삽입 인서트, 608ZZ 베어링
- **비용: 약 $350~430 (55만~65만원)** — 완제품 Microduck보다 싸지 않음. 이유는 가격이 아니라 학습+개조 자유

## 크기 검토 결과 (2026-08-29 결정)
- 희망 크기는 20x20x30cm(책상용)였으나 **42cm 원본 그대로 제작하기로 결정**
- 42cm는 "다리 편" 수치. BDX 계열은 무릎 굽히고 걸어서 실제 보행 높이는 더 낮음 → 책상에 둘 만함
- **오픈소스 DIY 2족은 전부 40cm대**(Zeroth-01 ~40cm, Bimo 45cm). 이유: 서보가 안 줄어듦.
  STS3215가 45.2x24.7x35mm인데 다리당 5개 직렬 → 다리 길이 하한이 물리적으로 고정됨
- Microduck만 25cm인 건 커스텀 액추에이터를 자체 개발했기 때문. 그래서 HW를 비공개한 것

### 보류한 대안: 0.7배 축소 설계 (나중에 2차 프로젝트로)
계산상으로는 성립함:
- 42cm x 0.7 = 29.4cm
- 서보 STS3215 -> **STS3032** (32x12x27.5mm, 4.5kg.cm @6V) 교체
- 필요 토크는 크기의 4제곱에 비례: 0.7^4 = 0.24 -> 19kg.cm x 0.24 = 약 4.6kg.cm 필요 ≒ STS3032의 4.5kg.cm
막는 요소:
- 전자부품은 안 줄어듦 (Pi Zero 2W 65x30mm, 18650 65x18mm) -> 몸통 재배치 필요
- 전압 다름 (STS3215 7.4V vs STS3032 6V) -> 전원 설계 변경
- Onshape CAD 전면 재설계 + URDF/MJCF 재생성 + RL 정책 재학습
- 서보 특성(백래시/마찰/지연)이 달라져 검증된 보상 함수가 안 통함 -> sim2real 맨땅에 헤딩
- 추가 3~6개월. **원본으로 한 번 완주한 뒤에 도전할 것**

## 진행 순서 (부품 지르기 전에 1번부터)
1. **시뮬레이션에서 먼저 걷게 만들기** (비용 0원) ← 지금 여기
2. 성공 확인 후 → 부품 주문 + 3D 프린팅
   - STL 36개(51개 출력) 이미 받아둠: `print/` — 체크리스트는 `print/PRINT_CHECKLIST.md`
   - PLA 15% 인필(약 990g) + 발바닥만 TPU 40%(약 34g). 최대 부품 head 199.8x197.3mm
   - 프린터 H2D/X1C 확정 → 베드 문제 없음. 예상 X1C 25~35h/6~7판, H2D 20~28h/4판
3. 조립 → 배선 → 서보 ID 설정
4. sim2real 튜닝 ← **진짜 벽. 2~6주 예상.** 저가 서보의 백래시/지연/마찰이 원인

예상 총 기간: 2~4개월

## 지금까지 한 것
- `Open_Duck_Playground/` 클론 완료 (RL 학습 환경, Apache-2.0)

## 다음에 할 것 (2026-08-31 갱신)
**학교 네트워크에서만 UBAI 접속 가능** → 학교 갈 때 아래를 한 번에 처리하고 오는 것이 목표.
절차 전문은 `ubai/README.md`. 요약:
1. 스크립트 업로드 → `bash ~/ubai/00_recon.sh` (정찰, 결과 저장해서 가져오기)
2. `bash ~/ubai/10_setup.sh` (uv + jax[cuda12] 설치, 15~25분)
3. `bash ~/ubai/20_verify_gpu.sh` (GPU 인식 확인)
4. 스모크 테스트 5M step → **step/sec 측정 후 300M 소요 시간 역산**
5. `sbatch --export=ALL,CHAIN=5 ~/ubai/30_train.sbatch` 던져놓고 귀가
6. 다음 방문 때 `bash ~/ubai/40_status.sh` → ONNX 회수 → 로컬 `mujoco_infer`로 재생

읽어볼 핵심 파일:
- `playground/open_duck_mini_v2/joystick.py` — 보상 함수 선택 (USE_IMITATION_REWARD 플래그)
- `playground/common/rewards.py` — 보상 정의
- `playground/common/randomize.py` — 도메인 랜덤화
- `playground/open_duck_mini_v2/constants.py` — 조인트/센서 정의

## 링크
- 메인 레포: https://github.com/apirrone/Open_Duck_Mini (v2 브랜치)
- 학습: https://github.com/apirrone/Open_Duck_Playground
- 실기 런타임: https://github.com/apirrone/Open_Duck_Mini_Runtime
- 레퍼런스 모션 생성: https://github.com/apirrone/Open_Duck_reference_motion_generator
- BAM 액추에이터 식별 (Rhoban) — sim2real 핵심
- 개요 정리(한국어): https://robotics.growbotics.ai/ko/projects/hardware/open-duck-mini-v2
- Discord: https://discord.gg/UtJZsgfQGe

## 시뮬레이션 환경 (완료, 2026-08-30)
- `.venv/` — CPU 전용 경량 환경 **1.3GB** (레포 기본 `uv sync`는 jax[cuda12]+tensorflow로 약 6GB. GPU 없으니 제외)
  - 설치: numpy mujoco onnxruntime jax jaxlib etils[epath] ml_collections mujoco-mjx playground
  - ※ PyPI 패키지명이 `mujoco_playground`가 아니라 **`playground`**임 (레포 pyproject와 동일)
- `BEST_WALK_ONNX_2.onnx` — 커뮤니티 검증 보행 정책 (884KB). 이거만 있으면 GPU 없이 실기 보행 가능
- 실행 (PYTHONPATH 필요, `-m`으로 실행해야 함):
  ```
  cd C:\school\2026_2\Microduck\Open_Duck_Playground
  set PYTHONPATH=C:\school\2026_2\Microduck\Open_Duck_Playground
  ..\.venv\Scripts\python.exe -m playground.open_duck_mini_v2.mujoco_infer -o ..\BEST_WALK_ONNX_2.onnx
  ```

### 조작키 (코드 주석은 AZERTY 기준이라 실제 키와 다름. 아래는 QWERTY 실제 키)
| 키 | 동작 |
|---|---|
| ↑ / ↓ | 전진 / 후진 (±0.15 m/s) |
| ← / → | 좌 / 우 게걸음 (±0.2 m/s) |
| **Q** | 좌회전 (주석엔 "a") |
| **E** | 우회전 |
| **P** / **;** | 보행 주파수 ±0.1 (주석엔 "m") |
| **H** | 머리 조종 모드 토글 |
| **R** | **완전 리셋 (직접 추가한 기능)** |
| Z 등 미매핑 키 | 속도 0으로 정지 |
- 키는 누르고 있는 게 아니라 **한 번 누르면 계속 유지**됨. 창 포커스 필요.
- 넘어뜨리기: 몸통 더블클릭 → Ctrl+드래그 (좌/우 버튼이 각각 토크/힘)

### 직접 수정한 것
- `mujoco_infer.py`에 `full_reset()` + **R키** 추가 (원본은 `mujoco_infer.py.orig`로 백업)
  - 이유: MuJoCo 기본 Backspace는 `mj_resetData`로 **qpos0(다리 편 자세)**로 되돌리는데,
    정책은 **"home" 키프레임(무릎 굽힌 자세)** 기준으로 학습됨 → Backspace 누르면 즉시 넘어짐
  - R키는 home 키프레임 복원 + 정책 내부 상태(last_action, imitation_phase, 필터) 전부 초기화

### 알게 된 것 (학습 과제 후보)
- **뒤집히면 스스로 못 일어남.** `standing.py`의 `fall_termination = get_gravity(data)[-1] < 0.0`
  → 학습 중 넘어지는 순간 에피소드가 종료돼서, 정책이 뒤집힌 상태를 한 번도 겪어본 적 없음
  - Microduck은 "스스로 일어나기"가 별도 정책으로 있음. Open Duck Mini엔 없음 → **직접 만들 첫 과제로 적합**
  - 방법: fall_termination 제거 → 뒤집힌 자세로 초기화 → 몸통 높이/중력벡터 기반 보상 설계
- **속도가 느린 건 설정이 아니라 학습 결과.** COMMANDS_RANGE_X=[-0.15,0.15]를 키워도
  정책이 그 범위 밖 명령을 본 적 없어서 넘어짐 → 빠른 보행은 재학습 필요
- joystick.py 보상 가중치: tracking_lin_vel=2.5, tracking_ang_vel=6.0, **alive=20.0**,
  torques=-1e-3, action_rate=-0.5, stand_still=-0.2, imitation=1.0
  → alive가 압도적으로 커서 "무슨 수를 써서든 안 넘어지기"에 극단적으로 최적화된 정책
