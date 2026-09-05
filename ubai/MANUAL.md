# UBAI 운영 매뉴얼 — Open Duck Mini v2

> **플레이스홀더 안내** — 공개 저장소이므로 접속 정보는 마스킹되어 있습니다.
> 사용 시 본인 값으로 바꾸세요.
>
> | 자리표시자 | 의미 |
> |---|---|
> | `<YOUR_ID>` | UBAI 계정 ID |
> | `<GATE1_IP>` / `<GATE2_IP>` | 게이트 노드 주소 (계정 발급 메일 참고) |
> | `<KEY_DIR>` | SSH 개인키(.pem)를 둔 로컬 디렉토리 |
>
> 개인키(`.pem`)는 **절대 저장소에 넣지 마세요.**


> 혼자서 처음부터 끝까지 할 수 있도록 쓴 문서. 막히면 "트러블슈팅" 표부터 볼 것.

---

## 0. 큰 그림

```
[학교 네트워크에서만 접속 가능]
        │
   ┌────┴─────────────────┐
   │  UBAI 클러스터        │   PPO 학습 (GPU, 헤드리스, 며칠 단위)
   │  gate1 → 계산노드     │   → 체크포인트 + ONNX 자동 저장
   └────┬─────────────────┘
        │  scp 로 .onnx 회수
   ┌────┴─────────────────┐
   │  내 노트북           │   걷는 모습 구경 / 조종 (CPU 로 충분)
   │  mujoco_infer        │
   └──────────────────────┘
```

**핵심 리듬**: 학교에서 `sbatch` 로 던져놓고 → 집에 감 (잡은 계속 돌아감) → 다음에 학교 가서 결과 회수.
SSH 를 끊어도 잡은 죽지 않는다. 이게 Slurm 을 쓰는 이유다.

---

## 1. 접속

```powershell
ssh <YOUR_ID>@<GATE1_IP> -i <KEY_DIR>\ubai-<YOUR_ID>.pem
```

| | |
|---|---|
| Gate1 | `<GATE1_IP>` |
| Gate2 | `<GATE2_IP>` (Gate1 이 안 될 때) |
| 계정 | `<YOUR_ID>` |
| 홈 | `/home1/<YOUR_ID>` |

**집에서는 절대 안 된다.** 사설 IP라 학교 네트워크 밖에서는 라우팅 자체가 안 된다. VPN 없음.
`Connection timed out` 이 뜨면 십중팔구 학교 밖이다.

---

## 2. 확인된 클러스터 사양 (2026-08-31)

| 파티션 | GPU | 노드 수 | 비고 |
|---|---|---|---|
| **gpu1** | RTX 3090 (24GB) x4 | 14 | **기본으로 쓰는 곳** |
| gpu2 / gpu6 | A10 x4 | 11 / 25 | |
| gpu3 | A6000 Ada (48GB) x4 | 10 | 메모리 부족할 때 |
| gpu4 / gpu5 | A6000 (48GB) x4 | 29 / 6 | gpu4 가 제일 큼 |
| cpu1 / cpu2 | 없음 | 10 / 10 | |

| 항목 | 값 |
|---|---|
| **잡 1개당 최대 시간** | **48시간** (`MaxWall=2-00:00:00`) ← 파티션은 무제한이지만 계정 한도가 이걸로 걸림 |
| 동시 실행 잡 | 10개 / 대기 포함 20개 |
| 최대 GPU | 12장 |
| NVIDIA 드라이버 | 595.45.04 (CUDA 12 wheel 문제없음) |
| 계산 노드 인터넷 | 됨 (pypi, github 모두 200) |
| 계산 노드 사양 | 48 CPU / 754GB RAM |
| 홈 | GPFS, 여유 넉넉 |

---

## 3. 설치 (이미 완료. 새로 깔 일 있을 때만)

```bash
bash ~/ubai/10_setup.sh
```
uv 설치 → 레포 클론 → Python 3.11 → `uv sync` (약 7.9GB). 5분 내외.

확인:
```bash
bash ~/ubai/20_verify_gpu.sh    # devices: [CudaDevice(id=0)] 나오면 성공
```

### ⚠️ 버전 고정이 핵심
이 레포는 `pyproject.toml` 의 의존성에 **상한이 하나도 없다.** 마지막 커밋이 2025-08-05 이라
그냥 `uv sync` 하면 최신 패키지가 깔리고 학습이 두 군데서 깨진다.

| 패키지 | 고정 버전 | 안 하면 |
|---|---|---|
| `jax` / `jaxlib` | **0.6.2** | jax 0.8+ 에서 `jax.device_put_replicated` 제거 → brax PPO `train()` 실패 |
| `brax` | **0.12.4** | 위 API 를 쓰는 쪽 |
| `mujoco` / `mujoco-mjx` | **3.3.4** | brax/playground 와 짝 |
| `playground` | **0.0.5** | 0.1.0+ 에서 `mujoco_playground._src.collision` 삭제 → `joystick.py` import 실패 |

`10_setup.sh` 가 `uv sync` 뒤에 이 핀을 다시 덮어씌운다. **직접 `uv sync` 만 돌리면 안 된다.**

> ℹ️ 노트북의 CPU 추론 환경(`C:\school6_2\Microduck\.venv`)은 이 제약과 무관하다.
> 추론(`mujoco_infer.py`)은 onnxruntime 만 쓰고 brax 를 타지 않아서 최신 버전으로도 잘 돈다. 건드리지 말 것.

> ⚠️ **`module load cuda` 하지 말 것.** `jax[cuda12]` 가 자기 CUDA를 wheel로 들고 온다.
> 클러스터 CUDA 11.6을 올리면 `LD_LIBRARY_PATH` 가 충돌해서 오히려 깨진다.

---

## 4. 학습 던지기

항상 `~/Open_Duck_Playground` 에서 제출한다 (로그가 `logs/` 에 떨어지도록).

```bash
cd ~/Open_Duck_Playground
```

**스모크 테스트** (설정 바꿨을 때 항상 이것부터. 20~40분):
```bash
sbatch --export=ALL,SEGMENT=5000000,TOTAL=5000000,CHAIN=0 ~/ubai/30_train.sbatch
```

**측정된 속도 (RTX 3090, 2026-08-31 실측)**: 약 **32,800 step/sec**
→ 300M step ≈ **2시간 30분**. JIT 컴파일 워밍업이 앞에 1분 40초 붙는다.
→ 48시간 한도에 한참 못 미치므로 300M 은 잡 하나로 완주된다.

**본 학습**:
```bash
sbatch --export=ALL,SEGMENT=300000000,TOTAL=300000000,CHAIN=1 ~/ubai/30_train.sbatch
```

### 조절 가능한 값
| 변수 | 기본값 | 뜻 |
|---|---|---|
| `SEGMENT` | 300000000 | 잡 1개가 돌릴 스텝 수 |
| `TOTAL` | 300000000 | 누적 목표 스텝 수. 도달하면 체인 자동 종료 |
| `CHAIN` | 0 | 뒤에 예약할 후속 잡 개수 (48h 초과 대비 보험) |
| `TASK` | flat_terrain_backlash | `flat_terrain` / `rough_terrain` / `flat_terrain_backlash` / `rough_terrain_backlash` |
| `REPO` | ~/Open_Duck_Playground | 레포 위치 |

파티션을 바꾸려면 제출할 때 덮어쓴다:
```bash
sbatch -p gpu3 --export=ALL,... ~/ubai/30_train.sbatch
```

### CHAIN 이 하는 일
잡이 시작할 때 `--dependency=afterany` 로 후속 잡을 미리 예약해둔다.
48시간에 잘리거나 노드가 죽어도, 후속 잡이 **가장 최신 체크포인트를 찾아서 이어받는다.**
목표 스텝(`TOTAL`)에 도달하면 `.done` 파일을 만들고 남은 후속 잡을 스스로 취소한다.

> ⚠️ brax 체크포인트는 **네트워크 파라미터만** 저장하고 Adam 옵티마이저 상태는 저장하지 않는다.
> 세그먼트 경계마다 모멘텀이 초기화되므로, 잘게 쪼갤수록 손해다. `SEGMENT` 는 48h 안에서 최대로.

---

## 5. 상태 확인

```bash
bash ~/ubai/40_status.sh
```

개별 명령:
```bash
squeue -u <YOUR_ID>                       # 내 잡 목록
squeue -u <YOUR_ID> --start                # 대기 잡 예상 시작 시각
scancel <잡번호>                          # 잡 취소
scancel -u <YOUR_ID>                       # 내 잡 전부 취소
sacct -u <YOUR_ID> -X --format=JobID,JobName,State,Elapsed,ExitCode | tail   # 끝난 잡 이력
tail -f ~/Open_Duck_Playground/logs/duck-rl-<잡번호>.out                     # 실시간 로그
```

`squeue` 의 STATE 읽는 법:
| 값 | 뜻 |
|---|---|
| `RUNNING` | 돌아가는 중 |
| `PENDING` + `(Resources)` | GPU 빌 때까지 대기. 정상 |
| `PENDING` + `(Priority)` | 다른 사람 잡이 우선. 정상 |
| `PENDING` + `(AssocMaxWallDurationPerJobLimit)` | **`--time` 이 48h 초과. 영영 안 돌아감** |
| `PENDING` + `(Dependency)` | 앞 잡 끝나길 기다리는 중 (CHAIN) |

---

## 6. 걷는 모습 보기

**학습 중에는 실시간으로 못 본다.** 학습은 GPU에서 8192개 환경을 한꺼번에 돌리는 배치 시뮬이라
"걷는 모습"이라는 화면 자체가 존재하지 않는다. 대신 이렇게 한다.

### 6-1. ONNX 받아서 노트북에서 재생 (이게 정석)

체크포인트를 저장할 때마다 **ONNX 가 자동으로 같이 저장된다.** 학습이 안 끝났어도 중간 정책을 받아서 볼 수 있다.

클러스터에서 목록 확인:
```bash
ls -1t ~/Open_Duck_Playground/checkpoints/*.onnx | head
```

노트북(PowerShell)에서 받기:
```powershell
scp -i <KEY_DIR>\ubai-<YOUR_ID>.pem "<YOUR_ID>@<GATE1_IP>:~/Open_Duck_Playground/checkpoints/*.onnx" C:\school\2026_2\Microduck\
```

재생:
```powershell
cd C:\school\2026_2\Microduck\Open_Duck_Playground
set PYTHONPATH=C:\school\2026_2\Microduck\Open_Duck_Playground
..\.venv\Scripts\python.exe -m playground.open_duck_mini_v2.mujoco_infer -o ..\<파일이름>.onnx
```

조작키는 [../NEXT_STEPS.md](../NEXT_STEPS.md) 의 "조작키" 표 참고. (↑↓ 전후진, ←→ 게걸음, Q/E 회전, R 리셋)

### 6-2. 학습 곡선 보기 (TensorBoard)

`checkpoints/` 에 tfevents 로그가 같이 쌓인다. 통째로 받아서:
```powershell
scp -i <KEY_DIR>\ubai-<YOUR_ID>.pem -r "<YOUR_ID>@<GATE1_IP>:~/Open_Duck_Playground/checkpoints" C:\school\2026_2\Microduck\ckpt_from_ubai
tensorboard --logdir C:\school\2026_2\Microduck\ckpt_from_ubai
```
`eval/episode_reward` 가 우상향하면 잘 되고 있는 것.

로그에서 바로 보고 싶으면:
```bash
grep "^STEP:" ~/Open_Duck_Playground/logs/duck-rl-<잡번호>.out | tail -20
```

### 6-3. 실시간으로 볼 수는 없나?

**없다.** 학습은 GPU 위에서 수천 개 환경을 동시에 굴리는 배치 시뮬레이션이라
"오리 한 마리가 걷는 화면"이라는 게 애초에 존재하지 않는다. 렌더링도 하지 않는다.
X11 forwarding 으로 억지로 창을 띄워도 볼 것이 없다.

대신 실시간으로 확인 가능한 건 **숫자**다 (학교 네트워크에서만):
```bash
tail -f ~/Open_Duck_Playground/logs/duck-rl-<잡번호>.out
```
`STEP: <스텝> reward: <보상>` 이 15초 간격으로 찍힌다. reward 가 올라가면 잘 되고 있는 것.

**집에서 볼 방법**: 학교를 떠나기 전에 `.onnx` 를 노트북에 받아두면,
집에서 그 정책으로 오리가 걷는 걸 얼마든지 볼 수 있다. 학습 진행 상황을 실시간으로 볼 수는 없다.

---

## 7. 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| `Connection timed out` | 학교 밖이다. 학교 네트워크로 가야 한다 |
| `sbatch: error: Batch script contains DOS line breaks` | 윈도우에서 편집해서 CRLF가 섞임. 서버에서 `sed -i 's/\r$//' 파일명` |
| `PENDING (AssocMaxWallDurationPerJobLimit)` | `--time` 이 48시간을 넘음. `#SBATCH --time` 을 `2-00:00:00` 이하로 |
| `PENDING` 이 오래감 | `squeue -u <YOUR_ID> --start` 로 예상 시각 확인. 급하면 `-p gpu4` 등 한산한 파티션으로 |
| `devices: [CpuDevice(...)]` | GPU를 못 잡음. `--gres=gpu:1` 빠졌거나 `module load cuda` 를 해버린 것 |
| `RESOURCE_EXHAUSTED` / OOM | GPU 메모리 부족. `-p gpu3` (A6000 Ada 48GB) 로 옮기거나 `XLA_PYTHON_CLIENT_MEM_FRACTION` 낮추기 |
| `conda: command not found` | `source $HOME/miniconda3/etc/profile.d/conda.sh` (이 프로젝트는 conda 안 씀. uv 씀) |
| `uv: command not found` | `export PATH="$HOME/.local/bin:$PATH"` |
| 잡이 바로 죽음 | `logs/duck-rl-<번호>.err` 를 먼저 볼 것 |
| `No module named 'mujoco_playground._src.collision'` | `playground` 패키지가 최신(0.1.0+)으로 깔림. `uv pip install "playground==0.0.5"` |
| `mujoco_menagerie not found. Downloading...` | 정상. 첫 실행 때 한 번만 받는다 (홈에 캐시됨) |
| 홈 용량 부족 | `du -sh ~/* \| sort -h \| tail` 로 범인 찾기. `.venv` 7.9GB, `~/.cache/uv` 도 큼 |

---

## 8. 하지 말 것

- ❌ **게이트 노드(gate1)에서 직접 학습 돌리기.** 로그인 노드다. 무거운 작업을 돌리면 다른 사용자에게 피해가 가고 계정이 정지될 수 있다. 계산은 반드시 `sbatch` 또는 `srun`.
  - 게이트에서 해도 되는 것: 파일 편집, git, 패키지 설치, 잡 제출/모니터링
- ❌ `srun --pty` 로 잡은 노드를 그대로 두고 나가기. 반드시 `exit` 으로 반납한다 (안 하면 GPU를 계속 점유).
- ❌ `module load cuda` (위 3번 참고)
- ❌ 스모크 테스트 없이 바로 300M 던지기. 20시간 뒤 마지막 단계에서 죽는 게 제일 아깝다.

---

## 9. 사사 표기 (의무)

논문·프로젝트·보고서에 반드시 넣어야 한다.

> **국문** 본 논문은 서울시립대학교 도시과학빅데이터AI연구원의 슈퍼컴퓨팅 자원을 지원 받아 수행되었습니다.
>
> **영문** The authors acknowledge the Urban Big data and AI Institute of the University of Seoul supercomputing resources (http://ubai.uos.ac.kr) made available for conducting the research reported in this paper.

---

## 10. 파일 목록

| 위치 | 내용 |
|---|---|
| `~/ubai/` (클러스터) | 실행 스크립트 |
| `~/Open_Duck_Playground/` | 레포 + `.venv` + `checkpoints/` + `logs/` |
| `C:\school\2026_2\Microduck\ubai\` | 스크립트 원본 (여기서 고쳐서 scp 로 올림) |
| `<KEY_DIR>\ubai-<YOUR_ID>.pem` | SSH 키 |

스크립트 고친 뒤 업로드:
```powershell
scp -i <KEY_DIR>\ubai-<YOUR_ID>.pem -r C:\school\2026_2\Microduck\ubai\* <YOUR_ID>@<GATE1_IP>:~/ubai/
```
> 업로드 후 `sbatch` 가 CRLF 에러를 내면 서버에서 `sed -i 's/\r$//' ~/ubai/*.sh ~/ubai/*.sbatch`
