# UBAI 학습 실행 절차 (Open Duck Mini v2)

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


## 접속 정보
| 항목 | 값 |
|---|---|
| Gate1 | `<GATE1_IP>` (Gate2: `<GATE2_IP>`) |
| 계정 | `<YOUR_ID>` |
| 키 | `<KEY_DIR>\ubai-<YOUR_ID>.pem` (icacls 권한 설정 완료) |
| 홈 | `/home1/<YOUR_ID>` (GPFS) |
| 제약 | **사설 IP → 학교 네트워크에서만 접속 가능. VPN 없음** |

```powershell
ssh <YOUR_ID>@<GATE1_IP> -i <KEY_DIR>\ubai-<YOUR_ID>.pem
```

## 클러스터 현황 (2026-08-26 확인)
| 파티션 | GPU | 노드 |
|---|---|---|
| gpu1 | RTX 3090 x4 | 14 |
| gpu2 / gpu6 | A10 x4 | 11 / 25 |
| gpu3 | **A6000 Ada x4** | 10 |
| gpu4 / gpu5 | A6000 x4 | 29 / 6 |
| cpu1 / cpu2 | - | 10 / 10 |

- 게이트 노드: 외부 인터넷 O, GPU X, 시스템 python 3.6.8
- `module`에 cuda/11.6.2 있으나 **쓰지 않음** (jax[cuda12]가 자체 CUDA wheel 동봉)
- 최대 walltime은 아직 미확인 → `00_recon.sh`로 확인

## 실행 순서

### 0. 스크립트 업로드 (윈도우 PowerShell)
```powershell
scp -i <KEY_DIR>\ubai-<YOUR_ID>.pem -r C:\school\2026_2\Microduck\ubai <YOUR_ID>@<GATE1_IP>:~/
```

### 1. 정찰 (2분) — 출력을 Claude에게 전달
```bash
bash ~/ubai/00_recon.sh 2>&1 | tee ~/recon.txt
```
확인 목적: **파티션 최대 walltime**, **컴퓨트 노드 드라이버 버전**(CUDA 12 wheel은 driver ≥ 525 필요), 홈 쿼터.

### 2. 환경 설치 (15~25분, 게이트 노드)
```bash
bash ~/ubai/10_setup.sh
```
uv 설치 → 레포 클론 → Python 3.11 → `uv sync` (jax[cuda12] + tensorflow, 약 7GB).

### 3. GPU 인식 확인 (5분)
```bash
bash ~/ubai/20_verify_gpu.sh
```
`devices: [CudaDevice(id=0)]` 나오면 성공. **여기서 실패하면 드라이버 버전 문제**.

### 4. 스모크 테스트 (20~40분) — 바로 300M 던지지 말 것
```bash
cd ~/Open_Duck_Playground
SEGMENT=5000000 TOTAL=5000000 CHAIN=0 sbatch --export=ALL,SEGMENT,TOTAL,CHAIN ~/ubai/30_train.sbatch
squeue -u <YOUR_ID>
```
확인할 것: ① GPU 실제 사용 ② `checkpoints/`에 디렉토리 + **`.onnx` 파일**이 떨어지는지 ③ 로그의 `STEP:` 간격으로 **step/sec 측정** → 300M 소요 시간 역산.

### 5. 본 학습 투입
```bash
cd ~/Open_Duck_Playground
sbatch --export=ALL,CHAIN=5,SEGMENT=100000000,TOTAL=300000000 ~/ubai/30_train.sbatch
```
walltime에 잘려도 후속 잡이 최신 체크포인트에서 자동 이어받음. 목표 스텝 도달하면 체인 스스로 종료.

### 6. 결과 회수 (다음 학교 방문 때)
```bash
bash ~/ubai/40_status.sh
```
```powershell
scp -i <KEY_DIR>\ubai-<YOUR_ID>.pem "<YOUR_ID>@<GATE1_IP>:~/Open_Duck_Playground/checkpoints/*.onnx" C:\school\2026_2\Microduck\
```
받은 ONNX를 로컬에서 재생:
```
..\.venv\Scripts\python.exe -m playground.open_duck_mini_v2.mujoco_infer -o ..\<파일>.onnx
```

## 주의
- **게이트 노드에서 직접 학습 금지.** 반드시 `sbatch`/`srun`.
- `srun --pty` 로 잡은 노드는 `exit` 로 반드시 반납.
- 체크포인트마다 ONNX가 자동 저장되므로, 중간에 끊겨도 그때까지의 정책은 건짐.
- **사사 표기 의무**: "본 논문은 서울시립대학교 도시과학빅데이터AI연구원의 슈퍼컴퓨팅 자원을 지원 받아 수행되었습니다."

## 알려진 한계
- brax PPO 체크포인트는 **네트워크 파라미터만** 저장하고 Adam 옵티마이저 상태는 저장하지 않음.
  → 세그먼트 경계에서 옵티마이저 모멘텀이 초기화됨. 학습률이 상수라 치명적이진 않지만,
  세그먼트를 잘게 쪼갤수록 손해. walltime이 허용하는 한 SEGMENT를 크게 잡을 것.
