# 기립(standup) 학습 — 서버에 올리는 법

> 접속 정보는 `ubai/MANUAL.md` 와 같은 자리표시자를 씁니다 (`<YOUR_ID>`, `<GATE1_IP>`, `<KEY_DIR>`).
> **학교 네트워크에서만 됩니다.** 집에서 `Connection timed out` 은 정상입니다.

---

## 0. 왜 다시 던지는가 (2026-09-15)

v2/v3/v4 (각 150M step) 를 `eval_standup.py` 로 각도별로 재봤더니 **45도를 넘으면 세 정책
전부 성공률 0%** 였습니다. 원인은 학습량이 아니라 태스크 설계 두 군데였고, 둘 다 고쳤습니다.

| 고친 것 | 내용 |
|---|---|
| 시작 자세 | `reset` 이 공중(z=0.22)에서 시작 + 90~160도가 전부 같은 자세로 수렴 + 180도는 물구나무(특이점). → 넘어지는 과정을 미리 굴려 만든 **바닥에 자리잡은 자세 512개** 에서 뽑도록 교체 |
| 보상 | `feet_contact` 가 `uprightness` 로 게이트돼 있어서, 누워 있는 동안(up=0)에는 **어떤 항목도 0점**. 기립의 전반부가 통째로 무보상이었음. → `ground_clear` (몸통·머리가 바닥에서 떨어짐) 추가하고 `feet_contact` 게이트를 그쪽으로 옮김 |

자세한 근거는 `SIM_NOTES.md` 의 2026-09-15 항목.

---

## 1. 서버로 보내야 하는 파일

서버 repo 는 upstream HEAD(`b9be205`) 기준이라 **로컬의 standup 커밋이 없습니다**
(`SIM_NOTES.md` "서버 코드가 git 에 없다"). git 으로 맞추려다 양방향 diff 를 만들지 말고
필요한 파일만 덮어씁니다.

```bash
scp -i <KEY_DIR>\ubai-<YOUR_ID>.pem ^
  Open_Duck_Playground\playground\open_duck_mini_v2\standup.py ^
  Open_Duck_Playground\playground\open_duck_mini_v2\standup_runner.py ^
  <YOUR_ID>@<GATE1_IP>:/home1/<YOUR_ID>/Open_Duck_Playground/playground/open_duck_mini_v2/

scp -i <KEY_DIR>\ubai-<YOUR_ID>.pem ^
  Open_Duck_Playground\playground\open_duck_mini_v2\xmls\open_duck_mini_v2_standup.xml ^
  Open_Duck_Playground\playground\open_duck_mini_v2\xmls\scene_standup.xml ^
  <YOUR_ID>@<GATE1_IP>:/home1/<YOUR_ID>/Open_Duck_Playground/playground/open_duck_mini_v2/xmls/

scp -i <KEY_DIR>\ubai-<YOUR_ID>.pem ^
  Open_Duck_Playground\playground\open_duck_mini_v2\data\standup_poses.npy ^
  <YOUR_ID>@<GATE1_IP>:/home1/<YOUR_ID>/Open_Duck_Playground/playground/open_duck_mini_v2/data/

scp -i <KEY_DIR>\ubai-<YOUR_ID>.pem ^
  ubai_standup\30_train_standup.sbatch ^
  <YOUR_ID>@<GATE1_IP>:/home1/<YOUR_ID>/ubai_standup/
```

⚠️ **`standup_poses.npy` 를 빼먹으면 잡이 즉시 죽습니다** (`FileNotFoundError`). 일부러 그렇게
해뒀습니다 — 없는 채로 돌면 예전 reset 으로 조용히 되돌아가는 게 아니라 그냥 안 됩니다.

---

## 2. 스모크 (2M step, 1분)

본학습 전에 반드시. 보상 항목이 하나 늘었으므로 `ground_clear` 가 메트릭에 뜨는지 봅니다.

```bash
ssh <YOUR_ID>@<GATE1_IP> -i <KEY_DIR>\ubai-<YOUR_ID>.pem
mkdir -p ~/ubai_standup ~/Open_Duck_Playground/logs
SEGMENT=2000000 TOTAL=2000000 OUTDIR=checkpoints_standup_smoke \
  sbatch --export=ALL,SEGMENT,TOTAL,OUTDIR $HOME/ubai_standup/30_train_standup.sbatch
```

```bash
squeue -u <YOUR_ID>
tail -40 ~/Open_Duck_Playground/logs/duck-standup-<JOBID>.out
```

확인할 것: `exit 0` / 보상이 0 이 아님 / `eval/episode_reward` 가 첫 평가에서 0 이 아닐 것.

---

## 3. 본학습 300M

v2~v4 는 150M 이었고 전부 45도에서 막혔습니다. 보상 설계가 바뀌었으니 같은 예산으로
비교해도 되지만, gpu1 기준 68,700 step/s 라 300M 이 **약 73분**입니다. 48시간 한도에
한참 못 미치므로 체인은 필요 없습니다 (`--restore_checkpoint_path` 는 orbax 포맷 문제로
**여전히 깨져 있습니다** — `SIM_NOTES.md` 발견 4).

```bash
sbatch --export=ALL,SEGMENT=300000000,TOTAL=300000000,OUTDIR=checkpoints_standup_v5 \
  $HOME/ubai_standup/30_train_standup.sbatch
```

> ⚠️ **시드 복제는 지금 못 돌립니다.** `SEED=1` 같은 걸 붙여도 아무 일도 안 일어납니다 —
> `standup_runner.py` 에도 `common/runner.py` 에도 시드 인자가 없고, PPO 설정은
> `brax_ppo_config` 기본값을 그대로 씁니다. 붙이면 **같은 학습이 이름만 다르게 두 번
> 돌아갑니다.** 09-12 의 `duck-frr-s2` 도 같은 이유로 실제 시드 복제가 아니었을 가능성이
> 높으니 그 비교는 다시 봐야 합니다.
>
> 고치려면 `common/runner.py` 에 `ppo_training_params["seed"]` 를 넣어야 하는데, 그 파일은
> 서버 쪽이 갈라져 있어서(미커밋 수정본) 덮어쓰면 걷기 쪽 작업을 날립니다. 서버 파일을
> 먼저 회수해 병합한 뒤에 손댈 것.

---

## 4. 회수하고 채점

```bash
scp -i <KEY_DIR>\ubai-<YOUR_ID>.pem ^
  "<YOUR_ID>@<GATE1_IP>:/home1/<YOUR_ID>/Open_Duck_Playground/checkpoints_standup_v5/*.onnx" ^
  C:\school\2026_2\Microduck\from_ubai\
```

눈으로 보기 전에 **숫자로** 봅니다. 뷰어는 "되는 것 같다" 까지밖에 안 나옵니다.

```bash
.venv\Scripts\python.exe eval_standup.py -o from_ubai\<파일>.onnx
```

판정 기준 (v2/v3/v4 는 45도 이상에서 전부 0/8):

| 결과 | 뜻 |
|---|---|
| 90도 이상에서 성공률 > 0 | 설계 수정이 먹혔다. 여기서부터 키운다 |
| 45도까지만 성공 | 보상은 고쳐졌지만 아직 부족. `ground_clear` 가중치를 올려볼 것 |
| 20도도 못 함 | 회귀. `standup_poses.npy` 가 서버에 갔는지부터 확인 |

뷰어로 보려면:

```bash
cd C:\school\2026_2\Microduck\Open_Duck_Playground
set PYTHONPATH=C:\school\2026_2\Microduck\Open_Duck_Playground
..\.venv\Scripts\python.exe -u -m playground.open_duck_mini_v2.mujoco_infer -o ..\from_ubai\<파일>.onnx --model_path playground\open_duck_mini_v2\xmls\scene_standup.xml
```
