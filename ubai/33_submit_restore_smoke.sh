#!/bin/bash
# 이어받기(--restore_checkpoint_path) 스모크 잡 1개 (2026-09-30 준비). gate1 에서 돌린다.
#
# 무엇을 확인하나: 포크 runner.py 의 이어받기 수정(fcb711f, SIM_NOTES 2026-09-29 "미뤄 둔 기록 정리")은
# 로컬에서 복원까지만 봤다 (되읽은 정책 = 같은 스텝 ONNX, 차이 8e-7). 복원한 뒤 학습 스텝은 노트북 CPU
# 메모리가 모자라 못 돌렸다. 이 잡이 그걸 본다.
#
# 설정: hp0dy2 300M 에서 이어받아, hp0dy2 와 똑같은 설정으로 2M 스텝 (실제로는 15회 평가 × 163840 = 2.29M).
#   같은 설정이라 첫 평가(STEP 0) 보상이 hp0dy2 의 학습 후반 평가 보상(206~255)과 같은 수준이어야 한다.
#   처음부터 배우면 STEP 0 은 15 다 (hp0dy2 994830 로그). 그래서 판정이 한 줄로 갈린다.
#   (09-29 노트에는 08-31 에서 이어받는다고 적었지만, 08-31 은 지금 joystick 과 보상 설정이 달라서
#    비교할 기준값이 없다. 코드 경로는 같다.)
#
# 순서:
#   1) 서버 runner.py 가 09-12 판(SEED 넣은 판)인지 해시로 확인
#   2) 백업하고 새 판(~/runner_restore.py, scp 로 올린 것)으로 교체
#      이어받기를 안 쓰는 잡에는 영향 없다 (restore 경로가 없으면 예전과 같은 코드를 탄다)
#   3) 로그인 노드 CPU 로 hp0dy2 체크포인트를 새 load_params 로 되읽어 본다 (몇 초. 학습 아님)
#   4) 출력 폴더가 비었는지 확인하고 제출 (walltime 1시간 — 짧게 잡아야 다른 잡 사이에 끼어 든다)
#
#   DRY=1 bash ~/ubai/33_submit_restore_smoke.sh       # 3) 까지만
#   bash ~/ubai/33_submit_restore_smoke.sh             # 제출까지
#   CHECK=<잡번호> bash ~/ubai/33_submit_restore_smoke.sh   # 끝난 뒤 판정
set -euo pipefail

REPO="${REPO:-$HOME/Open_Duck_Playground}"
DST="$REPO/playground/common/runner.py"
NEW="${NEW:-$HOME/runner_restore.py}"
# 09-12 서버 판 (from_ubai/server_wip_2026_09_12/runner.py), CR 을 뺀 sha256
OLD_SHA=9020c047bfa9a027d2857b66b762e96cc4e9bee92cc345eb3b39422825a646c6
WARM_CKPT="$REPO/checkpoints_hp0dy2/2026_09_28_102045_300482560"
OUT=checkpoints_restore_smoke
NAME=duck-restore
PASS_REWARD=100   # 처음부터면 STEP 0 이 15, hp0dy2 학습 후반 평가는 206~255

# ── 끝난 잡 판정 ──────────────────────────────────────────────
if [ -n "${CHECK:-}" ]; then
  O="$REPO/logs/$NAME-$CHECK.out"; E="$REPO/logs/$NAME-$CHECK.err"
  [ -f "$O" ] || { echo "!! $O 가 없다"; exit 1; }
  ok=1
  say() { if [ "$1" = 1 ]; then echo "  OK   $2"; else echo "  FAIL $2"; ok=0; fi; }
  echo "=== $NAME $CHECK ==="
  grep -h '^\[restore\]' "$O" | head -1 || true
  grep -q '^\[restore\].*count 3\.00' "$O" && r=1 || r=0
  say $r "[restore] 줄에 관측 정규화 count 3.00e+08 (hp0dy2 300M 을 읽었다)"
  grep -q '^\[warm\] 걷기 정책에서 웜스타트' "$O" && r=1 || r=0
  say $r "sbatch 가 웜스타트 경로를 탔다"
  steps=$(grep -c '^STEP:' "$O" || true)
  [ "$steps" -eq 15 ] && r=1 || r=0
  say $r "평가 15회 (지금 $steps)"
  first=$(grep '^STEP: 0 ' "$O" | awk '{print $4}' | head -1 || true)
  last=$(grep '^STEP:' "$O" | tail -1 | awk '{print $4}' || true)
  awk -v v="${first:-0}" -v t=$PASS_REWARD 'BEGIN{exit !(v>t)}' && r=1 || r=0
  say $r "STEP 0 보상 ${first:-없음} > $PASS_REWARD (복원이 먹었다. 처음부터면 15)"
  awk -v v="${last:-0}" -v t=$PASS_REWARD 'BEGIN{exit !(v>t)}' && r=1 || r=0
  say $r "마지막 평가 보상 ${last:-없음} > $PASS_REWARD (학습 스텝이 정책을 안 망쳤다)"
  grep -q '^=== end .* exit 0 ===' "$O" && r=1 || r=0
  say $r "exit 0"
  echo "--- 평가 보상 전부"
  grep '^STEP:' "$O" | awk '{printf "  %10s  %s\n", $2, $4}' || true
  if [ "$ok" = 1 ]; then
    echo "=== PASS: 이어받기 된다. 48시간 체인·토크 커리큘럼에 써도 된다 ==="
  else
    echo "=== FAIL. .err 끝:"; tail -30 "$E" 2>/dev/null || true; exit 1
  fi
  exit 0
fi

# --export=ALL 이라 셸에 남은 값이 잡으로 샌다.
unset LIN_VEL_X LIN_VEL_Y CMD_AXIS_ZERO HEAD_POS_W ALIVE_W HEAD_ACTION_SCALE \
      SEED FR_RAND_LO FR_RAND_HI WARM STEPS TASK

echo "=== 1-2) runner.py ==="
cur=$(tr -d '\r' < "$DST" | sha256sum | cut -c1-64)
if grep -q 'def load_params' "$DST"; then
  echo "[install] 이미 새 판이다 (load_params 있음). 교체 건너뜀"
elif [ "$cur" = "$OLD_SHA" ]; then
  [ -f "$NEW" ] || { echo "!! $NEW 가 없다. 윈도우에서 scp 부터"; exit 1; }
  grep -q 'def load_params' "$NEW" || { echo "!! $NEW 가 새 판이 아니다 (load_params 없음)"; exit 1; }
  bak="$DST.bak_$(date +%Y%m%d)"
  cp "$DST" "$bak"
  tr -d '\r' < "$NEW" > "$DST"
  echo "[install] 09-12 판 -> $bak 로 백업하고 교체"
else
  echo "!! 서버 runner.py 가 09-12 판이 아니다 (누가 고쳤다). 덮어쓰지 않는다. 차이:"
  if [ -f "$NEW" ]; then diff <(tr -d '\r' < "$DST") <(tr -d '\r' < "$NEW") | head -80 || true; fi
  exit 1
fi

echo "=== 3) 체크포인트 되읽기 (로그인 노드 CPU) ==="
[ -d "$WARM_CKPT" ] || { echo "!! $WARM_CKPT 가 없다"; exit 1; }
cd "$REPO"
out=$(JAX_PLATFORMS=cpu TF_CPP_MIN_LOG_LEVEL=2 .venv/bin/python - "$WARM_CKPT" 2>&1 <<'PY'
import sys
import jax
import numpy as np
from playground.common.runner import load_params
norm, policy, value = load_params(sys.argv[1])
count = float(np.asarray(norm.count))
n = lambda t: sum(np.asarray(x).size for x in jax.tree.leaves(t))
shapes = {k: np.asarray(x).shape for k, x in norm.mean.items()}
print(f"count {count:.4g} / obs {shapes} / policy {n(policy)} / value {n(value)} params")
assert abs(count - 3.005e8) < 1e6, count
assert shapes == {"state": (101,), "privileged_state": (212,)}, shapes
PY
) && rc=0 || rc=$?
printf '%s\n' "$out" | grep -v "Poly ref\|Processing\|cuInit\|oneDNN\|absl::InitializeLog" || true
[ "$rc" -eq 0 ] || { echo "!! 되읽기 실패 (exit $rc). 제출 안 함 (교체한 runner.py 는 그대로 둔다 — 이어받기를 안 쓰는 잡은 영향 없음)"; exit 1; }

if [ -n "$(ls -A "$REPO/$OUT" 2>/dev/null)" ]; then
  # 비어 있지 않으면 31_train_head.sbatch 가 WARM 대신 거기서 이어받는다 — 보려는 게 아니다.
  echo "!! $OUT 가 비어 있지 않다. 옮기거나 지우고 다시"; exit 1
fi

if [ "${DRY:-0}" = "1" ]; then
  echo "=== DRY=1: 여기까지. 제출은 DRY 없이 다시 ==="
  exit 0
fi

echo "=== 4) 제출 ==="
C="WARM=$WARM_CKPT,STEPS=2000000,TASK=flat_terrain_backlash,HEAD_POS_W=0.0,LIN_VEL_Y=0.2,FR_RAND_LO=1.40,FR_RAND_HI=1.90"
J=$(sbatch --parsable --job-name="$NAME" --partition=gpu1,gpu4,gpu5 --time=01:00:00 \
      --export="ALL,OUT=$OUT,$C" "$HOME/ubai/31_train_head.sbatch")
echo "$J $NAME"
echo "$(date '+%F %T') $J" >> "$HOME/ubai/restore_jobs.txt"
squeue -u "$USER"
echo
echo "끝나면 (러너가 -u 없이 돌아서 .out 은 끝날 때 한꺼번에 찬다):"
echo "  CHECK=$J bash ~/ubai/33_submit_restore_smoke.sh"
