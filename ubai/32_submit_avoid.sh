#!/bin/bash
# 회피용 걷기 정책 잡 4개 (2026-09-29 준비). gate1 에서 한 번 돌리면 끝난다.
#
# 무엇을 고치나: 지금까지 학습한 정책은 회피가 쓰는 동작 셋 — 후진, 제자리 게걸음,
# 제자리 선회 — 에서 멈춰 선다 (기본 정책 hp0dy2: 후진 -0.8 cm / 게걸음 0.2 cm / 12초).
# 원인 둘 (SIM_NOTES 2026-09-29):
#   1) 전진 범위 최대 0.222 로 학습한 정책은 전부 뒤로 못 걷는다 (910949 vs 910953).
#      -> LIN_VEL_X=0.15 (원본 범위)
#   2) 명령을 상자에서 균일하게 뽑아서 한 축짜리 명령이 1% 도 안 나온다.
#      -> CMD_AXIS_ZERO=0.4 (축마다 40% 확률로 0)
#
# | 잡            | LIN_VEL_X | CMD_AXIS_ZERO | 시드 | 역할                     |
# |---------------|-----------|---------------|------|--------------------------|
# | duck-avoid    | 0.15      | 0.4           | 0    | 본 설정                  |
# | duck-avoid-s2 | 0.15      | 0.4           | 1    | 본 설정 시드 복제        |
# | duck-dx15     | 0.15      | -             | 0    | 범위만 (마스킹 기여 분리) |
# | duck-mask     | (0.222)   | 0.4           | 0    | 마스킹만 (범위 기여 분리) |
# 나머지는 hp0dy2 와 같다 (HEAD_POS_W=0, LIN_VEL_Y=0.2, 토크 U(1.40,1.90), 300M, 처음부터).
# 즉 hp0dy2 가 "둘 다 안 한" 칸이고 이 넷과 2x2 를 이룬다.
#
# 순서:
#   1) 서버 joystick.py 가 09-28 판(hp0dy2 를 학습한 파일)인지 해시로 확인
#   2) 백업하고 새 판(~/joystick_avoid.py, scp 로 올린 것)으로 교체
#   3) 잡 설정 넷을 로그인 노드 CPU 로 찍어 본다 (명령 난수만, 몇 초씩. 학습 아님)
#   4) 출력 폴더가 비었는지 확인하고 제출
#
#   DRY=1 bash ~/ubai/32_submit_avoid.sh     # 3) 까지만
#   bash ~/ubai/32_submit_avoid.sh           # 제출까지
set -euo pipefail

REPO="${REPO:-$HOME/Open_Duck_Playground}"
DST="$REPO/playground/open_duck_mini_v2/joystick.py"
NEW="${NEW:-$HOME/joystick_avoid.py}"
# 09-28 서버 판 (from_ubai/server_wip_2026_09_28/joystick.py), CR 을 뺀 sha256
OLD_SHA=e04180d1a0fad150094fc9e96b3b514d370f003b2776ad23c40616fa8f232a60

# --export=ALL 이라 셸에 남은 값이 다른 잡으로 샌다 (duck-dx15 에 CMD_AXIS_ZERO 가 묻는 식).
unset LIN_VEL_X LIN_VEL_Y CMD_AXIS_ZERO HEAD_POS_W ALIVE_W HEAD_ACTION_SCALE \
      SEED FR_RAND_LO FR_RAND_HI OUT WARM STEPS TASK

echo "=== 1-2) joystick.py ==="
cur=$(tr -d '\r' < "$DST" | sha256sum | cut -c1-64)
if grep -q cmd_axis_zero "$DST"; then
  echo "[install] 이미 새 판이다 (cmd_axis_zero 있음). 교체 건너뜀"
elif [ "$cur" = "$OLD_SHA" ]; then
  [ -f "$NEW" ] || { echo "!! $NEW 가 없다. 윈도우에서 scp 부터"; exit 1; }
  grep -q cmd_axis_zero "$NEW" || { echo "!! $NEW 가 새 판이 아니다 (cmd_axis_zero 없음)"; exit 1; }
  cp "$DST" "$DST.bak_20260929"
  tr -d '\r' < "$NEW" > "$DST"
  echo "[install] 09-28 판 -> $DST.bak_20260929 로 백업하고 교체"
else
  echo "!! 서버 joystick.py 가 09-28 판이 아니다 (누가 고쳤다). 덮어쓰지 않는다. 차이:"
  if [ -f "$NEW" ]; then diff <(tr -d '\r' < "$DST") <(tr -d '\r' < "$NEW") | head -80 || true; fi
  exit 1
fi

echo "=== 3) 명령 분포 확인 (로그인 노드 CPU) ==="
cd "$REPO"
B="LIN_VEL_Y=0.2 HEAD_POS_W=0.0"
for extra in "LIN_VEL_X=0.15 CMD_AXIS_ZERO=0.4" "LIN_VEL_X=0.15" "CMD_AXIS_ZERO=0.4" ""; do
  echo "--- [$B $extra]"
  # shellcheck disable=SC2086
  out=$(env $B $extra JAX_PLATFORMS=cpu TF_CPP_MIN_LOG_LEVEL=2 \
        .venv/bin/python "$HOME/ubai/check_commands.py" 2>&1) && rc=0 || rc=$?
  printf '%s\n' "$out" | grep -v "Poly ref\|Processing" || true
  [ "$rc" -eq 0 ] || { echo "!! 분포 확인 실패 (exit $rc). 제출 안 함"; exit 1; }
done

OUTS="checkpoints_avoid checkpoints_avoid_s2 checkpoints_dx15 checkpoints_mask"
for d in $OUTS; do
  # 31_train_head.sbatch 는 출력 폴더에 진행분이 있으면 거기서 이어받으려 하는데,
  # --restore_checkpoint_path 가 깨져 있어서(SIM_NOTES 발견 4) 그대로 죽는다.
  if [ -n "$(ls -A "$REPO/$d" 2>/dev/null)" ]; then
    echo "!! $d 가 비어 있지 않다. 옮기거나 지우고 다시"; exit 1
  fi
done

if [ "${DRY:-0}" = "1" ]; then
  echo "=== DRY=1: 여기까지. 제출은 DRY 없이 다시 ==="
  exit 0
fi

echo "=== 4) 제출 ==="
C="WARM=none,STEPS=300000000,TASK=flat_terrain_backlash,HEAD_POS_W=0.0,LIN_VEL_Y=0.2,FR_RAND_LO=1.40,FR_RAND_HI=1.90"
sub() {
  sbatch --parsable --job-name="$1" --partition=gpu1,gpu4,gpu5 \
    --export="ALL,OUT=$2,$3,$C" "$HOME/ubai/31_train_head.sbatch"
}
J1=$(sub duck-avoid    checkpoints_avoid    "LIN_VEL_X=0.15,CMD_AXIS_ZERO=0.4")
J2=$(sub duck-avoid-s2 checkpoints_avoid_s2 "LIN_VEL_X=0.15,CMD_AXIS_ZERO=0.4,SEED=1")
J3=$(sub duck-dx15     checkpoints_dx15     "LIN_VEL_X=0.15")
J4=$(sub duck-mask     checkpoints_mask     "CMD_AXIS_ZERO=0.4")
echo "$J1 duck-avoid / $J2 duck-avoid-s2 / $J3 duck-dx15 / $J4 duck-mask"
echo "$(date '+%F %T') $J1 $J2 $J3 $J4" >> "$HOME/ubai/avoid_jobs.txt"
squeue -u "$USER"
echo
echo "시작하면 (1~2분) 로그 첫머리에서 설정이 먹었는지 본다:"
echo "  grep -h '\[joystick\] lin_vel\|\[randomize\]\|PPO params' logs/duck-avoid-$J1.out logs/duck-dx15-$J3.out | cut -c1-160"
