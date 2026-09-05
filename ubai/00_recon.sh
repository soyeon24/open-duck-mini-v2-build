#!/bin/bash
# UBAI gate1 로그인 노드에서 실행. 출력 전체를 복사해서 돌려주세요.
echo "########## 1. 파티션 / GPU / 최대 walltime ##########"
sinfo -o "%P %l %D %t %G %m" | column -t
echo
echo "########## 2. 내 계정 한도 (QOS / 동시 잡 수) ##########"
sacctmgr -n show assoc user=$USER format=Account,Partition,QOS%20,MaxJobs,MaxSubmit,GrpTRES%30 2>&1 | head
sacctmgr -n show qos format=Name,MaxWall,MaxTRESPU%30,MaxJobsPU 2>&1 | head
echo
echo "########## 3. 홈 쿼터 (jax+tensorflow 약 7GB 필요) ##########"
df -h "$HOME" | tail -1
mmlsquota --block-size auto 2>/dev/null | head -5 || quota -s 2>/dev/null | head -5 || echo "(쿼터 명령 없음)"
echo
echo "########## 4. 모듈 / 툴체인 ##########"
module avail 2>&1 | tr ' ' '\n' | grep -iE "^cuda|^cudnn|^python" | sort -u | head -20
for c in uv git curl wget nvidia-smi apptainer singularity; do
  printf "%-12s : %s\n" "$c" "$(command -v $c || echo '없음')"
done
echo
echo "########## 5. 컴퓨트 노드: 드라이버 버전 + 인터넷 (가장 중요) ##########"
srun -p "${PART:-gpu1}" --gres=gpu:1 -t 00:05:00 --pty bash -lc '
  echo "-- node: $(hostname)"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  echo -n "pypi   : "; curl -sI --max-time 8 https://pypi.org   | head -1 || echo BLOCKED
  echo -n "github : "; curl -sI --max-time 8 https://github.com | head -1 || echo BLOCKED
  nproc; free -g | head -2
' 2>&1 | head -25
echo "########## END ##########"
