#!/bin/bash
# UBAI gate1 (로그인 노드) 에서 1회 실행. 약 15~25분 (jax CUDA + tensorflow 다운로드).
# 게이트 노드엔 GPU 가 없으므로 여기서는 설치만 하고, 검증은 20_verify_gpu.sh 에서 함.
set -euo pipefail

REPO="$HOME/Open_Duck_Playground"
export UV_CACHE_DIR="$HOME/.cache/uv"

echo "=== 1) uv 설치 ==="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
grep -q 'HOME/.local/bin' "$HOME/.bash_profile" 2>/dev/null || \
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bash_profile"
uv --version

echo "=== 2) 레포 클론 ==="
if [ ! -d "$REPO/.git" ]; then
  git clone https://github.com/apirrone/Open_Duck_Playground.git "$REPO"
fi
cd "$REPO"
mkdir -p logs checkpoints

echo "=== 3) Python 3.11 + 의존성 설치 (약 7GB) ==="
# pyproject: requires-python>=3.11, jax[cuda12], tensorflow
# ※ CUDA 는 module load 하지 않음. jax[cuda12] 가 nvidia-*-cu12 wheel 로 자기 CUDA 를 들고 옴.
#    module 의 cuda/11.6.2 를 로드하면 LD_LIBRARY_PATH 가 충돌해서 오히려 깨짐.
uv python install 3.11
uv venv --python 3.11
uv sync

# ⚠️ 버전 고정 필수. pyproject 의 의존성에 상한이 전혀 없어서 uv 가 전부 최신을 끌어오는데,
#    이 레포의 마지막 커밋은 2025-08-05 이라 최신 조합에서는 학습이 두 군데서 깨진다:
#      1) playground 0.1.0+ : `mujoco_playground._src.collision` 삭제됨 → joystick.py import 실패
#      2) jax 0.8+          : `jax.device_put_replicated` 제거됨 → brax PPO 의 train() 에서 실패
#    아래는 레포 커밋 시점(2025-08) 과 맞는 조합. 스모크 테스트로 실제 학습 검증 완료(2026-08-31).
uv pip install   "jax[cuda12]==0.6.2" "jaxlib==0.6.2"   "brax==0.12.4"   "mujoco==3.3.4" "mujoco-mjx==3.3.4"   "playground==0.0.5"

echo "=== 4) 결과 ==="
du -sh "$REPO/.venv"
"$REPO/.venv/bin/python" -c "import jax, mujoco_playground, brax; print('jax', jax.__version__)"
df -h "$HOME" | tail -1
echo "=== 설치 완료. 다음: bash ubai/20_verify_gpu.sh ==="
