#!/bin/bash
# 계산 노드를 5분 잡아서 GPU 인식 확인. 게이트 노드에서 실행.
PART="${PART:-gpu1}"
srun -p "$PART" --gres=gpu:1 -c 4 -t 00:05:00 --pty bash -lc '
  cd "$HOME/Open_Duck_Playground"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  JAX_PLATFORMS=cuda .venv/bin/python -c "
import jax
print(\"jax\", jax.__version__)
print(\"devices:\", jax.devices())
import jax.numpy as jnp
print(\"matmul ok:\", float(jnp.ones((512,512)) @ jnp.ones((512,512))) if False else (jnp.ones((512,512))@jnp.ones((512,512))).sum())
"
'
