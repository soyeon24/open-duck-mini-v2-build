r"""carry · skate 학습 환경 스모크 (CPU, 학습 아님). 서버에 올리기 전에 로컬에서 본다.

brax 학습 래퍼(wrap_for_brax_training + 도메인 랜덤화)를 그대로 거쳐 4개 환경을 굴린다.
보는 것: 관측 크기, 보상이 유한한가, 종료가 걸리는가, carry 는 물건이 쟁반 위에서 시작하는가.

    .venv-train-cpu\Scripts\python.exe smoke_addons.py carry
    .venv-train-cpu\Scripts\python.exe smoke_addons.py skate
"""
import functools
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(ROOT, "Open_Duck_Playground"))
sys.path.insert(0, os.getcwd())

import jax  # noqa: E402
import jax.numpy as jp  # noqa: E402
import numpy as np  # noqa: E402
from mujoco_playground import wrapper  # noqa: E402

from playground.common import randomize  # noqa: E402

which = sys.argv[1] if len(sys.argv) > 1 else "carry"
N, STEPS = 4, int(os.environ.get("SMOKE_STEPS", "150"))

if which == "carry":
    from playground.open_duck_mini_v2 import carry
    cfg = carry.default_config()
    env = carry.Carry(cfg)
    rand = functools.partial(carry.carry_randomize, obj_body=env._obj_body,
                             obj_geom=env._obj_geom, mass_range=tuple(cfg.carry_mass_range))
else:
    from playground.open_duck_mini_v2 import skate
    env = skate.Skate(skate.default_config())
    rand = randomize.domain_randomize

print("obs", env.observation_size, "act", env.action_size)
keys = jax.random.split(jax.random.PRNGKey(0), N)
benv = wrapper.wrap_for_brax_training(env, episode_length=1000, action_repeat=1,
                                      randomization_fn=functools.partial(rand, rng=keys))
reset = jax.jit(benv.reset)
step = jax.jit(benv.step)
t0 = time.time()
st = reset(keys)
jax.block_until_ready(st.obs)
print(f"reset 컴파일+실행 {time.time() - t0:.1f}s")

rng = jax.random.PRNGKey(1)
tot_r, dones, t0 = np.zeros(N), np.zeros(N), time.time()
for k in range(STEPS):
    rng, a = jax.random.split(rng)
    act = 0.2 * jax.random.normal(a, (N, env.action_size))
    st = step(st, act)
    tot_r += np.asarray(st.reward)
    dones += np.asarray(st.done)
    if k == 0:
        jax.block_until_ready(st.obs)
        print(f"step 컴파일+실행 {time.time() - t0:.1f}s")
        t0 = time.time()
    if which == "carry" and k in (0, 25, 50, 100, STEPS - 1):
        rel = jax.vmap(env._obj_rel)(st.data)
        print(f"  step {k:4d} 물건(쟁반 기준, cm) env0 {np.round(np.asarray(rel[0]) * 100, 1)}"
              f"  carry {np.round(np.asarray(st.metrics['reward/carry']), 2)}")
    if which == "skate" and k in (0, 50, STEPS - 1):
        print(f"  step {k:4d} base z {np.round(np.asarray(st.data.qpos[:, 2]), 3)}"
              f"  접지(관측 끝 4칸 중 앞 2) {np.asarray(st.obs['state'][:, -4:-2])}")
print(f"{STEPS} 스텝 {time.time() - t0:.1f}s · 보상합 {np.round(tot_r, 2)} · 종료 횟수 {dones}")
assert np.all(np.isfinite(tot_r)), "보상이 NaN"
for k, v in st.metrics.items():
    if k.startswith(("reward/", "cost/")):
        print(f"  {k:28s} {np.round(np.asarray(v), 3)}")
print("OK")
