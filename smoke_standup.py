"""Local CPU smoke test for the standup task.

Does not train. Builds the env, resets it, steps it, and checks the things that
would otherwise only blow up an hour into a cluster job:

  - the module imports at all
  - Standup.__init__ reaches the base env with the standup scene
  - config.reward_config.scales could be replaced (ConfigDict not locked)
  - reset() really starts the robot on the ground, not upright
  - the mjx_env.State rebuilt in reset() is well formed
  - step() runs, rewards are finite, and falling does NOT end the episode
  - the observation layout still matches the walking task, so a policy trained
    here can be replayed with mujoco_infer.py

Run with the cluster-matched CPU venv:
    .venv-train-cpu\\Scripts\\python.exe smoke_standup.py
"""

import os, sys, time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# joystick.py 의 _post_init 은 playground/open_duck_mini_v2/data/polynomial_coefficients.pkl
# 을 상대경로로 읽는다. 레포 루트에서 실행해야 하므로 여기서 옮겨둔다
# (클러스터의 runner.py 도 같은 이유로 레포 루트에서 실행한다).
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Open_Duck_Playground")
os.chdir(REPO)
sys.path.insert(0, REPO)

import jax
import jax.numpy as jp
import numpy as np

jax.config.update("jax_platform_name", "cpu")

FAIL = []


def check(label, cond, detail=""):
    print(f"  [{'OK ' if cond else 'FAIL'}] {label}" + (f"  - {detail}" if detail else ""))
    if not cond:
        FAIL.append(label)


print("=" * 62)
print("1) 임포트")
t = time.time()
from playground.open_duck_mini_v2 import standup, joystick
print(f"  standup / joystick 임포트 OK  ({time.time()-t:.1f}s)")

print("\n2) 환경 생성")
t = time.time()
env = standup.Standup()
print(f"  Standup() 생성 OK  ({time.time()-t:.1f}s)")
check("보상 스케일 교체됨", "upright" in env._config.reward_config.scales,
      f"{list(env._config.reward_config.scales.keys())}")
check("fall_termination 제거 확인용 config", env._config.episode_length == 500,
      f"episode_length={env._config.episode_length}")

print("\n3) 관측 크기가 걷기 태스크와 동일한가 (뷰어 호환)")
walk = joystick.Joystick(task="flat_terrain")
so_s, so_w = env.observation_size, walk.observation_size
check("observation_size 일치", so_s == so_w, f"standup={so_s}  joystick={so_w}")
check("action_size 일치", env.action_size == walk.action_size,
      f"{env.action_size} vs {walk.action_size}")

print("\n4) reset - 넘어진 자세로 시작하는가")
t = time.time()
reset = jax.jit(env.reset)
state = reset(jax.random.PRNGKey(0))
print(f"  첫 reset (JIT 컴파일 포함) {time.time()-t:.1f}s")

ups, heights = [], []
for seed in range(8):
    s = reset(jax.random.PRNGKey(seed))
    ups.append(float(env.get_gravity(s.data)[-1]))
    heights.append(float(env.get_floating_base_qpos(s.data.qpos)[2]))
print(f"  upvector z (8회): {np.round(ups, 3)}")
check("전부 넘어진 채 시작 (upvector z < 0.2)", all(u < 0.2 for u in ups),
      f"최대 {max(ups):.3f}")
check("자세가 매번 다름 (랜덤성)", float(np.std(ups)) > 0.05, f"std={np.std(ups):.3f}")
check("obs 에 NaN 없음", not bool(jp.isnan(state.obs["state"]).any()))
check("명령이 0", float(jp.abs(state.info["command"]).max()) == 0.0)

print("\n5) step - 100스텝 굴려보기")
step = jax.jit(env.step)
t = time.time()
s = reset(jax.random.PRNGKey(3))
rews, dones = [], []
for i in range(100):
    act = jp.zeros(env.action_size)          # 정책 없이 0 액션
    s = step(s, act)
    rews.append(float(s.reward)); dones.append(float(s.done))
print(f"  100스텝 {time.time()-t:.1f}s")
check("보상이 유한", np.all(np.isfinite(rews)), f"평균 {np.mean(rews):.4f}")
check("qpos NaN 없음", not bool(jp.isnan(s.data.qpos).any()))
check("넘어져도 종료되지 않음", sum(dones) == 0, f"done 발생 {int(sum(dones))}회")
check("보상 항목 존재", len(s.metrics) > 0, f"{sorted(s.metrics.keys())}")

print("\n6) 보상이 자세에 반응하는가 (똑바로 서면 커져야 함)")
s_fallen = reset(jax.random.PRNGKey(1))
r_fallen = env._get_reward(s_fallen.data, jp.zeros(env.action_size), s_fallen.info, {},
                           jp.zeros(()), jp.zeros(2), jp.zeros(2))
# 똑바로 선 home 자세로 강제 세팅해서 비교
from mujoco_playground._src import mjx_env
qpos_up = env._init_q
data_up = mjx_env.init(env.mjx_model, qpos=qpos_up, qvel=jp.zeros(env.mjx_model.nv),
                       ctrl=env.get_actuator_joints_qpos(qpos_up))
r_up = env._get_reward(data_up, jp.zeros(env.action_size), s_fallen.info, {},
                       jp.zeros(()), jp.zeros(2), jp.zeros(2))
print(f"  넘어짐  upright={float(r_fallen['upright']):+.3f}  height={float(r_fallen['height']):+.3f}")
print(f"  서있음  upright={float(r_up['upright']):+.3f}  height={float(r_up['height']):+.3f}")
check("upright 보상이 서있을 때 더 큼", float(r_up["upright"]) > float(r_fallen["upright"]))
check("height 보상이 서있을 때 더 큼", float(r_up["height"]) > float(r_fallen["height"]))

print("\n" + "=" * 62)
if FAIL:
    print(f"실패 {len(FAIL)}건: {FAIL}")
    sys.exit(1)
print("전부 통과 - 클러스터에 올려도 됩니다")
