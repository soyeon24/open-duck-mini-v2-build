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

print("\n4) reset - 바닥에 누운 자세로 시작하는가")
t = time.time()
reset = jax.jit(env.reset)
state = reset(jax.random.PRNGKey(0))
print(f"  첫 reset (JIT 컴파일 포함) {time.time()-t:.1f}s")

ups, heights, speeds = [], [], []
for seed in range(16):
    s = reset(jax.random.PRNGKey(seed))
    ups.append(float(env.get_gravity(s.data)[-1]))
    heights.append(float(env.get_floating_base_qpos(s.data.qpos)[2]))
    speeds.append(float(jp.abs(s.data.qvel).max()))
print(f"  upvector z (16회): {np.round(ups, 3)}")
print(f"  몸통 높이 cm     : {np.round(np.array(heights) * 100, 1)}")
# 서 있는 자세는 up=+1.0 / 높이 15cm. 누운 자세는 up 이 0 근처 (몸통 z축이 수평)
# 이거나 음수(뒤집힘)고, 높이는 23cm(물구나무)를 넘지 않는다.
# RSI 를 넣은 뒤로 "전부 넘어진 채 시작" 은 더 이상 참이 아니다 — 절반은 일부러
# 거의 선 상태에서 시작한다. 확인해야 할 건 "서 있는 채로 시작하지 않는다" 가 아니라
# "누운 쪽도 제대로 섞여 있다" 쪽이다.
check("누운 자세에서 시작하는 경우가 충분히 있다", sum(u < 0.5 for u in ups) >= 5,
      f"up<0.5 인 시작 {sum(u < 0.5 for u in ups)}/16 회")
# 높이만으로는 서 있는지 누웠는지 못 가린다. 등을 대고 누운 자세의 base 높이가
# 14.7cm 로 서 있을 때(15.0cm)와 거의 같다 — base 원점이 몸통 박스 위쪽에 있어서다.
# 서 있다는 건 up=+1 이면서 높이가 나오는 것이므로 둘을 같이 봐야 한다.
check("서 있는 채로 시작한 개체가 없음",
      not any(u > 0.7 and h > 0.14 for u, h in zip(ups, heights)),
      f"최대 up {max(ups):+.3f}")
check("자세가 매번 다름 (랜덤성)", float(np.std(ups)) > 0.05, f"std={np.std(ups):.3f}")

# RSI 가 실제로 걸리는지. 절반은 정답 궤적 위의 상태에서 시작해야 하고, 그 상태에는
# 운동량이 있다. 속도가 전부 0 이면 REF_FRACTION 이 안 먹고 있다는 뜻이다.
moving = sum(v > 1e-6 for v in speeds)
print(f"  |qvel| 최대 (16회)  : {np.round(speeds, 2)}")
check("절반쯤이 정답 궤적 상태에서 시작 (속도 있음)", 3 <= moving <= 13,
      f"속도가 0 이 아닌 시작 {moving}/16 회")
check("거의 다 선 상태로 시작하는 경우가 있다 (RSI 의 핵심)",
      any(u > 0.75 for u in ups), f"최대 up {max(ups):+.3f}")
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

print("\n7) 보상 사다리 - 누워 있는 게 최적이 아닌가")
# v1~v4 를 망친 것은 항상 이 지점이었다. 항목 하나하나는 말이 되는데, 합쳐 놓으면
# "가만히 있기" 가 제일 이득인 배치가 된다. 그래서 중간 단계의 총점을 직접 찍어본다.
#
# 세 상태:
#   누움      바닥에 누운 자세 (자세 뱅크에서)
#   몸통 들림 같은 자세에서 몸통만 12cm 띄움. up 은 그대로 (아직 수평) 이지만
#            바닥 접촉이 사라진다 = 일어서기 전반부에서 실제로 일어나는 변화
#   서 있음   home 키프레임
scales = env._config.reward_config.scales


from mujoco_playground._src.collision import geoms_colliding  # noqa: E402


def score(data):
    contact = jp.array([
        geoms_colliding(data, gid, env._floor_geom_id) for gid in env._feet_geom_id
    ])
    r = env._get_reward(data, jp.zeros(env.action_size), s0.info, {},
                        jp.zeros(()), contact, contact)
    total = sum(float(r[k]) * float(scales[k]) for k in r)
    return total, float(r["ground_clear"]), float(env.get_gravity(data)[-1])


# 자세 하나만 보고 판정하면 안 된다. 512개 중 8개(1.6%)는 무릎·정강이에 걸려
# 몸통이 안 닿은 채 멈춘 자세라, 하필 그걸 뽑으면 `ground_clear` 가 처음부터 1 이고
# 사다리가 거꾸로 보인다 (실제로 seed 1 이 그랬다). 여러 번 뽑아 평균으로 본다.
lies, lifts, clears = [], [], []
for seed in range(8):
    s0 = reset(jax.random.PRNGKey(seed))
    t_lie, gc, _ = score(s0.data)
    # 같은 자세에서 몸통만 12cm 들어올린다. 관절 각도는 그대로이므로 up 은 변하지
    # 않는다 — 즉 "아직 수평인데 바닥에서 떨어졌다" 는, 일어서기 전반부 그 자체다.
    q = s0.data.qpos.at[2].add(0.12)
    lifted = mjx_env.init(env.mjx_model, qpos=q, qvel=jp.zeros(env.mjx_model.nv),
                          ctrl=env.get_actuator_joints_qpos(q))
    t_lift, _, _ = score(lifted)
    lies.append(t_lie), lifts.append(t_lift), clears.append(gc)

standing = mjx_env.init(env.mjx_model, qpos=env._init_q,
                        qvel=jp.zeros(env.mjx_model.nv),
                        ctrl=env.get_actuator_joints_qpos(env._init_q))
t_stand, gc_stand, up_stand = score(standing)

print(f"  누움(8회 평균)  총점 {np.mean(lies):+7.3f}   ground_clear 이 0 인 비율 "
      f"{(1 - np.mean(clears)) * 100:.0f}%")
print(f"  몸통 들림        총점 {np.mean(lifts):+7.3f}")
print(f"  서 있음          총점 {t_stand:+7.3f}   up {up_stand:+.2f}")

check("시작 자세는 대개 몸이 바닥에 닿아 있다", np.mean(clears) < 0.3,
      f"ground_clear=1 인 비율 {np.mean(clears) * 100:.0f}%")
check("몸통을 들면 점수가 오른다 (= 일어서기 전반부에 보상이 있다)",
      np.mean(lifts) > np.mean(lies) + 0.5,
      f"{np.mean(lies):+.3f} -> {np.mean(lifts):+.3f}  "
      f"(차이 {np.mean(lifts) - np.mean(lies):+.3f})")
check("서 있는 게 제일 높다", t_stand > np.mean(lifts),
      f"{np.mean(lifts):+.3f} -> {t_stand:+.3f}")

print("\n" + "=" * 62)
if FAIL:
    print(f"실패 {len(FAIL)}건: {FAIL}")
    sys.exit(1)
print("전부 통과 - 클러스터에 올려도 됩니다")
