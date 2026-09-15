"""학습 시작용 "실제로 바닥에 누운" 자세 묶음을 만든다.

`standup.py` 의 현재 reset 은 home 자세를 무작위 각도로 기울여 z=0.22 에서 놓고
**그 즉시** 에피소드를 시작한다. 세 가지가 잘못돼 있다:

1. 시작 시점에 로봇은 아직 공중에 있다. "누워 있는 상태에서 시작" 이 아니라
   "떨어지는 중간부터 시작" 이다.
2. 90~160도가 전부 같은 자세로 수렴한다. 각도를 넓게 뽑아 커리큘럼을 만든 의도였는데
   실제로는 한 자세를 반복해서 주고 있었다.
3. 정확히 180도는 머리(20x20cm 평판) 위에 올라앉은 **물구나무** 가 된다. 속도 0으로
   정확히 뒤집어 떨어뜨려야만 나오는 특이점이고, 발이 공중 35cm 에 뜬다.
   커리큘럼의 "제일 어려운 끝" 이 사실은 현실에 없는 자세였다.

그래서 넘어지는 과정을 미리 CPU 에서 굴려 **바닥에 자리잡은 자세만** 모아 둔다.
학습 쪽 reset 은 이 표에서 하나 골라 쓰기만 하면 된다 (MJX 에서 인덱싱 한 번이라
컴파일도 간단하고, 자세 분포를 눈으로 확인할 수 있다).

    ..\\.venv\\Scripts\\python.exe make_standup_poses.py
"""

import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Open_Duck_Playground")
sys.path.insert(0, REPO)

import mujoco  # noqa: E402
from etils import epath  # noqa: E402

SCENE = os.path.join(REPO, "playground/open_duck_mini_v2/xmls/scene_standup.xml")
OUT = os.path.join(REPO, "playground/open_duck_mini_v2/data/standup_poses.npy")

N_POSES = 512
SETTLE_SEC = 2.0
SIM_DT = 0.002

# 자리잡음 판정. 넘어지는 도중이나 튀는 중인 자세가 섞이면 "누운 상태에서 시작"
# 이라는 전제가 깨진다.
MAX_LIN_VEL = 0.05    # m/s
MAX_ANG_VEL = 0.30    # rad/s
MAX_UP = 0.70         # 이보다 똑바르면 안 넘어진 것 -> 버린다


def main():
    from playground.open_duck_mini_v2 import base as odm_base

    m = mujoco.MjModel.from_xml_string(epath.Path(SCENE).read_text(),
                                       assets=odm_base.get_assets())
    m.opt.timestep = SIM_DT
    d = mujoco.MjData(m)
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    home_ctrl = np.array(m.keyframe("home").ctrl)
    grav_adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "upvector")]

    # 관절 목표값도 흔들어야 한다. 늘 home 자세로 넘어지면 다리가 항상 같은 모양으로
    # 접힌 채 누워 있게 되고, 정책은 그 한 가지 초기 관절배치만 보게 된다.
    act_jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator(i).name)
               for i in range(m.nu)]
    lo = np.array([m.jnt_range[j][0] for j in act_jid])
    hi = np.array([m.jnt_range[j][1] for j in act_jid])

    rng = np.random.default_rng(0)
    poses, ups, kinds = [], [], []
    tries = 0

    while len(poses) < N_POSES and tries < N_POSES * 20:
        tries += 1
        mujoco.mj_resetDataKeyframe(m, d, key)

        # 넘어지는 방향은 균일하게, 각도는 90도 이상 (= 수평을 넘어간 진짜 전복).
        # 180도 근처는 물구나무로 빠지므로 175도에서 끊는다.
        phi = rng.uniform(0, 2 * np.pi)
        axis = np.array([np.cos(phi), np.sin(phi), 0.0])
        angle = rng.uniform(np.deg2rad(90), np.deg2rad(175))
        dq = np.zeros(4)
        mujoco.mju_axisAngle2Quat(dq, axis, angle)
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, d.qpos[3:7].copy(), dq)
        d.qpos[3:7] = out
        d.qpos[2] = rng.uniform(0.18, 0.26)

        ctrl = np.clip(home_ctrl + rng.normal(0, 0.25, m.nu), lo, hi)
        d.qpos[7:7 + m.nu] = ctrl
        d.qvel[:] = 0.0
        # 떨어뜨리면서 조금 밀고 돌린다 — 실제 넘어짐에는 병진·회전 속도가 있다.
        d.qvel[0:3] = rng.normal(0, 0.25, 3)
        d.qvel[3:6] = rng.normal(0, 1.5, 3)
        d.ctrl[:] = ctrl

        for _ in range(int(SETTLE_SEC / SIM_DT)):
            mujoco.mj_step(m, d)

        if not np.isfinite(d.qpos).all():
            continue
        if np.linalg.norm(d.qvel[0:3]) > MAX_LIN_VEL:
            continue
        if np.linalg.norm(d.qvel[3:6]) > MAX_ANG_VEL:
            continue
        up = float(d.sensordata[grav_adr + 2])
        if up > MAX_UP:
            continue          # 안 넘어졌다
        if d.qpos[2] > 0.30:
            continue          # 어딘가에 끼어서 떠 있다

        poses.append(d.qpos.copy())
        ups.append(up)
        # 분류는 기록용. 학습은 전부 섞어 쓴다.
        kinds.append("물구나무" if up < -0.85 else
                     "엎드림" if d.qpos[2] < 0.09 else
                     "옆/등")

    P = np.array(poses, dtype=np.float32)
    np.save(OUT, P)

    ups = np.array(ups)
    print(f"{len(P)}개 자세 저장 ({tries}회 시도)  ->  {OUT}")
    print(f"  qpos 차원 {P.shape[1]} (모델 nq={m.nq})")
    print(f"  up  최소 {ups.min():+.2f}  중앙 {np.median(ups):+.2f}  최대 {ups.max():+.2f}")
    print(f"  몸통 높이  {P[:, 2].min() * 100:.1f} ~ {P[:, 2].max() * 100:.1f} cm")
    for k in sorted(set(kinds)):
        n = kinds.count(k)
        print(f"  {k:<8} {n:4d}개 ({n / len(kinds) * 100:4.1f}%)")
    # up 분포를 구간별로 — 커리큘럼이 실제로 넓은지 여기서 확인한다.
    hist, edges = np.histogram(ups, bins=8, range=(-1, 1))
    print("  up 분포:", " ".join(f"[{edges[i]:+.2f}]{hist[i]:3d}" for i in range(8)))


if __name__ == "__main__":
    main()
