"""정답 궤적 위의 상태들을 뽑아 학습 시작점으로 쓸 수 있게 만든다 (RSI).

## 왜 이게 필요한가

v5 (300M, 자세 뱅크 + `ground_clear`) 도 자기 학습 분포에서 **0/120** 이었다. 보상을
고쳐도 안 됐다. `standup_feasibility.py` 가 찾아낸 정답 궤적을 따라가며 재보면 이유가
나온다:

```
3.0s  up +0.026
3.6s  up -0.702   <- 일어서기 전에 오히려 더 뒤집힌다
4.2s  up -0.556
4.8s  up +0.550
5.4s  up +0.838
```

**정답은 좋아지기 전에 나빠진다.** `upright` 가중치 5.0 이 그 구간을 강하게 벌하므로,
up 에 단조인 보상을 쓰는 한 PPO 는 이 경로를 못 찾는다. 지역 최소값이 아니라 **탐색의
문제**다. 게다가 `ground_clear` 는 이 궤적 **전 구간에서 0** 이다 (마지막에 up +0.84
가 돼도 몸통 뒷모서리가 바닥에 남아 있다) — 부분 점수를 주라고 넣은 항이 정작 알려진
정답 위에서 한 번도 안 켜진다.

보상을 더 만지는 건 답이 아니다. 그래서 **탐색을 건너뛴다**: 궤적 위의 여러 지점에서
에피소드를 시작시키면, 정책은 먼저 "거의 다 선 상태에서 마무리하기" 를 배우고, 가치
함수가 그 지식을 뒤로(=어려운 구간으로) 전파한다. DeepMimic 의 reference state
initialization 과 같은 수법이고, 여기서는 사람 모캡 대신 CEM 이 찾은 궤적을 쓴다.

## 만드는 것

`data/standup_refstates.npy`, 각 행이 `[qpos(21) | qvel(20)]` 인 (N, 41) 배열.
qvel 이 반드시 필요하다 — 궤적 중간 상태에는 운동량이 있고, 그걸 0 으로 두면 정책이
실제로는 존재하지 않는 상태에서 시작하게 된다.

    ..\\.venv\\Scripts\\python.exe make_standup_refstates.py
"""

import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(ROOT, "Open_Duck_Playground")
sys.path.insert(0, REPO)

import mujoco  # noqa: E402
from etils import epath  # noqa: E402

SCENE = os.path.join(REPO, "playground/open_duck_mini_v2/xmls/scene_standup.xml")
TRAJ = os.path.join(REPO, "playground/open_duck_mini_v2/data/standup_ref_traj.npz")
OUT = os.path.join(REPO, "playground/open_duck_mini_v2/data/standup_refstates.npy")

# 제어 스텝 몇 개마다 한 상태를 뽑을지. 궤적이 6초(300스텝)이므로 5 면 60개.
STRIDE = 5


def main():
    # 궤적 재생은 반드시 standup_feasibility 의 rollout 과 같은 제어 루프를 써야 한다.
    # 여기서 루프를 다시 쓰면 CEM 이 평가한 것과 다른 궤적이 나오고, 그러면 "정답 위의
    # 상태" 라는 전제가 깨진다. 그래서 그 모듈을 그대로 불러다 상수와 unpack 을 쓴다.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "sf", os.path.join(ROOT, "standup_feasibility.py"))
    sf = importlib.util.module_from_spec(spec)
    sys.argv = [sys.argv[0]]
    spec.loader.exec_module(sf)

    from playground.open_duck_mini_v2 import base as odm_base
    m = mujoco.MjModel.from_xml_string(epath.Path(SCENE).read_text(),
                                       assets=odm_base.get_assets())
    m.opt.timestep = sf.SIM_DT
    d = mujoco.MjData(m)

    z = np.load(TRAJ, allow_pickle=True)
    x, mode = z["x"], str(z["mode"])
    n_way, move_sec = int(z["waypoints"]), float(z["move_sec"])

    jid = sf._act_index(m)
    lo = np.array([m.jnt_range[j][0] for j in jid])
    hi = np.array([m.jnt_range[j][1] for j in jid])
    home = np.array(m.keyframe("home").ctrl)
    # 이 궤적은 마지막 waypoint 를 고정하지 않고 찾은 것이다 (--free_last).
    ways = sf.unpack(np.asarray(x), mode, n_way, lo, hi, home, False)

    grav = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "upvector")]

    mujoco.mj_resetData(m, d)
    d.qpos[:] = z["qpos0"]
    d.qvel[:] = z["qvel0"]
    d.ctrl[:] = home
    mujoco.mj_forward(m, d)

    n_move = int(move_sec / sf.CTRL_DT)
    n_hold = int(sf.HOLD_SEC / sf.CTRL_DT)
    prev, vlim = home.copy(), sf.MAX_MOTOR_VEL * sf.CTRL_DT

    states, ups = [], []
    for k in range(n_move + n_hold):
        if k < n_move:
            u = k / max(n_move - 1, 1) * (n_way - 1)
            i = min(int(u), n_way - 2)
            f = u - i
            tgt = ways[i] * (1 - f) + ways[i + 1] * f
        else:
            tgt = ways[-1]
        prev = np.clip(tgt, prev - vlim, prev + vlim)
        d.ctrl[:] = prev
        for _ in range(sf.DECIMATION):
            mujoco.mj_step(m, d)
        if k % STRIDE == 0:
            states.append(np.concatenate([d.qpos.copy(), d.qvel.copy()]))
            ups.append(float(d.sensordata[grav + 2]))

    S = np.array(states, dtype=np.float32)
    np.save(OUT, S)

    ups = np.array(ups)
    print(f"{len(S)}개 상태 저장 (제어스텝 {STRIDE}개마다)  ->  {OUT}")
    print(f"  차원 {S.shape[1]} = qpos {m.nq} + qvel {m.nv}")
    print(f"  up   {ups.min():+.2f} ~ {ups.max():+.2f}")
    print()
    print("  시각별 up (이 곡선이 아래로 꺼졌다 올라오는 게 핵심):")
    for i in range(0, len(ups), max(1, len(ups) // 12)):
        t = i * STRIDE * sf.CTRL_DT
        bar = "#" * int((ups[i] + 1) * 20)
        print(f"    {t:4.1f}s {ups[i]:+6.2f} |{bar}")
    n_up = int((ups > 0.8).sum())
    print(f"\n  up > 0.8 인 상태 {n_up}개 — 정책이 '마무리' 를 먼저 배울 수 있는 지점")
    if n_up == 0:
        print("  ⚠️ 거의 선 상태가 하나도 없다. RSI 의 이득이 작을 수 있다.")


if __name__ == "__main__":
    main()
