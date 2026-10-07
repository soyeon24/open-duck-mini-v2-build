r"""초파리 뇌 조종(fly_pilot)으로 걷는 사람을 따라가는지 잰다 — eval_follow_moving 과 같은 대본·같은 채점.

대본(Walker), 판정(summarize)은 eval_follow_moving.py 것을 그대로 가져온다. 다른 건 조종기뿐이다:
`MjInfer.follow_step` 대신 `fly_pilot.FlyPilot` (LC9 → 전뇌 LIF → DNp09·DNa02·MDN → 명령).
뇌는 사람이 안 보이면 찾지 않고(자극 0 → 조용), 가까우면 물러나는 배선도 없다. 그래서
"다가오기" 와 오래 놓치는 대본은 기존 추종(빈 바닥 15/15)보다 나쁠 것으로 본다 — 그걸 재는 판이다.

뇌가 실시간보다 ~40배(렌더 포함) 느려 한 판이 20분 남짓이라 판마다 프로세스 하나씩 동시에 돌린다.
판마다 궤적을 head_cam_out/fly_moving_<대본>_<속도>.npz 로 남긴다. 뷰어 재생:

    .venv\Scripts\python.exe eval_fly_moving.py                        # 빈 바닥 5개 x 0.1, 0.2 m/s
    .venv\Scripts\python.exe eval_fly_moving.py --scenario away --speeds 0.1
    .venv\Scripts\python.exe fly_pilot.py --replay head_cam_out\fly_moving_cross_0.1.npz
"""
import argparse
import concurrent.futures as cf
import os
import sys
import time

import numpy as np

import eval_follow_moving as efm   # chdir·경로 설정도 여기서 된다
import fly_pilot

import mujoco  # noqa: E402

OUT = fly_pilot.OUT


def run_one(job):
    name, speed, seed = job
    import contextlib
    import io
    import band_tracker
    from lif import Brain
    from playground.open_duck_mini_v2.mujoco_infer import MjInfer, resolve_policy

    a = fly_pilot.make_parser().parse_args([])
    a.seed = seed
    onnx, rr, dy, fr = resolve_policy(None)
    with contextlib.redirect_stdout(io.StringIO()):
        m = MjInfer(efm.SCENES[efm.scene_of(name)], efm.REFERENCE, onnx, False, rr, dy)
        m.model.actuator_forcerange[:] = np.array([-fr, fr])
        m.full_reset()
        m.direct_head = False
        m.heading_hold = False
        walker = efm.Walker(name, speed)
        walker.put(m.data)
        mujoco.mj_forward(m.model, m.data)
        m.render_head()
    pilot = fly_pilot.FlyPilot(m, Brain(seed=seed), a)

    ctrl_dt = m.sim_dt * m.decimation
    hold = efm.hold_of(name)
    walk_len = float(np.sum(np.hypot(*np.diff(walker.pts, axis=0).T))) if len(walker.pts) > 1 else 2.0
    t_max = walk_len / speed + 2.0 + hold
    log, plog, qpos, ptraj = [], [], [], []
    k, t0 = 0, time.time()
    while True:
        t = k * ctrl_dt
        base = m.get_floating_base_qpos(m.data.qpos)
        walker.step(t, ctrl_dt, base[:2])
        walker.put(m.data)
        pilot.step(band_tracker)
        for _ in range(m.decimation):
            mujoco.mj_step(m.model, m.data)
        base = m.get_floating_base_qpos(m.data.qpos)
        rel = walker.pos - base[:2]
        bear = (np.degrees(np.arctan2(rel[1], rel[0])) - m.body_yaw_deg() + 180.0) % 360.0 - 180.0
        L = pilot.last
        seen = np.isfinite(L["bearing"])
        d = float(np.hypot(rel[0], rel[1]))
        # eval_follow_moving.run_one 과 같은 열 (우회·닿음은 0)
        log.append((t, base[0], base[1], walker.pos[0], walker.pos[1], d, bear, seen, False,
                    float(m.get_gravity(m.data)[-1]), walker.done_t is None,
                    m.commands[0], m.commands[1], m.commands[2], False))
        # fly_pilot 재생용 열
        plog.append((t, d, L["bearing"], m.commands[0], m.commands[2], *L["lc9"],
                     *(L["R"][(dn, s)] for dn in fly_pilot.DNS for s in ("left", "right"))))
        qpos.append(m.data.qpos.copy())
        ptraj.append((walker.pos[0], walker.pos[1], walker.heading))
        k += 1
        if walker.done_t is not None and t >= walker.done_t + hold:
            break
        if t > t_max + 30.0:
            break
    log = np.array(log, dtype=float)
    path = os.path.join(OUT, f"fly_moving_{name}_{speed:.1f}.npz")
    np.savez(path, log=np.array(plog), qpos=np.array(qpos), person=np.array(ptraj[0][:2]),
             person_traj=np.array(ptraj), ctrl_dt=ctrl_dt, args=str(vars(a)))
    return name, speed, log, walker.done_t, time.time() - t0


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", nargs="+",
                    default=[n for n in efm.SCENARIOS if efm.scene_of(n) == "empty"])
    ap.add_argument("--speeds", type=float, nargs="+", default=[0.1, 0.2])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    jobs = [(n, v, args.seed) for n in args.scenario for v in args.speeds]
    print(f"{len(jobs)}판 · 동시 {min(args.workers, len(jobs))} · 판마다 20분 남짓", flush=True)
    outs = []
    with cf.ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as ex:
        for fut in cf.as_completed([ex.submit(run_one, j) for j in jobs]):
            outs.append(fut.result())
            n, v, _, _, w = outs[-1]
            print(f"  끝: {n} {v:.1f} m/s (벽시계 {w / 60:.0f}분)", flush=True)

    ctrl_dt = 0.02
    print(f"\n| 대본 | 사람 m/s | 놓침 % | 최장 놓침 s | 걷는 중 최대 m | 끝 m | 최소 m | 판정 |")
    print("|---|---|---|---|---|---|---|---|")
    ok = 0
    order = {n: i for i, n in enumerate(efm.SCENARIOS)}
    for name, speed, log, done_t, _ in sorted(outs, key=lambda r: (order[r[0]], r[1])):
        s = efm.summarize(log, done_t, ctrl_dt, name)
        ok += s["verdict"] == "OK"
        print(f"| {efm.SCENARIOS[name][0]} | {speed:.1f} | {s['miss']:.0f} | {s['longest']:.1f} "
              f"| {s['d_walk']:.2f} | {s['d_end']:.2f} | {s['d_min']:.2f} | {s['verdict']} |")
    print(f"\nOK {ok}/{len(outs)}  (기존 추종 follow_step 은 빈 바닥 15/15 — SIM_NOTES 09-30)")


if __name__ == "__main__":
    main()
