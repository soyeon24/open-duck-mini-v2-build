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
    name, speed, seed, head = job
    import contextlib
    import io
    import band_tracker
    from lif import Brain
    from playground.open_duck_mini_v2.mujoco_infer import MjInfer, resolve_policy

    a = fly_pilot.make_parser().parse_args([])
    a.seed = seed
    a.head = head
    a.dist_from = os.environ.get("FLY_DIST_FROM", "ground")
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
    plog, qpos, ptraj = [], [], []
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
        L = pilot.last
        d = float(np.hypot(rel[0], rel[1]))
        # fly_pilot 재생용 열. 채점 열은 efm_log 가 여기서 다시 만든다
        plog.append((t, d, L["bearing"], m.commands[0], m.commands[2], *L["lc9"],
                     *(L["R"][(dn, s)] for dn in fly_pilot.DNS for s in ("left", "right")), L["head"]))
        qpos.append(m.data.qpos.copy())
        ptraj.append((walker.pos[0], walker.pos[1], walker.heading))
        k += 1
        if walker.done_t is not None and t >= walker.done_t + hold:
            break
        if t > t_max + 30.0:
            break
    path = npz_path(name, speed, head)
    np.savez(path, log=np.array(plog), qpos=np.array(qpos), person=np.array(ptraj[0][:2]),
             person_traj=np.array(ptraj), ctrl_dt=ctrl_dt, args=str(vars(a)))
    return name, speed, time.time() - t0


def npz_path(name, speed, head):
    return os.path.join(OUT, f"fly_moving_{name}_{speed:.1f}_{'head' if head else 'nohead'}.npz")


def efm_log(path):
    """저장한 궤적에서 eval_follow_moving.summarize 가 받는 열을 다시 만든다.

    채점을 이 한 경로로만 한다 — 굴리는 쪽과 채점이 같은 파일을 본다. 그래서 중간에 죽어도
    끝난 판은 다시 굴리지 않고 채점된다 (10판 동시에 돌리다 메모리가 모자라 죽은 적 있다).
    """
    z = np.load(path)
    P, Q, T = z["log"], z["qpos"], z["person_traj"]
    t = P[:, 0]
    q = Q[:, 3:7]
    yaw = np.degrees(np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                                1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2)))
    rel = T[:, :2] - Q[:, :2]
    bear = (np.degrees(np.arctan2(rel[:, 1], rel[:, 0])) - yaw + 180.0) % 360.0 - 180.0
    up = 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)          # 몸통 z 축의 월드 z 성분
    moved = np.any(np.abs(np.diff(T[:, :2], axis=0)) > 1e-9, axis=1)
    last = int(np.nonzero(moved)[0][-1]) + 1 if moved.any() else 0
    done_t = float(t[last])
    moving = t < done_t
    log = np.column_stack([t, Q[:, 0], Q[:, 1], T[:, 0], T[:, 1], P[:, 1], bear,
                           np.isfinite(P[:, 2]), np.zeros_like(t), up, moving,
                           P[:, 3], np.zeros_like(t), P[:, 4], np.zeros_like(t)])
    return log, done_t


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", nargs="+",
                    default=[n for n in efm.SCENARIOS if efm.scene_of(n) == "empty"])
    ap.add_argument("--speeds", type=float, nargs="+", default=[0.1, 0.2])
    ap.add_argument("--seed", type=int, default=0)
    # 판마다 뇌(연결 1,509만 개)와 렌더러를 따로 올린다. 노트북 16 GB 에서 10개는 메모리가 모자랐다.
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--force", action="store_true", help="저장된 판도 다시 굴린다")
    ap.add_argument("--no_head", dest="head", action="store_false",
                    help="머리를 안 돌린다 (첫 비교 2/10 의 조건)")
    args = ap.parse_args()

    allj = [(n, v, args.seed, args.head) for n in args.scenario for v in args.speeds]
    jobs = [j for j in allj if args.force or not os.path.exists(npz_path(j[0], j[1], j[3]))]
    if len(jobs) < len(allj):
        print(f"저장된 {len(allj) - len(jobs)}판은 다시 안 굴린다 (--force 로 다시)")
    print(f"{len(jobs)}판 · 동시 {min(args.workers, len(jobs))} · 판마다 20분 남짓", flush=True)
    if jobs:
        with cf.ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as ex:
            for fut in cf.as_completed([ex.submit(run_one, j) for j in jobs]):
                n, v, w = fut.result()
                print(f"  끝: {n} {v:.1f} m/s (벽시계 {w / 60:.0f}분)", flush=True)
    outs = [(n, v, *efm_log(npz_path(n, v, h))) for n, v, _, h in allj]

    ctrl_dt = 0.02
    print(f"\n| 대본 | 사람 m/s | 놓침 % | 최장 놓침 s | 걷는 중 최대 m | 끝 m | 최소 m | 판정 |")
    print("|---|---|---|---|---|---|---|---|")
    ok = 0
    order = {n: i for i, n in enumerate(efm.SCENARIOS)}
    for name, speed, log, done_t in sorted(outs, key=lambda r: (order[r[0]], r[1])):
        s = efm.summarize(log, done_t, ctrl_dt, name)
        ok += s["verdict"] == "OK"
        print(f"| {efm.SCENARIOS[name][0]} | {speed:.1f} | {s['miss']:.0f} | {s['longest']:.1f} "
              f"| {s['d_walk']:.2f} | {s['d_end']:.2f} | {s['d_min']:.2f} | {s['verdict']} |")
    print(f"\n[머리 {'돌림' if args.head else '고정'}] OK {ok}/{len(outs)}"
          f"  (같은 10판: 기존 추종 follow_step 10/10, 머리 고정 뇌 2/10 — 2026-10-07)")


if __name__ == "__main__":
    main()
