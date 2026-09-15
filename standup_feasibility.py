"""기립이 물리적으로 가능한지 직접 찾아본다 (학습 없이, CPU, 뷰어 없음).

SIM_NOTES "남은 과제" 의 첫 줄:

    완전 전복 복구가 물리적으로 가능한지 검증. ... 불가능하면 학습을 더 해도 소용없다.

여태 이 질문에 답한 적이 없다. PPO 가 못 한 것과 할 수 없는 것은 다른데, 실패한
학습 결과만 보고는 둘을 구별할 수 없다. 그래서 정책을 빼고 **열린 루프 궤적**을
직접 최적화한다. 하나라도 일어서는 궤적이 나오면 물리적으로 가능하다는 뜻이고
(= PPO 를 더 돌릴 가치가 있다), 수천 번 찾아도 안 나오면 모델을 고쳐야 한다.

궤적 = 관절 목표값 waypoint K개 -> 선형보간 -> 50Hz 서보 (속도 제한 5.24 rad/s).
제어 주파수·속도 제한·힘 한계를 학습 환경과 똑같이 두는 게 핵심이다. 여기서
느슨하게 하면 "가능하다" 는 답이 나와도 정책이 재현할 수 없다.

탐색은 CEM (cross-entropy method). 경사가 없고 차원이 40~112 정도라 충분하고,
CMA-ES 와 달리 의존성이 없다.

    ..\\.venv\\Scripts\\python.exe standup_feasibility.py --pose supine
    ..\\.venv\\Scripts\\python.exe standup_feasibility.py --pose supine --mode full --iters 80
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Open_Duck_Playground")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import mujoco  # noqa: E402
from etils import epath  # noqa: E402

SCENE = os.path.join(REPO, "playground/open_duck_mini_v2/xmls/scene_standup.xml")

CTRL_DT = 0.02          # 50 Hz, 학습과 동일
SIM_DT = 0.002
DECIMATION = 10
MAX_MOTOR_VEL = 5.24    # rad/s, mujoco_infer.py 와 동일
ACTION_SCALE = 0.25     # mujoco_infer.py 와 동일
SETTLE_SEC = 1.5        # 넘어진 자세가 바닥에 자리잡을 때까지
HOLD_SEC = 2.0          # 궤적이 끝난 뒤 마지막 목표값을 유지하는 시간

# 성공 판정 (eval_standup.py 와 같은 기준)
UP_OK = 0.90
HEIGHT_OK = 0.12
TARGET_HEIGHT = 0.15

# 좌우 대칭 파라미터를 14개 액추에이터로 펴는 방법.
# home 자세를 보면 hip_yaw / hip_roll / hip_pitch 는 좌우 부호가 반대이고
# (left_hip_pitch -36.1°, right +36.4°), knee / ankle 은 부호가 같다
# (left_knee 78.4°, right 79.0°). 부호를 잘못 잡으면 대칭 탐색이 실제로는
# 몸을 비트는 궤적만 뒤지게 된다.
ACT = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
       "neck_pitch", "head_pitch", "head_yaw", "head_roll",
       "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]
SYM_NAMES = ["hip_pitch", "knee", "ankle", "neck_pitch", "head_pitch"]

# 시작 자세 = (기울임 축의 방위각, 기울임 각도).
#
# 각도를 180도로 두면 **축이 어디든 전부 같은 자세로 수렴한다** — 머리 충돌박스가
# 20x20cm 평판이라, 완전히 뒤집어 떨어뜨리면 그 위에 올라앉은 물구나무 자세에서
# 멈춘다 (up -0.99, 발이 공중 35cm). 그게 standup.py 의 현재 reset 이 180도에서
# 실제로 만들고 있는 자세인데, 이건 "누워 있는 상태" 가 아니라 속도 0으로 정확히
# 뒤집어 떨어뜨려야만 나오는 특이점이다.
#
# 진짜 누운 자세는 90~160도 구간에서 나온다 (이 구간은 전부 같은 자세로 수렴한다).
#   supine  등을 대고 누움      up 0.00, 가슴이 하늘 (fwd_z +1.00), 머리 18cm / 발 12cm
#   prone   엎드림              up 0.18, 가슴이 바닥 (fwd_z -0.98), 머리 5.5cm / 발 6.7cm
#   side    옆으로 누움         up -0.11
#   headstand 물구나무 (특이점) up -0.99
POSES = {
    "supine": (270.0, 140.0),
    "prone": (90.0, 140.0),
    "side": (0.0, 140.0),
    "headstand": (270.0, 180.0),
}

_M = None
_D = None
_POLICY = None
_NSTEPS = None


def _init_worker(handoff_onnx=None):
    """워커마다 자기 모델/데이터를 갖는다 (MjData 는 프로세스 간 공유 불가)."""
    global _M, _D, _POLICY, _NSTEPS
    from playground.open_duck_mini_v2 import base as odm_base
    _M = mujoco.MjModel.from_xml_string(epath.Path(SCENE).read_text(),
                                        assets=odm_base.get_assets())
    _M.opt.timestep = SIM_DT
    _D = mujoco.MjData(_M)
    if handoff_onnx:
        from playground.common.onnx_infer import OnnxInfer
        from playground.common.poly_reference_motion_numpy import PolyReferenceMotion
        _POLICY = OnnxInfer(handoff_onnx, awd=True)
        _NSTEPS = PolyReferenceMotion(
            os.path.join(REPO, "playground/open_duck_mini_v2/data/"
                               "polynomial_coefficients.pkl")).nb_steps_in_period


def _policy_obs(m, d, default, last, last2, last3, targets, phase):
    """mujoco_infer.py 의 get_obs 와 동일한 101차원 관측."""
    gyro_adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "gyro")]
    acc_adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR,
                                             "accelerometer")]
    acc = d.sensordata[acc_adr:acc_adr + 3].copy()
    acc[0] += 1.3
    jid = _act_index(m)
    qadr = np.array([m.jnt_qposadr[j] for j in jid])
    vadr = np.array([m.jnt_dofadr[j] for j in jid])
    lf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "foot_assembly")
    rf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "foot_assembly_2")
    fl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    con = [0.0, 0.0]
    for c in range(d.ncon):
        g1, g2 = d.contact[c].geom1, d.contact[c].geom2
        if fl not in (g1, g2):
            continue
        other = g2 if g1 == fl else g1
        b = m.geom_bodyid[other]
        if b == lf:
            con[0] = 1.0
        elif b == rf:
            con[1] = 1.0
    return np.concatenate([
        d.sensordata[gyro_adr:gyro_adr + 3], acc, np.zeros(7),
        d.qpos[qadr] - default, d.qvel[vadr] * 0.05,
        last, last2, last3, targets, con, phase,
    ])


def _act_index(m):
    return [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ACT]


def settled_pose(m, d, phi_deg, angle_deg=180.0, seed=0):
    """넘어진 자세를 만들고 바닥에 자리잡을 때까지 굴린다.

    standup.py 의 reset 은 z=0.22 에서 떨어뜨린 **직후** 에피소드를 시작한다. 그러면
    공중에 뜬 상태부터 시작이라 "누워 있는 상태에서 시작" 이 아니다. 여기서는
    실제로 바닥에 누운 뒤의 상태를 재기 위해 먼저 가라앉힌다.
    """
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(m, d, key)
    home_ctrl = np.array(m.keyframe("home").ctrl)

    phi = np.deg2rad(phi_deg)
    axis = np.array([np.cos(phi), np.sin(phi), 0.0])
    dq = np.zeros(4)
    mujoco.mju_axisAngle2Quat(dq, axis, np.deg2rad(angle_deg))
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, d.qpos[3:7].copy(), dq)
    d.qpos[3:7] = out
    d.qpos[2] = 0.22
    d.qvel[:] = 0.0
    d.ctrl[:] = home_ctrl
    for _ in range(int(SETTLE_SEC / SIM_DT)):
        mujoco.mj_step(m, d)
    return d.qpos.copy(), d.qvel.copy()


def unpack(x, mode, n_way, lo, hi, home, anchor_last=True):
    """정규화된 탐색 벡터 -> waypoint별 14개 관절 목표값 (rad)."""
    if mode == "sym":
        w = x.reshape(n_way, len(SYM_NAMES))
        full = np.zeros((n_way, 14))
        hp, kn, an, npch, hpch = w[:, 0], w[:, 1], w[:, 2], w[:, 3], w[:, 4]
        full[:, 0] = 0.0                 # left_hip_yaw
        full[:, 1] = 0.0                 # left_hip_roll
        full[:, 2] = -hp                 # left_hip_pitch  (좌우 부호 반대)
        full[:, 3] = kn
        full[:, 4] = an
        full[:, 5] = npch
        full[:, 6] = hpch
        full[:, 7] = 0.0                 # head_yaw
        full[:, 8] = 0.0                 # head_roll
        full[:, 9] = 0.0                 # right_hip_yaw
        full[:, 10] = 0.0                # right_hip_roll
        full[:, 11] = hp                 # right_hip_pitch
        full[:, 12] = kn
        full[:, 13] = an
        x = full
    else:
        x = x.reshape(n_way, 14)
    # [-1,1] -> 관절 가동범위. 범위 밖 목표값은 서보가 못 가므로 탐색에서 제외한다.
    ways = lo + (np.clip(x, -1.0, 1.0) + 1.0) * 0.5 * (hi - lo)

    # 마지막 waypoint 는 home(서 있는) 자세로 고정한다.
    #
    # 고정하지 않으면 탐색이 "일어나는 동작" 과 "서 있을 수 있는 최종 자세" 두 가지를
    # 동시에 찾아야 하고, 실제로 그러면 웅크린 자세(up 0.75 / 높이 12cm)에서 수렴해
    # 멈춘다 — 8세대 만에 0.86 에서 붙박이었다. home 은 이미 알고 있는 안정한 기립
    # 자세이고, 학습 쪽 attention 보상도 어차피 이쪽으로 끌어당긴다. 알고 있는 답을
    # 탐색 변수로 두지 않는 것이다.
    if anchor_last:
        ways[-1] = home
    return ways


def rollout(args):
    """궤적 하나를 굴리고 (점수, 성공여부, 최고 up) 을 돌려준다."""
    x, mode, n_way, move_sec, qpos0, qvel0, forcerange, anchor_last = args
    m, d = _M, _D
    if forcerange is not None:
        m.actuator_forcerange[:] = forcerange

    jid = _act_index(m)
    lo = np.array([m.jnt_range[j][0] for j in jid])
    hi = np.array([m.jnt_range[j][1] for j in jid])
    home = np.array(m.keyframe("home").ctrl)
    grav_adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "upvector")]

    ways = unpack(np.asarray(x), mode, n_way, lo, hi, home, anchor_last)

    # qpos/qvel 만 덮어쓰면 안 된다. MjData 는 워커 안에서 재사용되는데 솔버의
    # warmstart(qacc_warmstart)와 접촉 캐시가 이전 rollout 에서 그대로 남아 있어,
    # 같은 파라미터를 같은 워커에서 두 번 굴려도 다른 점수가 나온다. 실제로 탐색
    # 최고값(0.095)과 재현값(0.036)이 어긋나서 발견했다. 탐색이 그 잡음을 신호로
    # 착각하면 엘리트 선택이 통째로 망가진다.
    mujoco.mj_resetData(m, d)
    d.qpos[:] = qpos0
    d.qvel[:] = qvel0
    d.ctrl[:] = home
    mujoco.mj_forward(m, d)

    n_move = int(move_sec / CTRL_DT)
    n_hold = int(HOLD_SEC / CTRL_DT)
    prev = home.copy()
    last = last2 = last3 = np.zeros(m.nu)
    phase_i = 0.0
    ups = np.zeros(n_move + n_hold)
    hs = np.zeros(n_move + n_hold)
    vlim = MAX_MOTOR_VEL * CTRL_DT

    for k in range(n_move + n_hold):
        if k < n_move:
            # waypoint 사이를 선형보간. 계단으로 주면 서보 속도 제한에 걸려
            # 실제로 따라가는 궤적이 파라미터와 달라지고, 탐색이 헛돈다.
            u = k / max(n_move - 1, 1) * (n_way - 1)
            i = min(int(u), n_way - 2)
            f = u - i
            tgt = ways[i] * (1 - f) + ways[i + 1] * f
        elif _POLICY is None:
            tgt = ways[-1]   # 마지막 자세를 유지 -> 순간적인 몸부림은 점수를 못 받는다
        else:
            # 인계 구간: 걷기 정책에 명령 0 으로 맡긴다.
            #
            # 열린 루프만으로는 여기서 반드시 넘어진다 — 실제로 탐색을 돌려보면
            # 몸통을 up 0.996 까지 세워 올리지만 그 자세를 못 지킨다. 균형은
            # 피드백이 하는 일이고 고정된 궤적에는 피드백이 없다. 그래서 "기립이
            # 가능한가" 를 열린 루프 하나로 판정하면 물리적 한계가 아니라
            # 파라미터화의 한계를 재게 된다.
            #
            # Microduck 도 기립과 보행을 별개 정책으로 출고한다. 여기서도 같은
            # 구조로, 세워 올리는 일만 궤적이 하고 버티는 일은 정책이 한다.
            if k == n_move:      # 인계 시점에 정책 내부 상태를 현재 목표값으로 맞춘다
                last = last2 = last3 = (prev - home) / ACTION_SCALE
                phase_i = 0.0
            phase_i = (phase_i + 1.0) % _NSTEPS
            ph = phase_i / _NSTEPS * 2 * np.pi
            obs = _policy_obs(m, d, home, last, last2, last3, prev,
                              np.array([np.cos(ph), np.sin(ph)]))
            act = _POLICY.infer(obs)
            last3, last2, last = last2, last, act.copy()
            tgt = home + act * ACTION_SCALE
        cmd = np.clip(tgt, prev - vlim, prev + vlim)
        prev = cmd
        d.ctrl[:] = cmd
        for _ in range(DECIMATION):
            mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all():
            return -10.0, False, -1.0
        ups[k] = d.sensordata[grav_adr + 2]
        hs[k] = d.qpos[2]

    tail = int(1.0 / CTRL_DT)
    up_t, h_t = ups[-tail:], hs[-tail:]
    # 주 항: 끝에서 똑바로(up) 이면서 높이도 나와야 1.0. 둘을 곱해서 머리로 서는
    # 자세(높이는 높지만 up 은 음수)가 점수를 못 받게 한다.
    primary = float(np.mean(np.clip(up_t, 0, 1) * np.clip(h_t / TARGET_HEIGHT, 0, 1)))
    # 보조 항: 탐색 초기에는 아무도 못 일어나므로 주 항이 전부 0 이라 방향이 없다.
    # "등에서 굴러 나오기라도 했는가" 를 최고 up 으로 준다.
    shaping = 0.3 * float(np.max(ups))
    settle_tail = int(1.5 / CTRL_DT)
    success = bool((ups[-settle_tail:] > UP_OK).all() and (hs[-settle_tail:] > HEIGHT_OK).all())
    return primary + shaping, success, float(np.max(ups))


def cem(pool, mode, n_way, move_sec, qpos0, qvel0, iters, pop, elite_frac, seed,
        forcerange, anchor_last):
    dim = n_way * (len(SYM_NAMES) if mode == "sym" else 14)
    rng = np.random.default_rng(seed)
    mu = np.zeros(dim)
    sigma = np.ones(dim) * 0.7
    n_elite = max(4, int(pop * elite_frac))
    best = (-1e9, None, False)

    for it in range(iters):
        X = np.clip(mu + sigma * rng.standard_normal((pop, dim)), -1.0, 1.0)
        out = pool.map(rollout,
                       [(x, mode, n_way, move_sec, qpos0, qvel0, forcerange, anchor_last)
                        for x in X],
                       chunksize=max(1, pop // max(pool._processes, 1)))
        scores = np.array([o[0] for o in out])
        order = np.argsort(-scores)
        el = X[order[:n_elite]]
        mu = el.mean(0)
        # 분산 하한이 없으면 몇 세대 만에 붕괴해서 지역해에 갇힌다.
        sigma = np.maximum(el.std(0), 0.05)
        if scores[order[0]] > best[0]:
            best = (float(scores[order[0]]), X[order[0]].copy(), out[order[0]][1])
        n_ok = sum(o[1] for o in out)
        print(f"  세대 {it + 1:3d}/{iters}  최고 {scores[order[0]]:.3f}  "
              f"상위평균 {scores[order[:n_elite]].mean():.3f}  "
              f"성공 {n_ok:3d}/{pop}  최고 up {max(o[2] for o in out):+.3f}  σ̄ {sigma.mean():.3f}")
        if n_ok >= pop * 0.5:
            print("  → 개체군 절반이 성공. 조기 종료.")
            break
    return best


def replay(m, d, args):
    """저장해둔 궤적을 다시 굴리면서 자세를 기록하고, 원하면 필름스트립을 만든다.

    굴리는 일은 rollout() 을 그대로 쓴다. 여기에 제어 루프를 한 벌 더 쓰면
    탐색이 본 것과 다른 궤적을 보게 되고, 그러면 그림이 증거가 못 된다.
    """
    z = np.load(args.replay, allow_pickle=True)
    _init_worker(args.handoff)
    x, mode = z["x"], str(z["mode"])
    n_way, move_sec = int(z["waypoints"]), float(z["move_sec"])
    qpos0, qvel0 = z["qpos0"], z["qvel0"]

    score, ok, peak = rollout((x, mode, n_way, move_sec, qpos0, qvel0,
                               None, not args.free_last))
    print(f"재생: {args.replay}")
    print(f"  시작 {str(z['pose'])} / {mode} / waypoint {n_way} / 동작 {move_sec}s")
    print(f"  점수 {score:.3f}  성공 {ok}  최고 up {peak:+.3f}")
    if not args.film:
        return

    # 다시 한 번 굴리면서 등간격으로 그림을 뽑는다. rollout 은 자세를 안 돌려주므로
    # 여기서는 같은 제어 입력을 재현하기 위해 ways 를 다시 계산해 직접 굴린다.
    import PIL.Image as Image
    jid = _act_index(m)
    lo = np.array([m.jnt_range[j][0] for j in jid])
    hi = np.array([m.jnt_range[j][1] for j in jid])
    home = np.array(m.keyframe("home").ctrl)
    ways = unpack(np.asarray(x), mode, n_way, lo, hi, home, not args.free_last)

    n_move, n_hold = int(move_sec / CTRL_DT), int(HOLD_SEC / CTRL_DT)
    shots = np.linspace(0, n_move + n_hold - 1, 8).astype(int)
    r = mujoco.Renderer(m, height=420, width=460)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 1.15, -10, 120
    cam.lookat[:] = [0, 0, 0.12]

    mujoco.mj_resetData(m, d)
    d.qpos[:] = qpos0
    d.qvel[:] = qvel0
    d.ctrl[:] = home
    mujoco.mj_forward(m, d)
    prev, vlim = home.copy(), MAX_MOTOR_VEL * CTRL_DT
    last = last2 = last3 = np.zeros(m.nu)
    phase_i = 0.0
    tiles = []
    for k in range(n_move + n_hold):
        if k < n_move:
            u = k / max(n_move - 1, 1) * (n_way - 1)
            i = min(int(u), n_way - 2)
            f = u - i
            tgt = ways[i] * (1 - f) + ways[i + 1] * f
        elif _POLICY is None:
            tgt = ways[-1]
        else:
            if k == n_move:
                last = last2 = last3 = (prev - home) / ACTION_SCALE
                phase_i = 0.0
            phase_i = (phase_i + 1.0) % _NSTEPS
            ph = phase_i / _NSTEPS * 2 * np.pi
            act = _POLICY.infer(_policy_obs(m, d, home, last, last2, last3, prev,
                                            np.array([np.cos(ph), np.sin(ph)])))
            last3, last2, last = last2, last, act.copy()
            tgt = home + act * ACTION_SCALE
        prev = np.clip(tgt, prev - vlim, prev + vlim)
        d.ctrl[:] = prev
        for _ in range(DECIMATION):
            mujoco.mj_step(m, d)
        if k in shots:
            cam.lookat[0], cam.lookat[1] = d.qpos[0], d.qpos[1]
            r.update_scene(d, cam)
            tiles.append((k * CTRL_DT, Image.fromarray(r.render())))

    W, H = tiles[0][1].size
    sheet = Image.new("RGB", (W * len(tiles), H), "white")
    for i, (_, im) in enumerate(tiles):
        sheet.paste(im, (i * W, 0))
    sheet.save(args.film)
    print(f"  필름스트립 저장: {args.film}  ({', '.join(f'{t:.1f}s' for t, _ in tiles)})")


def main():
    p = argparse.ArgumentParser(description="기립 물리적 가능성 검증")
    p.add_argument("--pose", default="supine", choices=list(POSES),
                   help="supine=등을 대고 누움, prone=엎드림, side=옆으로 누움")
    p.add_argument("--mode", default="sym", choices=["sym", "full"],
                   help="sym=좌우대칭 5축, full=14축 전부")
    p.add_argument("--waypoints", type=int, default=8)
    p.add_argument("--move_sec", type=float, default=4.0)
    p.add_argument("--iters", type=int, default=60)
    p.add_argument("--pop", type=int, default=256)
    p.add_argument("--elite", type=float, default=0.125)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=0,
                   help="병렬 워커 수 (0 = 코어수-2). 노트북을 쓰면서 돌릴 거면 줄일 것")
    p.add_argument("--force", type=float, default=None,
                   help="액추에이터 힘 한계를 이 값으로 덮어쓴다 (기본: 모델값 3.23)")
    p.add_argument("--handoff", default=None,
                   help="궤적이 끝난 뒤 버티는 일을 맡길 걷기 정책 .onnx "
                        "(없으면 마지막 자세를 그대로 유지 = 열린 루프)")
    p.add_argument("--free_last", action="store_true",
                   help="마지막 waypoint 를 home 에 고정하지 않고 탐색 변수로 둔다")
    p.add_argument("--save", default=None, help="찾은 궤적을 이 .npz 로 저장")
    p.add_argument("--replay", default=None,
                   help="탐색 대신 저장해둔 .npz 궤적을 다시 굴린다")
    p.add_argument("--film", default=None,
                   help="--replay 와 함께. 등간격 8컷 필름스트립을 이 .png 로 저장")
    args = p.parse_args()

    from playground.open_duck_mini_v2 import base as odm_base
    m = mujoco.MjModel.from_xml_string(epath.Path(SCENE).read_text(),
                                       assets=odm_base.get_assets())
    m.opt.timestep = SIM_DT
    d = mujoco.MjData(m)
    grav_adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "upvector")]

    if args.replay:
        replay(m, d, args)
        return

    print("=" * 74)
    print("시작 자세 조사 (기울여 떨어뜨린 뒤 바닥에 가라앉힌 상태)")
    for name, (phi, ang) in POSES.items():
        qpos, qvel = settled_pose(m, d, phi, ang)
        up = d.sensordata[grav_adr + 2]
        touching = set()
        for c in range(d.ncon):
            for g in (d.contact[c].geom1, d.contact[c].geom2):
                gn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
                if gn and gn != "floor":
                    touching.add(gn.replace("_ground_collision", ""))
        mark = "  <-- 선택" if name == args.pose else ""
        print(f"  {name:<9} ({ang:3.0f}도) up {up:+.3f}  몸통높이 {qpos[2] * 100:5.1f}cm  "
              f"바닥접촉 {sorted(touching)}{mark}")

    qpos0, qvel0 = settled_pose(m, d, *POSES[args.pose])
    forcerange = None
    if args.force is not None:
        forcerange = np.tile([-args.force, args.force], (m.nu, 1))
        print(f"\n액추에이터 힘 한계를 ±{args.force} N·m 로 덮어씀 (모델 기본 ±3.23)")

    dim = args.waypoints * (len(SYM_NAMES) if args.mode == "sym" else 14)
    print(f"\n탐색: {args.mode} / waypoint {args.waypoints}개 / {dim}차원 / "
          f"동작 {args.move_sec}s + 유지 {HOLD_SEC}s / "
          f"개체군 {args.pop} × {args.iters}세대 = {args.pop * args.iters}회 시뮬")
    print("=" * 74)

    t0 = time.time()
    n_workers = args.workers or max(1, (os.cpu_count() or 4) - 2)
    with Pool(processes=n_workers,
              initializer=_init_worker, initargs=(args.handoff,)) as pool:
        score, x, ok = cem(pool, args.mode, args.waypoints, args.move_sec,
                           qpos0, qvel0, args.iters, args.pop, args.elite,
                           args.seed, forcerange, not args.free_last)
        print("=" * 74)
        print(f"최고 점수 {score:.3f}  ({time.time() - t0:.0f}s)")
        if x is None:
            print("결과 없음")
            return
        # 최고 궤적을 다시 굴려 판정값을 뽑는다 (CEM 루프 안의 값은 세대별 최고라
        # 여러 번 재현되는지 확인이 안 된다). 워커 안에서 돌려야 조건이 같다 —
        # 메인 프로세스에는 _init_worker 가 만든 모델이 없다.
        s, succ, peak = pool.map(rollout, [(x, args.mode, args.waypoints, args.move_sec,
                                            qpos0, qvel0, forcerange,
                                            not args.free_last)])[0]
        print(f"재현: 점수 {s:.3f}  성공 {succ}  최고 up {peak:+.3f}")
        print()
        if succ:
            print(f"판정: {args.pose} 에서 기립은 **물리적으로 가능**하다.")
            print("      PPO 가 못 찾은 것이지 못 하는 게 아니다 -> 학습 설계를 고칠 가치가 있다.")
        elif peak > 0.5:
            print(f"판정: 몸을 뒤집는 데까지는 간다 (최고 up {peak:+.3f}). 서는 데서 막힌다.")
        else:
            print(f"판정: 등에서 굴러 나오지도 못한다 (최고 up {peak:+.3f}).")
            print("      학습을 더 돌려도 소용없을 가능성이 높다. 모델(머리 충돌박스 등)을 의심할 것.")

    if args.save:
        np.savez(args.save, x=x, mode=args.mode, waypoints=args.waypoints,
                 move_sec=args.move_sec, pose=args.pose, qpos0=qpos0, qvel0=qvel0)
        print(f"\n저장: {args.save}")


if __name__ == "__main__":
    main()
