"""279M vs 300M 체크포인트 비교 (README "Next" 1번 정리용).

세 런(head 910949 / fast 910953 / rough 910954) 전부에서 **마지막 300M 체크포인트가
279M 보다 총 reward 가 낮게** 나왔다. 셋 다 그러면 우연이 아닐 수 있다. 가설 둘:

  (A) alive 쏠림  - 후반부에 alive(20.0) 를 버는 쪽으로 기울어 덜 움직인다.
                    -> 생존은 같거나 낫고 명령 추종만 나빠진다.
  (B) 보상 해킹    - 후반부에 떠는 등 퇴화 동작으로 보상을 긁는다.
                    -> action rate 가 튀고 토크 포화가 늘어난다.

둘은 지표가 다르므로 뷰어로 눈대중하지 않고 갈라낸다. 반증도 가능하다:
차이가 전부 오차 범위면 "그냥 노이즈, 279M 쓰면 됨" 이 답이다.

하네스는 torque_probe.py / torque_derate_test.py 와 동일 - 뷰어만 가짜로 갈아끼우고
제어 루프는 원본을 그대로 돌린다. 토크 피크는 물리 스텝(500Hz)에서 나오므로
mj_step 도 감싼다.

주의: head(910949) 는 옛 명령범위로 학습됐고 fast/rough 는 레퍼런스 정합 범위다.
      ref_range 를 잘못 주면 학습 범위 밖 명령이 들어가 넘어진다 (SIM_NOTES 참조).
      "hi"/"lo" 로 쓰면 각 정책 자신의 범위가 들어가므로 쌍 내부 비교는 공정하다.
      런끼리의 절대속도 비교는 범위가 달라 무의미 - 추종률(실제/명령)로만 볼 것.
"""
import numpy as np
import mujoco
import mujoco.viewer
import playground.open_duck_mini_v2.mujoco_infer as MI

SCENE = "playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash_collision.xml"
REF = "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"

CTRL_HZ = 50.0
DECIM = 10          # 물리 500Hz / 제어 50Hz
SETTLE = 50         # 구간마다 앞 1초는 과도구간이라 버린다
T_STALL = 1.86      # STS3215 7.4V 정격 (torque_probe.py 와 동일)
R2D = 57.29578
HEAD = {"neck_pitch": 5, "head_pitch": 6, "head_yaw": 7, "head_roll": 8}

# (라벨, 제어스텝, [vx, vy, wz])  총 1500스텝 = 30초
PHASES = [
    ("정지",        200, [0.0,  0.0,  0.0]),
    ("전진 최대",   500, ["hi", 0.0,  0.0]),
    ("좌 게걸음",   250, [0.0,  "hi", 0.0]),
    ("좌회전",      250, [0.0,  0.0,  "hi"]),
    ("전진+좌회전", 300, ["hi", 0.0,  "hi"]),
]


class Done(Exception):
    pass


def yaw_of(q):
    """free joint 쿼터니언(w,x,y,z) -> yaw(rad)."""
    w, x, y, z = q
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def run_one(onnx, ref_range):
    mj = MI.MjInfer(SCENE, REF, onnx, False, ref_range)
    mj.direct_head = False          # 머리도 정책에 맡긴다 (실기 조건)

    base = mj._floating_base_qpos_addr
    rng = [mj.COMMANDS_RANGE_X, mj.COMMANDS_RANGE_Y, mj.COMMANDS_RANGE_THETA]

    bounds, acc = [], 0
    for _, n, _ in PHASES:
        bounds.append((acc, acc + n))
        acc += n

    st = {"i": 0, "phase": -1}
    trace = []      # (phase, x, y, yaw, up, *head4, *action)
    force, qvel, fph = [], [], []
    cmd_used = [None] * len(PHASES)
    orig_step = mujoco.mj_step

    def rec_step(model, data, *a, **k):
        orig_step(model, data, *a, **k)
        if st["phase"] >= 0:
            force.append(data.actuator_force.copy())
            qvel.append(data.qvel[mj.actuator_qvel_addr].copy())
            fph.append(st["phase"])

    def sync():
        i = st["i"]
        st["i"] += 1
        if i >= acc:
            raise Done
        for p, (s, _e) in enumerate(bounds):
            if i == s:
                c = list(PHASES[p][2])
                for k in range(3):
                    if c[k] == "hi":
                        c[k] = rng[k][1]
                    elif c[k] == "lo":
                        c[k] = rng[k][0]
                c = [float(v) for v in c]
                cmd_used[p] = c
                mj.commands = c + [0.0, 0.0, 0.0, 0.0]
                st["phase"] = p
        q = mj.data.qpos
        hq = mj.get_actuator_joints_qpos(q)
        trace.append(np.concatenate((
            [st["phase"], q[base], q[base + 1],
             yaw_of(q[base + 3:base + 7]), mj.get_gravity(mj.data)[-1]],
            [hq[j] for j in HEAD.values()],
            mj.last_action.copy())))

    fake = type("V", (), {"sync": staticmethod(sync),
                          "__enter__": lambda s: s,
                          "__exit__": lambda *a: False})()
    mujoco.viewer.launch_passive = lambda *a, **k: fake
    mujoco.mj_step = rec_step
    MI.time.sleep = lambda *a: None
    mj.full_reset()
    try:
        mj.run()
    except Done:
        pass
    finally:
        mujoco.mj_step = orig_step

    return (np.array(trace), np.array(force), np.abs(np.array(qvel)),
            np.array(fph), cmd_used, mj.actuator_names)


def analyse(tr, f, w, fph, cmd_used, names):
    """구간별 추종/부드러움/토크 + 전체 낙상."""
    ph = tr[:, 0]
    up = tr[:, 4]
    act = tr[:, 9:]

    fall_idx = np.where(up < 0.5)[0]
    fall_at = fall_idx[0] / CTRL_HZ if len(fall_idx) else None

    leg = [i for i, n in enumerate(names) if not n.startswith(("neck", "head"))]
    out = {"fall_at": fall_at, "min_up": up.min(), "phases": {}}

    for p, (label, _n, _c) in enumerate(PHASES):
        idx = np.where(ph == p)[0]
        if len(idx) <= SETTLE + 10:
            continue
        k = idx[SETTLE:]
        # 넘어진 뒤 수치는 의미 없다
        if fall_at is not None:
            k = k[k < fall_idx[0]]
            if len(k) < 20:
                continue

        x, y, yaw = tr[k, 1], tr[k, 2], tr[k, 3]
        dt = 1.0 / CTRL_HZ
        dx, dy = np.diff(x), np.diff(y)
        h = yaw[:-1]
        v_fwd = ((dx * np.cos(h) + dy * np.sin(h)) / dt).mean()
        v_lat = ((-dx * np.sin(h) + dy * np.cos(h)) / dt).mean()
        w_yaw = (np.unwrap(yaw)[-1] - np.unwrap(yaw)[0]) / ((len(k) - 1) * dt)

        # action rate: 제어스텝당 평균 |Δaction|. 떨면 커진다
        arate = np.abs(np.diff(act[k], axis=0)).mean()

        fk = np.where(fph == p)[0][SETTLE * DECIM:]
        if fall_at is not None:
            fk = fk[fk < fall_idx[0] * DECIM]
        if len(fk) > 10:
            a_leg = np.abs(f[np.ix_(fk, leg)])
            tq_peak, tq_over = a_leg.max(), 100 * (a_leg > T_STALL).mean()
        else:
            tq_peak = tq_over = float("nan")

        c = cmd_used[p]
        out["phases"][label] = dict(
            cmd=c, v_fwd=v_fwd, v_lat=v_lat, w_yaw=w_yaw,
            arate=arate, tq_peak=tq_peak, tq_over=tq_over)

    idx0 = np.where(ph == 0)[0][SETTLE:]
    out["head0"] = {n: tr[idx0, 5 + i].mean() * R2D for i, n in enumerate(HEAD)}
    return out


# (런, 체크포인트, 파일, ref_range)  head 만 옛 범위다
RUNS = [
    ("head 910949",  "279M", "../from_ubai/head_2026_09_04_140108_279019520.onnx",  False),
    ("head 910949",  "300M", "../from_ubai/head_2026_09_04_140643_300482560.onnx",  False),
    ("fast 910953",  "279M", "../from_ubai/fast_2026_09_04_140838_279019520.onnx",  True),
    ("fast 910953",  "300M", "../from_ubai/fast_2026_09_04_141412_300482560.onnx",  True),
    ("rough 910954", "279M", "../from_ubai/rough_2026_09_04_183038_279019520.onnx", True),
    ("rough 910954", "300M", "../from_ubai/rough_2026_09_04_185617_300482560.onnx", True),
]

res = {}
for run, ckpt, onnx, rr in RUNS:
    try:
        res[(run, ckpt)] = analyse(*run_one(onnx, rr))
        print("done %s %s" % (run, ckpt), flush=True)
    except Exception as e:
        print("FAIL %s %s: %s: %s" % (run, ckpt, type(e).__name__, e), flush=True)

runs = []
for run, ckpt, _o, _r in RUNS:
    if run not in runs:
        runs.append(run)

W = 96
print("\n" + "=" * W)
print("279M vs 300M  (평지 충돌체 씬, 30초: 정지4/전진10/게걸음5/좌회전5/전진+회전6)")
print("=" * W)

print("\n[1] 생존 - alive 쏠림(A)이면 300M 이 더 잘 버텨야 한다")
print("%-14s %6s %12s %9s" % ("런", "ckpt", "넘어짐", "최저 up"))
for run in runs:
    for ckpt in ("279M", "300M"):
        r = res.get((run, ckpt))
        if not r:
            continue
        print("%-14s %6s %12s %9.2f" % (
            run, ckpt,
            "%.1fs" % r["fall_at"] if r["fall_at"] is not None else "안 넘어짐",
            r["min_up"]))

print("\n[2] 명령 추종률 = 실제/명령. 1.0 이 정답, 낮으면 덜 움직인 것 (가설 A)")
print("%-14s %6s %14s %9s %9s %9s" %
      ("런", "ckpt", "구간", "명령", "실제", "추종률"))
KEY = {"전진 최대": ("v_fwd", 0), "좌 게걸음": ("v_lat", 1),
       "좌회전": ("w_yaw", 2), "전진+좌회전": ("v_fwd", 0)}
for run in runs:
    for label, (fld, ci) in KEY.items():
        row = []
        for ckpt in ("279M", "300M"):
            r = res.get((run, ckpt))
            row.append(r["phases"].get(label) if r else None)
        for ckpt, d in zip(("279M", "300M"), row):
            if not d:
                continue
            c, a = d["cmd"][ci], d[fld]
            print("%-14s %6s %14s %9.3f %9.3f %9s" % (
                run, ckpt, label, c, a,
                "%.2f" % (a / c) if abs(c) > 1e-6 else "-"))

print("\n[3] 부드러움 action rate (제어스텝당 평균 |Δaction|) - 크면 떠는 것 (가설 B)")
print("%-14s %6s %11s %11s %11s %11s" %
      ("런", "ckpt", "정지", "전진 최대", "좌회전", "전진+좌회전"))
for run in runs:
    for ckpt in ("279M", "300M"):
        r = res.get((run, ckpt))
        if not r:
            continue
        v = [r["phases"].get(l, {}).get("arate", float("nan"))
             for l in ("정지", "전진 최대", "좌회전", "전진+좌회전")]
        print("%-14s %6s %11.4f %11.4f %11.4f %11.4f" % (run, ckpt, *v))

print("\n[4] 다리 토크 - 정격 %.2f N·m 초과 비율. 1순위(forcerange 재학습) 근거" % T_STALL)
print("%-14s %6s %11s %11s %11s %11s" %
      ("런", "ckpt", "피크Nm", "전진>정격%", "회전>정격%", "복합>정격%"))
for run in runs:
    for ckpt in ("279M", "300M"):
        r = res.get((run, ckpt))
        if not r:
            continue
        pk = max([d["tq_peak"] for d in r["phases"].values()
                  if d["tq_peak"] == d["tq_peak"]] or [float("nan")])
        o = [r["phases"].get(l, {}).get("tq_over", float("nan"))
             for l in ("전진 최대", "좌회전", "전진+좌회전")]
        print("%-14s %6s %11.2f %11.2f %11.2f %11.2f" % (run, ckpt, pk, *o))

print("\n[5] 정지 자세 머리 오차 (도, 정답 0)")
print("%-14s %6s %11s %11s %11s %11s" %
      ("런", "ckpt", "neck_pitch", "head_pitch", "head_yaw", "head_roll"))
for run in runs:
    for ckpt in ("279M", "300M"):
        r = res.get((run, ckpt))
        if not r:
            continue
        h = r["head0"]
        print("%-14s %6s %+11.1f %+11.1f %+11.1f %+11.1f" % (
            run, ckpt, h["neck_pitch"], h["head_pitch"],
            h["head_yaw"], h["head_roll"]))
