"""서보 정격으로 깎았을 때도 걷는가 (HARDWARE_PREP.md §12 결정 실험).

torque_probe.py 는 "sim 이 정격을 넘는다" 는 통계를 냈다. 하지만 통계만으로는
실기에서 걷는지 못 정한다 — 정책이 내는 건 토크가 아니라 위치 목표고, 실물 서보는
못 내는 토크에서 그냥 추종이 나빠질 뿐이다. 그래서 직접 잘라 보고 걷는지 본다.

model.actuator_forcerange 만 바꾸고 나머지는 동일. 정책·모델·명령 전부 그대로.
"""
import numpy as np
import mujoco
import mujoco.viewer
import playground.open_duck_mini_v2.mujoco_infer as MI

SCENE = "playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash_collision.xml"
REF = "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"

CTRL_HZ = 50.0
PHASES = [        # (라벨, 제어스텝 수, 명령)
    ("정지",   150, [0.0,  0.0, 0.0]),
    ("전진",   750, ["hi", 0.0, 0.0]),
    ("좌회전", 300, [0.0,  0.0, "hi"]),
    ("전진",   300, ["hi", 0.0, 0.0]),
]
FORCERANGES = [3.23, 2.80, 2.40, 2.10, 1.86, 1.60]


class Done(Exception):
    pass


def run_one(onnx, ref_range, fmax):
    mj = MI.MjInfer(SCENE, REF, onnx, False, ref_range)
    mj.direct_head = False
    mj.model.actuator_forcerange[:] = np.array([-fmax, fmax])

    base = mj._floating_base_qpos_addr
    bounds, acc = [], 0
    for _, n, _ in PHASES:
        bounds.append((acc, acc + n))
        acc += n

    st = {"i": 0}
    trace = []   # (제어스텝, x, y, up)

    def sync():
        i = st["i"]
        st["i"] += 1
        if i >= acc:
            raise Done
        for p, (s, e) in enumerate(bounds):
            if i == s:
                cmd = list(PHASES[p][2])
                rng = [mj.COMMANDS_RANGE_X, mj.COMMANDS_RANGE_Y, mj.COMMANDS_RANGE_THETA]
                for k in range(3):
                    if cmd[k] == "hi":
                        cmd[k] = rng[k][1]
                    elif cmd[k] == "lo":
                        cmd[k] = rng[k][0]
                mj.commands = [float(v) for v in cmd] + [0.0, 0.0, 0.0, 0.0]
        q = mj.data.qpos[base:base + 3]
        trace.append((i, q[0], q[1], mj.get_gravity(mj.data)[-1]))

    fake = type("V", (), {"sync": staticmethod(sync),
                          "__enter__": lambda s: s,
                          "__exit__": lambda *a: False})()
    mujoco.viewer.launch_passive = lambda *a, **k: fake
    MI.time.sleep = lambda *a: None
    mj.full_reset()
    try:
        mj.run()
    except Done:
        pass

    cmd_vx = mj.COMMANDS_RANGE_X[1]
    tr = np.array(trace)
    fall = np.where(tr[:, 3] < 0.0)[0]
    fall_at = tr[fall[0], 0] / CTRL_HZ if len(fall) else None

    # 전진 구간(1번)에서의 실제 전진 속도. 넘어지기 전까지만 본다.
    s, e = bounds[1]
    seg = tr[(tr[:, 0] >= s + 50) & (tr[:, 0] < e)]
    if len(fall):
        seg = seg[seg[:, 0] < tr[fall[0], 0]]
    if len(seg) > 20:
        dist = np.hypot(seg[-1, 1] - seg[0, 1], seg[-1, 2] - seg[0, 2])
        vx = dist / ((seg[-1, 0] - seg[0, 0]) / CTRL_HZ)
    else:
        vx = float("nan")
    return fall_at, vx, cmd_vx, tr[:, 3].min()


RUNS = [
    ("BEST_WALK_ONNX_2", "../BEST_WALK_ONNX_2.onnx", False),
    ("hw8 931375 (-8.0)", "../from_ubai/hw8_2026_09_06_151227_300482560.onnx", True),
]

rows = []
for name, onnx, rr in RUNS:
    for fmax in FORCERANGES:
        try:
            rows.append((name, fmax) + run_one(onnx, rr, fmax))
        except Exception as e:
            print("FAIL %s @%.2f: %s: %s" % (name, fmax, type(e).__name__, e))

print("\n" + "=" * 74)
print("forcerange 를 깎으면 걷는가  (총 30초: 정지3 / 전진15 / 좌회전6 / 전진6)")
print("=" * 74)
print("%-20s %8s %10s %10s %9s %8s" %
      ("정책", "forceNm", "넘어짐", "전진 m/s", "명령 m/s", "최저 up"))
for name, fmax, fall_at, vx, cmd_vx, minup in rows:
    print("%-20s %8.2f %10s %10s %9.3f %8.2f" % (
        name, fmax,
        "%.1fs" % fall_at if fall_at is not None else "안 넘어짐",
        "%.3f" % vx if vx == vx else "-",
        cmd_vx, minup))
