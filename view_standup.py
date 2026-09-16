"""기립 정책을 눈으로 보는 뷰어 (누운 자세에서 시작).

`mujoco_infer.py` 는 home 키프레임(서 있는 자세)에서 시작한다. 기립 정책을 거기에
띄우면 이미 서 있는 상태라 일어서는 걸 볼 수가 없다. 이 스크립트는 `standup.py` 가
학습에 쓰는 **자세 뱅크**에서 시작하고, 키로 얼마든지 다시 눕힐 수 있다.

제어 루프는 `eval_standup.py` 의 `StandupEval` 을 그대로 쓴다 (import). 뷰어가 채점기와
다른 루프를 돌면 "눈으로는 되는데 숫자로는 안 된다" 같은 상황이 생기고, 그러면 둘 다
믿을 수 없게 된다.

    ..\\.venv\\Scripts\\python.exe view_standup.py -o from_ubai\\v5_300482560.onnx

키:
    SPACE  자세 뱅크에서 새로 뽑아 눕힌다
    R      정답 궤적 위의 상태에서 시작 — 누를 때마다 시간 순으로 한 칸씩
    T      궤적 끝(거의 다 선 상태)으로 바로 — 마무리를 할 줄 아는지 보는 시험
    N      다음 자세 (뱅크를 순서대로)
    P      일시정지 / 재개
    0      home(서 있는) 자세로
"""

import argparse
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# eval_standup 은 import 시점에 레포 루트로 chdir 한다 (polynomial_coefficients.pkl 을
# 상대경로로 읽기 때문). 그래서 이 import 하나로 경로 준비까지 끝난다.
from eval_standup import StandupEval, UP_OK, HEIGHT_OK  # noqa: E402

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402

REF_STATES = "playground/open_duck_mini_v2/data/standup_refstates.npy"


def main():
    p = argparse.ArgumentParser(description="기립 정책 뷰어 (누운 자세에서 시작)")
    p.add_argument("-o", "--onnx", required=True)
    p.add_argument("--start", type=int, default=0, help="자세 뱅크에서 시작할 인덱스")
    args = p.parse_args()

    path = args.onnx if os.path.isabs(args.onnx) else os.path.join(ROOT, args.onnx)
    ev = StandupEval(path)
    bank = np.load(os.path.join(os.getcwd(), "playground/open_duck_mini_v2/data/standup_poses.npy"))
    refs = None
    if os.path.exists(REF_STATES):
        refs = np.load(REF_STATES)

    state = {"idx": args.start, "paused": False, "t0": time.time(), "mode": "bank",
             "ref": -1, "last_report": 0.0, "pending": None}

    def put_bank(i):
        state["idx"] = int(i) % len(bank)
        ev.reset_from_bank(state["idx"])
        state["mode"] = "bank"
        state["t0"] = time.time()
        state["last_report"] = 0.0
        print(f"[뱅크 {state['idx']:3d}] up {ev.get_gravity(ev.data)[-1]:+.2f}")

    def put_ref(i=None):
        """레퍼런스 상태를 **시간 순서대로** 준다.

        무작위로 뽑으면 안 된다. 60개 중 거의 다 선 상태(up>0.8)는 10개뿐이라,
        제일 보고 싶은 구간이 1/6 확률로만 나온다. 순서대로 밟으면 궤적을 따라
        걸으면서 정책이 정확히 어디서부터 못 버티는지 볼 수 있다.
        """
        if refs is None:
            print("레퍼런스 상태 파일이 없습니다 (make_standup_refstates.py)")
            return
        if i is None:
            state["ref"] = (state["ref"] + 1) % len(refs)
        else:
            state["ref"] = int(i) % len(refs)
        i = state["ref"]
        nq = ev.model.nq
        ev.data.qpos[:] = refs[i][:nq]
        ev.data.qvel[:] = refs[i][nq:]
        ev.data.ctrl[:] = ev.get_actuator_joints_qpos(ev.data.qpos)
        mujoco.mj_forward(ev.model, ev.data)
        # 정책 내부 상태도 같이 초기화해야 한다. 안 그러면 이전 에피소드의
        # last_action / motor_targets 이 남아 관측이 실제 상태와 어긋난다.
        ev.last_action = np.zeros(ev.num_dofs)
        ev.last_last_action = np.zeros(ev.num_dofs)
        ev.last_last_last_action = np.zeros(ev.num_dofs)
        ev.motor_targets = np.array(ev.default_actuator).copy()
        ev.prev_motor_targets = np.array(ev.default_actuator).copy()
        ev.imitation_i = 0.0
        ev.imitation_phase = np.array([0.0, 0.0])
        state["mode"] = "ref"
        state["t0"] = time.time()
        state["last_report"] = 0.0
        # 레퍼런스는 제어스텝 5개(=0.1초)마다 뽑았으므로 인덱스 i 는 궤적의 0.1*i 초.
        print(f"[레퍼런스 {i:2d}/{len(refs)}  궤적 {i * 0.1:.1f}초 지점] "
              f"up {ev.get_gravity(ev.data)[-1]:+.2f}")

    def put_home():
        key = mujoco.mj_name2id(ev.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(ev.model, ev.data, key)
        ev.data.ctrl[:] = ev.default_actuator
        mujoco.mj_forward(ev.model, ev.data)
        state["mode"] = "home"
        state["t0"] = time.time()
        state["last_report"] = 0.0
        print("[home] 서 있는 자세")

    def on_key(k):
        """**요청만 남기고 아무것도 실행하지 않는다.**

        이 콜백은 뷰어 스레드에서 불린다. 여기서 qpos 를 쓰거나 mj_forward 를 부르면
        메인 루프의 mj_step 과 같은 MjData 를 동시에 건드리게 되고, 키를 빠르게 연타하면
        실제로 세그폴트로 죽는다 (exit 139 로 겪었다). 적용은 메인 루프가 한다.
        """
        c = chr(k) if 32 <= k < 127 else ""
        if k == 32:
            state["pending"] = ("bank", int(np.random.randint(len(bank))))
        elif c in "nN":
            state["pending"] = ("bank", state["idx"] + 1)
        elif c in "rR":
            state["pending"] = ("ref", None)
        elif c in "tT":
            # 궤적의 마지막 = 거의 다 선 상태. 여기서 못 버티면 "마무리조차
            # 못 한다" 는 뜻이라, RSI 가 먹힐지 가늠하는 제일 빠른 시험이다.
            state["pending"] = ("ref", (len(refs) - 3) if refs is not None else 0)
        elif c in "pP":
            state["paused"] = not state["paused"]
            print("일시정지" if state["paused"] else "재개")
        elif c == "0":
            state["pending"] = ("home", None)

    put_bank(args.start)
    print()
    print("  SPACE 새 자세 / N 다음 / R 레퍼런스 순서대로 / T 거의 선 상태 / P 일시정지 / 0 home")
    print("  성공 판정: up > %.2f 이고 몸통 높이 > %.0fcm" % (UP_OK, HEIGHT_OK * 100))
    print()

    dt = ev.sim_dt * ev.decimation
    with mujoco.viewer.launch_passive(ev.model, ev.data, show_left_ui=False,
                                      show_right_ui=False, key_callback=on_key) as v:
        while v.is_running():
            t_start = time.time()

            # 키 요청은 여기서만 처리한다 (위 on_key 주석 참조).
            req = state["pending"]
            if req is not None:
                state["pending"] = None
                kind, arg = req
                if kind == "bank":
                    put_bank(arg)
                elif kind == "ref":
                    put_ref(arg)
                else:
                    put_home()

            if not state["paused"]:
                ev.imitation_i = (ev.imitation_i + 1.0) % ev.PRM.nb_steps_in_period
                ph = ev.imitation_i / ev.PRM.nb_steps_in_period * 2 * np.pi
                ev.imitation_phase = np.array([np.cos(ph), np.sin(ph)])

                action = ev.policy.infer(ev.get_obs())
                ev.last_last_last_action = ev.last_last_action.copy()
                ev.last_last_action = ev.last_action.copy()
                ev.last_action = action.copy()

                targets = ev.default_actuator + action * ev.action_scale
                lim = ev.max_motor_velocity * dt
                ev.motor_targets = np.clip(targets, ev.prev_motor_targets - lim,
                                           ev.prev_motor_targets + lim)
                ev.prev_motor_targets = ev.motor_targets.copy()
                ev.data.ctrl[:] = ev.motor_targets
                for _ in range(ev.decimation):
                    mujoco.mj_step(ev.model, ev.data)

            v.sync()

            el = time.time() - state["t0"]
            if el - state["last_report"] > 1.0:
                state["last_report"] = el
                up = float(ev.get_gravity(ev.data)[-1])
                h = float(ev.get_floating_base_qpos(ev.data.qpos)[2])
                ok = "서 있음" if (up > UP_OK and h > HEIGHT_OK) else ""
                print(f"  {el:4.1f}s  up {up:+.3f}  높이 {h*100:5.1f}cm  {ok}")

            # 실시간으로 돌린다. 안 그러면 CPU 가 낼 수 있는 만큼 빨라져서
            # 사람이 보기엔 순식간에 지나간다.
            slack = dt - (time.time() - t_start)
            if slack > 0:
                time.sleep(slack)


if __name__ == "__main__":
    main()
