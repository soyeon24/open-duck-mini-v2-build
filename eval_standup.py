"""기립 정책을 시작 각도별로 채점한다 (뷰어 없음, CPU).

SIM_NOTES 의 "v2 결과 — 버티지만 완전히 넘어지면 못 일어남" 을 눈이 아니라 숫자로
확인하기 위한 스크립트. 뷰어로 보면 "되는 것 같다/아닌 것 같다" 밖에 안 나오는데,
어느 각도에서 벽이 생기는지 모르면 다음 학습을 어떻게 고쳐야 할지도 모른다.

제어 루프는 `mujoco_infer.py` 와 동일하다 (50Hz, action_scale 0.25, 서보 속도 제한
5.24 rad/s). 뷰어만 뺐다. 여기서 루프를 다르게 만들면 뷰어에서 보는 것과 다른
결과가 나와서 측정 자체가 쓸모없어진다.

리셋은 `standup.py` 의 reset 과 동일하다 — 수평축 기준 random 방향, 지정 각도만큼
기울여 z=0.22 에서 낙하.

    ..\\.venv\\Scripts\\python.exe eval_standup.py -o from_ubai\\v4_final_151388160.onnx
"""

import argparse
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Open_Duck_Playground")
os.chdir(REPO)
sys.path.insert(0, REPO)

import mujoco  # noqa: E402

from playground.common.onnx_infer import OnnxInfer  # noqa: E402
from playground.common.poly_reference_motion_numpy import PolyReferenceMotion  # noqa: E402
from playground.open_duck_mini_v2.mujoco_infer_base import MJInferBase  # noqa: E402

SCENE = "playground/open_duck_mini_v2/xmls/scene_standup.xml"
REFERENCE = "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
POSE_BANK = "playground/open_duck_mini_v2/data/standup_poses.npy"

DROP_HEIGHT = 0.22   # standup.py 와 동일
TARGET_HEIGHT = 0.15  # home 키프레임의 몸통 높이

# 성공 판정. 마지막 1.5초 동안 계속 서 있어야 한다.
#
# "최종 프레임에 서 있으면 성공" 으로 하면 일어나다가 넘어지는 중에 우연히 통과하는
# 프레임이 잡힌다. 반대로 평균만 보면 잠깐 넘어졌다 다시 일어난 것도 실패로 친다.
# 구간 내내 조건을 만족할 것을 요구하는 게 "일어서서 그 자세를 유지" 에 제일 가깝다.
SETTLE_SEC = 1.5
UP_OK = 0.90        # 몸통 z축과 연직의 각도 약 26도 이내
HEIGHT_OK = 0.12    # home(0.15)의 80%. 무릎을 굽힌 채 서 있어도 통과


class StandupEval(MJInferBase):
    def __init__(self, onnx_path, scene=SCENE):
        super().__init__(scene)
        self.sim_dt = 0.002
        self.decimation = 10
        self.action_scale = 0.25
        self.dof_vel_scale = 0.05
        self.max_motor_velocity = 5.24  # rad/s
        self.policy = OnnxInfer(onnx_path, awd=True)
        self.PRM = PolyReferenceMotion(REFERENCE)
        self.home_qpos = np.array(self.model.keyframe("home").qpos)
        self._bank = None

    def reset_fallen(self, angle, rng):
        """standup.py 의 reset 과 같은 자세로 시작한다.

        축은 반드시 수평면(xy)에서 뽑아야 한다. 구면에서 균일하게 뽑으면 축이 연직에
        가까울 때 아무리 큰 각도를 줘도 요(yaw)만 돌고 안 넘어져서, 각도 인자가
        의미를 잃는다 (standup.py 의 v1 실패 2번).
        """
        qpos = self.home_qpos.copy()
        phi = rng.uniform(0.0, 2.0 * np.pi)
        axis = np.array([np.cos(phi), np.sin(phi), 0.0])
        dq = np.zeros(4)
        mujoco.mju_axisAngle2Quat(dq, axis, angle)
        base = qpos[self._floating_base_qpos_addr + 3 : self._floating_base_qpos_addr + 7]
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, base, dq)
        qpos[self._floating_base_qpos_addr + 3 : self._floating_base_qpos_addr + 7] = out
        qpos[self._floating_base_qpos_addr + 2] = DROP_HEIGHT

        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.get_actuator_joints_qpos(qpos)
        mujoco.mj_forward(self.model, self.data)

        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)
        self.motor_targets = np.array(self.default_actuator).copy()
        self.prev_motor_targets = np.array(self.default_actuator).copy()
        self.imitation_i = 0.0
        self.imitation_phase = np.array([0.0, 0.0])

    def reset_from_bank(self, idx):
        """학습이 실제로 쓰는 시작 자세 (`standup.py` 의 reset 과 같은 표).

        각도 스윕(`reset_fallen`)은 **구버전 reset 을 재현한 것**이라 공중 z=0.22 에서
        시작한다. v5 는 바닥에 자리잡은 자세로 학습했으므로, 각도 스윕으로 재면
        학습한 적 없는 분포에서 시험하는 게 된다. v2~v4 와 나란히 비교하려면 각도
        스윕이 맞고, "누워 있다가 일어서는가" 를 물으려면 이쪽이 맞다.
        """
        if self._bank is None:
            self._bank = np.load(os.path.join(REPO, POSE_BANK))
        qpos = self._bank[idx % len(self._bank)]

        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.get_actuator_joints_qpos(qpos)
        mujoco.mj_forward(self.model, self.data)

        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)
        self.motor_targets = np.array(self.default_actuator).copy()
        self.prev_motor_targets = np.array(self.default_actuator).copy()
        self.imitation_i = 0.0
        self.imitation_phase = np.array([0.0, 0.0])
        return float(self.get_gravity(self.data)[-1])

    def get_obs(self):
        d = self.data
        accel = self.get_accelerometer(d).copy()
        accel[0] += 1.3
        contacts = self.get_feet_contacts(d)
        return np.concatenate([
            self.get_gyro(d),
            accel,
            np.zeros(7),  # command: 기립 태스크는 항상 0
            self.get_actuator_joints_qpos(d.qpos) - self.default_actuator,
            self.get_actuator_joints_qvel(d.qvel) * self.dof_vel_scale,
            self.last_action,
            self.last_last_action,
            self.last_last_last_action,
            self.motor_targets,
            contacts,
            self.imitation_phase,
        ])

    def rollout(self, angle, seed, duration, bank_idx=None):
        """한 번 굴리고 (up 이력, height 이력) 을 돌려준다.

        bank_idx 가 주어지면 자세 뱅크에서 시작하고, 아니면 각도로 기울여 떨어뜨린다.
        """
        if bank_idx is None:
            self.reset_fallen(angle, np.random.default_rng(seed))
        else:
            self.reset_from_bank(bank_idx)
        n_ctrl = int(duration / (self.sim_dt * self.decimation))
        ups = np.zeros(n_ctrl)
        heights = np.zeros(n_ctrl)

        for k in range(n_ctrl):
            # 학습 때 imitation_phase 는 명령이 0이어도 계속 돈다
            # (standup 은 Joystick.step 을 물려받고 USE_IMITATION_REWARD=True).
            # 여기서 0 으로 고정하면 정책이 겪어본 적 없는 입력이 된다.
            self.imitation_i = (self.imitation_i + 1.0) % self.PRM.nb_steps_in_period
            ph = self.imitation_i / self.PRM.nb_steps_in_period * 2 * np.pi
            self.imitation_phase = np.array([np.cos(ph), np.sin(ph)])

            action = self.policy.infer(self.get_obs())
            self.last_last_last_action = self.last_last_action.copy()
            self.last_last_action = self.last_action.copy()
            self.last_action = action.copy()

            targets = self.default_actuator + action * self.action_scale
            lim = self.max_motor_velocity * (self.sim_dt * self.decimation)
            self.motor_targets = np.clip(
                targets, self.prev_motor_targets - lim, self.prev_motor_targets + lim
            )
            self.prev_motor_targets = self.motor_targets.copy()
            self.data.ctrl[:] = self.motor_targets

            for _ in range(self.decimation):
                mujoco.mj_step(self.model, self.data)

            ups[k] = self.get_gravity(self.data)[-1]
            heights[k] = self.get_floating_base_qpos(self.data.qpos)[2]

        return ups, heights


def main():
    p = argparse.ArgumentParser(description="기립 정책 각도별 성공률 측정")
    p.add_argument("-o", "--onnx", required=True, nargs="+",
                   help="정책 파일. 여러 개 주면 같은 조건으로 나란히 잰다")
    p.add_argument("--seeds", type=int, default=8, help="각도당 시도 횟수")
    p.add_argument("--duration", type=float, default=10.0, help="에피소드 길이(초)")
    p.add_argument("--angles", type=float, nargs="+",
                   default=[30, 60, 90, 120, 150, 180],
                   help="시작 기울기(도)")
    p.add_argument("--bank", type=int, default=0, metavar="N",
                   help="각도 스윕 대신 자세 뱅크에서 N개를 뽑아 잰다 "
                        "(= 학습이 실제로 쓰는 시작 분포)")
    args = p.parse_args()

    # 아래에서 레포 루트로 chdir 했으므로 상대경로는 프로젝트 루트 기준으로 되돌린다.
    root = os.path.dirname(os.path.abspath(__file__))
    settle = int(SETTLE_SEC / 0.02)

    for onnx in args.onnx:
        path = onnx if os.path.isabs(onnx) else os.path.join(root, onnx)
        print("=" * 72)
        print(f"정책: {os.path.basename(path)}   시도 {args.seeds}회/각도, {args.duration:.0f}초")
        ev = StandupEval(path)

        if args.bank:
            ok, rows = 0, []
            for i in range(args.bank):
                up0 = ev.reset_from_bank(i)
                ups, hs = ev.rollout(0.0, 0, args.duration, bank_idx=i)
                tail_up, tail_h = ups[-settle:], hs[-settle:]
                s_ok = (tail_up > UP_OK).all() and (tail_h > HEIGHT_OK).all()
                ok += s_ok
                rows.append((up0, float(tail_up.mean()), float(tail_h.mean()), s_ok))
            print(f"자세 뱅크 {args.bank}개 (학습이 쓰는 시작 분포)")
            print(f"  성공 {ok}/{args.bank}  ({ok / args.bank * 100:.1f}%)")
            # 시작 자세가 얼마나 뒤집혀 있었는지로 나눠 본다. 전부 0% 여도
            # 어느 구간이 되고 어느 구간이 안 되는지는 봐야 다음 수가 나온다.
            import collections
            buckets = collections.defaultdict(list)
            for up0, fu, fh, s_ok in rows:
                k = ("물구나무 up<-0.85" if up0 < -0.85 else
                     "옆/등 -0.85~0.3" if up0 < 0.3 else "거의 엎드림 up>0.3")
                buckets[k].append((fu, fh, s_ok))
            for k in sorted(buckets):
                v = buckets[k]
                print(f"  {k:<18} {len(v):3d}개  성공 {sum(x[2] for x in v):3d}  "
                      f"최종 up {np.mean([x[0] for x in v]):+.3f}  "
                      f"높이 {np.mean([x[1] for x in v]) * 100:5.1f}cm")
            continue

        print(f"{'시작각':>6} {'성공률':>8} {'최종 up':>16} {'최종 높이(cm)':>16} {'최고 up':>9}")

        for deg in args.angles:
            ok = 0
            fin_up, fin_h, peak_up = [], [], []
            for s in range(args.seeds):
                ups, hs = ev.rollout(np.deg2rad(deg), 1000 + s, args.duration)
                tail_up, tail_h = ups[-settle:], hs[-settle:]
                if (tail_up > UP_OK).all() and (tail_h > HEIGHT_OK).all():
                    ok += 1
                fin_up.append(tail_up.mean())
                fin_h.append(tail_h.mean())
                peak_up.append(ups.max())
            print(f"{deg:5.0f}° {ok:3d}/{args.seeds:<4d} "
                  f"{np.mean(fin_up):+7.3f} ± {np.std(fin_up):.3f} "
                  f"{np.mean(fin_h) * 100:9.1f} ± {np.std(fin_h) * 100:.1f} "
                  f"{np.mean(peak_up):+8.3f}")


if __name__ == "__main__":
    main()
