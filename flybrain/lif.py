"""초파리 전뇌 LIF 모델 (Shiu et al. 2024, Nature) 의 numpy 판.

원본은 Brian2 (github.com/philshiu/Drosophila_brain_model, model.py). 식과 값은 그대로 옮겼다:

    dv/dt = (v_0 - v + g) / t_mbr      (불응기 동안 정지)
    dg/dt = -g / tau                   (불응기 동안 정지)
    v > v_th  →  v = v_rst, g = 0, 불응기 t_rfc
    시냅스: 전 뉴런이 쏘면 t_dly 뒤 후 뉴런 g += w_syn × (부호 붙은 시냅스 수)
    자극: 포아송 r_poi Hz, 한 번 들어올 때마다 v += w_syn × f_poi (= 68.75 mV, 한 방에 문턱을 넘는다)

    v_0 = v_rst = -52 mV, v_th = -45 mV, t_mbr = 20 ms, tau = 5 ms, t_rfc = 2.2 ms,
    t_dly = 1.8 ms, w_syn = 0.275 mV, f_poi = 250, dt = 0.1 ms (Brian2 기본)

선형 부분은 Brian2 의 'exact' 적분처럼 정확해(解)로 한 스텝씩 민다. 자발 발화가 없는 모델이라
자극을 안 주면 뇌 전체가 조용하다 — 무엇이 움직였든 자극에서 온 것이다.

스파이크 전달은 이벤트 방식이다: 그 스텝에 쏜 뉴런의 CSR 행만 모아 더한다.
"""
import os

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "brain_783.npz")

PARAMS = dict(v_0=-52.0, v_rst=-52.0, v_th=-45.0, t_mbr=20.0, tau=5.0, t_rfc=2.2,
              t_dly=1.8, w_syn=0.275, f_poi=250.0)


class Brain:
    def __init__(self, path=DATA, dt=0.1, seed=0, arrays=None, **overrides):
        p = dict(PARAMS, **overrides)
        self.p, self.dt = p, dt
        z = arrays if arrays is not None else np.load(path, allow_pickle=False)
        self.indptr, self.indices = z["indptr"], z["indices"].astype(np.int32)
        self.w = (z["data"] * p["w_syn"]).astype(np.float32)
        self.cell_type, self.side = z["cell_type"], z["side"]
        self.hemibrain_type = z["hemibrain_type"] if "hemibrain_type" in z else self.cell_type
        self.n = len(self.indptr) - 1
        self.rng = np.random.default_rng(seed)

        # 정확해 계수 (u = v - v_0)
        a = np.exp(-dt / p["t_mbr"])
        b = np.exp(-dt / p["tau"])
        self.a, self.b = np.float32(a), np.float32(b)
        self.c = np.float32(p["tau"] / (p["tau"] - p["t_mbr"]) * (b - a))
        self.ref_steps = int(round(p["t_rfc"] / dt))
        self.delay_steps = int(round(p["t_dly"] / dt))
        self.poi_kick = np.float32(p["w_syn"] * p["f_poi"])
        self.reset()

    # ── 상태 ──────────────────────────────────────────────────────────────
    def reset(self):
        self.u = np.zeros(self.n, np.float32)      # v - v_0
        self.g = np.zeros(self.n, np.float32)
        self.ref = np.zeros(self.n, np.int16)      # 남은 불응기 스텝
        self.queue = [np.empty(0, np.int64) for _ in range(self.delay_steps)]
        self.t = 0
        self.stim_idx = np.empty(0, np.int64)
        self.stim_p = np.empty(0, np.float32)

    def find(self, name, side=None):
        """세포 유형 이름(cell_type, 없으면 hemibrain_type)과 좌우로 뉴런 번호를 찾는다."""
        m = self.cell_type == name
        if not m.any():
            m = self.hemibrain_type == name
        if side is not None:
            m &= self.side == side
        return np.nonzero(m)[0]

    def set_stim(self, rates):
        """rates: {뉴런 번호 배열(또는 튜플 키): Hz}. 이전 자극은 지운다."""
        idx, pr = [], []
        for ids, hz in rates:
            if hz <= 0 or len(ids) == 0:
                continue
            idx.append(np.asarray(ids, np.int64))
            pr.append(np.full(len(ids), hz * self.dt * 1e-3, np.float32))
        self.stim_idx = np.concatenate(idx) if idx else np.empty(0, np.int64)
        self.stim_p = np.concatenate(pr) if pr else np.empty(0, np.float32)

    # ── 적분 ──────────────────────────────────────────────────────────────
    def _deliver(self, spk):
        if len(spk) == 0:
            return
        s, e = self.indptr[spk], self.indptr[spk + 1]
        lens = e - s
        tot = int(lens.sum())
        if tot == 0:
            return
        off = np.repeat(s - np.concatenate(([0], np.cumsum(lens)[:-1])), lens)
        k = off + np.arange(tot)
        post = self.indices[k]
        if tot < 20000:
            np.add.at(self.g, post, self.w[k])
        else:
            self.g += np.bincount(post, weights=self.w[k], minlength=self.n).astype(np.float32)

    def step(self):
        """dt 한 스텝. 이번 스텝에 쏜 뉴런 번호를 돌려준다."""
        # 지연 끝난 스파이크 도착
        self._deliver(self.queue[self.t % self.delay_steps])
        # 막전위 적분 (불응기 아닌 뉴런만)
        free = self.ref == 0
        u, g = self.u, self.g
        u_new = u * self.a + g * self.c
        np.copyto(u, u_new, where=free)
        np.multiply(g, self.b, out=g, where=free)
        self.ref[~free] -= 1
        # 포아송 자극
        if len(self.stim_idx):
            hit = self.stim_idx[self.rng.random(len(self.stim_idx)) < self.stim_p]
            u[hit] += self.poi_kick
        # 문턱
        spk = np.nonzero((u > self.p["v_th"] - self.p["v_0"]) & (self.ref == 0))[0]
        if len(spk):
            u[spk] = self.p["v_rst"] - self.p["v_0"]
            g[spk] = 0.0
            self.ref[spk] = self.ref_steps
        self.queue[self.t % self.delay_steps] = spk
        self.t += 1
        return spk

    def run(self, ms, count=None):
        """ms 만큼 굴린다. count 를 주면 (n,) 배열에 뉴런별 스파이크 수를 더한다."""
        if count is None:
            count = np.zeros(self.n, np.int32)
        for _ in range(int(round(ms / self.dt))):
            spk = self.step()
            if len(spk):
                count[spk] += 1
        return count
