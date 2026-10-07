"""FlyWire 783 커넥톰을 시뮬레이터용 npz 하나로 바꾼다 (한 번만).

pandas + pyarrow 가 있는 파이썬으로 돌린다 (프로젝트 .venv 에는 없어서 시스템 파이썬):

    python flybrain/build_brain.py

입력 (flybrain/data/, 받는 곳은 flybrain/README.md):
    Completeness_783.csv       뉴런 목록. 행 순서가 곧 뉴런 번호 (Shiu et al. 2024 의 flyid2i)
    Connectivity_783.parquet   연결. 'Excitatory x Connectivity' = 부호 붙은 시냅스 수
    neuron_annotations.tsv     세포 유형·좌우 (Schlegel et al. 2024, 783 판)
출력:
    brain_783.npz   W 를 CSR(행 = 시냅스 전 뉴런)로, 뉴런별 cell_type / side / super_class
"""
import os
import sys

import numpy as np
import pandas as pd

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def main():
    comp = pd.read_csv(os.path.join(D, "Completeness_783.csv"), index_col=0)
    n = len(comp)
    con = pd.read_parquet(os.path.join(D, "Connectivity_783.parquet"),
                          columns=["Presynaptic_Index", "Postsynaptic_Index",
                                   "Excitatory x Connectivity"])
    pre = con["Presynaptic_Index"].to_numpy(np.int64)
    post = con["Postsynaptic_Index"].to_numpy(np.int32)
    w = con["Excitatory x Connectivity"].to_numpy(np.float32)
    order = np.argsort(pre, kind="stable")
    pre, post, w = pre[order], post[order], w[order]
    indptr = np.zeros(n + 1, np.int64)
    np.add.at(indptr, pre + 1, 1)
    indptr = np.cumsum(indptr)

    ann = pd.read_csv(os.path.join(D, "neuron_annotations.tsv"), sep="\t", low_memory=False)
    ann = ann.drop_duplicates("root_id").set_index("root_id")
    ann = ann.reindex(comp.index)
    def col(c):
        return ann[c].fillna("").astype(str).to_numpy().astype("U")
    out = os.path.join(D, "brain_783.npz")
    np.savez(out, indptr=indptr, indices=post, data=w, root_id=comp.index.to_numpy(np.int64),
             cell_type=col("cell_type"), hemibrain_type=col("hemibrain_type"),
             side=col("side"), super_class=col("super_class"))
    print(f"뉴런 {n} · 연결 {len(w)} · 주석 있는 뉴런 {np.mean(col('cell_type') != ''):.1%} → {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
