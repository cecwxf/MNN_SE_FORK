#!/usr/bin/env python3
"""Generate a tiny ONNX model: Y = X @ W + B

This is used to create a minimal, reproducible test model for validating MNN OpenCL execution on PoCL.

Outputs:
  - tiny_matmul_add.onnx (by default in repo root)

Requirements:
  - onnx
  - numpy

Install example:
  pip install onnx numpy
"""

import argparse
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tiny_matmul_add.onnx", help="Output ONNX file")
    ap.add_argument("--m", type=int, default=1)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    # Shapes
    # X: [M, K]
    # W: [K, N]
    # B: [N] (broadcast to [M, N])
    M, K, N = args.m, args.k, args.n

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [M, K])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, N])

    # Deterministic weights/bias for easy checking
    W_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    if W_np.shape != (K, N):
        # generate sequential values if user picks different K/N
        W_np = np.arange(K * N, dtype=np.float32).reshape(K, N) + 1.0

    B_np = np.array([0.5, -0.25, 1.0], dtype=np.float32)
    if B_np.shape != (N,):
        B_np = (np.arange(N, dtype=np.float32) * 0.1).astype(np.float32)

    W = numpy_helper.from_array(W_np, name="W")
    B = numpy_helper.from_array(B_np, name="B")

    matmul = helper.make_node("MatMul", ["X", "W"], ["Z"], name="matmul")
    add = helper.make_node("Add", ["Z", "B"], ["Y"], name="add")

    graph = helper.make_graph(
        nodes=[matmul, add],
        name="tiny_matmul_add",
        inputs=[X],
        outputs=[Y],
        initializer=[W, B],
    )

    model = helper.make_model(graph, producer_name="mnn_pocl_test")
    onnx.checker.check_model(model)
    onnx.save(model, args.out)
    print(f"Wrote {args.out} (X:[{M},{K}] W:[{K},{N}] B:[{N}] -> Y:[{M},{N}])")


if __name__ == "__main__":
    main()
