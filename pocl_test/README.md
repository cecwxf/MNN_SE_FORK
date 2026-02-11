# MNN + PoCL (OpenCL CPU) Validation

This directory provides a reproducible way to validate that **MNN OpenCL backend** can run on **PoCL** (Portable Computing Language).

It includes:
- a small **OpenCL smoke test** (`pocl_smoke.cpp`) to prove the ICD/device is working
- small **MNN runtime creation** checks
- a tiny model test (`tiny_matmul_add.mnn`) to prove MNN can execute a simple graph on OpenCL

> These are developer validation utilities; not part of MNN public API.

---

## 1) Prerequisites

### 1.1 Install PoCL and register system ICD

You need a working OpenCL ICD loader + PoCL installed. The typical system setup is:

- `/etc/OpenCL/vendors/pocl.icd` contains an absolute path to PoCL's OpenCL library, e.g.
  ```
  /usr/local/lib64/libpocl.so.2.15.0
  ```

You can validate PoCL is visible with `clinfo` (optional) or by running the `pocl_smoke` demo after build.

### 1.2 Build tools

- `cmake`
- `ninja`
- a C++ compiler

---

## 2) How to build (one command)

From repo root:

```bash
bash pocl_test/build_opencl_pocl.sh
```

This will:
- configure MNN with OpenCL enabled
- build `libMNN.so`
- build validation executables: `pocl_smoke`, `run_mnn_opencl`, `run_mnn_opencl_model`
- run `pocl_smoke`
- run the model demo if `tiny_matmul_add.mnn` exists

### Strict validation (recommended)

To ensure there is **no silent CPU fallback** during testing, export:

```bash
export MNN_STRICT_NO_CPU_RUNTIME=1
export MNN_STRICT_OPENCL_NO_CPU_OP=1
```

---

## 3) How `tiny_matmul_add.mnn` was generated

There are many ways; the simplest reproducible path is:

1) Create a tiny ONNX model (MatMul + Add).
2) Convert ONNX -> MNN using `MNNConvert`.

### Option A: If you already have `MNNConvert`

If your build enables the converter tool, you can do:

```bash
# Example (adjust paths/tools)
MNNConvert -f ONNX --modelFile tiny_matmul_add.onnx --MNNModel tiny_matmul_add.mnn --bizCode MNN
```

### Option B: Generate ONNX with Python (example)

Create `tiny_matmul_add.onnx` using Python + onnx (pseudo example):

```python
# create a tiny graph: Y = X @ W + B
# then export as ONNX
```

(If you want, we can add a concrete python script here; keeping this repo minimal for now.)

---

## 4) How to test

### 4.1 OpenCL / PoCL smoke

After build:

```bash
./build_pocl_opencl/pocl_smoke
```

Expected: it should list a platform named "Portable Computing Language" and produce a simple numeric result.

### 4.2 Run MNN OpenCL model demo

```bash
./build_pocl_opencl/run_mnn_opencl_model ./tiny_matmul_add.mnn
```

Expected: OpenCL runtime creation logs and output values like:

- `CLRuntime creation OK.`
- `runSession rc=0`
- `y=[6.5,6.75,9]` (depends on the model constants)

---

## 5) Source files

- `pocl_smoke.cpp`: OpenCL platform/device enumeration + tiny kernel
- `run_mnn_opencl.cpp`: basic MNN OpenCL runtime path
- `run_mnn_opencl_model.cpp`: load/run a `.mnn` model on OpenCL

