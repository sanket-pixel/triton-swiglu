---
# High-Performance Fused SwiGLU Kernel with Triton

This repository provides a complete implementation and tutorial of a high-performance fused SwiGLU kernel using OpenAI Triton. By leveraging kernel fusion, this implementation achieves a 15–40% performance improvement over a standard PyTorch version by reducing memory bandwidth bottlenecks and kernel launch overhead.

---

## What is SwiGLU?

SwiGLU is a variant of the Gated Linear Unit (GLU) activation function used in modern large language models such as LLaMA and Mixtral. It is defined as:
```angular2html
SwiGLU(x, W, V) = SiLU(x * W) * (x * V)
```


where SiLU is the Sigmoid Linear Unit activation, and ⊙ denotes element-wise multiplication.

---

## Motivation

A naive PyTorch implementation of SwiGLU performs the operations as separate steps, which introduces several performance issues:

1. **Memory Bandwidth Bottleneck**: The input tensor is read twice for the two matrix multiplications.
2. **Intermediate Tensors**: Large intermediate results are written to and read from GPU memory, slowing down execution.
3. **Kernel Launch Overhead**: Each separate operation launches a GPU kernel, which accumulates latency.

These factors prevent the GPU from reaching its maximum throughput.

---

## Kernel Fusion with Triton

This project addresses these bottlenecks by implementing a single, fused kernel in Triton that performs:

- Both matrix multiplications
- SiLU activation
- Element-wise multiplication

The input tensor is read only once from memory, intermediate results are kept in fast on-chip memory, and only the final output is written back. Triton also provides auto-tuning to find the optimal kernel configuration for a given GPU and input size.

---

## Project Structure

```

baseline_torch_swiglu.py     # Naive PyTorch implementation (baseline)
fused_triton_swiglu.py       # High-performance fused Triton kernel
benchmark.py                 # Correctness check and performance benchmark
requirements.txt             # Required Python packages
README.md                    # Project documentation

````

---

## Setup and Usage

### Create a virtual environment
```bash
python3 -m venv .venv
````

### Activate the environment

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the benchmark

```bash
python benchmark.py
```

This will verify correctness of the fused kernel and compare performance against the PyTorch baseline. Results and a performance plot are saved in the current directory.

---

## Performance

The fused Triton kernel consistently outperforms the naive PyTorch implementation for most practical input sizes. For very large input sizes (for example, M=8192), PyTorch may become slightly faster because its matmul operations rely on highly-optimized cuBLAS routines, which excel in compute-bound scenarios. However, for typical latency-sensitive workloads, the fused kernel provides a significant speedup.

---

## Key Takeaways

* Fusing operations reduces memory traffic and kernel launch overhead.
* Keeping intermediate computations in on-chip memory improves GPU utilization.
* Auto-tuning in Triton ensures the kernel runs efficiently on a variety of GPUs.
* Overall, this approach achieves a 15–40% speedup for most input sizes relevant to LLMs.

---
