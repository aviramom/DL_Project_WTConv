# WTConv-CUDA: Fused CUDA Kernels for Wavelet Convolutions

Custom C++/CUDA kernels for the WTConv layer: the same outputs and gradients as the PyTorch reference, with up to **~10x lower inference latency** and **~2.4x higher training throughput** in a full ResNet-18 pipeline.

## Background

**WTConv** ([Finder et al., *Wavelet Convolutions for Large Receptive Fields*, ECCV 2024](https://arxiv.org/abs/2407.05848)) gives a CNN a large receptive field cheaply. It applies a Haar wavelet transform (DWT), runs a small depthwise convolution on each frequency sub-band, and recombines the results with the inverse transform (IDWT). Repeating this over several wavelet levels grows the receptive field, while the parameter count only grows logarithmically.

The [official implementation](https://github.com/BGU-CS-VIL/WTConv) (vendored here under `Reference/`, MIT license) builds the layer from a chain of standard PyTorch ops. Each step launches its own kernel and writes intermediate tensors to global memory, so the layer is much slower than a plain convolution with the same FLOP count.

## What this project adds

Four drop-in replacements for `WTConv2d`, each with a hand-written forward **and backward** pass:

| Implementation | Directory | Key ideas |
|---|---|---|
| C++ (CPU) | `cpp_source/` | OpenMP-parallel reference port (forward/backward) |
| `cuda` | `cuda_source/` | Direct CUDA port: separate Haar DWT, IDWT and depthwise-conv kernels |
| `cuda_opt` | `optimized_cuda_source/` | **Kernel fusion**: DWT and depthwise conv in one shared-memory tiled kernel, IDWT fused with the residual add, fused conv-backward and IDWT, parallel-reduction weight gradient |
| `cuda_opt2` | `optimized2_cuda_source/` | `cuda_opt` plus a reworked backward pass and a tuned forward kernel (32x8 blocks for coalescing, weights cached in shared memory, unrolled loops for fixed K), with padding fixes for odd spatial sizes |

Also included:
- **Model surgery** (`model_utils/model_surgery.py`): recursively swaps compatible `nn.Conv2d` layers (k > 1, `in == out` channels) in a torchvision model for any of the implementations above.
- **Correctness tests** (`tests/`): forward outputs and autograd gradients are checked against the PyTorch reference.
- **End-to-end benchmark suite** (`run_experiment_full.py`): trains and evaluates ResNet-18 for every implementation, wavelet level and seed, then aggregates the results into CSVs and plots.

## Results

ResNet-18 (10 classes) on Imagenette at 224x224, batch size 64, 6 epochs, mean over 3 seeds (10, 2, 367). The GPU is the run labelled `6000` in `old_runs/`. Source files are `full_analysis_output/inference_summary_by_level.csv` and `full_analysis_output/training_summary_by_level.csv`. Speedups are computed from those means.

**Inference latency (ms / batch, lower is better)**

| WT levels | Reference (PyTorch) | `cuda` | `cuda_opt` | `cuda_opt2` | Speedup (`cuda_opt` vs ref) |
|---|---|---|---|---|---|
| 1 | 25.48 | 6.26 | **4.81** | 4.87 | 5.3x |
| 2 | 46.06 | 7.37 | **5.69** | 5.73 | 8.1x |
| 3 | 68.14 | 8.51 | **6.35** | 6.43 | 10.7x |
| 4 | 87.58 | 11.52 | **8.39** | 8.45 | 10.4x |

**Training throughput (images / s, higher is better)**

| WT levels | Reference (PyTorch) | `cuda` | `cuda_opt` | `cuda_opt2` | Speedup (`cuda_opt2` vs ref) |
|---|---|---|---|---|---|
| 1 | 1221.8 | 1347.6 | 1464.3 | **2061.1** | 1.69x |
| 2 | 800.7 | 1115.7 | 1233.0 | **1741.9** | 2.18x |
| 3 | 618.8 | 986.6 | 1099.0 | **1518.0** | 2.45x |
| 4 | 509.1 | 794.7 | 880.8 | **1231.7** | 2.42x |

Per-step timings tell the same story. At level 4 the forward pass takes 87.9 ms for the reference and 8.8 ms for `cuda_opt2`. At level 1, `cuda_opt2`'s backward pass takes 12.8 ms against 25.4 ms for the reference. The custom kernels use more VRAM than the reference, for example 4833 MB against 3186 MB at level 4.

The reference's latency grows steeply with the number of wavelet levels, while the fused kernels grow much more slowly. Top-1 accuracy after 6 epochs is in the 64–75% range for every implementation (see the CSV for means and std).

Plots (scalability, throughput, kernel breakdown, speedup heatmaps) are in `full_analysis_output/`. Earlier runs on RTX 3090 and RTX 4090 GPUs are in `old_runs/`.

## Setup

Requirements:
- NVIDIA GPU with CUDA (compute capability 6.0+) and a matching CUDA toolkit
- GCC (Linux) or MSVC (Windows) on the PATH
- Python 3.10, PyTorch >= 2.0

Install the dependencies:

```bash
conda create -n wt_conv python=3.10 && conda activate wt_conv
pip install -r requirements.txt
```

Build all extensions. This runs `setup.py build_ext --inplace` in each source directory:

```bash
python analysis/compile_utils.py
```

You should see `SUCCESS` for `cpp_source`, `cuda_source`, `optimized_cuda_source` and `optimized2_cuda_source`.

> **Windows:** if the build fails with "Access Denied", close any Python process (Jupyter, pytest) that still holds the `.pyd` files.

## Usage

**Correctness tests.** These check outputs and gradients against PyTorch autograd:

```bash
pytest tests/
```

**Layer micro-benchmark** (forward and backward, all implementations):

```bash
python analysis/benchmark_backward.py --batch 32 --channels 128 --kernel 5 --iters 100
```

Flags: `--batch` (16), `--channels` (64), `--height`/`--width` (64), `--kernel` (5), `--levels` (1), `--iters` (50), `--warmup` (30), `--no-build`.

**Full end-to-end experiment** (train, then inference, then analysis):

```bash
# Reproduce the reported results
python run_experiment_full.py --epochs 6 --seeds 10 2 367 --levels 1 2 3 4

# Quick sanity check
python run_experiment_full.py --epochs 1 --levels 1 --impls cuda_opt2
```

Other flags: `--batch-size` (64), `--model` (`resnet18` | `resnet50`), `--impls`, `--device`, `--skip-train`, `--skip-inference`, `--no-cleanup` (keep `.pth` checkpoints), `--no-report-clean`.

> **Note:** by default the suite **wipes** `training_reports/`, `inference_reports/` and `full_analysis_output/` before it starts. Pass `--no-report-clean` to keep the existing results.

Outputs:
- `full_analysis_output/training_summary_by_level.csv`: throughput, VRAM, forward and backward ms
- `full_analysis_output/inference_summary_by_level.csv`: latency, throughput, accuracy
- `full_analysis_output/*.png`: plots

## Repository layout

```text
├── cpp_source/               C++ / OpenMP CPU extension
├── cuda_source/              Baseline CUDA extension
├── optimized_cuda_source/    Fused CUDA extension (cuda_opt)
├── optimized2_cuda_source/   Fused CUDA extension with optimized fwd/bwd (cuda_opt2)
├── Reference/                Original PyTorch WTConv (BGU-CS-VIL/WTConv, MIT)
├── model_utils/              Conv2d -> WTConv model surgery
├── full_model_tests/         ResNet training / inference / analysis scripts
├── analysis/                 Build utility, layer benchmarks, profiling
├── tests/                    pytest correctness suite
├── run_experiment_full.py    End-to-end experiment driver
├── analyze_full_results.py   Aggregates reports into CSVs and plots
├── training_reports/         Per-run training JSON (latest run)
├── inference_reports/        Inference benchmark JSON/TXT (latest run)
├── full_analysis_output/     Aggregated CSVs and plots (latest run)
└── old_runs/                 Earlier runs (RTX 3090, RTX 4090, sanity check)
```

## Authors

- **Dayan** ([@Dayan135](https://github.com/Dayan135)): project lead. Wrote the C++ and CUDA implementations, the fused `cuda_opt` kernels including the optimized backward pass, multi-level support, the auto-compilation tooling and the pytest suite.
- **Tal** ([@Talmal6](https://github.com/Talmal6)): wrote the CUDA backpropagation and the backward benchmark, rewrote the `cuda_opt2` forward and backward kernels, fixed the odd-dimension padding and added the correctness tests.
- **Omer Aviram** ([@aviramom](https://github.com/aviramom)): built the evaluation side: model surgery, the ResNet training and inference pipelines on Imagenette, the end-to-end experiment suite (`run_experiment_full.py`), the results analysis and plots, and the multi-GPU benchmark runs.

## Acknowledgements

Built on [BGU-CS-VIL/WTConv](https://github.com/BGU-CS-VIL/WTConv) by Shahaf E. Finder, Roy Amoyal, Eran Treister and Oren Freifeld (MIT License).
