# llama.cpp for the AMD XDNA2 NPU

A fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) that processes prompts on the NPU of Ryzen AI 300 and Ryzen AI Max chips and
generates tokens on the CPU. Everything else is stock llama.cpp: `llama-server` (OpenAI-compatible API), `llama-cli`, `llama-bench` and GGUF
models work as before, and the NPU is used automatically.

What you get: on Qwen3.5 models, long prompts are processed faster than on the iGPU (Vulkan or ROCm) at about half the power. Qwen3.5-4B, 4.3k-token prompt, power-saver
profile: 560 tok/s at 16 W, against 528 tok/s at 33 W for Vulkan. What you do not get: faster token generation (19 tok/s against 24 on the
iGPU), or a speedup on Gemma 4 12B, where the iGPU is faster. See [Benchmarks](#benchmarks).

## Quick start

```sh
sudo pacman -S xrt xrt-plugin-amdxdna        # Arch; other distributions: docs/backend/XDNA2.md
git clone https://github.com/mistrjirka/llama.cpp-xdna2 && cd llama.cpp-xdna2
scripts/xdna2-setup.sh                       # checks the NPU stack, installs the kernel compiler (no root, ~1.4 GB)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_XDNA2=ON && cmake --build build -j
./build/bin/llama-server -m model.gguf -c 16384    # OpenAI API at http://127.0.0.1:8080/v1
```

`llama-server --list-devices` should list `XDNA2: AMD XDNA2 NPU`. The first prompt with a new model takes 20-45 s longer while the NPU kernels
compile (cached in `~/.cache/ggml-xdna2` afterwards). To benchmark: `llama-bench -m model.gguf -fa 1 -ub 2048 -p 512,2048 -n 64` (the server
uses `-ub 2048` by default, `llama-bench` needs it spelled out).

Other distributions, all environment variables and the full method: [docs/backend/XDNA2.md](docs/backend/XDNA2.md).

## Requirements

- AMD XDNA2 NPU: Ryzen AI 300 (Strix Point, Krackan Point) or Ryzen AI Max (Strix Halo). Not XDNA1 (Ryzen 7040/8040).
- Linux 6.14+ with the `amdxdna` driver, x86-64 with AVX-512, XRT with the amdxdna plugin, Python 3.11-3.14, GCC or Clang.
- 64 MiB of lockable memory for XRT (`ulimit -Hl`); `scripts/xdna2-setup.sh` checks it.

Tested only on an HP OmniBook Ultra 14 (Ryzen AI 9 365, 32 GB), Arch Linux, kernel 7.2. The Ubuntu and Fedora steps come from AMD's driver
documentation and have not been run. Windows is not supported.

## Benchmarks

Same laptop, one engine at a time, each through its own OpenAI-compatible server and the same client. Prompt: a 4.3-4.5k-token chat request,
time to first token. Generation: 256 tokens. Power is the whole APU package (CPU, iGPU, NPU, memory controller) as reported by the amdgpu
driver, not the wall. Numbers are the second of two runs, about 5 % apart. Power-saver profile (the laptop default); the performance profile is
in the [guide](docs/backend/XDNA2.md#performance).

| Model | Engine | Prompt, tok/s at W | Generation, tok/s at W |
|---|---|---:|---:|
| Qwen3.5-4B Q4_K_M | **this fork** | **560** at 16.3 | 19.2 at 16.4 |
| | llama.cpp Vulkan (iGPU) | 528 at 33.2 | **23.6** at 32.4 |
| | llama.cpp ROCm (iGPU) | 454 at 31.0 | 22.3 at 31.4 |
| | FastFlowLM (NPU) | 479 at 12.9 | 15.2 at 12.0 |
| Ornith-1.5-9B Q4_K_M | **this fork** | **408** at 18.3 | 11.9 at 17.8 |
| | llama.cpp Vulkan (iGPU) | 307 at 30.7 | **15.1** at 29.7 |
| | llama.cpp ROCm (iGPU) | 287 at 29.7 | 13.6 at 29.4 |
| | FastFlowLM (NPU), Qwen3.5-9B | 338 at 11.8 | 9.3 at 11.5 |
| Gemma 4 12B QAT Q4_0 | this fork | 136 at 17.8 | 9.2 at 18.4 |
| | llama.cpp Vulkan (iGPU) | 261 at 29.2 | **11.3** at 29.7 |
| | llama.cpp ROCm (iGPU) | 179 at 28.8 | 10.4 at 29.2 |
| | FastFlowLM (NPU) | **278** at 16.3 | 6.8 at 14.3 |

- Qwen3.5 prompts: fastest engine, at roughly half the iGPU's energy per token (4B: 29 mJ/token against 63 for Vulkan). FastFlowLM, the NPU
  engine [Lemonade](https://github.com/lemonade-sdk/lemonade) uses on Linux, is a little more frugal on the 4B but slower.
- Gemma 4 12B prompts: slowest of the four, because its global-attention layers run on the CPU.
- Generation: the iGPU is fastest everywhere; this fork uses about half its power in power-saver, about the same in the performance profile.
- The models are not byte-identical (FastFlowLM runs its own conversions, Qwen3.5-9B in place of Ornith, a fine-tune of it), and output quality
  was not compared.

## How it works

| Work | Runs on |
|---|---|
| Weight matmuls for prompts (Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, Q8_0) | NPU |
| Attention for prompts (head size 128/256, 4 or 8 KV heads; batches of 32+ tokens, 1024+ context) | NPU |
| Other prompt work (norms, RoPE, Qwen3.5 linear attention, other weight formats, mixture-of-experts) | CPU |
| Token generation and batches of up to 8 tokens | CPU (AVX-512), reading the NPU's weight copy in place |

Generation stays on the CPU because one token stream is bound by memory bandwidth, and the cores read DRAM faster (~70 GB/s) than the NPU
(~53 GB/s). The NPU computes in bfp16 (8-bit mantissas, shared exponent per 8 values): perplexity stays within 0.2-0.6 % of the CPU build, but
outputs are not bit-identical. Kernels are generated per model shape with [mlir-aie](https://github.com/Xilinx/mlir-aie) and compiled on first
use. The backend is in `ggml/src/ggml-xdna2/`.

## Troubleshooting

- **Prompts run at about 100 tok/s instead of 500+:** the NPU could not be opened, and the log says why. Usually the locked-memory limit is
  under 64 MiB: set `LimitMEMLOCK=infinity` in the service unit, or `memlock unlimited` in `/etc/security/limits.d`, and log in again.
- **`no kernel ... and no toolchain to build it`:** run `scripts/xdna2-setup.sh`, or set `GGML_XDNA2_MLIR_AIE_ENV` to a venv with `mlir_aie` and `llvm-aie`.
- **CMake: XRT not found:** install XRT. On Ubuntu and Fedora, build [amd/xdna-driver](https://github.com/amd/xdna-driver) and run
  `source /opt/xilinx/xrt/setup.sh`, then configure again.
- **Rule out the NPU:** `GGML_XDNA2_DISABLE=1` runs the same build on the CPU only.

## Limitations

- Generation is 0-5 % slower than a plain CPU build, about 13 % slower with Q4_0 weights (Llama 3.2 3B).
- Gemma 4: attention with head size 512, and all attention in E2B and E4B, runs on the CPU, so long prompts are slow.

## License

MIT, see [LICENSE](LICENSE). Built on llama.cpp and ggml; kernels use [mlir-aie](https://github.com/Xilinx/mlir-aie) and the Peano compiler
([llvm-aie](https://github.com/Xilinx/llvm-aie)); the device is driven through [XRT](https://github.com/Xilinx/XRT).
