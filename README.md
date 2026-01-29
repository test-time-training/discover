<p align="center">
  <h1 align="center">🔬 TTT-Discover</h1>
  <h3 align="center">Learning to Discover at Test Time</h3>
</p>





---

**TTT-Discover** performs reinforcement learning at test time, allowing the LLM to continue training with experience specific to the problem at hand. We achieve **new state-of-the-art** across mathematics, GPU kernels, algorithms, and biology.

<p align="center">
  <img src="assets/figure1.svg" width="800">
</p>

## Key Results

<div align="center">

|                  | **Mathematics**<br>Erdős Overlap ↓ | **Kernel A100**<br>TriMul ↓ | **Kernel H100**<br>TriMul ↓ | **Algorithms**<br>AtCoder ↑ | **Biology**<br>Denoising ↑ |
|------------------|:----------------------------------:|:---------------------------:|:---------------------------:|:---------------------------:|:--------------------------:|
| Best Human       | 0.380927                           | 4531 μs                     | 1371 μs                     | 566,997                     | 0.64                       |
| Prev. Best AI    | 0.380924                           | —                           | —                           | 558,026                     | —                          |
| **TTT-Discover** | **0.380876**                       | **2198 μs**                 | **1161 μs**                 | **567,062**                 | **0.71**                   |

</div>

## Installation

```bash
pip install -r requirements/requirements-math.txt
```

Set environment variables:

```bash
export TINKER_API_KEY="..."      
export WANDB_API_KEY="..."       
export WANDB_ENTITY="..."        
```

Task-specific requirements:
- GPU kernels: `requirements/requirements-gpumode.txt`
- AtCoder: `requirements/requirements-ale.txt`  
- Denoising: `requirements/denoising/requirements-denoising.txt` (see [README](requirements/denoising/README.md))

## Quick Start

Requires SLURM. Launch AC1 (autocorrelation inequality) on 4 nodes:

```bash
python main_tinker_submitit.py \
    --nodes 4 \
    --partition default \
    --cpus-per-task 100 \
    env=ac1 \
    model_name="openai/gpt-oss-120b" \
    sampler_type=puct_backprop \
    initial_exp_type=random \
    num_epochs=50 \
    wandb_project="my-project" \
    wandb_name="ac1-run-1"
```

Or use a preconfigured script:

```bash
bash scripts/tinker/ac1.sh
```

See [docs/launching.md](docs/launching.md) for all parameters and [docs/intro.md](docs/intro.md) for adding new tasks.

## Domains

<details>
<summary><b>Mathematics</b> — Classic open problems in combinatorics and analysis</summary>

<p align="center">
  <img src="assets/erdos.png" width="800">
</p>

<div align="center">

| Task | Erdős Min. Overlap ↓ | Autocorr. (AC1) ↓ | Autocorr. (AC2) ↑ |
|------|:--------------------:|:-----------------:|:-----------------:|
| Best Human | 0.380927 | 1.50973 | 0.9015 |
| Prev. Best AI | 0.380924 | 1.50314 | 0.9610 |
| **TTT-Discover** | **0.380876** | **1.50287** | 0.9591 |

</div>

</details>

<details>
<summary><b>Kernel Engineering</b> — GPUMode TriMul competition for triangular matrix multiplication</summary>

<div align="center">

| Task | A100 ↓ | H100 ↓ | B200 ↓ | MI300x ↓ |
|------|:------:|:------:|:------:|:--------:|
| Best Human | 4531 μs | 1371 μs | 1005 μs | 2462 μs |
| **TTT-Discover** | **2198 μs** | **1161 μs** | **905 μs** | **1596 μs** |

</div>

</details>

<details>
<summary><b>Algorithm Engineering</b> — AtCoder Heuristic Contests on real-world optimization [<a href="https://atcoder.jp/contests/ahc039/submissions/72633477">AHC39</a>] [<a href="https://atcoder.jp/contests/ahc058/submissions/72633508">AHC58</a>]</summary>

<div align="center">

| Task | AHC39 (Geometry) ↑ | AHC58 (Scheduling) ↑ |
|------|:------------------:|:--------------------:|
| Best Human | 566,997 | 847,674,723 |
| Prev. Best AI | 558,026 | 848,373,282 |
| **TTT-Discover** | **567,062** | **848,414,228** |

</div>

</details>

<details>
<summary><b>Biology</b> — Single-cell RNA-seq denoising on OpenProblems benchmark</summary>

<div align="center">

| Task | PBMC ↑ | Tabula ↑ |
|------|:------:|:--------:|
| Best Human | 0.64 | 0.64 |
| **TTT-Discover** | **0.71** | **0.73** |

</div>

</details>

## Acknowledgments

This work builds on several outstanding projects and communities:

- **[GPU Mode](https://github.com/gpu-mode)** — Community for GPU kernel optimization and the TriMul competition
- **[ALE-Bench](https://github.com/PLACEHOLDER)** — AtCoder-based benchmark for LLM evaluation
- **[AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)** — DeepMind's evolutionary coding agent
- **[OpenEvolve](https://github.com/codelion/openevolve)** — Open-source implementation of AlphaEvolve
- **[Tinker](https://github.com/PLACEHOLDER)** — LLM training recipes and RL framework

## Citation

```bibtex
@article{ttt-discover2026,
  title   = {Learning to Discover at Test Time},
  author  = {Yuksekgonul, Mert and Koceja, Daniel and Li, Xinhao 
             and Bianchi, Federico and McCaleb, Jed and Wang, Xiaolong 
             and Kautz, Jan and Choi, Yejin and Zou, James 
             and Guestrin, Carlos and Sun, Yu},
  journal = {arXiv preprint arXiv:2601.16175},
  year    = {2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

