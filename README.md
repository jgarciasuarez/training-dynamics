# Active-Subspace Dynamics of Gradient Descent

Code accompanying the preprint **"Active-Subspace Dynamics of Gradient Descent: A Geometric Account of Feature Learning and Weight Decay"** 
by Joaquín García Suárez (École Polytechnique Fédérale de Lausanne).

## About

This work studies feature learning in overparameterized neural networks through the dynamics of the **active parameter subspace**: the low-dimensional subspace of parameter space spanned by the network Jacobian. 
Training acts as a trajectory on the Grassmannian of parameter subspaces, driven by the loss Hessian, which in turn depends on weight decay. 
In controlled teacher–student regression experiments (full-batch gradient descent, MSE loss), weight decay keeps the active subspace rotating after the training loss has plateaued and the networks eventually generalize, whereas without weight decay the active subspace freezes and generalization fails. 
This provides a candidate account of delayed generalization (grokking) and of the association between weight decay and grokking.

## Contents

| Path | Description |
|---|---|
| `paper/JGS_Active_Subspace_2026.pdf` | Preprint (PDF). |
| `notebooks/running_saving_template.ipynb` | Single-run training template: generates the benchmark, trains one configuration, saves checkpoints and training history. |
| `notebooks/create_paper_images.ipynb` | Postprocessing: recomputes active-subspace diagnostics from the checkpoints and reproduces all data-driven figures of the paper. |

## Setup

Requires Python ≥ 3.10 (tested with Python 3.11):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Both CPU and GPU setups are supported (CUDA detected automatically); 
all diagnostics expected to run in double precision (`float64`).

## Reproducing the results

> Launch Jupyter from the repository root — the notebooks resolve all input/output paths relative to the current working directory.

1. Run `notebooks/running_saving_template.ipynb` three times, once per
   canonical configuration:

   | Run | `STUDENT_ALPHA` | `LAMBDA_REG` |
   |---|---|---|
   | weight decay, α = 2 | `2.0` | `1e-4` |
   | no weight decay, α = 2 | `2.0` | `0.0` |
   | weight decay, α = 3 | `3.0` | `1e-4` |

   Each run trains a 5→100→100→5 tanh MLP on a fixed teacher-generated
   regression benchmark (100 train / 100 test examples) for 100k steps of
   full-batch gradient descent, and writes benchmark tensors, initial weights,
   training history, and checkpoints (every 100 steps) to
   `benchmarks/omnigrok_regression/<RUN_ID>/` and
   `results/omnigrok_regression/<RUN_ID>/`.

2. Run `notebooks/create_paper_images.ipynb`. It loads the three runs,
   recomputes the active-subspace diagnostics (`D_t`, `E_t`, `C_t`, NTK
   eigenvalues, `E_t(k)`), and reproduces the main-text figures on weight
   decay and initialization scale, plus the appendix figures.

Generated `benchmarks/` and `results/` directories are git-ignored. Training
one run takes minutes on GPU (tens of minutes on CPU); the diagnostic notebook
is the more demanding step and benefits strongly from a GPU.

## Data availability

The benchmark data and trained checkpoints for the three canonical runs are archived on Zenodo (see [Citation](#citation)). 
To reproduce the paper figures without retraining, download the two archives and unpack them into `benchmarks/`and `results/` at the repository root:

```bash
mkdir -p benchmarks results
unzip benchmarks.zip -d benchmarks/
unzip results.zip -d results/
```

Then run `notebooks/create_paper_images.ipynb` directly.

## Citation

If you use this code or the preprint, please cite:

```bibtex
@misc{garcia_suarez2026active,
  author       = {Garc{\'i}a Su{\'a}rez, Joaqu{\'i}n},
  title        = {{Active-Subspace Dynamics of Gradient Descent: A Geometric
                   Account of Feature Learning and Weight Decay}},
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.23099714},
  url          = {https://doi.org/10.5281/zenodo.23099714}
}
```

## Funding

The support from the Swiss National Science Foundation (SNSF) through Ambizione Grant 216341, "Data-Driven Computational Friction" 
is gratefully acknowledged.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).
