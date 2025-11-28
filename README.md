# dlm_optim

This repository contains the code used for our experiments on masking probability distributions in Diffusion Language Models (DLMs). The implementation supports ablations on masking expectation, masking variance, and a basic curriculum learning setup.

## Requirements

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Running Experiments

This project provides convenience scripts for running the main experiments.

### Expectation Ablation Experiments

To run experiments that vary the expected masking probability (`w`):

```bash
./run_train_p.sh
```

This script simply wraps:

```bash
python train_diffusion.py --p <value>
```

### Variance (κ) Ablation Experiments

To run experiments that vary the Beta–Binomial variance parameter `kappa`:

```bash
./run_train_kappa.sh
```

This script calls:

```bash
python train_diffusion.py --kappa <value>
```

## Training Script Arguments

You can also run experiments manually using:

```bash
python train_diffusion.py
```

The key arguments are:

- `--p` (float):  
  Expected masking probability \( w \).  
  Used for expectation ablation experiments.  
  Default: `None`.

- `--disable_log` (flag):  
  Disable logging to Weights & Biases.

- `--kappa` (int):  
  Controls the variance of the Beta--Binomial masking distribution.  
  Default: `20`.

- `--curriculum` (flag):  
  Enables curriculum learning, where the masking distribution increases over training.

## Notes

- The convenience scripts provide preset configurations used in the experiments.  
- Logging is enabled by default unless `--disable_log` is passed.  
- Curriculum learning is optional and can be activated independently of other ablations.

---

For reproducibility, the full implementation is provided as-is. Contributions and extensions are welcome.
