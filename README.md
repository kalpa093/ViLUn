# ViLUn (Villain-Guided Learning for Unlearning)

This repository contains the experimental artifact for **ViLUn: Villain-Guided Learning for Forget-Free Unlearning**. ViLUn studies requester-owner machine unlearning where the requester keeps the forget data local, trains a compact headless villain backbone, and the model owner performs feature-space orthogonal repulsion without receiving raw forget samples.

## Environment

We tested the artifact under the following environment:

| Component | Version |
| --- | --- |
| OS | Ubuntu 22.04.5 LTS (jammy) |
| Kernel | Linux 6.8.0-107-generic |
| Python | 3.9.7 |
| pip | 21.2.4 |
| GPU | 3 x NVIDIA GeForce RTX 4090, 24 GB each |
| NVIDIA Driver | 535.230.02 |
| System CUDA | 12.2 |
| PyTorch | 2.7.0 |
| PyTorch CUDA | 12.6 |
| cuDNN | 90501 |

A minimal setup is:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For GPU runs, install the PyTorch build that matches your CUDA/driver setup from the official PyTorch instructions before installing the remaining packages. On the tested server, `nvidia-smi` reported CUDA 12.2, while PyTorch was built with CUDA 12.6 and `torch.cuda.is_available()` returned `True`.

The artifact intentionally keeps `requirements.txt` minimal. The code directly uses PyTorch/TorchVision, NumPy, Pillow, Optuna, and HuggingFace Transformers. `accelerate` is included because the LLM extension loads HuggingFace models with `device_map`.

```text
torch==2.7.0
torchvision==0.22.0
numpy==1.26.4
Pillow==8.4.0
optuna==4.7.0
transformers==4.49.0
accelerate==1.9.0
```

The LLM extension uses HuggingFace `transformers` and may require model access, enough GPU memory, and a configured HuggingFace cache/token depending on the selected model.

## Data

By default scripts use:

```bash
DATA_DIR=./data
ROOT_DIR=./vilun_runs
```

`torchvision` datasets such as CIFAR-10 and CIFAR-100 are downloaded automatically. TinyImageNet should be placed under `DATA_DIR` in the standard TinyImageNet layout expected by the loaders.

All outputs are written under `ROOT_DIR`:

```text
vilun_runs/
  saved_models/
  history/
  logs/
  .done/
```

The `.done` directory lets scripts skip completed jobs when rerun.

## Configuration

All shell scripts share `scripts/common.sh`. You can override defaults with environment variables:

```bash
GPUS="0 1 2" SEED=42 ROOT_DIR=./vilun_runs DATA_DIR=./data ./scripts/run_02_main_sample.sh
```

Common variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `GPUS` | `0 1 2` | GPU IDs used by the launcher |
| `DATASETS` | `cifar10 cifar100 tinyimagenet` | Main image datasets |
| `TRAIN_EPOCHS` | `200` | Original/retrain training epochs |
| `UNLEARN_EPOCHS` | `50` | ViLUn owner-side update epochs |
| `DEFAULT_ALPHA` | `1` | Retain KL weight |
| `DEFAULT_BETA` | `2` | Feature repulsion weight |
| `DEFAULT_HELDOUT_RATIO` | `0.05` | Held-out auxiliary ratio |
| `DEFAULT_RETAIN_RATIO` | `0.05` | Retain subset ratio |
| `LLM_ORIG_MODEL` | `Qwen/Qwen2.5-3B` | LLM classifier backbone for the extension |

## Running Experiments

Run everything:

```bash
chmod +x run_vilun.sh scripts/*.sh
./run_vilun.sh
```

For most users, running stage by stage is easier to monitor.

### 0. Prepare Original Models, Villains, and Retrain References

```bash
./scripts/run_00_prepare.sh
```

This stage trains or loads original models, prepares forget indices, trains compact headless villain backbones, and runs retrain references used as the oracle.

### 1. Baselines

```bash
./scripts/run_01_baselines.sh
```

Runs Gradient Ascent, SISA, SalUn, Prototype Surgery, and DELETE on the main sample-level setting.

### 2. Main Sample-Level ViLUn Results

```bash
./scripts/run_02_main_sample.sh
```

Runs privacy-preserving `ViLUn` and direct-forget `ViLUn_f` on CIFAR-10, CIFAR-100, and TinyImageNet with the default dataset/model pairs:

```text
CIFAR-10     -> ResNet18
CIFAR-100    -> ResNet50
TinyImageNet -> ResNet50
```

### 3. Configuration Sensitivity

```bash
./scripts/run_03_config_sensitivity.sh
```

Runs the alpha-beta sensitivity grid, held-out data ratio study, and retain subset scalability study.

Defaults:

```bash
ALPHA_VALUES="0.5 1 2 4"
BETA_VALUES="0.5 1 2 4"
HELDOUT_RATIOS="0.05 0.1 0.2 0.5"
RETAIN_RATIOS="0.05 0.1 0.5"
```

### 4. Architecture Robustness

```bash
./scripts/run_04_architecture.sh
```

Runs the original/villain architecture grid for the image classifiers.

### 5. Privacy and MIA

```bash
./scripts/run_05_privacy_mia.sh
```

Runs membership-inference evaluation for the unlearned models and the transferred villain artifact. The villain artifact experiment compares the full villain model and the headless villain backbone.

### 6. Ablation and Scope Analysis

```bash
./scripts/run_06_ablation_scope.sh
```

Runs the random villain ablation and the class-level scope experiment.

To run only the random villain ablation:

```bash
RUN_CLASS_LEVEL=0 ./scripts/run_06_ablation_scope.sh
```

To run only class-level scope:

```bash
RUN_RANDOM_VILLAIN=0 ./scripts/run_06_ablation_scope.sh
```

Class-level scope uses class `0` by default. Override it with:

```bash
CLASS_TARGET_ID=3 ./scripts/run_06_ablation_scope.sh
```

### 7. LLM Extension and MIA

```bash
./scripts/run_07_llm_mia.sh
```

Runs the LLM-based classifier extension and then evaluates membership
inference attacks on the saved LLM unlearned models. By default, this
uses `Qwen/Qwen2.5-3B` with compact villain architectures listed in
`LLM_EXPERTS`.

To run only the LLM MIA step after models have already been generated:

```bash
RUN_LLM_UNLEARN=0 RUN_LLM_MIA=1 ./scripts/run_07_llm_mia.sh
```

To change the LLM backbone:

```bash
LLM_ORIG_MODEL="Qwen/Qwen2.5-7B" ./scripts/run_07_llm_mia.sh
```

## Useful Single Commands

Run a small smoke test on CIFAR-10 only:

```bash
DATASETS="cifar10" GPUS="0" TRAIN_EPOCHS=1 UNLEARN_EPOCHS=1 ./scripts/run_02_main_sample.sh
```

Run only main ViLUn and skip expensive baselines:

```bash
./scripts/run_00_prepare.sh
./scripts/run_02_main_sample.sh
```

Run image experiments without the final LLM stage:

```bash
./scripts/run_00_prepare.sh
./scripts/run_01_baselines.sh
./scripts/run_02_main_sample.sh
./scripts/run_03_config_sensitivity.sh
./scripts/run_04_architecture.sh
./scripts/run_05_privacy_mia.sh
./scripts/run_06_ablation_scope.sh
```

## Output Files

Important CSV summaries are written to:

```text
${ROOT_DIR}/history/summary_retrain_ga.csv
${ROOT_DIR}/history/summary_heldout.csv
${ROOT_DIR}/history/summary_vilun.csv
${ROOT_DIR}/history/summary_mia.csv
${ROOT_DIR}/history/summary_vilun_llm.csv
${ROOT_DIR}/history/summary_mia_llm.csv
```

Per-epoch traces and logs are saved under:

```text
${ROOT_DIR}/history/
${ROOT_DIR}/logs/
```

## Notes

`ViLUn` uses:

```text
alpha * L_retain + beta * L_hrep
```

where `L_retain` is KL consistency against the frozen original model on a small retain subset and `L_hrep` is held-out feature-space orthogonal repulsion against the headless villain backbone.

`ViLUn_f` uses the same retain KL loss but applies feature repulsion directly on the forget samples:

```text
alpha * L_retain + beta * L_frep
```

The model owner receives headless villain backbone parameters rather than raw forget data or a full classification head.
