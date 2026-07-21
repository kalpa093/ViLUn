# ViLUn: Villain-Guided Learning for Unlearning

This repository is the experimental artifact for ViLUn. It contains the code
used for the image-domain main results, privacy evaluation, architecture and
configuration studies, the Qwen2.5-based extension, small forget-set analysis,
and the requester-side training ablation.

ViLUn keeps the forget set with the requester. The requester trains a compact
villain model and transfers only its headless backbone; the model owner then
uses feature-space orthogonal repulsion without receiving raw forget samples.

## Environment

The experiments were developed on Ubuntu 22.04 with Python 3.9.7 and three
NVIDIA RTX 4090 GPUs. The tested package versions are pinned in
`requirements.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Install a PyTorch build compatible with the local CUDA driver when the pinned
wheel is unavailable for the target platform. The Qwen2.5 experiment also
requires access to the selected HuggingFace checkpoint and sufficient GPU
memory.

## Data

Scripts use `./data` by default. CIFAR-10 and CIFAR-100 are downloaded by
TorchVision. TinyImageNet must use the following standard layout:

```text
data/tiny-imagenet-200/
  train/
  val/
  wnids.txt
```

Generated checkpoints, indices, histories, and logs are written below
`./vilun_runs` and are excluded from Git.

## Main Evaluation

The main image-domain setting uses ResNet18 for CIFAR-10, ResNet50 for
CIFAR-100 and TinyImageNet, and an RNN villain. All reported unlearning
checkpoints use the final fixed epoch rather than selecting an epoch with the
Retrain oracle.

Run the stages in order:

```bash
bash scripts/run_00_prepare.sh
bash scripts/run_01_baselines.sh
bash scripts/run_02_main_sample.sh
bash scripts/run_05_privacy_mia.sh
```

The stages respectively prepare original, villain, and Retrain models; run GA,
SISA, SalUn, PS, and DELETE; run ViLUn and ViLUn_f; and evaluate output- and
representation-based membership leakage for the unlearned and villain models.

Default settings can be overridden through environment variables:

```bash
SEED=42 GPUS="0 1 2" DATA_DIR=./data ROOT_DIR=./vilun_runs \
  bash scripts/run_02_main_sample.sh
```

Important defaults are `alpha=1`, `beta=2`, feature margin `-0.2`, held-out
ratio `0.05`, retain-subset ratio `0.05`, unlearning learning rate `1e-4`, and
50 owner-side update epochs. Baseline learning rates and training budgets are
defined explicitly in `scripts/common.sh` and match the paper appendix.

## Multi-Seed Statistics

The statistical evaluation repeats the complete main pipeline with paired
seeds. Every seed has an isolated output directory and completed stages are
resumable.

```bash
SEEDS="42 43 44 45 46" GPUS="0 1 2" bash run_multiseed_main.sh
```

To run only new seeds and analyze later:

```bash
SEEDS="43 44 45 46" GPUS="0 1 2" RUN_ANALYSIS=0 \
  bash run_multiseed_main.sh

python3 analyze_multiseed_main.py \
  --root ./vilun_runs/multiseed \
  --seeds 42 43 44 45 46
```

The analyzer reports sample mean, standard deviation, variance, and 95% CI for
accuracy and privacy metrics. It computes seed-matched Total MAE to Retrain,
uses dataset-seed pairs as blocks in a Friedman test, and applies two-sided
paired Wilcoxon tests against ViLUn with Holm correction and paired
rank-biserial effect sizes.

Outputs are written to `vilun_runs/multiseed/analysis/`:

```text
main_accuracy_mean_std.csv
main_privacy_mean_std.csv
main_overall_total_mae_mean_std.csv
friedman_total_mae.csv
wilcoxon_holm_total_mae.csv
```

## Additional Experiments

### Configuration sensitivity

Runs the alpha-beta grid, held-out ratio study, and retain-subset study for the
settings reported in the appendix.

```bash
bash scripts/run_03_config_sensitivity.sh
```

### Architecture robustness

Runs the CIFAR-10 `5 x 5` original/villain architecture grid for ViLUn.

```bash
bash scripts/run_04_architecture.sh
```

### Requester-side training ablation and class-level scope

```bash
bash scripts/run_06_ablation_scope.sh
```

The two components can be selected independently:

```bash
RUN_CLASS_LEVEL=0 bash scripts/run_06_ablation_scope.sh
RUN_RANDOM_VILLAIN=0 bash scripts/run_06_ablation_scope.sh
```

### Qwen2.5-based classifier

Prepares one shared Qwen2.5 original checkpoint and runs Retrain, ViLUn, and
ViLUn_f with CNN, MLP, RNN, ResNet18, and ResNet50 villains.

```bash
bash scripts/run_07_llm.sh
```

The default model is `Qwen/Qwen2.5-3B` with eight transformer layers, matching
the paper implementation.

### Small forget sets

Runs ViLUn and ViLUn_f at forget ratios of 1%, 0.5%, 0.1%, 0.05%, and 0.01%,
followed by ratio-matched Retrain references.

```bash
bash run_small_forget.sh
bash run_small_forget_retrain.sh
```

This experiment uses `alpha=1` and `beta=8` by default.

## Repository Structure

```text
vilun.py, vilun_f.py       ViLUn and direct-forget ViLUn_f
pretrain_models.py         original and villain preparation
retrain_ga.py              Retrain and Gradient Ascent
sisa_train.py              SISA
salun.py                    SalUn
ps.py                       Prototype Surgery
delete.py                   DELETE
evaluate_mia.py             output-, representation-, and gradient-based MIA
vilun_llm.py                Qwen2.5-based image-classification extension
scripts/                    paper experiment launchers
```

ViLUn optimizes

```text
alpha * L_retain + beta * L_hrep
```

where `L_retain` is KL consistency with the frozen original model and `L_hrep`
is held-out feature-space repulsion from the headless villain. ViLUn_f uses the
same retain term but computes the repulsion term directly on forget samples.
