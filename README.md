# MCC-TTA: Mitigating Class Confusion for Federated Test-Time Adaptation of Vision-Language Models

This repository contains the official implementation of **MCC-TTA**, accepted to ACCV 2026. MCC-TTA is a CLIP-based federated test-time adaptation method for reducing class confusion under domain and corruption shifts. The main method is implemented in `mcc_tta_runner.py`. The runner supports the CLIP backbones `RN50` and `ViT-B/16`; both backbones use the same dataset-specific configuration files.

## Log

- Main MCC-TTA runner and configs are provided.
- Baseline runners are organized under `baseline/`.
- Dataset preparation and run commands are documented below.

## Prepare Data

### Download Or Generate Data

**VLCS and TerraIncognita**

We use the DomainBed format for VLCS and TerraIncognita. Please prepare these datasets following the [DomainBed](https://github.com/facebookresearch/DomainBed) project.

**CIFAR-10-C and CIFAR-100-C**

For corruption shifts, we evaluate on CIFAR-10-C and CIFAR-100-C at severity level 5. The corruption streams are generated from the full 60,000 clean images of CIFAR-10/100 instead of only the standard 10,000 test images.

You can generate the full corruption datasets with the official CIFAR-C generation code from the [Hendrycks robustness repository](https://github.com/hendrycks/robustness/blob/master/ImageNet-C/create_c/make_cifar_c.py). This repository also includes `make_cifar_c.py` for building the full CIFAR-C files locally.

### Data Layout

Arrange the local data as follows and pass the parent directory with `--data-root`. The data itself is not included in this repository.

```text
${data_root}
|- domainbed
|  |- VLCS
|  |  |- Caltech101
|  |  |- LabelMe
|  |  |- SUN09
|  |  `- VOC2007
|  `- terra_incognita
|     |- location_100
|     |- location_38
|     |- location_43
|     `- location_46
`- corruption
   |- CIFAR-10-C-Full
   |  |- brightness.npy
   |  |- contrast.npy
   |  |- ...
   |  `- labels.npy
   `- CIFAR-100-C-Full
      |- brightness.npy
      |- contrast.npy
      |- ...
      `- labels.npy
```

## Evaluation Protocol

We evaluate MCC-TTA under two heterogeneous federated TTA scenarios:

- Domain shifts: `VLCS`, `TerraIncognita`
- Corruption shifts: `CIFAR10CFull`, `CIFAR100CFull`

For each domain benchmark, each of its four domains is split into `m = 10` clients, giving 40 clients per benchmark.

For CIFAR-C benchmarks, each of the 19 corruption streams at severity level 5 is split into clients. Each stream contains 60,000 samples, for 1,140,000 evaluated samples per benchmark. CIFAR-10-C uses `m = 10` clients per corruption and CIFAR-100-C uses `m = 3` clients per corruption.

Default synchronization frequencies:

- VLCS / TerraIncognita: `--sync-freq 10`
- CIFAR-10-C / CIFAR-100-C: `--sync-freq 100`

## Run MCC-TTA

Use `mcc_tta_runner.py` as the main entry point.

### VLCS

```bash
python mcc_tta_runner.py \
  --config configs \
  --datasets VLCS \
  --data-root ./dataset/ \
  --backbone "ViT-B/16" \
  --num-clients 10 \
  --part-rate 1.0 \
  --sync-freq 10 \
  --seed 1 \
  --cache-features
```

### TerraIncognita

```bash
python mcc_tta_runner.py \
  --config configs \
  --datasets TerraIncognita \
  --data-root ./dataset/ \
  --backbone "ViT-B/16" \
  --num-clients 10 \
  --part-rate 1.0 \
  --sync-freq 10 \
  --seed 1 \
  --cache-features
```

### CIFAR-10-C

```bash
python mcc_tta_runner.py \
  --config configs \
  --datasets CIFAR10CFull \
  --data-root ./dataset/ \
  --backbone "ViT-B/16" \
  --num-clients 10 \
  --part-rate 1.0 \
  --sync-freq 100 \
  --seed 1 \
  --cache-features
```

### CIFAR-100-C

```bash
python mcc_tta_runner.py \
  --config configs \
  --datasets CIFAR100CFull \
  --data-root ./dataset/ \
  --backbone "ViT-B/16" \
  --num-clients 3 \
  --part-rate 1.0 \
  --sync-freq 100 \
  --seed 1 \
  --cache-features
```

## Cache Image Embeddings

The runner supports `--cache-features`, which stores CLIP image embeddings under `cached_features/` to speed up repeated experiments. Cached features are local experiment artifacts and should not be uploaded to GitHub.

## Baselines

Baseline implementations are kept under `baseline/`, with their configs under `baseline/configs/`.

```text
baseline/
|- *_runner.py
`- configs/
```

Each baseline has four configs: VLCS, TerraIncognita, CIFAR-10-C, and CIFAR-100-C. The MCC-TTA runner likewise uses four dataset-specific configs shared by `RN50` and `ViT-B/16`. For `tda_runner.py` and `dmn_zs_runner.py`, local/global behavior is controlled by the corresponding config.

Baseline commands are optional for normal MCC-TTA use. A typical pattern is:

```bash
python baseline/stata_runner.py \
  --config baseline/configs \
  --datasets VLCS \
  --data-root ./dataset/ \
  --backbone "ViT-B/16" \
  --num-clients 10 \
  --part-rate 1.0 \
  --sync-freq 10 \
  --seed 1 \
  --cache-features
```

## Repository Structure

```text
MCC-TTA/
|- mcc_tta_runner.py
|- configs/
|  |- mcc_tta_vlcs.yaml
|  |- mcc_tta_terra.yaml
|  |- mcc_tta_cifar10c.yaml
|  `- mcc_tta_cifar100c.yaml
|- baseline/
|- datasets/
|- clip/
|- scripts/
|- make_cifar_c.py
|- requirements.txt
`- README.md
```

The repository contains dataset loader code, not the benchmark data. Dataset files, cached features, and generated experiment results are kept outside GitHub.

## Notes For GitHub Submission

Upload source code, configs, `requirements.txt`, and this README. Do not upload large data or generated artifacts.

Recommended files to exclude:

- `dataset/`
- `cached_features/`
- `results/`
- `wandb/`
- `__pycache__/`

## Acknowledgements

This codebase uses the [DomainBed](https://github.com/facebookresearch/DomainBed) benchmark format and the official CIFAR-C generation code from [Hendrycks robustness](https://github.com/hendrycks/robustness).
