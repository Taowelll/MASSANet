# MASSANet

Official research code release for the paper:

**MASSANet: Multiscale adaptive spectral-spatial dual-path attention network for motor imagery decoding from EEG**

This repository contains the training, evaluation, model, and preprocessing code aligned with the paper settings.

## Recommended environment

- Python: `3.12.4`
- PyTorch: `2.7.0`

## Datasets

- BCI Competition IV 2a: [https://www.bbci.de/competition/iv/#datasets](https://www.bbci.de/competition/iv/#datasets)
- BCI Competition IV 2b: [https://www.bbci.de/competition/iv/#datasets](https://www.bbci.de/competition/iv/#datasets)
- OpenBMI: [http://gigadb.org/dataset/view/id/100542/File_page](http://gigadb.org/dataset/view/id/100542/File_page)


## Installation

Install the basic dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

If you use OpenBMI through MOABB, also install:

```bash
pip install moabb
```

## Repository structure

- `training.py`: training entry point
- `evaluation.py`: checkpoint evaluation script
- `configs/`: dataset-specific experiment settings
- `model/`: MASSANet backbone and Lightning wrapper
- `preprocessing/`: dataset loading and augmentation
- `utils/`: setup and training helpers

## Configuration

- Set experiment parameters in `configs/*.yaml`.
- Set the model wrapper in `model/litmodel.py`.
- Choose the dataset configuration with `--config_name`.

Supported configuration names:

- `bcicompet2a_config`
- `bcicompet2b_config`
- `OpenBMI_config`

## Training

Train all subjects for BCI Competition IV 2b:

```bash
python training.py --config_name bcicompet2b_config
```

Notes:

- `subject_num` follows the checkpoint naming convention used in code, such as `S00`, `S01`, and so on.
- `fold_num` is one-based, from `1` to `10`.

## Evaluation

Evaluate a trained run:

```bash
python evaluation.py --config_name bcicompet2b_config --ckpt_path <run_name>
```

## Citation

If you find this repository useful, please cite:

```bibtex
@article{LUO2026104939,
  title = {MASSANet: Multiscale adaptive spectral-spatial dual-path attention network for motor imagery decoding from EEG},
  journal = {Information Processing \& Management},
  volume = {63},
  number = {8},
  pages = {104939},
  year = {2026},
  issn = {0306-4573},
  doi = {10.1016/j.ipm.2026.104939},
  url = {https://www.sciencedirect.com/science/article/pii/S0306457326003304}
}
```
