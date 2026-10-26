# CoBand: RGB-Guided Complementary Frequency Band Routing for Efficient RGB-Thermal Fusion in Semantic Segmentation

Official code of the paper (Information Fusion, under review).

CoBand fuses RGB and thermal features at the two high-resolution encoder stages. It splits the thermal feature
with a fixed Gaussian low-pass filter into a low band (large warm regions) and a high band (edges and thin
structures), refines each band with its own spatial attention, and mixes the two bands with a per-channel
gate that is conditioned on the RGB features. The deep stages use a lightweight semantic fusion block (LSF)
and a PANet decoder; both encoders are ConvNeXt V2.

```
model.py          CoBandNet, CoBand module, FGF, LSF, ablation fusions, PANet decoder
losses.py         training objective
data/             MFNet, PST900 and IRON datasets
train.py          training
evaluate.py       single-scale test-set evaluation (paper protocol)
benchmark.py      parameters, MACs, latency
scripts/          training commands of every model in the paper
configs/          configurations of the paper models
tools/            IRON split files
```

## Installation

```bash
conda create -n coband python=3.12 && conda activate coband
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

The ImageNet weights of the ConvNeXt V2 encoders are downloaded by `timm` on first use.

## Datasets

**MFNet** (480 x 640, 9 classes). Download the dataset release used by
[RTFNet](https://github.com/yuxiangsun/RTFNet) (`images/`, `labels/`, `train.txt`, `val.txt`, `test.txt`).
As in prior work we train on the training and validation images:

```bash
cat train.txt val.txt > train_.txt
```

**PST900** (720 x 1280, 5 classes). Download from the
[official repository](https://github.com/ShreyasSkandanS/pst900_thermal_rgb); the expected layout is
`train/{rgb,thermal,labels}` and `test/{rgb,thermal,labels}`.

**IRON** (512 x 640, binary freespace). Download from the [official repository](https://github.com/wsnbws/IRON)
(`aligned_rgb/`, `images/`, `anns/`) and create the sequence split of the dataset paper:

```bash
python tools/make_iron_splits.py --data_root /path/to/IRON_Dataset
```

## Trained models

| Dataset | Model | Encoders | Result (single scale) | Weights |
|---|---|---|---|---|
| MFNet | CoBand-N | ConvNeXt V2-Nano | 62.14 mIoU (day 53.79, night 63.87) | coming soon |
| PST900 | CoBand-N | ConvNeXt V2-Nano | 86.51 mIoU | coming soon |
| PST900 | CoBand-T | ConvNeXt V2-Tiny | 87.45 mIoU | coming soon |
| PST900 | CoBand-B | ConvNeXt V2-Base | 89.43 mIoU | coming soon |
| IRON | CoBand-N | ConvNeXt V2-Nano | 85.45 freespace IoU (low light 77.69), 91.26 mIoU | coming soon |

Each model is a folder with `config.json` and `best_model.ema.pth` (EMA weights).

## Evaluation

```bash
python evaluate.py --run weights/mfnet_coband --data_root /path/to/MFNet
```

`--run` also accepts the experiment folder written by `train.py`. Inference uses fp16 autocast; models
trained with bf16 (`amp_dtype: bf16`, e.g. CoBand-B) are evaluated in fp32. For MFNet the script also reports
day and night images, for IRON the freespace IoU on all frames, per scene and on low-light frames (mean RGB
intensity below 0.2).

## Training

```bash
MFNET_ROOT=/path/to/MFNet           bash scripts/train_mfnet.sh  coband      0   # <variant> <gpu>
PST900_ROOT=/path/to/PST900         bash scripts/train_pst900.sh coband_nano 0
IRON_ROOT=/path/to/IRON_Dataset     bash scripts/train_iron.sh   coband      0
```

The variants of each script are the models of the paper (main models, single-modality references and
ablations); the arguments are taken from the configurations of our runs. The gate ablations on IRON keep the
CoBand filters and change only the band gate (`--coband_gate`): `no_rgb_context` (gate on the band statistics
only), `static_alpha` (one learned weight per channel that sees no input) and `stop_gradient` (RGB context
without gradient into the RGB encoder). Runs are written to
`experiments/<dataset>_<variant>/` (`config.json`, `log.txt`, TensorBoard logs and the checkpoint with the
best test mIoU, `checkpoints/best_model.ema.pth`). As in most prior work on these benchmarks, and as stated in
the paper, the reported checkpoint is the one with the best test mIoU. Each script trains for the number of
epochs of our run; where a run was stopped earlier or a shorter budget is stated in the paper, the script uses
that number of epochs (it always contains the reported epoch).

### Training pipeline of the paper models

All models of the paper were trained with `--paper_pipeline`, which switches on five details of our training
code:

| Option | Effect |
|---|---|
| `--rank_weighted_lovasz` | the Lovasz term is the rank-weighted absolute-error term described in the paper, not Lovasz-Softmax |
| `--strong_gauss_noise` | Gaussian noise with standard deviation in [0.20, 0.44] of the intensity range |
| `--ema_params_only` | the EMA tracks the parameters; BatchNorm statistics are those of the raw model |
| `--ir_lr_by_block` | layer-wise learning-rate decay of the thermal encoder follows the block index within a stage rather than the stage |
| `--no_worker_reseed` | the random generators of the augmentation pipelines are not reseeded in the DataLoader workers |

The last option has two consequences. With albumentations 2.x every augmentation pipeline has its own random
generator; without reseeding, all DataLoader workers start from the same state, so the workers produce the same
sequence of augmentations and this sequence repeats every epoch. In addition, albumentations initializes these
generators from operating-system entropy and not from `--seed`, so two runs with `--paper_pipeline` and the same
seed see different augmentations. Retraining therefore reproduces the procedure but not the exact numbers of the
paper; the released weights reproduce them exactly. Without `--paper_pipeline`, every worker and epoch gets its
own augmentation stream derived from `--seed`, and runs are reproducible up to the non-determinism of some
CUDA kernels.

## Efficiency

```bash
python benchmark.py --config configs/mfnet_coband.json
```

Batch 1, 50 warm-up and 300 timed iterations; reports parameters, MACs (with `thop`) and latency in fp32 and fp16.

## Citation

```bibtex
@article{canitez2026coband,
  title   = {CoBand: RGB-Guided Complementary Frequency Band Routing for Efficient RGB-Thermal Fusion in Semantic Segmentation},
  author  = {Can{\i}tez, {\.I}smail Emre and Erkent, {\"O}zg{\"u}r},
  journal = {Information Fusion},
  year    = {2026},
  note    = {under review}
}
```

## Acknowledgements

The encoders come from [timm](https://github.com/huggingface/pytorch-image-models) (ConvNeXt V2). We thank the
authors of MFNet, PST900 and IRON for releasing their datasets.
