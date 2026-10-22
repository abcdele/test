# MIA Video Action Detection

This repository provides the paper's main model, training, inference, Frame-mAP and Video-mAP paths. It retains the ResNet-50 Kinetics-400 initialization, MIA temporal alignment, Interaction Token fusion, the dual Gaussian Mask, and the bidirectional Mamba detection head. Default training settings are 7 epochs for UCF101-24 and 10 for J-HMDB-21, with `batch_size=6` and `accumulation_steps=18`.

## Installation

Use Python 3.10 and install a PyTorch/torchvision build matching the local CUDA version. Then install the project dependencies:

```bash
pip install -r requirements.txt
pip install --no-build-isolation -r requirements-cuda.txt
```

Place the OpenMMLab Kinetics-400 TSN ResNet-50 checkpoint at the configured default path:

```bash
mkdir -p pretrained
wget -O pretrained/tsn_r50_8xb32_1x1x8_100e_kinetics400_rgb_20220818-2692d16c.pth \
  https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_8xb32_1x1x8_100e_kinetics400_rgb/tsn_r50_8xb32_1x1x8_100e_kinetics400_rgb_20220818-2692d16c.pth
```

The default configuration requires a compatible pretrained checkpoint. Normal CUDA training uses the official Mamba backend.

## Dataset files and official splits

Prepare the UCF101-24 and J-HMDB-21 datasets using their official training and testing partitions.

Expected data layout:

```text
data_root/
  rgb-images/<class>/<video>/<frame>.jpg|png
  labels/<class>/<video>/<frame>.txt
```

Each nonempty annotation file contains `class_id x1 y1 x2 y2`, with 1-based class IDs and pixel coordinates.

Users should prepare `train.txt` and `test.txt` according to the official dataset partitions. Each line contains a frame path relative to `rgb-images/`, for example:

```text
Basketball/v_Basketball_g01_c01/00001.jpg
Basketball/v_Basketball_g01_c01/00002.jpg
```

Frames listed for training or Frame-mAP evaluation must have corresponding annotation files.

UCF101-24 uses its official training and testing lists. Corrected annotations for Video-mAP evaluation are available from [corrected-UCF101-Annots](https://github.com/gurkirt/corrected-UCF101-Annots).

J-HMDB-21 uses three official splits, each trained and evaluated independently.

## Training and per-epoch evaluation

The historical selection procedure trains on the official training list and evaluates on the official test list after every epoch. `best.pt` is selected by that test-set Frame-mAP. The test-selected checkpoint is not an independent validation result; the official test scores have been used for model selection.

UCF101-24:

```bash
python main.py --cfg cfg/ucf24.json --root /path/to/ucf24 \
  --train-list /path/to/ucf24_split1_lists/train.txt \
  --test-list /path/to/ucf24_split1_lists/test.txt \
  --annotation-version ucf24-existing-labels \
  --output runs/ucf24_split1
```

J-HMDB-21, run separately for each official split:

```bash
for split in 1 2 3; do
  python main.py --cfg cfg/jhmdb21.json --root /path/to/JHMDB-21 \
    --train-list "/path/to/jhmdb21_split${split}_lists/train.txt" \
    --test-list "/path/to/jhmdb21_split${split}_lists/test.txt" \
    --annotation-version "split${split}-existing-labels" \
    --output "runs/jhmdb21_split${split}"
done
```

## Frame-mAP evaluation

Frame-mAP uses the supplied frame list and defaults to `<root>/labels`. An alternate label tree can be selected explicitly with `--labels-root`:

```bash
python evaluate.py --checkpoint runs/ucf24_split1/best.pt --root /path/to/ucf24 \
  --list /path/to/ucf24_split1_lists/test.txt \
  --annotation-version ucf24-existing-labels \
  --output results/ucf24_frame_map --export-voc
```

This evaluates the official test frames; if the same test list was used each epoch to select `best.pt`, report the resulting score as a test-set model-selection score, not independent validation. Detection class IDs in exported JSON are zero-based.

## Standard Video-mAP workflow

Run `predict_video.py` on all RGB frames in the official test videos, including frames without label files, then pass its JSON to `evaluate_video.py`. UCF101-24 supports Video-mAP@0.1, 0.2, 0.5 and 0.75. J-HMDB-21 supports @0.2, 0.5 and 0.75; evaluate its three official splits separately.

```bash
RUN_DIR=/path/to/new/ucf24_video_map_run
mkdir -p "$RUN_DIR"
python predict_video.py --checkpoint /path/to/ucf24_best.pt \
  --dataset ucf24 --split 1 --root /path/to/ucf24 \
  --output "$RUN_DIR/frame_detections.json" --device cuda --batch-size 1 --workers 0
python evaluate_video.py --dataset ucf24 --split 1 --root /path/to/ucf24 \
  --predictions "$RUN_DIR/frame_detections.json" \
  --coordinate-space input --input-size 224 224 \
  --output "$RUN_DIR/evaluation"
```

For J-HMDB-21, use a checkpoint and new output directory for each split, changing `--split` to 1, 2 or 3. The full protocol is documented in [Video-mAP details](evaluation_ucf24_jhmdb/VIDEO_MAP.md).

