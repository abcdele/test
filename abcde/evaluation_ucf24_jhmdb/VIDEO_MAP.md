# Video-mAP evaluation

`evaluate_video.py` evaluates saved frame predictions on CPU, without model
weights or inference. Existing `evaluate.py` and Frame-mAP training validation
are unchanged. NumPy, Pillow and SciPy are required (SciPy loads UCF MAT files).

## Commands

UCF101-24 official split 1; reports Video-mAP at 0.1, 0.2, 0.5, 0.75:

```bash
python evaluate_video.py --dataset ucf24 --split 1 \
  --root /path/to/ucf24 --predictions /path/to/frame_detections.json \
  --coordinate-space input --input-size 224 224 \
  --output /path/to/NEW_video_results
```

J-HMDB-21; reports Video-mAP at 0.2, 0.5, 0.75:

```bash
python evaluate_video.py --dataset jhmdb21 --split 2 \
  --root /path/to/JHMDB-21 --predictions /path/to/split2_frame_detections.json \
  --coordinate-space input --input-size 224 224 \
  --output /path/to/NEW_split2_video_results
```

Choose `--split 1`, `2`, or `3` explicitly. Each invocation evaluates one split;
there is no automatic three-split average. For original-image pixel boxes use
`--coordinate-space original` and omit `--input-size`. Input size is HEIGHT WIDTH
and must match the detector/export, not an assumed default. These input-space
adapters assume the project's direct resize, not letterboxing or cropping.

The output directory must not exist. It receives `video_metrics.json` (mAP/AP in
[0,1], per-class GT tube counts, split and protocol metadata) and
`predicted_tubes.json` (original-pixel tubes, actual one-based frames, zero-based
labels, mean confidence). No existing experiment outputs are overwritten.

## Inputs and coverage

The JSON input is the list format exported by existing `evaluate.py`:

```json
[
  {"image_id":"Basketball/v_Basketball_g01_c01/00001.jpg",
   "boxes":[[10,20,60,100]], "labels":[0], "scores":[0.9]},
  {"image_id":"Basketball/v_Basketball_g01_c01/00002.jpg",
   "boxes":[], "labels":[], "scores":[]}
]
```

Supply **all frames of all test videos**, including background frames and frames
with no detections. A missing record is an error, not an implicit empty frame.
Extra videos, wrong splits, repeated frames, nonfinite values, invalid class IDs,
invalid boxes and inconsistent shapes are rejected. Input labels are zero-based;
annotation files are one-based. Scores are the final per-class confidence
exported by the detector; the adapter does not multiply actionness again.
Frames are parsed numerically, grouped by full class/video path, and validated
as complete 1..N sequences. Prediction row order has no effect on frame order.
The adapter reads the first image of each video for original dimensions, as does
the dataset convention of fixed dimensions within a video.

UCF layout:

- `splitfiles/testlist01.txt`: one `class/video` per test video.
- `splitfiles/finalAnnots.mat`: `annot` records with `num_imgs`, `name`, `tubes`.
  Each tube has `sf`, `ef`, `class`, `boxes`; `boxes` is xywh. The adapter casts to
  float before addition (preventing uint8 overflow), then uses x2=x+w, y2=y+h,
  matching the reference. It preserves each actor instance and temporal extent.
  `--annotations` can specify another compatible MAT file explicitly.
- `rgb-images/class/video/00001.jpg` (PNG/JPEG also supported).

J-HMDB layout:

- `splitfiles/splits/<action>_test_split{1,2,3}.txt`: official filename and flag
  (1=train, 2=test). Class names follow the reference's 21-class ordering.
- `labels/class/video/00001.txt`: one annotated actor per frame, in the project's
  `class_id x1 y1 x2 y2` original-pixel format. These trimmed videos form one GT
  tube each. Some RGB videos include unannotated frames after the action interval;
  those label files are absent and the GT tube ends at its last annotated frame.
  Missing labels inside the annotated interval, multiple actors or mismatched
  classes fail instead of guessing identity associations. The differently
  structured J-HMDB `finalAnnots.mat` is deliberately not interpreted as UCF MAT.
- `rgb-images/class/video/00001.png` (JPG/JPEG also supported).

Older frame caches may only cover annotated frames or another split. They are
not silently treated as full-video predictions; export the correct full test
frame list before computing Video-mAP. This command cannot verify the model's
training split from a bare prediction file; callers must supply predictions from
the corresponding split's model.

## Reference and compatibility fixes

The existing project contained only Frame-mAP. We reuse the public
[wei-tim/YOWO](https://github.com/wei-tim/YOWO/tree/9d202c1694dfa883e67fd8c5455a2b7ad779684f)
Video-AP protocol, not a newly invented tracking metric. The adapted functions in
`yowo_linking.py` come from `core/eval_results.py` and `core/utils.py`, under
CC BY-NC 4.0; full license is in `YOWO_LICENSE.txt`.

- Per-video, per-class Viterbi association uses edge score
  `IoU + s1 + s2 + 0.5*s1*s2`, nearest-frame filling for explicitly empty frames,
  at most three paths, and tube NMS at 0.3. There is no test-GT-based temporal
  trimming. This is the reference linking choice, not an assertion that every
  paper uses identical tube generation. Linking choices affect reported scores.
- Tube score is mean frame confidence. Link-edge spatial IoU uses continuous
  widths, as in reference `bbox_iou`; evaluation spatial IoU uses inclusive-pixel
  `+1` widths, as in reference `iou3dt`. Predictions and GT use original pixels.
- Spatiotemporal IoU = mean per-frame spatial IoU on the temporal intersection
  multiplied by inclusive temporal intersection/union. It is not volume IoU.
- Predictions are globally sorted by score per class, matched to the best
  remaining GT in the same video, each GT at most once. AP is integral VOC AP
  with precision envelope (not VOC07 11-point). Full official splits include all
  classes; absent classes in synthetic/subset inputs are reported as null and
  excluded from the mean. A present class with no predictions gets AP=0.
- Corrected reference edge cases: single-frame intersections count; actual frame
  IDs survive association; empty predictions are valid; GT is not mutated across
  thresholds; single-frame paths rank by confidence; NumPy deprecated aliases
  and unused stale edge indexing are removed. Exact recall uses GT count without
  the reference's unnecessary `1e-5` epsilon, so perfect AP is exactly 1.

## CPU tests

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m unittest tests.test_video_map -v
```

Tests use synthetic images, annotations and detections in temporary directories.
No checkpoint, CUDA, training process or existing result is accessed by tests.
