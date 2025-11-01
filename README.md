# VSDLM
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17494543.svg)](https://doi.org/10.5281/zenodo.17494543) ![GitHub License](https://img.shields.io/github/license/pinto0309/vsdlm) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/PINTO0309/vsdlm)

Visual-only speech detection driven by lip movements.

There are countless situations where you can't hear the audio, and it's really frustrating.

https://github.com/user-attachments/assets/c1813290-e7a6-4ce1-b44d-4dcfad6f8837

|Variant|Size|F1|ONNX|
|:-:|:-:|:-:|:-:|
|P|112 KB|0.9502|[Download](https://github.com/PINTO0309/VSDLM/releases/download/onnx/vsdlm_p.onnx)|
|N|176 KB|0.9586|[Download](https://github.com/PINTO0309/VSDLM/releases/download/onnx/vsdlm_n.onnx)|
|S|494 KB|0.9696|[Download](https://github.com/PINTO0309/VSDLM/releases/download/onnx/vsdlm_s.onnx)|
|M|1.7 MB|0.9801|[Download](https://github.com/PINTO0309/VSDLM/releases/download/onnx/vsdlm_m.onnx)|
|L|6.4 MB|0.9891|[Download](https://github.com/PINTO0309/VSDLM/releases/download/onnx/vsdlm_l.onnx)|

## Training Pipeline

- Use the images located under `dataset/output/002_xxxx_front_yyyyyy` together with their annotations in `dataset/output/002_xxxx_front.csv`.
- Only samples with `class_id` equal to 1 (closed) or 2 (open) are used. Label 0 (unknown) is dropped, and the remaining labels are remapped to 0 (closed) and 1 (open).
- Every augmented image that originates from the same `still_image` stays in the same split to prevent leakage.
- The training loop relies on `BCEWithLogitsLoss`, `pos_weight`, and a `WeightedRandomSampler` to stabilise optimisation under class imbalance; inference produces sigmoid probabilities.
- Training history, validation metrics, optional test predictions, checkpoints, configuration JSON, and ONNX exports are produced automatically.
- Per-epoch checkpoints named like `vsdlm_epoch_0001.pt` are retained (latest 10), as well as the best checkpoints named `vsdlm_best_epoch0004_f10.9321.pt` (also latest 10).
- The backbone can be switched with `--arch_variant`. Supported combinations with `--head_variant` are:

  | `--arch_variant` | Default (`--head_variant auto`) | Explicitly selectable heads | Remarks |
  |------------------|-----------------------------|---------------------------|------|
  | `baseline`       | `avg`                       | `avg`, `avgmax_mlp`       | When using `transformer`/`mlp_mixer`, you need to adjust the height and width of the feature map so that they are divisible by `--token_mixer_grid` (if left as is, an exception will occur during ONNX conversion or inference). |
  | `inverted_se`    | `avgmax_mlp`                | `avg`, `avgmax_mlp`       | When using `transformer`/`mlp_mixer`, it is necessary to adjust `--token_mixer_grid` as above. |
  | `convnext`       | `transformer`               | `avg`, `avgmax_mlp`, `transformer`, `mlp_mixer` | For both heads, the grid must be divisible by the feature map (default `3x2` fits with 30x48 input). |
- The classification head is selected with `--head_variant` (`avg`, `avgmax_mlp`, `transformer`, `mlp_mixer`, or `auto` which derives a sensible default from the backbone).
- Mixed precision can be enabled with `--use_amp` when CUDA is available.
- Resume training with `--resume path/to/vsdlm_epoch_XXXX.pt`; all optimiser/scheduler/AMP states and history are restored.
- Loss/accuracy/F1 metrics are logged to TensorBoard under `output_dir`, and `tqdm` progress bars expose per-epoch progress for train/val/test loops.

### 1. Training

Baseline depthwise-separable CNN:

```bash
uv run python -m vsdlm train \
--data_root dataset/data \
--output_dir runs/vsdlm \
--epochs 50 \
--batch_size 256 \
--train_ratio 0.8 \
--val_ratio 0.2 \
--image_size 30x48 \
--base_channels 32 \
--num_blocks 4 \
--arch_variant baseline \
--seed 42 \
--device auto \
--use_amp
```

Inverted residual + SE variant (recommended for higher capacity):

```bash
uv run python -m vsdlm train \
--data_root dataset/data \
--output_dir runs/vsdlm_is \
--epochs 50 \
--batch_size 256 \
--train_ratio 0.8 \
--val_ratio 0.2 \
--image_size 30x48 \
--base_channels 32 \
--num_blocks 4 \
--arch_variant inverted_se \
--head_variant avgmax_mlp \
--seed 42 \
--device auto \
--use_amp
```

ConvNeXt-style backbone with transformer head over pooled tokens:

```bash
uv run python -m vsdlm train \
--data_root dataset/data \
--output_dir runs/vsdlm_convnext \
--epochs 50 \
--batch_size 256 \
--train_ratio 0.8 \
--val_ratio 0.2 \
--image_size 30x48 \
--base_channels 32 \
--num_blocks 4 \
--arch_variant convnext \
--head_variant transformer \
--token_mixer_grid 3x2 \
--seed 42 \
--device auto \
--use_amp
```

- Outputs include the latest 10 `vsdlm_epoch_*.pt`, the latest 10 `vsdlm_best_epochXXXX_f1YYYY.pt` (highest validation F1, or training F1 when no validation split), `history.json`, `summary.json`, optional `test_predictions.csv`, and `train.log`.
- After every epoch a confusion matrix and ROC curve are saved under `runs/vsdlm/diagnostics/<split>/confusion_<split>_epochXXXX.png` and `roc_<split>_epochXXXX.png`.
- `--image_size` accepts either a single integer for square crops (e.g. `--image_size 48`) or `HEIGHTxWIDTH` to resize non-square frames (e.g. `--image_size 64x48`).
- Add `--resume <checkpoint>` to continue from an earlier epoch. Remember that `--epochs` indicates the desired total epoch count (e.g. resuming `--epochs 40` after training to epoch 30 will run 10 additional epochs).
- Launch TensorBoard with:
  ```bash
  tensorboard --logdir runs/vsdlm
  ```

### 2. Inference

```bash
uv run python -m vsdlm predict \
--checkpoint runs/vsdlm/vsdlm_best_epoch0004_f10.9321.pt \
--inputs dataset/output/002_0005_front_028001 \
--output runs/vsdlm/predictions.csv
```

- `--inputs` accepts files and/or directories (recursively scanned).
- The resulting CSV contains raw logits and sigmoid probabilities (`prob_open`).

### 3. Webcam Demo

```bash
uv run python -m vsdlm webcam \
--checkpoint runs/vsdlm/vsdlm_best_epoch0004_f10.9321.pt \
--camera_index 0 \
--mirror
```

- Press `q` or `Esc` to stop the session.
- The live overlay shows the open-mouth probability and predicted label in real time.
- Use `--device cuda` to force GPU inference and `--window_name` to customise the OpenCV window title.

### 4. ONNX Export

```bash
uv run python -m vsdlm exportonnx \
--checkpoint runs/vsdlm/vsdlm_best_epoch0004_f10.9321.pt \
--output runs/vsdlm/vsdlm.onnx \
--opset 17

uv run python -m vsdlm webcam_onnx --model vsdlm.onnx --camera_index 0 --provider cuda --detector_provider tensorrt
```

- The saved graph exposes `images` as input and `prob_open` as output (batch dimension is dynamic); probabilities can be consumed directly.
- After exporting, the tool runs `onnxsim` for simplification and rewrites any remaining BatchNormalization nodes into affine `Mul`/`Add` primitives. If simplification fails, a warning is emitted and the unsimplified model is preserved.

## Datasets

1. **GRID Audio-Visual Speech Corpus**
   34 speakers × 1000 sentences (frontal video + audio). Speech segments are cleanly separated. Creative exploitation: treat sentences as “speaking” and silence/random segments as “non-speaking”. License: CC BY 4.0.
   https://zenodo.org/records/3625687
   ```bibtex
   Cooke, M., Barker, J., Cunningham, S., & Shao, X. (2006).
   The Grid Audio-Visual Speech Corpus (1.0) [Data set].
   Zenodo. https://doi.org/10.5281/zenodo.3625687
   ```

2. **Lombard GRID (Audio-Visual Lombard Speech Corpus)**
   54 speakers × 100 sentences (half “normal”, half “Lombard”), captured from frontal and profile views. All clips contain speech; create negatives by mixing pauses or other data. Lombard speech features exaggerated mouth motion, improving robustness. License: CC BY 4.0.
   https://spandh.dcs.shef.ac.uk/avlombard
   ```bibtex
   Najwa Alghamdi, Steve Maddock, Ricard Marxer, Jon Barker and Guy J. Brown,
   A corpus of audio-visual Lombard speech with frontal and profile views,
   The Journal of the Acoustical Society of America 143, EL523 (2018);
   https://doi.org/10.1121/1.5042758
   ```

### Dataset Preparation
#### Lombard GRID (Audio-Visual Lombard Speech Corpus)

```bash
./dataset/01_rename_lombardgrid.sh
```
```bash
# Single file
uv run python dataset/03_batch_mouth_labeler.py \
--src_file dataset/lombardgrid/001_0002_front_002097.mov \
--output_dir dataset/output_lombardgrid \
--threshold_front 0.25 \
--threshold_side 0.55 \
--min_kpt_score 0.15

# Batch processing
uv run python dataset/03_batch_mouth_labeler.py \
--src_dir dataset/lombardgrid \
--output_dir dataset/output_lombardgrid \
--threshold_front 0.25 \
--threshold_side 0.55 \
--min_kpt_score 0.15
```
```
=== Processing Summary ===
+---+-------------------------------+--------+
| # | Description                   |  Value |
+---+-------------------------------+--------+
| 1 | Videos without unknown frames |  10208 |
| 2 | Videos with unknown frames    |    568 |
+---+-------------------------------+--------+
| 3 | Total processed videos        |  10776 |
+---+-------------------------------+--------+

+---+-------------------------------+--------+
| # | Description                   |  Value |
+---+-------------------------------+--------+
| 4 | Total unknown frames          |   6340 |
| 5 | Total mouth closed frames     | 200749 |
| 6 | Total mouth open frames       | 455480 |
+---+-------------------------------+--------+
| 7 | Total frames                  | 662569 |
+---+-------------------------------+--------+
8. Histogram (dataset-wide ratios)
   - Unknown      | #                                        |   1.0% (6340)
   - Mouth closed | ############                             |  30.3% (200749)
   - Mouth open   | ###########################              |  68.7% (455480)
```

#### GRID Audio-Visual Speech Corpus

```bash
./dataset/02_rename_grid_audio_visual.sh
```
```bash
# Single file
uv run python dataset/03_batch_mouth_labeler.py \
--src_file dataset/grid_audio_visual_speech_corpus/002_0001_front_000001.mpg \
--output_dir dataset/output_grid_audio_visual_speech_corpus \
--threshold_front 0.15 \
--min_kpt_score 0.15

# Batch processing
uv run python dataset/03_batch_mouth_labeler.py \
--src_dir dataset/grid_audio_visual_speech_corpus \
--output_dir dataset/output_grid_audio_visual_speech_corpus \
--threshold_front 0.15 \
--min_kpt_score 0.15

# Still-only mode
uv run python dataset/03_batch_mouth_labeler.py \
--src_dir dataset/grid_audio_visual_speech_corpus \
--output_dir dataset/output_grid_audio_visual_speech_corpus \
--threshold_front 0.15 \
--min_kpt_score 0.15 \
--still_only
```

```bash
cd dataset
uv run python 05_random_pick.py
```
```bash
uv run python 06_face_augmentation.py \
--input_dir output_grid_audio_visual_speech_corpus_still_image_partial \
--output_dir outputs_face_aug
```
```bash
uv run python 07_demo_deimv2_onnx_wholebody34_with_edges.py \
-i outputs_face_aug/ \
-m deimv2_dinov3_s_wholebody34_1750query_n_batch_640x640.onnx \
-ep tensorrt \
-dwk \
-dtk \
-dti \
-dhd \
-dhm \
-dlr \
-dgm \
-dnm \
-drc 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 20 21 22 23 24 25 26 27 28 29 30 31 32 33 \
--crop_size 48 \
--crop_disable_padding \
--crop_margin_top 2 \
--crop_margin_bottom 6 \
--crop_margin_left 2 \
--crop_margin_right 2

count=351473 | height mean=29.67px median=29.00px | width mean=48.00px median=48.00px
```

## Arch

<img width="300" alt="vsdlm_s" src="https://github.com/user-attachments/assets/f2d42767-43f0-4433-ae62-882a70fb65cc" />

## Citation

If you find this project useful, please consider citing:

```bibtex
@software{hyodo2025vsdlm,
  author    = {Katsuya Hyodo},
  title     = {PINTO0309/VSDLM},
  month     = {10},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.17494543},
  url       = {https://github.com/PINTO0309/vsdlm},
  abstract  = {Visual only speech detection by lip movement.},
}
```

## Acknowledgements

1. https://zenodo.org/records/3625687 - CC BY 4.0 License
2. https://spandh.dcs.shef.ac.uk/avlombard - CC BY 4.0 License
3. https://github.com/hhj1897/face_alignment - MIT License
4. https://github.com/hhj1897/face_detection - MIT License
5. https://github.com/PINTO0309/Face_Mask_Augmentation - MIT License
6. https://github.com/PINTO0309/PINTO_model_zoo/tree/main/472_DEIMv2-Wholebody34 - Apache 2.0
