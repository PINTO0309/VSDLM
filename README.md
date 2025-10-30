# VSDLM
Visual only speech detection by lip movement.

- Datasets

  1. GRID Audio-Visual Speech Corpus
    Contents: 34 speakers x 1000 sentences each (frontal video + audio). Speech sections are clearly separated by sentence.
    Usage: Assuming there is a mouth ROI, it is easy to binarize by creating sentence clips = "speaking" and silent sections or shuffling to create "not speaking".
    License: CC BY 4.0 (commercial use permitted with attribution). Please indicate the source on the distribution page.
    https://zenodo.org/records/3625687
      ```bibtex
      Cooke, M., Barker, J., Cunningham, S., & Shao, X. (2006).
      The Grid Audio-Visual Speech Corpus (1.0) [Data set].
      Zenodo. https://doi.org/10.5281/zenodo.3625687
      ```
  2. Lombard GRID（Audio-Visual Lombard Speech Corpus）
    Contents: 54 speakers x 100 sentences each (50 each for "normal" and "Lombard"). Two perspectives: frontal and side.
    Uses: All clips are speech, so Positives (speaking) can be easily created. Negatives are created by mixing pauses and other data. Speech in noisy environments (Lombard speech) has large mouth movements, making it effective for robust mouth movement detection.
    License: CC BY 4.0 (specified on the official page).
    https://spandh.dcs.shef.ac.uk/avlombard
      ```bibtex
      Najwa Alghamdi, Steve Maddock, Ricard Marxer, Jon Barker and Guy J. Brown,
      A corpus of audio-visual Lombard speech with frontal and profile views,
      The Journal of the Acoustical Society of America 143, EL523 (2018);
      https://doi.org/10.1121/1.5042758
      ```

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

# Process videos in a folder in bulk
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

# Process videos in a folder in bulk
uv run python dataset/03_batch_mouth_labeler.py \
--src_dir dataset/grid_audio_visual_speech_corpus \
--output_dir dataset/output_grid_audio_visual_speech_corpus \
--threshold_front 0.15 \
--min_kpt_score 0.15

# Still only mode and Process videos in a folder in bulk
uv run python dataset/03_batch_mouth_labeler.py \
--src_dir dataset/grid_audio_visual_speech_corpus \
--output_dir dataset/output_grid_audio_visual_speech_corpus \
--threshold_front 0.15 \
--min_kpt_score 0.15 \
--still_only
```

```bash
cd dataset
uv run python 05_face_augmentation.py \
--image test.png \
--output_dir outputs_face_aug
```
```bash
uv run python 06_demo_deimv2_onnx_wholebody34_with_edges.py \
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
-drc 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 20 21 22 23 24 25 26 27 28 29 30 31 32 33
```

## Acknowledgements

1. https://github.com/hhj1897/face_alignment MIT license
2. https://github.com/hhj1897/face_detection MIT license
3. https://github.com/PINTO0309/Face_Mask_Augmentation MIT license
