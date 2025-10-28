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

```
lombardgrid_{front or side}/{talker}_x_xxxxxx.mov

{dataset:03d}_{talker:04d}_{front or side}_{serial_number:04d}.mov
```
