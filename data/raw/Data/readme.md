## Datasets for Classification of Chloroplast Cells

Last edited 12.06.2023

---

## Synthetic Data

We have different synthetic datasets.
Each of them contains 1000 train images, equally distributed among the four classes and 100 test images.
The structure is as follows

```
├── train
│   ├── diamond
│   ├── gyroid
│   ├── lonsdaleite
│   ├── primitive
├── test
│   ├── diamond
│   ├── gyroid
│   ├── lonsdaleite
│   ├── primitive
```

In each folder, the file `Image_summary.txt` contains all parameters (such as slice height, orientation angle etc.) for each image.
These information may come handy when analyzing and understanding the performance in later stages of the project.

### Set1

This contains views of the different classes with a fixed scale, i.e. each image captures the same volume of the cell tissue.
The `Set1_clean`\-dataset is a good starting point to explore the data and to train a first basic model.

In addition, the folder `Set1_noisy` has different kind of noise applied to the images from the clean-dataset.
The extent of noise is randomly chosen for each image and documented in `noisy.txt` (again, for analyze purposes).

### Set2

Same settings as Set1, but now the parameter controlling the scale (slice_height, slice_width) is no longer fixed.
This results in more variance regarding the 'size' of the different structures.

### Set 3 - multiview

Same parameter settings as Set2. One sample consists of three images/views which have the same parameter configuration but a different orientation (hkl indices).

Filename pattern: \[Class_name\]\_\[Sample_number\]\_\[view_number\]

For each view within a sample, the noise is the same.

---

## Real Data

The available real datasets will be described here.

---