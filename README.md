# Reference-Based Printing Defect Detection and Classification

**AMME5710 — Computer Vision and Image Processing**  
**Major Project — Source Code and Experimental Materials**

## 1. Project Overview

This project investigates reference-based printing defect detection and classification using classical image processing and a Support Vector Machine (SVM).

The work proceeds through three experimental stages:

1. **Benchmark experiments:** compare grayscale, RGB, hybrid, and exploratory SSIM-based image difference methods; train and evaluate a five-class SVM using annotated defect regions.
2. **Custom digital test:** detect, separate, classify, and annotate multiple simulated printing defects in a digitally designed test image.
3. **Real printing test:** compare smartphone-scanned reference and defective prints using ECC image registration, RGB differencing, region grouping, and the previously trained SVM.

The final classifier recognises five defect categories: **Black Spot, Color Shift, Hair, Line, and Pinhole**. The Friction category was explored initially but excluded from the final five-class classifier.

## 2. Project Directory Structure

```text
AMME5710_Major_Project/
|
|-- README.md                    # This guide
|-- notebooks/                   # Jupyter source code for all experiments
|   |-- 01_AbsoluteDifference.ipynb
|   |-- 02_AbsoluteDifference_Batch_evaluation.ipynb
|   |-- 03_RGBDifference_Batch_evaluation.ipynb
|   |-- 04_ssim_detection.ipynb
|   |-- 05_printing_defect_svm.ipynb
|   |-- 06_hybrid_difference.ipynb
|   |-- 07_digital_Test.ipynb
|   `-- 08_real_printing_test.ipynb
|-- dataset/                     # Taktpixel2025PD-CD benchmark dataset
|   |-- train/
|   |   |-- A/                  # Reference images
|   |   |-- B/                  # Defective images
|   |   `-- OUT/                # Ground-truth masks
|   |-- val/
|   |   |-- A/
|   |   |-- B/
|   |   `-- OUT/
|   `-- test/
|       |-- A/
|       |-- B/
|       `-- OUT/
|-- Custom_test/                 # Custom digital reference and defective images
|-- real_scan/                   # Real printed samples and smartphone scans
|-- results/                     # Evaluation tables, figures, masks, annotations
`-- svm_classes.joblib           # Saved five-class SVM model
```

The project may also contain `.ipynb_checkpoints/`, an automatically generated Jupyter backup folder. It is not required to reproduce the project.

### Dataset

`dataset/` contains the Taktpixel2025PD-CD dataset, organised into training, validation, and test splits. Each matching filename appears in:

- `A/` — defect-free reference image;
- `B/` — corresponding defective image;
- `OUT/` — pixel-level ground-truth defect mask.

The complete original dataset contains six defect categories. The final five-class classifier excludes Friction.

**Dataset reference:** Taktpixel2025PD-CD (2025)  
Zenodo: https://doi.org/10.5281/zenodo.15318946  
Project repository: https://github.com/taktpixel/Taktpixel2025PD-CD

The dataset is third-party material and remains subject to its original licence and attribution requirements.

### Custom_test

`Custom_test/` contains the custom digital test materials, including a defect-free reference design and a version containing simulated printing defects. These images are used in Notebook 07 to demonstrate multi-region detection and classification.

### real_scan

`real_scan/` contains images of physically printed reference and defective designs captured using a smartphone and a document-scanning application. These are the input materials for Notebook 08. Some files may be intermediate registered images or scan variants.

### results

`results/` contains saved experimental outputs, such as threshold comparisons, difference masks, evaluation tables, figures, aligned images, connected-component visualisations, and final defect annotations. Subfolders correspond to different experiments where applicable.

### Saved classifier

`svm_classes.joblib` is the saved five-class SVM pipeline, including feature standardisation. It is intended for use with the **same 334-dimensional feature extraction procedure** implemented during model training.

## 3. Notebook Guide

The `notebooks/` directory contains these eight Jupyter Notebook files (shown with their original filenames):

| Exact filename | Purpose |
|---|---|
| `01_AbsoluteDifference.ipynb` | Initial grayscale absolute-difference experiments on paired images. |
| `02_AbsoluteDifference_Batch_evaluation.ipynb` | Batch validation of grayscale threshold-based defect localisation. |
| `03_RGBDifference_Batch_evaluation.ipynb` | Maximum-channel colour difference, threshold sweep, and segmentation metrics. |
| `04_ssim_detection.ipynb` | Exploratory SSIM-based defect localisation and qualitative observations. |
| `05_printing_defect_svm.ipynb` | Feature extraction; five-class RBF-SVM training, validation, test evaluation, and confusion matrix. |
| `06_hybrid_difference.ipynb` | Weighted grayscale/RGB difference comparison on the validation data. |
| `07_digital_Test.ipynb` | Custom digital-image differencing, connected components, region grouping, SVM inference, and annotation. |
| `08_real_printing_test.ipynb` | Smartphone-scan registration using ECC, difference masking, region grouping, SVM inference, and real-scan analysis. |

The numbers indicate the experimental sequence. Notebook 05 generates the classifier subsequently used by Notebooks 07 and 08. Notebooks 01–04 and 06 document the alternative localisation methods and comparisons.

## 4. Software Requirements

The work was developed using Python and Jupyter Notebook, primarily with:

- `numpy`
- `opencv-python` (`cv2`)
- `matplotlib`
- `scikit-image`
- `scikit-learn`
- `pandas`
- `joblib`

To install the main dependencies:

```bash
python -m pip install jupyter numpy opencv-python matplotlib scikit-image scikit-learn pandas joblib
```

## 5. How to Run

1. Extract the full ZIP archive while preserving its folder structure.
2. Open a terminal in the project directory and launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

3. Open the required file in `notebooks/`.
4. **Update local file paths before running.** Some notebooks use absolute Windows paths from the original development computer (for example, `D:\AMME5710_MajorProject\...`). They must be changed to match the extracted project's location. **Note that the directory name displayed in the supplied project folder is `AMME5710_Major_Project` (with an underscore), while some development-time code paths may use `AMME5710_MajorProject` (without that underscore).**
5. For benchmark experiments, ensure the `dataset/train/`, `dataset/val/`, and `dataset/test/` directories contain matching `A/`, `B/`, and `OUT/` files.
6. Run Notebook 05 to regenerate the model if necessary. Alternatively, use the supplied `svm_classes.joblib` file for the application notebooks.
7. Run Notebook 07 with the files in `Custom_test/`, and Notebook 08 with the files in `real_scan/`.

**Important:** Check the filename used in any `joblib.load(...)` or `joblib.dump(...)` statement. The model supplied at the project root is named `svm_classes.joblib`; update code paths if an earlier notebook version refers to a different name.

The notebooks document experimental development and may contain exploratory cells. Their outputs are included as a record of the experiments. Executing a notebook from a newly restarted kernel may require running its setup, function-definition, and data-loading cells first.

## 6. Feature Extraction and Model

Each candidate defect is represented using a **334-dimensional handcrafted feature vector**:

| Feature group | Dimensions | Description |
|---|---:|---|
| HOG | 324 | Local gradient orientations and image structure |
| Colour differences | 6 | Mean and standard deviation of absolute BGR differences |
| Shape | 4 | Relative area, relative width, relative height, and aspect ratio |
| **Total** | **334** | Input to the five-class RBF-SVM |

The classifier uses a `StandardScaler` followed by an RBF-kernel SVM (`C=1.0`, balanced class weights, default `gamma=scale`).

The SVM predicts the **class of an extracted region**. Region localisation is a separate stage performed using image differencing and connected components.

## 7. Main Experimental Results

| Experiment | Reported result |
|---|---|
| Grayscale difference, original six-category validation | Pixel-level F1 = **0.6215**, IoU = **0.4509** |
| RGB maximum-channel difference, original six-category validation | Pixel-level F1 = **0.7245**, IoU = **0.5681** (`T=40`) |
| Five-class SVM, GT-guided validation | Classification accuracy = **94.01%** (484 samples) |
| Five-class SVM, GT-guided test | Classification accuracy = **94.92%** (492 samples); macro F1 approximately **0.93** |
| Custom digital image | Multi-region localisation, grouping, classification, and annotation demonstrated |
| Real printed and scanned image | ECC registration, RGB difference, grouped candidate regions, and classification demonstrated, with some misclassifications |

**Evaluation distinction:** The 94.92% SVM test accuracy uses ground-truth masks to extract defect regions and measures classification given accurate localisation. It does **not** represent end-to-end automatic detection accuracy. The digital and real-scan experiments instead use predicted masks; their qualitative results should be interpreted separately.

## 8. Known Limitations

- A reference image is required for the current detection pipeline.
- Difference masks are sensitive to image misalignment, scanning variation, and print texture.
- Some fragmented regions in the custom digital and real-scan demonstrations were merged using manually specified regions of interest (ROIs).
- The classifier was trained using ground-truth-defined regions, whereas application inputs use estimated regions.
- The classifier has no explicit background or unknown-defect class, so false-positive regions may still receive one of the five labels.
- The real-scan experiment is exploratory and does not provide an independently annotated pixel-level ground truth or a generalisable end-to-end accuracy estimate.

## 9. Generative AI Declaration

ChatGPT (OpenAI) was used to assist with Python code development and debugging, interpretation of experimental results, and language and report-structure refinement. Experimental execution and the reported measurements were carried out as part of the project. The author is responsible for reviewing the final submission and ensuring that its contents and results are accurate.

## 10. Submission and Repository Notes

If sending the project by email, first compress the root folder into one ZIP archive and check the actual resulting file size against the sender's and recipient's attachment limits. For larger archives, share a GitHub repository or a cloud-storage link instead of attaching the archive directly.

If using GitHub, include the eight notebooks, this README, the saved model, custom digital test images, real-scan input images, and representative results. The public Taktpixel dataset may be linked instead of uploaded in full. GitHub does not accept ordinary repository files larger than 100 MB without alternative large-file handling, and the dataset's original licence and attribution requirements still apply.

Remove the automatically generated `.ipynb_checkpoints/` folder from the submission if it is not needed. Do not include unrelated temporary files or private information.
