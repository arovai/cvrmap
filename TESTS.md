# CVRmap Tests

This document outlines the installation, prerequisites, and test cases for running **CVRmap** locally.

---

## 📌 Prerequisites

Before running CVRmap, ensure the following:

1. **Installation**:
   Install CVRmap using `pip` by following the instructions in the [main README file](README.md).

2. **Data Requirements**:
   - **Raw data**: Ensure raw physiological data is present in the `rawdata/` directory.
   - **Preprocessed data**: Use fMRIPrep-preprocessed data (e.g., from [OpenNeuro datasets](https://openneuro.org/datasets/ds004604/versions/2.0.0)).
   - **Special cases**: Additional prerequisites may apply for specific use cases.

---

## 🧪 Test Cases

Below are the commands to run CVRmap in various configurations.

---

### 1️⃣ **Basic Local Install Test**
Run CVRmap on raw data with fMRIPrep derivatives:

```bash
cvrmap rawdata derivatives/cvrmap_4.4.1_test1 participant \
  --participant-label 004 \
  --task gas \
  --derivatives fmriprep=derivatives/fmriprep_21.0.4/
```

---

### 2️⃣ **Custom Config File Test**
Use a custom configuration file (`code/configs/cvrmap/config.json`):

```bash
cvrmap rawdata derivatives/cvrmap_4.4.1_test2 participant \
  --participant-label 004 \
  --task gas \
  --derivatives fmriprep=derivatives/fmriprep_21.0.4/ \
  --config code/configs/cvrmap/config.json
```

---

### 3️⃣ **ROI Probe Test**
Test ROI probe functionality with custom coordinates and radius:

```bash
cvrmap rawdata derivatives/cvrmap_4.4.1_test3 participant \
  --participant-label 004 \
  --task gas \
  --derivatives fmriprep=derivatives/fmriprep_21.0.4/ \
  --roi-probe \
  --roi-coordinates -0.5 -60.3 15.8 \
  --roi-radius 12
```

---

### 4️⃣ **Direct fMRIPrep Input with ROI Probe**
#### **Coordinates Only**
Run CVRmap directly on fMRIPrep derivatives with ROI probe coordinates:

```bash
cvrmap derivatives/fmriprep_21.0.4/ derivatives/cvrmap_4.4.1_test4 participant \
  --participant-label 004 \
  --task gas \
  --roi-probe \
  --roi-coordinates -0.5 -60.3 15.8 \
  --roi-radius 12
```

#### **Coordinates with Filtering**
Add a bandpass filter to the ROI probe:

```bash
cvrmap derivatives/fmriprep_21.0.4/ derivatives/cvrmap_4.4.1_test4 participant \
  --participant-label 004 \
  --task gas \
  --roi-probe \
  --roi-coordinates -0.5 -60.3 15.8 \
  --roi-radius 12 \
  --probe-bandpass-filter \
  --probe-highpass 0.018
```

---
---
### 5️⃣ **ROI Mask Test**
Use a custom ROI mask (requires `derivatives/masks/space-MNI_desc-arteries_mask.nii.gz`):

```bash
cvrmap derivatives/fmriprep_21.0.4/ derivatives/cvrmap_4.4.1_test5 participant \
  --participant-label 004 \
  --task gas \
  --roi-probe \
  --roi-mask derivatives/masks/space-MNI_desc-arteries_mask.nii.gz
```

---
---
### 6️⃣ **ROI Atlas Test**
Use an atlas file (requires `/opt/fsl/data/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr50-2mm.nii.gz`) and specify a region ID:

```bash
cvrmap derivatives/fmriprep_21.0.4/ derivatives/cvrmap_4.4.1_test6 participant \
  --participant-label 004 \
  --task gas \
  --roi-probe \
  --roi-atlas /opt/fsl/data/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr50-2mm.nii.gz \
  --roi-region-id 43
```
