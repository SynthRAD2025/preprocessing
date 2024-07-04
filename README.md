# synthRAD2025 pre-processing
This repository contains the code for pre-processing the data for the [synthRAD2025 challenge](https://synthrad2025.grand-challenge.org/). The data preprocessing is performed in two stages:
1. **Stage 1** contains all pre-processing steps carried out locally in each data providing center. This includes the following steps:
    - **Data conversion**: All image data is converted to .nii.gz and .nrrd format.
    - **Rigid registration**: CBCT and MR images are registered to the corresponding CT images.
    - **Defacing**: Datasets with visible faces in the images are defaced.
    - **Resampling**: Images are resampled to a common voxel size.

2. **Stage 2** contains all steps that are carried out for the entire dataset at once and includes the following steps:
    - **Deformable Image Registration**: MR/CBCT images are deformably registered to the CT
     - **Cropping**: CBCT/CT/MR and structures are cropped to the same size.
     - **Validation**: The preprocessed data is validated to ensure that the preprocessing steps have been carried out correctly.
     - **Dataset creation**: The preprocessed data is seperated into training/validation and test datasets. Furthermore separate datasets for centers and tasks are created.

# Requirements

The code is written/tested in Python 3.12.3 The following packages are required to run the code:

- SimpleITK
- numpy
- nibabel
- scipy
- totalsegmentator
- matplotlib
- csv

To convert RT structs to .nrrd files plastimatch is required. Plastimatch can be downloaded from [here](https://plastimatch.org/). The path to plastimatch should be added to the system path.

# Usage

The code is organized in two main files: [stage1.py](./stage1.py) and [stage2.py](./stage2.py). The code can be run by executing the following command:

```python stage1.py config.csv```

[config.csv](./stage1_config.csv) is a configuration file that contains the paths to the input data and the parameters for the preprocessing steps. The configuration file contains a header in the first row each further row contains configuration for a single patient. The configuration file should contain the following columns:

| column        | description           | parsed as|
| ------------- |-------------| -------|
| **ID**        | A unique patient ID in the synhtRAD2025 format: [Task][Region][Center][001-999].| str |
| **task**      | *1* for Task 1 (MR-to-CT) and *2* for Task 2 (CBCT-to-CT)      |int|
| **region**    |  *HN* for head and neck, *AB* for abdomen, *TH* for thorax  |  string |
| **ct_path**   | path to CT image, can be a dicom directory or a single file compatible with SimpleITK (e.g .mha, .nrrd, .nii.gz, ...) | string |
| **input_path**| path to MR/CBCT image, can be a dicom directory or a single file compatible with SimpleITK (e.g .mha, .nrrd, .nii.gz, ...) | string |
| **struct_path**| path to RTstruct file, can be left empty if no structure file is available| string |
| **output_dir**| path to output directory, if directory does not exist it will be generated| string |
| **defacing**| *True* if defacing is required, *False* otherwise| bool |
| **registration**| path to registration parameter file, registration files are provided in [configs](./configs/)| bool |
| **reg_fovmask**| *True* if a FOV mask should be used for registration, *False* otherwise| bool |
| **background**| intensity of background, usually 0 for MR and -1024 for CBCT, but can vary between centers/regions and can influence FOV masking| float |
| **order**| order of axis, usually should be [Sagittal, Coronal, Axial], if reordering is required indicate order using following notation: e.g. [2,1,0] reverses the order | array |
| **flip**| flip an axis, usually should be [False, False, False], if flipping is required indicate flip using following notation: e.g. [True, False, False] flips the first axis | array |
| **resample**| resamples to a uniform voxel size, indicate the target voxel size with array, e.g. [1,1,3] results in in-plane voxel size of 1mm x 1mm and slice thickness of 3 mm | array |
| **mr_overlap_correction**| *True* if MR overlap correction (some centers have artificial override to match multiple scans) is required, *False* otherwise| bool |
| **intensity_shift**| shifts the intensity of an image, usually 0, but can be required for CBCTs| float |





