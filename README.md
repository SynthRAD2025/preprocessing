# pre-processing
This repository contains the code for pre-processing the data for the synthRAD2025 challenge. The data preprocessing is performed in two stages:
1. **Stage 1**: This stage contains all pre-processing steps carried out locally in each data providing center. This includes the following steps:
    - **Data conversion**: All image data is converted to .mha and .nrrd format.
    - **Rigid registration**: CBCT and MR images are registered to the corresponding CT images.
    - **Defacing**: Datasets with visible faces in the images are defaced.
    
2. **Stage 2**: This stage contains all steps that are carried out for the entire dataset at once and includes the following steps:
    - **Deformable Image Registration**: MR/CBCT images are deformably registered to the CT
     - **Cropping**: CBCT/CT/MR and structures are cropped to the same size.
     - **Validation**: The preprocessed data is validated to ensure that the preprocessing steps have been carried out correctly.
     --**Dataset creation**: The preprocessed data is seperated into training/validation and test datasets. Furthermore separate datasets for centers and tasks are created.