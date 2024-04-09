import SimpleITK as sitk
import pyplastimatch
import numpy as np

def read_image(image_path:str)->sitk.Image:
    image = sitk.ReadImage(image_path)
    return image

def read_dicom_image(image_path:str)->sitk.Image:
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(image_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image

def save_image(image:sitk.Image, image_path:str):
    sitk.WriteImage(image, image_path)

def convert_rtstruct_to_nrrd(rtstruct_path:str, nrrd_dir_path:str):
    convert_args_ct = {"input" :            rtstruct_path,
                        "output-prefix" :   nrrd_dir_path,
                        "prefix-format" :   'nrrd',
                        }
    pyplastimatch.convert(**convert_args_ct)

def register(fixed:sitk.Image, moving:sitk.Image, parameter:sitk.ParameterMap):
    # Perform registration based on parameter file
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetParameterMap(parameter)
    elastixImageFilter.PrintParameterMap()
    elastixImageFilter.SetFixedImage(moving)  # due to FOV differences CT first registered to MR an inverted in the end
    elastixImageFilter.SetMovingImage(fixed)
    elastixImageFilter.LogToConsoleOn()
    elastixImageFilter.LogToFileOff()
    elastixImageFilter.Execute()

    # convert to itk transform format
    transform = elastixImageFilter.GetTransformParameterMap(0)
    x = transform.values()
    center = np.array((x[0])).astype(np.float64)
    rigid = np.array((x[22])).astype(np.float64)
    transform_itk = sitk.Euler3DTransform()
    transform_itk.SetParameters(rigid)
    transform_itk.SetCenter(center)
    transform_itk.SetComputeZYX(False)

    # save itk transform to correct MR mask later
    output = str(output)
    transform_itk.WriteTransform(str(output.split('.')[:-2][0]) + '_parameters.txt')
    #transform_itk.WriteTransform(str('registration_parameters.txt'))

    ##invert transform to get MR registered to CT
    inverse = transform_itk.GetInverse()

    ## check if moving image is an mr or cbct
    min_moving = np.amin(sitk.GetArrayFromImage(moving))
    if min_moving <-500:
        background = -1000
    else:
        background = 0

    ##transform MR image
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(fixed)
    resample.SetTransform(inverse)
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetDefaultPixelValue(background)
    registered_image = resample.Execute(moving)

    return registered_image