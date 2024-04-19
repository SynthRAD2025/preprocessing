import SimpleITK as sitk
import pyplastimatch
import numpy as np
import nibabel as nib
from typing import List,Union
from scipy.signal import find_peaks
from totalsegmentator.python_api import totalsegmentator

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
    sitk.WriteImage(image, image_path,useCompression=True)

def convert_rtstruct_to_nrrd(rtstruct_path:str, nrrd_dir_path:str):
    convert_args_ct = {"input" :            rtstruct_path,
                        "output-prefix" :   nrrd_dir_path,
                        "prefix-format" :   'nrrd',
                        }
    pyplastimatch.convert(**convert_args_ct)

def rigid_registration(fixed:sitk.Image, moving:sitk.Image, parameter)->Union[sitk.Image,sitk.Transform]:
    # Perform registration based on parameter file
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetParameterMap(parameter)
    #elastixImageFilter.PrintParameterMap()
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
    #transform_itk.WriteTransform(output)
    #transform_itk.WriteTransform(str('registration_parameters.txt'))

    ##invert transform to get MR registered to CT
    inverse_transform = transform_itk.GetInverse()

    ## check if moving image is an mr or cbct
    min_moving = np.amin(sitk.GetArrayFromImage(moving))
    if min_moving <-200:
        background = -1000
    else:
        background = 0

    ##transform moving image
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(fixed)
    resample.SetTransform(inverse_transform)
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetDefaultPixelValue(background)
    registered_image = resample.Execute(moving)

    return registered_image,inverse_transform

def correct_orientation(input_image:sitk.Image,order=[0,1,2],flip=[False,False,False]):
    # Correct orientation of image, default is to do nothing
    image_permuted = sitk.PermuteAxes(input_image, order)
    image_flipped = sitk.Flip(image_permuted,flip)
    image_flipped.SetDirection([1,0,0,0,1,0,0,0,1])
    return image_flipped

def nib_to_sitk(nib_image)->sitk.Image:
    img_nib_np = nib_image.get_fdata()
    nib_header = nib_image.header
    img_nib_np = np.swapaxes(img_nib_np,0,2)
    img_sitk = sitk.GetImageFromArray(img_nib_np)
    img_sitk.SetSpacing((float(nib_header['pixdim'][1]),float(nib_header['pixdim'][2]),float(nib_header['pixdim'][3])))
    img_sitk.SetOrigin((float(nib_header['srow_x'][3])*(-1),float(nib_header['srow_y'][3])*(-1),float(nib_header['srow_z'][3])))
    img_sitk.SetDirection((1,0,0,0,1,0,0,0,1))
    
    return img_sitk

def sitk_to_nib(sitk_image:sitk.Image):
    def make_affine(simpleITKImage):
        # get affine transform in LPS
        c = [simpleITKImage.TransformContinuousIndexToPhysicalPoint(p)
            for p in ((1, 0, 0),
                    (0, 1, 0),
                    (0, 0, 1),
                    (0, 0, 0))]
        c = np.array(c)
        affine = np.concatenate([
            np.concatenate([c[0:3] - c[3:], c[3:]], axis=0),
            [[0.], [0.], [0.], [1.]]
        ], axis=1)
        affine = np.transpose(affine)
        # convert to RAS to match nibabel
        affine = np.matmul(np.diag([-1., -1., 1., 1.]), affine)
        return affine

    affine = make_affine(sitk_image)
    header = nib.Nifti1Header()
    header.set_xyzt_units('mm', 'sec')
    img_nib = nib.Nifti1Image(np.swapaxes(sitk.GetArrayFromImage(sitk_image),2,0), affine, header)

    return img_nib

def segment_defacing(ct_image:sitk.Image,structures=['brain','skull'])->Union[sitk.Image,sitk.Image]:
    # Segment brain and skull from ct image for defacing using totalsegmentator
    ct_nib = sitk_to_nib(ct_image)
    segmentation = totalsegmentator(ct_nib,output=None,roi_subset=structures,quiet=True)
    structures = nib_to_sitk(segmentation)
    structures_np = sitk.GetArrayFromImage(structures)
    brain_np = np.copy(structures_np)
    brain_np[brain_np!=90]=0
    skull_np = np.copy(structures_np)
    skull_np[skull_np!=91]=0
    
    brain = sitk.GetImageFromArray(brain_np)
    brain.CopyInformation(structures)
    skull = sitk.GetImageFromArray(skull_np)
    skull.CopyInformation(structures)
    
    return brain,skull

def defacing(brain_mask:sitk.Image, skull_mask:sitk.Image)->sitk.Image:
    # Create defacing mask based on brain and external ROI
    brain_mask_np = sitk.GetArrayFromImage(brain_mask)
    skull_mask_np = sitk.GetArrayFromImage(skull_mask)
    dims = brain_mask_np.shape

    # find brain POI
    brain_central = brain_mask_np[:,:,int(dims[2]/2)]
    surface = []
    for l in range(dims[0]):
        array = np.where(brain_central[l,:] != 0)
        if array[0].size == 0:
            surface.append(np.nan)
        else:
            surface.append(np.min(array))
    x_brain = np.nanmin(surface)
    y_brain = np.nanargmin(surface)

    # find skull POI
    skull_central = skull_mask_np[:,:,int(dims[2]/2)]
    coords = np.where(skull_central != 0)
    bbox = (np.min(coords[0]), np.max(coords[0]), np.min(coords[1]), np.max(coords[1]))
    x_skull = bbox[2]
    y_skull = bbox[0]

    # create defacing mask
    k = (y_brain - y_skull)/(x_brain - x_skull)
    d = (y_brain - k*x_brain)
    face = np.zeros_like(brain_mask_np)
    for l in range(dims[2]):
        for i in range(dims[1]):
            for j in range(y_skull,dims[0]):
                if j > k*i + d:
                    face[j,i,l] = 1
    defacing_mask = sitk.GetImageFromArray(face)
    defacing_mask.CopyInformation(brain_mask)
    defacing_mask = sitk.Cast(defacing_mask, sitk.sitkUInt8)
    return defacing_mask
