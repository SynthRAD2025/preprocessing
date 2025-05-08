import tempfile
import shutil
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import csv
import subprocess
import SimpleITK as sitk
import numpy as np
import nibabel as nib
import math
from typing import Union
from totalsegmentator.python_api import totalsegmentator
from scipy import ndimage

def read_image(image_path:str,log=False)->sitk.Image:
    """
    Read an image from the specified image path using SimpleITK.

    Parameters:
    image_path (str): The path to the image file. All ITK file formats can be loaded.

    Returns:
    sitk.Image: The loaded image.

    """
    if os.path.isdir(image_path):
        image = read_dicom_image(image_path,log)
    elif os.path.isfile(image_path):   
        image = sitk.ReadImage(image_path)
    
    if log != False:
        log.info(f'Image sucessfully read from {image_path}')
    
    return image

def read_dicom_image(image_path:str,log=False)->sitk.Image:
    """
    Reads a DICOM image from the specified image path using SimpleITK.

    Parameters:
    image_path (str): The path to the DICOM image.

    Returns:
    sitk.Image: The loaded DICOM image.
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(image_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    if log != False:
        log.info(f'DICOM image sucessfuly read from {image_path}')
    return image

def save_image(image:sitk.Image, image_path:str, compression:bool=True, dtype:str=None, log=False):
    """
    Save the given SimpleITK image to the specified file path.
    
    Args:
        image (sitk.Image): The SimpleITK image to be saved.
        image_path (str): The file path where the image will be saved.
        compression(bool): Whether to use compression when saving the image. Default is True.
        dtype(sitk.PixelIDValueEnum): Default is None. Allowed dtypes: float32 and int16
    """
    if dtype != None:
        if image.GetPixelIDTypeAsString() != '32-bit float':
            image = sitk.Cast(image,sitk.sitkFloat32)
        image = sitk.Round(image)
        if dtype == 'float32':
            image = sitk.Cast(image,sitk.sitkFloat32)
        elif dtype == 'int16':
            image = sitk.Cast(image,sitk.sitkInt16)
        else:
            raise ValueError('Invalid dtype/not implemented. Allowed dtypes: float32 and int16')
    sitk.WriteImage(image, image_path, useCompression=compression)
    if log != False:
        log.info(f'Image saved to {image_path}')

def convert_rtstruct_to_nrrd(rtstruct_path:str, output_dir:str,plastimatch_path=None,log=False):
    """
    Converts an RTSTRUCT file to NRRD format using plastimatch. plastimatch needs to be installed in the system

    Parameters:
    rtstruct_path (str): The path to the RTSTRUCT file.
    output_dir (str): The directory path where the NRRD files will be saved.
    plastimatch_path (str, optional): The path to the plastimatch executable. Defaults to None.

    Returns:
    None
    """
    output_dir = os.path.join(output_dir,'structures')
    
    if plastimatch_path == None:
        command = ['plastimatch','convert','--input',rtstruct_path,'--output-prefix',output_dir,'--prefix-format','nrrd']
    else:
        command = [plastimatch_path,'convert','--input',rtstruct_path,'--output-prefix',output_dir,'--prefix-format','nrrd']
    subprocess.run(command)
    if log != False:
        log.info(f'RT struct {rtstruct_path} converted, saved to {output_dir}')

def rigid_registration(fixed:sitk.Image, moving:sitk.Image, parameter_file, mask=None, default_value = 0,log=False)->Union[sitk.Image,sitk.Transform]:
    """
    Perform rigid registration between a fixed image and a moving image using the given parameter file.

    Parameters:
    fixed (sitk.Image): The fixed image to register.
    moving (sitk.Image): The moving image to register.
    parameter_file (str): The path to the parameter file for the registration.
    mask (sitk.Image, optional): The mask image to be used during registration. Defaults to None.
    default_value (float, optional): The default pixel value for the resampled image. Defaults to 0.
    log (bool, optional): Whether to log the registration process. Defaults to False.

    Returns:
    Tuple[sitk.Image, sitk.Transform]: A tuple containing the registered image and the inverse transform.

    """
    temp_dir = tempfile.mkdtemp()
    current_directory = os.getcwd()
    os.chdir(temp_dir)
    
    parameter = sitk.ReadParameterFile(parameter_file)
    
    # Perform registration based on parameter file
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetParameterMap(parameter)
    elastixImageFilter.SetFixedImage(moving)  # due to FOV differences CT first registered to MR an inverted in the end
    elastixImageFilter.SetMovingImage(fixed)
    if mask != None:
        elastixImageFilter.SetFixedMask(mask)
    elastixImageFilter.LogToConsoleOn()
    elastixImageFilter.LogToFileOff()
    elastixImageFilter.SetNumberOfThreads(16)
    elastixImageFilter.Execute()
    elastixImageFilter.SetOutputDirectory(temp_dir)

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

    ##transform moving image
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(fixed)
    resample.SetTransform(inverse_transform)
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetDefaultPixelValue(float(default_value))
    registered_image = resample.Execute(moving)
    os.chdir(current_directory)
    shutil.rmtree(temp_dir)
    if log != False:
        log.info(f'Rigid registration performed using parameter file {parameter_file}')
    
    return registered_image,inverse_transform

def deformable_registration(fixed:sitk.Image, moving:sitk.Image, parameter_file, mask=None, default_value=0, log=False)->Union[sitk.Image,sitk.Transform]:
    """
    Perform deformable registration between a fixed image and a moving image using the specified parameter file.

    Args:
        fixed (sitk.Image): The fixed image to register to.
        moving (sitk.Image): The moving image to be registered.
        parameter_file (str): The path to the parameter file containing the registration parameters.
        mask (sitk.Image, optional): The mask image to restrict the registration. Defaults to None.
        default_value (int, optional): The default value to use for pixels outside the moving image. Defaults to 0.
        log (bool, optional): Whether to log the registration process. Defaults to False.

    Returns:
        Union[sitk.Image, sitk.Transform]: The registered moving image and the transform parameter map.

    """
    
    if log != False:
        log.info(f'Starting deformable registration using parameter file {parameter_file}')
        
    temp_dir = tempfile.mkdtemp()
    current_directory = os.getcwd()
    os.chdir(temp_dir)

    parameter = sitk.ReadParameterFile(parameter_file)

    # Perform registration based on parameter file
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetParameterMap(parameter)
    elastixImageFilter.SetFixedImage(fixed)  # due to FOV differences CT first registered to MR an inverted in the end
    elastixImageFilter.SetMovingImage(moving)
    if mask != None:
        elastixImageFilter.SetFixedMask(mask)
    elastixImageFilter.LogToConsoleOn()
    elastixImageFilter.LogToFileOff()
    elastixImageFilter.SetNumberOfThreads(16)
    elastixImageFilter.Execute()
    
    moving_def = elastixImageFilter.GetResultImage()
    transform = elastixImageFilter.GetTransformParameterMap()[0]
    
    os.chdir(current_directory)
    shutil.rmtree(temp_dir)
    
    if log != False:
        log.info(f'Deformable registration successful!')
    
    return moving_def, transform

def correct_image_properties(input_image:sitk.Image, order=[0,1,2], flip=[False,False,False], intensity_shift=None, data_type=None, mr_overlap_correction=False, log=False):
    """
    Corrects the properties of an input image based on the specified parameters.

    Parameters:
    input_image (sitk.Image): The input image to be corrected.
    order (list[int]): The order of axes permutation. Default is [0, 1, 2].
    flip (list[bool]): The flip status for each axis. Default is [False, False, False].
    intensity_shift (float): The intensity shift to be applied to the image. Default is None.
    data_type (sitk.PixelIDValueEnum): The desired data type of the image. Default is None.
    mr_overlap_correction (bool): Flag indicating whether to perform MR overlap correction. Default is False.
    log (bool): Flag indicating whether to log the orientation correction. Default is False.

    Returns:
    sitk.Image: The corrected image.
    """
    
    image = sitk.PermuteAxes(input_image, order)
    image = sitk.Flip(image,flip)
    image.SetDirection([1,0,0,0,1,0,0,0,1])
    
    if data_type != None:
        image = sitk.Cast(image,data_type)
        
    if intensity_shift != None:
        image = image + intensity_shift
    
    if mr_overlap_correction:
        image[image==4000] = 0
        
    if log !=False:
        log.info(f'Orientation corrected using order = {order} and flip = {flip}')
    return image

def clip_image(image:sitk.Image,lower_bound:float, upper_bound:float, log=False)->sitk.Image:
    # clip an image using SimpleITK
    if log != False:
        log.info(f'Clipping image between {lower_bound} and {upper_bound}')
    image[image<lower_bound] = lower_bound
    image[image>upper_bound] = upper_bound
    return image

def nib_to_sitk(nib_image) -> sitk.Image:
    """
    Convert a NIfTI image to a SimpleITK image.

    Args:
        nib_image: The NIfTI image to be converted.

    Returns:
        The converted SimpleITK image.
    """
    img_nib_np = nib_image.get_fdata()
    nib_header = nib_image.header
    img_nib_np = np.swapaxes(img_nib_np, 0, 2)
    img_sitk = sitk.GetImageFromArray(img_nib_np)
    img_sitk.SetSpacing((float(nib_header['pixdim'][1]), float(nib_header['pixdim'][2]), float(nib_header['pixdim'][3])))
    img_sitk.SetOrigin((float(nib_header['srow_x'][3]) * (-1), float(nib_header['srow_y'][3]) * (-1), float(nib_header['srow_z'][3])))
    img_sitk.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

    return img_sitk

def sitk_to_nib(sitk_image:sitk.Image):
    """
    Convert a SimpleITK image to a NIfTI image using nibabel.

    Args:
        sitk_image (sitk.Image): The SimpleITK image to be converted.

    Returns:
        nib.Nifti1Image: The converted NIfTI image.
    """
    
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

def segment_defacing(ct_image:sitk.Image,structures=['brain','skull'],log=False)->Union[sitk.Image,sitk.Image]:
    """
    Generates brain and skull masks for defacing based on the ct image using totalsegmentator.

    Args:
        ct_image (sitk.Image): The CT image that should be defaced.
        structures (list): The structures that should be segmented.

    Returns:
        sitk.Image: brain mask.
        sitk.Image: skull mask.

    """
    # Segment brain and skull from ct image for defacing using totalsegmentator
    ct_nib = sitk_to_nib(ct_image)
    segmentation = totalsegmentator(ct_nib,output=None,roi_subset=structures,quiet=True,fast=True)
    structures = nib_to_sitk(segmentation)
    structures_np = sitk.GetArrayFromImage(structures)
    brain_np = np.copy(structures_np)
    brain_np[brain_np!=90]=0
    skull_np = np.copy(structures_np)
    skull_np[skull_np!=91]=0
    
    brain = sitk.GetImageFromArray(brain_np)
    brain.CopyInformation(ct_image)
    skull = sitk.GetImageFromArray(skull_np)
    skull.CopyInformation(ct_image)
    if log != False:
        log.info(f'Brain and skull masks generated for defacing')
    return brain,skull

def defacing(brain_mask:sitk.Image, skull_mask:sitk.Image,version='v1',log=False)->sitk.Image:
    """
    Applies defacing to the brain image based on the brain and skull masks.

    Args:
        brain_mask (sitk.Image): The brain mask image.
        skull_mask (sitk.Image): The skull mask image.

    Returns:
        sitk.Image: defacing mask.

    """
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

    print(x_brain,y_brain)
    
    # find skull POI
    skull_central = skull_mask_np[:,:,int(dims[2]/2)]
    if version == 'v1':
        coords = np.where(skull_central != 0)
        bbox = (np.min(coords[0]), np.max(coords[0]), np.min(coords[1]), np.max(coords[1]))
        x_skull = bbox[2]
        y_skull = bbox[0]
    
    if version == 'v2':
        # different approach to localize mandible
        # find indices of first non-zero value in the skull_central array
        nonzero_indices = np.argwhere(skull_central != 0)
        if nonzero_indices.size > 0:
            first_nonzero_index = nonzero_indices[0]
        x_skull = first_nonzero_index[1]
        y_skull = first_nonzero_index[0]
    
    # create defacing mask
    k = (y_brain - y_skull)/(x_brain - x_skull)
    d = (y_brain - k*x_brain)
    face = np.zeros_like(brain_mask_np)
    for l in range(dims[2]):
        for i in range(dims[1]):
            for j in range(y_skull,y_brain):
                if version == 'v1':
                    if j > k*i + d:
                        face[j,i,l] = 1
                elif version == 'v2':
                    if j > k*i + d and k > 0:
                        face[j,i,l] = 1
                    elif j < k*i + d and k < 0:
                        face[j,i,l] = 1

    defacing_mask = sitk.GetImageFromArray(face)
    defacing_mask.CopyInformation(brain_mask)
    defacing_mask = sitk.Cast(defacing_mask, sitk.sitkUInt8)
    if log != False:
        log.info(f'Defacing mask generated with following parameters: x_brain = {x_brain}, y_brain = {y_brain}, x_skull = {x_skull}, y_skull = {y_skull}')
    return defacing_mask

# this is not used
# def segment_outline(ct_image:sitk.Image,fast=False,log=False)->sitk.Image:
#     """
#     Segment the patient outline from a CT/CBCT image using totalsegmentator.

#     Parameters:
#     - ct_image (sitk.Image): The input CT image.
#     - fast (bool): Whether to use fast mode for segmentation. Default is False.

#     Returns:
#     - sitk.Image: The segmented patient outline image.
#     """
#     ct_nib = sitk_to_nib(ct_image)
#     segmentation = totalsegmentator(ct_nib,task='body',fast=fast,output=None,quiet=True)
#     structures = nib_to_sitk(segmentation)
#     structures_np = sitk.GetArrayFromImage(structures)
#     outline_np = np.copy(structures_np)
#     outline_np[outline_np!=0]=1
    
#     outline = sitk.GetImageFromArray(outline_np)
#     outline.CopyInformation(structures)
#     return outline

def get_cbct_fov(cbct:sitk.Image,background:int=0,log=False)->sitk.Image:
    """
    Generate a field of view (FOV) mask for a given CBCT image.

    Parameters:
    - cbct (sitk.Image): The input CBCT image.
    - background (int): The intensity value used to define the background. Default is 0.

    Returns:
    - fov_mask (sitk.Image): The generated FOV mask.

    """
    cbct_np = sitk.GetArrayFromImage(cbct)
    cbct_np[cbct_np>background] = 1
    cbct_np[cbct_np<=background] = 0
    fov_mask_np = np.zeros(cbct_np.shape)
    for i in range(cbct_np.shape[0]):
        slice = cbct_np[i,:,:]
        y, x = np.indices((slice.shape))
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])
        r = np.hypot(x - center[0], y - center[1])
        bins = np.arange(0, r.max() + 1, 1)
        radial_mean = ndimage.mean(cbct_np[i,:,:], labels=np.digitize(r, bins), index=np.arange(1, len(bins)))
        if not radial_mean.any():
            continue
        else:
            mask_radius = np.where(radial_mean>0)[0][-1]
            size = cbct_np.shape
            y, x = np.ogrid[-size[1]//2:size[1]//2, -size[2]//2:size[2]//2]
            fov_mask_np[i,:,:] = x**2 + y**2 <= mask_radius**2
    
    fov_mask = sitk.GetImageFromArray(fov_mask_np)
    fov_mask.CopyInformation(cbct)
    fov_mask = sitk.Cast(fov_mask,sitk.sitkUInt8)
    if log != False:
        log.info(f'CBCT FOV mask generated using background = {background}')
    return fov_mask

def get_cbct_fov_v2(cbct:sitk.Image,background:int=0,log=False)->sitk.Image:
    # find the center and radius of a circular binary mask, allowing off-center positioning
    cbct_np = sitk.GetArrayFromImage(cbct)
    cbct_np[cbct_np>background] = 1
    cbct_np[cbct_np<=background] = 0
    fov_mask_np = np.zeros(cbct_np.shape)
    
    # Store centers and radii for all slices
    centers = []
    radii = []
    
    # First pass - collect all centers and radii
    for i in range(cbct_np.shape[0]):
        slice = cbct_np[i,:,:]
        if np.any(slice == 1):
            coords = np.argwhere(slice == 1)
            if len(coords) > 0:
                center = np.mean(coords, axis=0)
                distances = np.sqrt((coords[:,0] - center[0])**2 + (coords[:,1] - center[1])**2)
                mask_radius = np.max(distances) - 1  # Reduce radius by 1 pixel
                centers.append(center)
                radii.append(mask_radius)
            else:
                centers.append(None)
                radii.append(0)
        else:
            centers.append(None)
            radii.append(0)
    
    # Smooth the radii using median filter
    radii = np.array(radii)
    valid_radii = radii[radii > 0]
    if len(valid_radii) > 0:
        median_radius = np.median(valid_radii)
        # Replace zeros with median value for interpolation
        radii[radii == 0] = median_radius
        # Apply median filter to smooth out variations
        smoothed_radii = ndimage.median_filter(radii, size=5)
    
    # Second pass - create masks using smoothed values
    for i in range(cbct_np.shape[0]):
        if centers[i] is not None:
            y, x = np.indices(cbct_np.shape[1:])
            circle = (y - centers[i][0])**2 + (x - centers[i][1])**2 <= smoothed_radii[i]**2
            fov_mask_np[i,:,:] = circle
    
    fov_mask = sitk.GetImageFromArray(fov_mask_np)
    fov_mask.CopyInformation(cbct)
    fov_mask = sitk.Cast(fov_mask,sitk.sitkUInt8)
    if log != False:
        log.info(f'CBCT FOV mask v2 generated using background = {background}')
    return fov_mask

def get_mr_fov(mr:sitk.Image)->sitk.Image:
    """
    Get the field of view (FOV) of a given MR image.

    Parameters:
    mr (sitk.Image): The input MR image.

    Returns:
    sitk.Image: The FOV image.

    """
    mr_np = sitk.GetArrayFromImage(mr)
    fov_np = np.copy(mr_np)
    fov_np[fov_np == 0] = 0
    fov_np[fov_np != 0] = 1

    r = np.any(fov_np, axis=(1, 2))
    c = np.any(fov_np, axis=(0, 2))
    z = np.any(fov_np, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    bbox_img = np.zeros_like(fov_np)
    bbox_img[rmin:rmax+1, cmin:cmax+1, zmin:zmax+1] = 1

    fov_sitk = sitk.GetImageFromArray(bbox_img)
    fov_sitk.CopyInformation(mr)
    fov_sitk = sitk.Cast(fov_sitk,sitk.sitkUInt8)
    return fov_sitk
        
        
def apply_transform(image: sitk.Image, transform: sitk.Transform, ref_image:sitk.Image,interpolator:str='nearest') -> sitk.Image:
    """
    Applies the given transform to the input image and returns the transformed image.

    Parameters:
    image (sitk.Image): The input image to be transformed.
    transform (sitk.Transform): The transform to be applied to the image.
    ref_image (sitk.Image): The reference image used for setting the output spacing, size, direction, and origin.
    interpolator (str, optional): The type of interpolator to be used during resampling. 
                                  Valid options are 'nearest', 'linear', and 'bspline'. 
                                  Defaults to 'nearest'.

    Returns:
    sitk.Image: The transformed image.

    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(ref_image.GetSpacing())
    resampler.SetSize(ref_image.GetSize())
    resampler.SetOutputDirection(ref_image.GetDirection())
    resampler.SetOutputOrigin(ref_image.GetOrigin())
    resampler.SetTransform(transform)
    if interpolator == 'nearest':
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    elif interpolator == 'linear':
        resampler.SetInterpolator(sitk.sitkLinear)
    elif interpolator == 'bspline':
        resampler.SetInterpolator(sitk.sitkBSpline)
    else:
        print(resampler)
    transformed_image = resampler.Execute(image)
    return transformed_image

## Why did i do this?
# def crop_image(image:sitk.Image, bbox:tuple,dilation:int)->sitk.Image:
#     """
#     Crop the input image based on the given bounding box and dilation.

#     Parameters:
#     image (sitk.Image): The input image to be cropped.
#     bbox (tuple): The bounding box coordinates in the format (x_min, y_min, z_min, x_max, y_max, z_max).
#     dilation (int): The amount of dilation to be applied to the bounding box.

#     Returns:
#     sitk.Image: The cropped image.

#     """
#     start_index = [int(bbox[0]-dilation), int(bbox[1]-dilation), int(bbox[2]-dilation)]
#     size = [int(bbox[3] - bbox[0]+dilation*2), int(bbox[4] - bbox[1]+dilation*2), int(bbox[5] - bbox[2] +dilation*2)]
#     roi_filter = sitk.RegionOfInterestImageFilter()
#     roi_filter.SetIndex(start_index)
#     roi_filter.SetSize(size)
#     cropped_image = roi_filter.Execute(image)
#     return cropped_image

def get_bounding_box(image:sitk.Image)->tuple:
    """
    Calculate the bounding box coordinates of a given binary image.

    Parameters:
    image (sitk.Image): The input image.

    Returns:
    tuple: A tuple containing the coordinates of the bounding box in the format (xmin, ymin, zmin, xmax, ymax, zmax).
    """
    image_np = sitk.GetArrayFromImage(image)
    z, y, x = np.where(image_np)
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    zmin = np.min(z)
    zmax = np.max(z)

    bbox = [xmin, ymin, zmin, xmax, ymax, zmax]
    return bbox

def mask_image(image:sitk.Image, mask:sitk.Image, mask_value = -1000)->sitk.Image:
    """
    Masks the input image using the provided mask image.

    Parameters:
    image (sitk.Image): The input image to be masked.
    mask (sitk.Image): The mask image used for masking.
    mask_value (int, optional): The value to be assigned to the pixels outside the mask. Default is -1000.

    Returns:
    sitk.Image: The masked image.
    """
    mask = sitk.Cast(mask, sitk.sitkUInt8)
    masked_image = sitk.Mask(image, mask, outsideValue=mask_value)
    return masked_image

def resample_reference(image, ref_image, default_value=0, log=None)->sitk.Image:
    """
    Resamples the given image to the grid of a reference image.

    Parameters:
    - image: SimpleITK.Image
        The input struct to be resampled.
    - ref_image: SimpleITK.Image
        reference image for resampling.

    Returns:
    - resampled_image: SimpleITK.Image
        The resampled struct.

    """
    # Create a resample filter
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_image)
    resampler.SetDefaultPixelValue(default_value)
    # Apply the resampling
    resampled_image = resampler.Execute(image)
    
    if log != None:
        log.info(f'Struct resampled to reference image!')
    return resampled_image

def stitch_image(image_inside: sitk.Image, image_outside: sitk.Image, mask: sitk.Image) -> sitk.Image:
    """
    Stitches the `image_inside` and `image_outside` based on the `mask`.
    Also resamples the inside image and mask to be on same grid as outside image.

    Args:
        image_inside (sitk.Image): The image to be stitched inside the mask.
        image_outside (sitk.Image): The image to be stitched outside the mask.
        mask (sitk.Image): The mask used to determine the stitching region.

    Returns:
        sitk.Image: The stitched image.

    """
    image_inside = resample_reference(image_inside,image_outside,default_value=0)
    mask = resample_reference(mask,image_outside,default_value=0)
    mask = sitk.BinaryErode(mask, (1, 1, 1))
    image_inside_np = sitk.GetArrayFromImage(image_inside)
    mask_np = sitk.GetArrayFromImage(mask)
    image_outside_np = sitk.GetArrayFromImage(image_outside)

    # stitch images
    image_stitched_np = image_outside_np * (mask_np == 0) + image_inside_np * (mask_np > 0)
    image_stitched = sitk.GetImageFromArray(image_stitched_np)
    image_stitched.CopyInformation(image_outside)

    return image_stitched

def resample_image(image, new_spacing=[1.0, 1.0, 1.0],log=False)->sitk.Image:
    """
    Resamples the given image to a new spacing.

    Parameters:
    - image: SimpleITK.Image
        The input image to be resampled.
    - new_spacing: list, optional
        The desired spacing for the resampled image. Default is [1.0, 1.0, 1.0].

    Returns:
    - resampled_image: SimpleITK.Image
        The resampled image.

    """
    # Get the original image's spacing and size
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    # Calculate the new size based on the original and new spacing
    new_size = [int(round(osz*osp/nsp)) for osz, osp, nsp in zip(original_size, original_spacing, new_spacing)]

    # Create a resample filter
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(image.GetPixelIDValue())

    # Apply the resampling
    resampled_image = resampler.Execute(image)
    if log != False:
        log.info(f'Image resampled to new spacing {new_spacing}')
    return resampled_image

def rescale_image(image: sitk.Image, fov: sitk.Image, shift: float, clip: tuple, log=None) -> sitk.Image:
    """
    Rescale the given image by shifting and clipping the pixel values.

    Parameters:
    - image: SimpleITK.Image
        The input image to be rescaled.
    - shift: float
        The shift value to be applied to the pixel values.
    - clip: tuple
        The lower and upper bounds used for clipping the pixel values.

    Returns:
    - rescaled_image: SimpleITK.Image
        The rescaled image.

    """
    rescaled_image = image + shift
    rescaled_image = clip_image(rescaled_image, clip[0], clip[1])
    rescaled_image[fov==0]=clip[0]
    if log != None:
        log.info(f'Image rescaled with shift = {shift} and clip = {clip}')
    return rescaled_image


def cone_correction(fov:sitk.Image,log=None):
    """
    Apply cone correction to the given field of view (FOV) image. Only used for Task2.
    
    Parameters:
    - fov (sitk.Image): The input field of view image.
    - log (bool, optional): Whether to log cone correction information. Defaults to None.

    Returns:
    - sitk.Image: The cone-corrected field of view image.
    """
    fov_np = sitk.GetArrayFromImage(fov)
    fov_shape = fov_np.shape
    area = np.zeros(fov_shape[0])
    for i in range(fov_shape[0]):
        area[i] = np.sum(fov_np[i,:,:])
    area = area / np.max(area)
    area = [1 if i > 0.95 else 0 for i in area]
    full = np.argwhere(area)
    full_I = np.min(full)
    full_S = np.max(full)
    fov[:,:,:full_I] = 0
    fov[:,:,full_S+1:] = 0
    if log != None:
        log.info(f'Cone correction applied to FOV with full_I = {full_I} and full_S = {full_S}') 
    return fov

def segment_outline(input:sitk.Image,threshold:float=0.30,log=False)->sitk.Image:
    """
    Segment the outline of a given input image.

    Parameters:
    input (sitk.Image): The input image to segment.
    threshold (float): A relative threshold value for segmentation, 
                       can be used in case holes are appearing in the mask or too much 
                       of surrounding elements are included in the mask. Default is 0.30.

    Returns:
    sitk.Image: The segmented outline image.
    """
    
    # get patient outline segmentation
    input_np = sitk.GetArrayFromImage(input)
    
    #find range of values in image
    background = np.percentile(input_np, 2.5)
    high = np.percentile(input_np, 97.5)

    # create mask
    mask = input_np > background + threshold*(high-background)
    struct_erosion = np.ones((1,10,10))
    struct_dilation = np.ones((1,10,10))
    mask = ndimage.binary_erosion(mask,structure=struct_erosion).astype(mask.dtype)
    mask = ndimage.binary_dilation(mask,structure=struct_dilation).astype(mask.dtype)
    mask = ndimage.binary_erosion(mask,structure=struct_erosion).astype(mask.dtype)
    mask = ndimage.binary_dilation(mask,structure=struct_dilation).astype(mask.dtype)
    mask = sitk.ConnectedComponent(sitk.GetImageFromArray(mask.astype(int)))
    sorted_component_image = sitk.RelabelComponent(mask, sortByObjectSize=True)
    largest_component_binary_image = sorted_component_image == 1
    mask = largest_component_binary_image
    mask = sitk.BinaryMorphologicalClosing(largest_component_binary_image, (8, 8, 8))
    mask = sitk.BinaryFillhole(mask)
    mask.CopyInformation(input)
    mask = sitk.Cast(mask,sitk.sitkUInt8)
    
    # 2D axial hole filling 
    mask_np = sitk.GetArrayFromImage(mask)
    mask_np_filled = np.zeros_like(mask_np)
    for i in range(mask_np.shape[0]):
        mask_np_filled[i] = ndimage.binary_fill_holes(mask_np[i])

    mask_filled = sitk.GetImageFromArray(mask_np_filled)
    mask_filled.CopyInformation(mask)
    mask_filled = sitk.Cast(mask_filled, sitk.sitkUInt8)

    if log != False:
        log.info(f'Patient outline segmented using threshold {threshold}')
    
    return mask_filled

def postprocess_outline(mask:sitk.Image, fov:sitk.Image, dilation_radius:int=10,IS_correction=None, defacing_correction = None,log=False)->sitk.Image:
    """
    Postprocesses the input mask by dilating it and multiplying it with the field of view (FOV) image.

    Parameters:
    - mask (sitk.Image): The input mask image.
    - fov (sitk.Image): The field of view (FOV) image.
    - dilation_radius (int): The radius used for dilation. Default is 10. Dilation is only performed in-plane (2D)
    - IS_correction (int or None): The number of slices to be corrected in the inferior-superior direction. Default is None.
    - defacing_correction (sitk.Image or None): The defacing mask which is used to correct the patient outline (limit dilation in that area). Default is None.
    - cone_correction (None): Not yet implemented.

    Returns:
    - mask_final (sitk.Image): The postprocessed mask image.

    """
    if log != False:
        if IS_correction != None:
            IS_correction_str = True
        else:
            IS_correction_str = False
        if defacing_correction != None:
            defacing_correction_str = True
        else:
            defacing_correction_str = False
            
        log.info(f'Starting postprocessing of patient outline mask with IS_correction = {IS_correction_str}, defacing_correction = {defacing_correction_str}')
    
    # dilate mask 
    dilate = sitk.BinaryDilateImageFilter()
    dilate.SetKernelType(sitk.sitkBall)
    dilate.SetKernelRadius((dilation_radius,dilation_radius,0))
    mask_dilated = dilate.Execute(mask)
    
    # multiply with FOV to ensure there is no mask outside of FOV
    mask_final = mask_dilated*fov
    
    if IS_correction != None:
        bbox = get_bounding_box(mask_final)
        try:
            if bbox[2] > 0:
                bbox[2] = bbox[2]-1
            if bbox[5] <= mask_final.GetSize()[2]-1:
                bbox[5] = bbox[5]+1
            mask_final[:,:,bbox[2]:bbox[2]+IS_correction] = 0
            mask_final[:,:,bbox[5]-IS_correction:bbox[5]] = 0
        except Exception as e:
            print(f"Error in IS correction occured: {e}\n fallback to previous implementation" )
            if bbox[2] > 0:
                bbox[2] = bbox[2]-1
            if bbox[5] < mask_final.GetSize()[2]-1:
                bbox[5] = bbox[5]+1
            mask_final[:,:,bbox[2]:bbox[2]+IS_correction] = 0
            mask_final[:,:,bbox[5]-IS_correction:bbox[5]] = 0
        
    
    if defacing_correction != None:
        defacing_np = sitk.GetArrayFromImage(defacing_correction)
        mask_final_np = sitk.GetArrayFromImage(mask_final)
        mask_final_np[defacing_np==1]=0
        mask_final = sitk.GetImageFromArray(mask_final_np)
        mask_final.CopyInformation(mask)
    
    return mask_final

def crop_image(image:sitk.Image, mask:sitk.Image, margin:int=20) -> sitk.Image:
    """
    Crop the input image based on the boudning box of a provided mask.

    Args:
        image (sitk.Image): The input image to be cropped.
        mask (sitk.Image): The mask used to determine the cropping boundaries.
        margin (int, optional): The margin added to the cropping boundaries. Defaults to 20.

    Returns:
        sitk.Image: The cropped image.
    """
    
    # get 3D bounding box of mask
    img = sitk.GetArrayFromImage(mask)

    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    I, S = np.where(r)[0][[0, -1]]
    A, P = np.where(c)[0][[0, -1]]
    L, R = np.where(z)[0][[0, -1]]

    dims = np.shape(img)
    
    # add margin for cropping
    if margin is not None:
        if I - int(margin/3) >= 0:
            I = I - int(margin/3)
        if S + int(margin/3) < dims[0]:
            S = S + int(margin/3)
        else:
            S = dims[0] - 1
        if A - margin >= 0:
            A = A - margin
        if P + margin < dims[1]:
            P = P + margin
        else:
            P = dims[1] - 1
        if L - margin >= 0:
            L = L - margin
        if R + margin < dims[2]:
            R = R + margin
        else:
            R = dims[2] - 1
    
    # crop image
    cropper = sitk.CropImageFilter()
    cropper.SetLowerBoundaryCropSize((int(L), int(A), int(I)))
    cropper.SetUpperBoundaryCropSize((int(dims[2] - R), int(dims[1] - P), int(dims[0] - S)))
    image_cropped = cropper.Execute(image)
    
    return image_cropped

def warp_structure(structure:sitk.Image,transform):
    """
    Warps the structures with the transform generater by the deformable image registration.
    Also performs minor smoothing of the outline to reduce artifacts from b-spline registration

    Args:
        structure (sitk.Image): The input structure image to be transformed.
        transform: The transform parameter map to be applied.

    Returns:
        sitk.Image: The transformed structure image.
    """
    
    # read transform and change interpolator to nearest neighbor
    transform['FinalBSplineInterpolationOrder']='0'  

    # create bspline transformix filter
    transformer = sitk.TransformixImageFilter()
    transformer.SetTransformParameterMap(transform)
    transformer.LogToConsoleOff()
    transformer.LogToFileOff()
    transformer.SetMovingImage(structure)
    transformer.Execute()
    transformed_mask = transformer.GetResultImage()
    
    ## post-process mask to slightly smooth the edges
    transformed_mask = sitk.Threshold(sitk.Cast(transformed_mask, sitk.sitkUInt16),0,1)
    transformed_mask = sitk.BinaryDilate(transformed_mask, (2,2,2), sitk.sitkBall)
    transformed_mask = sitk.BinaryErode(transformed_mask, (2,2,2),sitk.sitkBall)
    
    return transformed_mask

def get_inverse_deformation(deformation, fixed_image):
    
    temp_dir = tempfile.mkdtemp()
    current_directory = os.getcwd()
    os.chdir(temp_dir)
    
    invert_params = sitk.ReadParameterFile('/workspace/code/preprocessing/configs/param_inverse_def.txt')
    elastix = sitk.ElastixImageFilter()
    elastix.SetParameterMap(invert_params)
    elastix.SetInitialTransformParameterFileName(deformation)
    elastix.SetFixedImage(fixed_image)
    elastix.SetMovingImage(fixed_image)
    elastix.LogToConsoleOff()
    elastix.LogToFileOff()
    elastix.SetNumberOfThreads(16)
    elastix.Execute()
    inverted_def = elastix.GetTransformParameterMap()[0]
    inverted_def['InitialTransformParametersFileName'] = ['NoInitialTransform']
    
    os.chdir(current_directory)
    shutil.rmtree(temp_dir)
    
    return inverted_def
    
def invert_structure(struct, rigid_reg, inverted_def_reg,ct):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ct)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(rigid_reg)
    
    struct_rigid = resampler.Execute(struct)
    struct_inverted = warp_structure(struct_rigid,inverted_def_reg)
    return struct_rigid, struct_inverted

def preprocess_structures(patient,input,ct_s1,fov_s1,fov,rigid_reg,transform_def,mask,log=None):
    output_dir = patient['output_dir']
    region = patient['region']
    invert = patient['invert_structures']
    
    #make list of all relevant structures
    structures = os.listdir(os.path.join(output_dir,'structures'))
    structures = [struct for struct in structures if 
                    struct.endswith('.nrrd') or 
                    struct.endswith('.nii') or 
                    struct.endswith('.nii.gz')]
    structures = [struct for struct in structures if not 
                    struct.endswith('_s2.nrrd') and not 
                    struct.endswith('_s2_def.nrrd') and not
                    struct.endswith('_stitched.nrrd')]
    if invert:
        transform_def_inv = get_inverse_deformation(os.path.join(output_dir,'transform_def.txt'),input)
    if patient['defacing_correction'] == True:
            face = read_image(os.path.join(patient['output_dir'],'defacing_mask.nii.gz'),log=log)
    #pre-process each structure
    for struct in structures:
        struct_path = os.path.join(output_dir,'structures',struct)
        struct_img = read_image(struct_path,log=log)
        struct_img_orig = resample_reference(struct_img,ct_s1)
        if invert:
            struct_rigid, struct_inv = invert_structure(struct_img,rigid_reg,transform_def_inv,ct_s1)
            struct_deformed = struct_rigid
            struct_deformed = crop_image(struct_deformed,fov_s1)
            struct_deformed = mask_image(struct_deformed,fov,0)
            struct_img = struct_inv
            #struct_deformed = mask_image(struct_deformed,fov,0)
            #struct_img = mask_image(struct_img,fov,0)
            struct_stitched = stitch_image(struct_deformed, struct_img_orig, mask)
        else:
            if region == 'HN':
                struct_img_orig[face == 1] = 0
            struct_img = crop_image(struct_img_orig,fov_s1)
            struct_img = mask_image(struct_img,fov,0)
            struct_deformed = warp_structure(struct_img_orig,transform_def)
            struct_deformed = resample_reference(struct_deformed,fov,0)
            struct_deformed = mask_image(struct_deformed,fov,0)
            struct_stitched = stitch_image(struct_deformed, struct_img_orig, mask)
        save_image(struct_stitched,os.path.join(output_dir,'structures',struct.split('.')[0]+'_stitched.nrrd'))
        save_image(struct_img,os.path.join(output_dir,'structures',struct.split('.')[0]+'_s2.nrrd'))
        save_image(struct_deformed,os.path.join(output_dir,'structures',struct.split('.')[0]+'_s2_def.nrrd'))
    

def generate_overview_png(ct:sitk.Image,input:sitk.Image,mask:sitk.Image,patient_dict:dict)->None:
    """
    Generate an overview PNG image showing slices from different orientations of the input CT, input image, and mask.

    Parameters:
    ct (sitk.Image): The CT image.
    input (sitk.Image): The input image.
    mask (sitk.Image): The mask image.
    output_dir (str): The directory to save the overview PNG image.

    Returns:
    None
    """
    
    shape = np.shape(sitk.GetArrayFromImage(ct))
    background_ct = np.percentile(sitk.GetArrayFromImage(ct), 0.1)
    high_ct = np.percentile(sitk.GetArrayFromImage(ct), 99.9)
    background_input = np.percentile(sitk.GetArrayFromImage(input), 0.1)
    high_input = np.percentile(sitk.GetArrayFromImage(input), 99.9)

    slice_sag = shape[2]//2
    slice_cor = shape[1]//2
    slice_ax = shape[0]//2
    
    #calculate final size of figure so minimal white space in figure
    sag_len = slice_sag*2
    cor_len = slice_cor*2
    ax_len = slice_ax*2
    height_ratios = [cor_len/cor_len,((ax_len*3)/cor_len)*sag_len/cor_len,ax_len*3/cor_len]
    x_len = (sag_len/cor_len)*3
    y_len = np.array(height_ratios).sum()/x_len
    gridspec_kw={'width_ratios':[1,1,1],'height_ratios':height_ratios}
    
    size=13
    fig,ax = plt.subplots(3,3,figsize=(size,y_len*size),gridspec_kw=gridspec_kw)
    
    ax[0,0].imshow(sitk.GetArrayFromImage(ct)[slice_ax,:,:],cmap='gray',vmin=background_ct,vmax=high_ct)
    ax[0,1].imshow(sitk.GetArrayFromImage(input)[slice_ax,:,:],cmap='gray',vmin=background_input,vmax=high_input)
    ax[0,1].contour(sitk.GetArrayFromImage(mask)[slice_ax,:,:],levels=[0.5],colors='r')
    ax[0,2].imshow(sitk.GetArrayFromImage(input)[slice_ax,:,:],cmap='Reds',alpha=0.5,vmin=background_input,vmax=high_input)
    ax[0,2].imshow(sitk.GetArrayFromImage(ct)[slice_ax,:,:],cmap='Blues',alpha=0.5,vmin=background_ct,vmax=high_ct)
    
    ax[1,0].imshow(sitk.GetArrayFromImage(ct)[::-1,:,slice_sag],cmap='gray',aspect=3,vmin=background_ct,vmax=high_ct)
    ax[1,1].imshow(sitk.GetArrayFromImage(input)[::-1,:,slice_sag],cmap='gray',aspect=3,vmin=background_input,vmax=high_input)
    ax[1,1].contour(sitk.GetArrayFromImage(mask)[::-1,:,slice_sag],levels=[0.5],colors='r')
    ax[1,2].imshow(sitk.GetArrayFromImage(input)[::-1,:,slice_sag],cmap='Reds',aspect=3,alpha=0.5,vmin=background_input,vmax=high_input)
    ax[1,2].imshow(sitk.GetArrayFromImage(ct)[::-1,:,slice_sag],cmap='Blues',aspect=3,alpha=0.5,vmin=background_ct,vmax=high_ct)  

    ax[2,0].imshow(sitk.GetArrayFromImage(ct)[::-1,slice_cor,:],cmap='gray',aspect=3,vmin=background_ct,vmax=high_ct)
    ax[2,1].imshow(sitk.GetArrayFromImage(input)[::-1,slice_cor,:],cmap='gray',aspect=3,vmin=background_input,vmax=high_input)
    ax[2,1].contour(sitk.GetArrayFromImage(mask)[::-1,slice_cor,:],levels=[0.5],colors='r')
    ax[2,2].imshow(sitk.GetArrayFromImage(input)[::-1,slice_cor,:],cmap='Reds',aspect=3,alpha=0.5,vmin=background_input,vmax=high_input)
    ax[2,2].imshow(sitk.GetArrayFromImage(ct)[::-1,slice_cor,:],cmap='Blues',aspect=3,alpha=0.5,vmin=background_ct,vmax=high_ct)

    def add_text(ax,text):
        props = dict(facecolor='white', alpha=0.9, edgecolor='white', boxstyle='round,pad=0.5')
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,verticalalignment='top', bbox=props)
    
    def add_patient(ax,text):
        props = dict(facecolor='white', alpha=0.9, edgecolor='white', boxstyle='round,pad=0.5')
        ax.text(0.95, 0.95, text, transform=ax.transAxes, fontsize=10,verticalalignment='top',horizontalalignment='right',bbox=props)

    for r,ax_row in enumerate(ax):
        for c,a in enumerate(ax_row):
            a.set_xticks([])
            a.set_yticks([])
            if c == 0:
                add_text(a,'CT')
                add_patient(a,patient_dict['ID'])
                a.set_ylabel('Axial' if r == 0 else 'Sagittal' if r == 1 else 'Coronal',fontsize=12,fontweight='bold')
            if c == 1:
                add_text(a,'Input + Mask')
                add_patient(a,patient_dict['ID'])
            if c == 2:
                add_text(a,'Overlay')
                add_patient(a,patient_dict['ID'])
    
    fig.subplots_adjust(wspace=0.02,hspace=0.02)
    plt.tight_layout()
    plt.savefig(os.path.join(patient_dict['output_dir'],f'{patient_dict['ID']}.png'),dpi=300,bbox_inches='tight')

def generate_overview_val_test(ct:sitk.Image,input:sitk.Image,ct_deformed:sitk.Image, mask:sitk.Image,patient_dict:dict)->None:
    """
    Generate an overview PNG image showing slices from different orientations of the input CT, input image, and mask.

    Parameters:
    ct (sitk.Image): The CT image.
    input (sitk.Image): The input image.
    mask (sitk.Image): The mask image.
    output_dir (str): The directory to save the overview PNG image.

    Returns:
    None
    """
    
    shape = np.shape(sitk.GetArrayFromImage(ct))
    background_ct = np.percentile(sitk.GetArrayFromImage(ct), 0.1)
    high_ct = np.percentile(sitk.GetArrayFromImage(ct), 99.9)
    background_input = np.percentile(sitk.GetArrayFromImage(input), 0.1)
    high_input = np.percentile(sitk.GetArrayFromImage(input), 99.9)

    slice_sag = shape[2]//2
    slice_cor = shape[1]//2
    slice_ax = shape[0]//2
    
    #calculate final size of figure so minimal white space in figure
    sag_len = slice_sag*2
    cor_len = slice_cor*2
    ax_len = slice_ax*2
    height_ratios = [cor_len/cor_len,((ax_len*3)/cor_len)*sag_len/cor_len,ax_len*3/cor_len]
    x_len = (sag_len/cor_len)*5
    y_len = np.array(height_ratios).sum()/x_len
    gridspec_kw={'width_ratios':[1,1,1,1,1],'height_ratios':height_ratios}
    
    size=20
    fig,ax = plt.subplots(3,5,figsize=(size,y_len*size),gridspec_kw=gridspec_kw)
    
    ax[0,1].imshow(sitk.GetArrayFromImage(ct)[slice_ax,:,:],cmap='gray',vmin=background_ct,vmax=high_ct)
    ax[0,0].imshow(sitk.GetArrayFromImage(input)[slice_ax,:,:],cmap='gray',vmin=background_input,vmax=high_input)
    ax[0,0].contour(sitk.GetArrayFromImage(mask)[slice_ax,:,:],levels=[0.5],colors='r')
    ax[0,3].imshow(sitk.GetArrayFromImage(input)[slice_ax,:,:],cmap='Reds',alpha=0.5,vmin=background_input,vmax=high_input)
    ax[0,3].imshow(sitk.GetArrayFromImage(ct)[slice_ax,:,:],cmap='Blues',alpha=0.5,vmin=background_ct,vmax=high_ct)
    ax[0,2].imshow(sitk.GetArrayFromImage(ct_deformed)[slice_ax,:,:],cmap='gray',vmin=background_ct,vmax=high_ct)
    ax[0,4].imshow(sitk.GetArrayFromImage(input)[slice_ax,:,:],cmap='Reds',alpha=0.5,vmin=background_input,vmax=high_input)
    ax[0,4].imshow(sitk.GetArrayFromImage(ct_deformed)[slice_ax,:,:],cmap='Blues',alpha=0.5,vmin=background_ct,vmax=high_ct)
    
    ax[1,1].imshow(sitk.GetArrayFromImage(ct)[::-1,:,slice_sag],cmap='gray',aspect=3,vmin=background_ct,vmax=high_ct)
    ax[1,0].imshow(sitk.GetArrayFromImage(input)[::-1,:,slice_sag],cmap='gray',aspect=3,vmin=background_input,vmax=high_input)
    ax[1,0].contour(sitk.GetArrayFromImage(mask)[::-1,:,slice_sag],levels=[0.5],colors='r')
    ax[1,3].imshow(sitk.GetArrayFromImage(input)[::-1,:,slice_sag],cmap='Reds',aspect=3,alpha=0.5,vmin=background_input,vmax=high_input)
    ax[1,3].imshow(sitk.GetArrayFromImage(ct)[::-1,:,slice_sag],cmap='Blues',aspect=3,alpha=0.5,vmin=background_ct,vmax=high_ct)  
    ax[1,2].imshow(sitk.GetArrayFromImage(ct_deformed)[::-1,:,slice_sag],cmap='gray',aspect=3,vmin=background_ct,vmax=high_ct)
    ax[1,4].imshow(sitk.GetArrayFromImage(input)[::-1,:,slice_sag],cmap='Reds',aspect=3,alpha=0.5,vmin=background_input,vmax=high_input)
    ax[1,4].imshow(sitk.GetArrayFromImage(ct_deformed)[::-1,:,slice_sag],cmap='Blues',aspect=3,alpha=0.5,vmin=background_ct,vmax=high_ct)
    
    ax[2,1].imshow(sitk.GetArrayFromImage(ct)[::-1,slice_cor,:],cmap='gray',aspect=3,vmin=background_ct,vmax=high_ct)
    ax[2,0].imshow(sitk.GetArrayFromImage(input)[::-1,slice_cor,:],cmap='gray',aspect=3,vmin=background_input,vmax=high_input)
    ax[2,0].contour(sitk.GetArrayFromImage(mask)[::-1,slice_cor,:],levels=[0.5],colors='r')
    ax[2,3].imshow(sitk.GetArrayFromImage(input)[::-1,slice_cor,:],cmap='Reds',aspect=3,alpha=0.5,vmin=background_input,vmax=high_input)
    ax[2,3].imshow(sitk.GetArrayFromImage(ct)[::-1,slice_cor,:],cmap='Blues',aspect=3,alpha=0.5,vmin=background_ct,vmax=high_ct)
    ax[2,2].imshow(sitk.GetArrayFromImage(ct_deformed)[::-1,slice_cor,:],cmap='gray',aspect=3,vmin=background_ct,vmax=high_ct)
    ax[2,4].imshow(sitk.GetArrayFromImage(input)[::-1,slice_cor,:],cmap='Reds',aspect=3,alpha=0.5,vmin=background_input,vmax=high_input)
    ax[2,4].imshow(sitk.GetArrayFromImage(ct_deformed)[::-1,slice_cor,:],cmap='Blues',aspect=3,alpha=0.5,vmin=background_ct,vmax=high_ct)
    
    def add_text(ax,text):
        props = dict(facecolor='white', alpha=0.9, edgecolor='white', boxstyle='round,pad=0.5')
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,verticalalignment='top', bbox=props)
    
    def add_patient(ax,text):
        props = dict(facecolor='white', alpha=0.9, edgecolor='white', boxstyle='round,pad=0.5')
        ax.text(0.95, 0.95, text, transform=ax.transAxes, fontsize=10,verticalalignment='top',horizontalalignment='right',bbox=props)

    for r,ax_row in enumerate(ax):
        for c,a in enumerate(ax_row):
            a.set_xticks([])
            a.set_yticks([])
            if c == 0:
                add_text(a,'Input + Mask')
                add_patient(a,patient_dict['ID'])
                a.set_ylabel('Axial' if r == 0 else 'Sagittal' if r == 1 else 'Coronal',fontsize=12,fontweight='bold')
            if c == 1:
                add_text(a,'CT')
                add_patient(a,patient_dict['ID'])
            if c == 2:
                add_text(a,'CT def')
                add_patient(a,patient_dict['ID'])
            if c == 3:
                add_text(a,'Overlay')
                add_patient(a,patient_dict['ID'])
            if c == 4:
                add_text(a,'Overlay def')
                add_patient(a,patient_dict['ID'])
    
    fig.subplots_adjust(wspace=0.02,hspace=0.02)
    plt.tight_layout()
    plt.savefig(os.path.join(patient_dict['output_dir'],f'{patient_dict['ID']}_def.png'),dpi=300,bbox_inches='tight')


def generate_overview_planning(ct:sitk.Image,input:sitk.Image,ct_deformed:sitk.Image, mask:sitk.Image,patient_dict:dict)->None:
    """
    Generate an overview PNG image showing slices from different orientations of the input CT, input image, and mask.

    Parameters:
    ct (sitk.Image): The CT image.
    input (sitk.Image): The input image.
    mask (sitk.Image): The mask image.
    output_dir (str): The directory to save the overview PNG image.

    Returns:
    None
    """
    
    shape = np.shape(sitk.GetArrayFromImage(ct))
    background_ct = np.percentile(sitk.GetArrayFromImage(ct), 0.1)
    high_ct = np.percentile(sitk.GetArrayFromImage(ct), 99.9)
    background_input = np.percentile(sitk.GetArrayFromImage(input), 0.1)
    high_input = np.percentile(sitk.GetArrayFromImage(input), 99.9)

    slice_sag = shape[2]//2
    slice_cor = shape[1]//2
    slice_ax = shape[0]//2
    
    #calculate final size of figure so minimal white space in figure
    sag_len = slice_sag*2
    cor_len = slice_cor*2
    ax_len = slice_ax*2
    height_ratios = [cor_len/cor_len,((ax_len*3)/cor_len)*sag_len/cor_len,ax_len*3/cor_len]
    x_len = (sag_len/cor_len)*5
    y_len = np.array(height_ratios).sum()/x_len
    gridspec_kw={'width_ratios':[1,1,1,1,1],'height_ratios':height_ratios}
    
    size=20
    fig,ax = plt.subplots(3,5,figsize=(size,y_len*size),gridspec_kw=gridspec_kw)
    
    ax[0,1].imshow(sitk.GetArrayFromImage(ct)[slice_ax,:,:],cmap='gray',vmin=background_ct,vmax=high_ct)
    ax[0,0].imshow(sitk.GetArrayFromImage(input)[slice_ax,:,:],cmap='gray',vmin=background_input,vmax=high_input)
    ax[0,0].contour(sitk.GetArrayFromImage(mask)[slice_ax,:,:],levels=[0.5],colors='r')
    ax[0,3].imshow(sitk.GetArrayFromImage(input)[slice_ax,:,:],cmap='Reds',alpha=0.5,vmin=background_input,vmax=high_input)
    ax[0,3].imshow(sitk.GetArrayFromImage(ct)[slice_ax,:,:],cmap='Blues',alpha=0.5,vmin=background_ct,vmax=high_ct)
    ax[0,2].imshow(sitk.GetArrayFromImage(ct_deformed)[slice_ax,:,:],cmap='gray',vmin=background_ct,vmax=high_ct)
    ax[0,4].imshow(sitk.GetArrayFromImage(input)[slice_ax,:,:],cmap='Reds',alpha=0.5,vmin=background_input,vmax=high_input)
    ax[0,4].imshow(sitk.GetArrayFromImage(ct_deformed)[slice_ax,:,:],cmap='Blues',alpha=0.5,vmin=background_ct,vmax=high_ct)
    
    ax[1,1].imshow(sitk.GetArrayFromImage(ct)[::-1,:,slice_sag],cmap='gray',aspect=3,vmin=background_ct,vmax=high_ct)
    ax[1,0].imshow(sitk.GetArrayFromImage(input)[::-1,:,slice_sag],cmap='gray',aspect=3,vmin=background_input,vmax=high_input)
    ax[1,0].contour(sitk.GetArrayFromImage(mask)[::-1,:,slice_sag],levels=[0.5],colors='r')
    ax[1,3].imshow(sitk.GetArrayFromImage(input)[::-1,:,slice_sag],cmap='Reds',aspect=3,alpha=0.5,vmin=background_input,vmax=high_input)
    ax[1,3].imshow(sitk.GetArrayFromImage(ct)[::-1,:,slice_sag],cmap='Blues',aspect=3,alpha=0.5,vmin=background_ct,vmax=high_ct)  
    ax[1,2].imshow(sitk.GetArrayFromImage(ct_deformed)[::-1,:,slice_sag],cmap='gray',aspect=3,vmin=background_ct,vmax=high_ct)
    ax[1,4].imshow(sitk.GetArrayFromImage(input)[::-1,:,slice_sag],cmap='Reds',aspect=3,alpha=0.5,vmin=background_input,vmax=high_input)
    ax[1,4].imshow(sitk.GetArrayFromImage(ct_deformed)[::-1,:,slice_sag],cmap='Blues',aspect=3,alpha=0.5,vmin=background_ct,vmax=high_ct)
    
    ax[2,1].imshow(sitk.GetArrayFromImage(ct)[::-1,slice_cor,:],cmap='gray',aspect=3,vmin=background_ct,vmax=high_ct)
    ax[2,0].imshow(sitk.GetArrayFromImage(input)[::-1,slice_cor,:],cmap='gray',aspect=3,vmin=background_input,vmax=high_input)
    ax[2,0].contour(sitk.GetArrayFromImage(mask)[::-1,slice_cor,:],levels=[0.5],colors='r')
    ax[2,3].imshow(sitk.GetArrayFromImage(input)[::-1,slice_cor,:],cmap='Reds',aspect=3,alpha=0.5,vmin=background_input,vmax=high_input)
    ax[2,3].imshow(sitk.GetArrayFromImage(ct)[::-1,slice_cor,:],cmap='Blues',aspect=3,alpha=0.5,vmin=background_ct,vmax=high_ct)
    ax[2,2].imshow(sitk.GetArrayFromImage(ct_deformed)[::-1,slice_cor,:],cmap='gray',aspect=3,vmin=background_ct,vmax=high_ct)
    ax[2,4].imshow(sitk.GetArrayFromImage(input)[::-1,slice_cor,:],cmap='Reds',aspect=3,alpha=0.5,vmin=background_input,vmax=high_input)
    ax[2,4].imshow(sitk.GetArrayFromImage(ct_deformed)[::-1,slice_cor,:],cmap='Blues',aspect=3,alpha=0.5,vmin=background_ct,vmax=high_ct)
    
    try:
        structures = os.listdir(os.path.join(patient_dict['output_dir'],'structures'))
        structures = [i for i in structures if i.endswith('s2_def.nrrd')]
        colormap = plt.get_cmap('tab20')
        num_colors = 20
        colors = [mcolors.rgb2hex(colormap((i / num_colors))) for i in range(num_colors)]
        lines = [0 for i in range(len(structures))]
        for i, struct in enumerate(structures):
            struct = sitk.ReadImage(os.path.join(patient_dict['output_dir'],'structures',struct))
            struct = sitk.GetArrayFromImage(struct)
            ax[0,2].contour(struct[slice_ax,:,:],alpha=0.5, colors=colors[i%len(colors)],linewidths=0.5)
            ax[1,2].contour(struct[::-1,:,slice_sag],alpha=0.5, colors=colors[i%len(colors)],linewidths=0.5)
            ax[2,2].contour(struct[::-1,slice_cor,:],alpha=0.5, colors=colors[i%len(colors)],linewidths=0.5)
            lines[i]=ax[2,2].plot(0,0,color=colors[i%len(colors)],label='_'.join(structures[i].split('_')[0:-2])[:20])
        rows=math.ceil(len(structures)/8)
        fig.legend(loc='lower center',bbox_to_anchor=(0.5, -0.026*rows),ncol=8,fontsize=10)
    except:
        print('No structures found...')
    
    def add_text(ax,text):
        props = dict(facecolor='white', alpha=0.9, edgecolor='white', boxstyle='round,pad=0.5')
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,verticalalignment='top', bbox=props)
    
    def add_patient(ax,text):
        props = dict(facecolor='white', alpha=0.9, edgecolor='white', boxstyle='round,pad=0.5')
        ax.text(0.95, 0.95, text, transform=ax.transAxes, fontsize=10,verticalalignment='top',horizontalalignment='right',bbox=props)

    for r,ax_row in enumerate(ax):
        for c,a in enumerate(ax_row):
            a.set_xticks([])
            a.set_yticks([])
            if c == 0:
                add_text(a,'Input + Mask')
                add_patient(a,patient_dict['ID'])
                a.set_ylabel('Axial' if r == 0 else 'Sagittal' if r == 1 else 'Coronal',fontsize=12,fontweight='bold')
            if c == 1:
                add_text(a,'CT')
                add_patient(a,patient_dict['ID'])
            if c == 2:
                add_text(a,'CT def')
                add_patient(a,patient_dict['ID'])
            if c == 3:
                add_text(a,'Overlay')
                add_patient(a,patient_dict['ID'])
            if c == 4:
                add_text(a,'Overlay def')
                add_patient(a,patient_dict['ID'])
    
    fig.subplots_adjust(wspace=0.02,hspace=0.02)
    plt.tight_layout()
    plt.savefig(os.path.join(patient_dict['output_dir'],f'{patient_dict['ID']}_planning.png'),dpi=300,bbox_inches='tight')

def generate_overview_stage1(ct:sitk.Image,input:sitk.Image,output_dir:str)->None:
    """
    Generate an overview PNG image showing slices from different orientations of the input CT and input image after stage 1 preprocessing.

    Parameters:
    ct (sitk.Image): The CT image.
    input (sitk.Image): The input image.
    output_dir (str): The directory to save the overview PNG image.

    Returns:
    None
    """
    
    shape = np.shape(sitk.GetArrayFromImage(ct))
    background_ct = np.percentile(sitk.GetArrayFromImage(ct), 0.1)
    high_ct = np.percentile(sitk.GetArrayFromImage(ct), 99.9)
    background_input = np.percentile(sitk.GetArrayFromImage(input), 0.1)
    high_input = np.percentile(sitk.GetArrayFromImage(input), 99.9)

    slice_sag = shape[2]//2
    slice_cor = shape[1]//2
    slice_ax = shape[0]//2
    fig,ax = plt.subplots(3,3,figsize=(15,15))
    ax[0,0].imshow(sitk.GetArrayFromImage(ct)[::-1,:,slice_sag],cmap='gray',aspect=3,vmin=background_ct,vmax=high_ct)
    ax[0,1].imshow(sitk.GetArrayFromImage(input)[::-1,:,slice_sag],cmap='gray',aspect=3,vmin=background_input,vmax=high_input)
    ax[0,2].imshow(sitk.GetArrayFromImage(input)[::-1,:,slice_sag],cmap='Reds',aspect=3,alpha=0.5,vmin=background_input,vmax=high_input)
    ax[0,2].imshow(sitk.GetArrayFromImage(ct)[::-1,:,slice_sag],cmap='Blues',aspect=3,alpha=0.5,vmin=background_ct,vmax=high_ct)  

    ax[1,0].imshow(sitk.GetArrayFromImage(ct)[::-1,slice_cor,:],cmap='gray',aspect=3,vmin=background_ct,vmax=high_ct)
    ax[1,1].imshow(sitk.GetArrayFromImage(input)[::-1,slice_cor,:],cmap='gray',aspect=3,vmin=background_input,vmax=high_input)
    ax[1,2].imshow(sitk.GetArrayFromImage(input)[::-1,slice_cor,:],cmap='Reds',aspect=3,alpha=0.5,vmin=background_input,vmax=high_input)
    ax[1,2].imshow(sitk.GetArrayFromImage(ct)[::-1,slice_cor,:],cmap='Blues',aspect=3,alpha=0.5,vmin=background_ct,vmax=high_ct)

    ax[2,0].imshow(sitk.GetArrayFromImage(ct)[slice_ax,:,:],cmap='gray',vmin=background_ct,vmax=high_ct)
    ax[2,1].imshow(sitk.GetArrayFromImage(input)[slice_ax,:,:],cmap='gray',vmin=background_input,vmax=high_input)
    ax[2,2].imshow(sitk.GetArrayFromImage(input)[slice_ax,:,:],cmap='Reds',alpha=0.5,vmin=background_input,vmax=high_input)
    ax[2,2].imshow(sitk.GetArrayFromImage(ct)[slice_ax,:,:],cmap='Blues',alpha=0.5,vmin=background_ct,vmax=high_ct)
    
    plt.savefig(os.path.join(output_dir,'overview_s1.png'),dpi=300,bbox_inches='tight')

def read_csv_lines(file):
    """
    Reads a CSV file and returns a list of lines.

    Args:
        file (str): The path to the CSV file.

    Returns:
        list: A list of lines, where each line is represented as a list of values.

    """
    lines = []
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            lines.append(row)
    return lines

def csv_to_dict(file):
    """
    Convert a CSV file to a dictionary.

    Args:
        file (str): The path to the CSV file.

    Returns:
        dict: A dictionary representation of the CSV file, where the keys are the values in the first column
              and the values are dictionaries representing each row, with keys as the column names and values
              as the corresponding values in the row.
    """
    lines = read_csv_lines(file)
    # convert lines in dict
    patients_dict = {}
    for line in lines[1:]:
        if line[0][0]=='#':
            continue
        patients_dict[line[0]] = {}
        for i in range(0, len(line)):
            patients_dict[line[0]][lines[0][i]] = line[i]
        try:
            patients_dict[line[0]]['task'] = int(patients_dict[line[0]]['task'])
        except:
            pass
        try:
            patients_dict[line[0]]['background'] = float(patients_dict[line[0]]['background'])
        except:
            pass
        try:
            patients_dict[line[0]]['defacing'] = True if patients_dict[line[0]]['defacing'] == 'True' or patients_dict[line[0]]['defacing'] == 'TRUE' else False
        except:
            pass
        try:
            patients_dict[line[0]]['order'] = [int(i) for i in patients_dict[line[0]]['order'].strip('[]').split(',')]
        except:
            pass
        try:
            patients_dict[line[0]]['flip'] = [True if x == 'True' else False for x in patients_dict[line[0]]['flip'].strip('[]').split(',')]
        except:
            pass
        try:
            patients_dict[line[0]]['intensity_shift'] = float(patients_dict[line[0]]['intensity_shift'])
        except:
            pass
        try:
            patients_dict[line[0]]['reg_fovmask'] = True if patients_dict[line[0]]['reg_fovmask'] == 'True' or patients_dict[line[0]]['reg_fovmask'] == 'TRUE' else False
        except:
            pass
        try:
            patients_dict[line[0]]['mr_overlap_correction'] = True if patients_dict[line[0]]['mr_overlap_correction'] == 'True' or patients_dict[line[0]]['mr_overlap_correction'] == 'TRUE' else False
        except:
            pass
        try:
            patients_dict[line[0]]['resample'] = [float(i) for i in patients_dict[line[0]]['resample'].strip('[]').split(',')]
        except:
            pass
        try:
            patients_dict[line[0]]['mask_thresh'] = float(patients_dict[line[0]]['mask_thresh'])
        except:
            pass
        try:
            patients_dict[line[0]]['defacing_correction'] = True if patients_dict[line[0]]['defacing_correction'] == 'True' or patients_dict[line[0]]['defacing_correction'] == 'TRUE' else False
        except:
            pass
        try:
            patients_dict[line[0]]['IS_correction'] = True if patients_dict[line[0]]['IS_correction'] == 'True' or patients_dict[line[0]]['IS_correction'] == 'TRUE' else False
        except:
            pass
        try:
            patients_dict[line[0]]['cone_correction'] = True if patients_dict[line[0]]['cone_correction'] == 'True' or patients_dict[line[0]]['cone_correction'] == 'TRUE' else False
        except:
            pass
        try:
            patients_dict[line[0]]['invert_structures'] = True if patients_dict[line[0]]['invert_structures'] == 'True' or patients_dict[line[0]]['invert_structures'] == 'TRUE' else False
        except:
            pass


    return patients_dict