import SimpleITK as sitk
#import pyplastimatch
import numpy as np
import nibabel as nib
from typing import List,Union
from scipy.signal import find_peaks
from totalsegmentator.python_api import totalsegmentator
from scipy import ndimage

def read_image(image_path:str)->sitk.Image:
    """
    Read an image from the specified image path using SimpleITK.

    Parameters:
    image_path (str): The path to the image file. All ITK file formats can be loaded.

    Returns:
    sitk.Image: The loaded image.

    """
    image = sitk.ReadImage(image_path)
    return image

def read_dicom_image(image_path:str)->sitk.Image:
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
    return image

def save_image(image:sitk.Image, image_path:str,compression=True):
    """
    Save the given SimpleITK image to the specified file path.
    
    Args:
        image (sitk.Image): The SimpleITK image to be saved.
        image_path (str): The file path where the image will be saved.
    """
    sitk.WriteImage(image, image_path, useCompression=compression)

# def convert_rtstruct_to_nrrd(rtstruct_path:str, nrrd_dir_path:str):
#     """
#     Converts an RTSTRUCT file to NRRD format.

#     Parameters:
#     rtstruct_path (str): The path to the RTSTRUCT file.
#     nrrd_dir_path (str): The directory path where the NRRD files will be saved.

#     Returns:
#     None
#     """
#     convert_args_ct = {"input" :            rtstruct_path,
#                         "output-prefix" :   nrrd_dir_path,
#                         "prefix-format" :   'nrrd',
#                         }
#     pyplastimatch.convert(**convert_args_ct)

def rigid_registration(fixed:sitk.Image, moving:sitk.Image, parameter)->Union[sitk.Image,sitk.Transform]:
    """
    Perform rigid registration between a fixed image and a moving image using the given parameter file.

    Parameters:
    fixed (sitk.Image): The fixed image to register.
    moving (sitk.Image): The moving image to register.
    parameter: The parameter file for the registration.

    Returns:
    Tuple[sitk.Image, sitk.Transform]: A tuple containing the registered image and the inverse transform.

    """
    
    # Perform registration based on parameter file
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetParameterMap(parameter)
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
        background = 0
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

def deformable_registration(fixed:sitk.Image, moving:sitk.Image, parameter)->Union[sitk.Image,sitk.Transform]:
    
    # Perform registration based on parameter file
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetParameterMap(parameter)
    elastixImageFilter.SetFixedImage(fixed)  # due to FOV differences CT first registered to MR an inverted in the end
    elastixImageFilter.SetMovingImage(moving)
    elastixImageFilter.LogToConsoleOn()
    elastixImageFilter.LogToFileOff()
    elastixImageFilter.Execute()
    deformed = elastixImageFilter.GetResultImage()

    return deformed

def correct_orientation(input_image:sitk.Image,order=[0,1,2],flip=[False,False,False]):
    """
    Corrects the orientation of an input image based on the specified order and flip parameters.

    Parameters:
    input_image (sitk.Image): The input image to be corrected.
    order (list[int]): The order of axes for permuting the image. Default is [0, 1, 2].
    flip (list[bool]): The flip status for each axis. Default is [False, False, False].

    Returns:
    sitk.Image: The corrected image with the specified orientation.
    """
    image_permuted = sitk.PermuteAxes(input_image, order)
    image_flipped = sitk.Flip(image_permuted,flip)
    image_flipped.SetDirection([1,0,0,0,1,0,0,0,1])
    return image_flipped

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

def segment_defacing(ct_image:sitk.Image,structures=['brain','skull'])->Union[sitk.Image,sitk.Image]:
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
    segmentation = totalsegmentator(ct_nib,output=None,roi_subset=structures,quiet=True)
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
    
    return brain,skull

def defacing(brain_mask:sitk.Image, skull_mask:sitk.Image)->sitk.Image:
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
            for j in range(y_skull,y_brain):
                if j > k*i + d:
                    face[j,i,l] = 1
    defacing_mask = sitk.GetImageFromArray(face)
    defacing_mask.CopyInformation(brain_mask)
    defacing_mask = sitk.Cast(defacing_mask, sitk.sitkUInt8)
    return defacing_mask

def segment_outline(ct_image:sitk.Image,fast=False)->sitk.Image:
    """
    Segment the patient outline from a CT/CBCT image using totalsegmentator.

    Parameters:
    - ct_image (sitk.Image): The input CT image.
    - fast (bool): Whether to use fast mode for segmentation. Default is False.

    Returns:
    - sitk.Image: The segmented patient outline image.
    """
    ct_nib = sitk_to_nib(ct_image)
    segmentation = totalsegmentator(ct_nib,task='body',fast=fast,output=None,quiet=True)
    structures = nib_to_sitk(segmentation)
    structures_np = sitk.GetArrayFromImage(structures)
    outline_np = np.copy(structures_np)
    outline_np[outline_np!=0]=1
    
    outline = sitk.GetImageFromArray(outline_np)
    outline.CopyInformation(structures)
    return outline

def get_cbct_fov(cbct:sitk.Image)->sitk.Image:
    """
    Generate a field of view (FOV) mask for a given CBCT image.

    Args:
        cbct (sitk.Image): The input CBCT image.

    Returns:
        sitk.Image: The FOV mask image.

    """
    cbct_np = sitk.GetArrayFromImage(cbct)
    cbct_np[cbct_np>0] = 1
    cbct_np[cbct_np<=0] = 0
    fov_mask_np = np.zeros(cbct_np.shape)
    for i in range(cbct_np.shape[0]):
        slice = cbct_np[i,:,:]
        y, x = np.indices((slice.shape))
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])
        r = np.hypot(x - center[0], y - center[1])
        bins = np.arange(0, r.max() + 1, 1)
        radial_mean = ndimage.mean(cbct_np[i,:,:], labels=np.digitize(r, bins), index=np.arange(1, len(bins)))
        mask_radius = np.where(radial_mean>0.5)[0][-1]
        size = cbct_np.shape
        y, x = np.ogrid[-size[1]//2:size[1]//2, -size[2]//2:size[2]//2]
        fov_mask_np[i,:,:] = x**2 + y**2 <= mask_radius**2
    
    fov_mask = sitk.GetImageFromArray(fov_mask_np)
    fov_mask.CopyInformation(cbct)
    fov_mask = sitk.Cast(fov_mask,sitk.sitkUInt8)
    return fov_mask

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

def crop_image(image:sitk.Image, bbox:tuple,dilation:int)->sitk.Image:
    """
    Crop the input image based on the given bounding box and dilation.

    Parameters:
    image (sitk.Image): The input image to be cropped.
    bbox (tuple): The bounding box coordinates in the format (x_min, y_min, z_min, x_max, y_max, z_max).
    dilation (int): The amount of dilation to be applied to the bounding box.

    Returns:
    sitk.Image: The cropped image.

    """
    start_index = [int(bbox[0]-dilation), int(bbox[1]-dilation), int(bbox[2]-dilation)]
    size = [int(bbox[3] - bbox[0]+dilation*2), int(bbox[4] - bbox[1]+dilation*2), int(bbox[5] - bbox[2] +dilation*2)]
    roi_filter = sitk.RegionOfInterestImageFilter()
    roi_filter.SetIndex(start_index)
    roi_filter.SetSize(size)
    cropped_image = roi_filter.Execute(image)
    return cropped_image

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
