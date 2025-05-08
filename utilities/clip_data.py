import shutil
import os
import fnmatch
import SimpleITK as sitk

# change the path to where the data is stored
dataset_path = '/workspace/data/full/releases/synthRAD2025_Task2_Train'
clipped_dataset_path = '/workspace/data/full/releases/synthRAD2025_Task2_Train_clipped'

def read_image(image_path:str)->sitk.Image:
    image = sitk.ReadImage(image_path)
    return image

def save_image(image:sitk.Image, image_path:str, compression:bool=True, dtype:str=None):
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

def clip_image(image:sitk.Image,lower_bound:float, upper_bound:float)->sitk.Image:
    # clip an image using SimpleITK
    image[image<lower_bound] = lower_bound
    image[image>upper_bound] = upper_bound
    return image

task = os.listdir(dataset_path)[0]
regions = os.listdir(os.path.join(dataset_path, task))

for region in regions:
    region_path = os.path.join(dataset_path, task, region)
    clipped_region_path = os.path.join(clipped_dataset_path, task, region)
    os.makedirs(clipped_region_path, exist_ok=True)
    for root, dirs, files in os.walk(region_path):
        for file in files:
            if file in ['ct.mha','cbct.mha']:
                file_path = os.path.join(root, file)
                image = read_image(file_path)
                image = clip_image(image, -1000, 3072)
                save_image(image, os.path.join(clipped_region_path, file), compression=True, dtype='int16')
            elif file in ['mr.mha']:
                file_path = os.path.join(root, file)
                image = read_image(file_path)
                image = clip_image(image, -1000, 3072)
                save_image(image, os.path.join(clipped_region_path, file), compression=True, dtype='int16')
    


