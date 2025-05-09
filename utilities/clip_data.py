import shutil
import os
import fnmatch
import SimpleITK as sitk

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
    image[image<lower_bound] = lower_bound
    image[image>upper_bound] = upper_bound
    return image

### change theses path for source (non-clipped) and destination (clipped)  ###
### use the same path twice if you want to overvwrite the original data    ###
dataset_root = '/workspace/data/full/releases/non-clipped/'
dataset_dest_root = '/workspace/data/full/releases/clipped/'

datasets = [
    'synthRAD2025_Task1_Train',
    'synthRAD2025_Task2_Train',
    'synthRAD2025_Task1_Train_D',
    'synthRAD2025_Task2_Train_D',
]
datasets = os.listdir(dataset_root)

for dataset in datasets:
    dataset_path = os.path.join(dataset_root, dataset)
    dataset_dest_path = os.path.join(dataset_dest_root, dataset)

    task = os.listdir(dataset_path)
    if len(task) > 1:
        # copy license
        file = fnmatch.filter(task, '*.txt')[0]
        if not os.path.exists(os.path.join(dataset_dest_path, file)):
            os.makedirs(dataset_dest_path, exist_ok=True)
            shutil.copy(os.path.join(dataset_path, file), os.path.join(dataset_dest_path, file))
        task = fnmatch.filter(task, 'Task*')[0]
    else:
        task = task[0]
    regions = os.listdir(os.path.join(dataset_path, task))

    for region in regions:
        region_path = os.path.join(dataset_path, task, region)
        for root, dirs, files in os.walk(region_path):
            print(f'Processing {root}')
            root_dest = root.replace(dataset_path, dataset_dest_path)
            os.makedirs(root_dest, exist_ok=True)
            for file in files:
                file_path = os.path.join(root, file)
                dest_path = os.path.join(root_dest, file)
                if file in ['ct.mha','cbct.mha','ct_def.mha']:
                    image = read_image(file_path)
                    image = clip_image(image, -1024, 3071)
                    save_image(image, dest_path, compression=True, dtype='int16')
                else:
                    if not os.path.exists(dest_path):
                        shutil.copy(file_path, dest_path)
        


