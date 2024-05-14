import preprocessing_utils as utils
import os 
import SimpleITK as sitk
import logging

## set up logging to console and file
log_file = 'stage2.log'

logger = logging.getLogger()
logger.setLevel(logging.INFO) 

file_handler = logging.FileHandler(log_file,mode = 'a')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info('Starting stage 2 preprocessing')

patient_list = []

## THIS SECTION HAS TO BE MODIFIED FOR EACH CENTER ##

## EXAMPLE for a single patient, for an entire dataset use a loop to populate patient_list with a dict for each patient
name = '2THA203'
region = 'Thorax'

patient = {}
patient['name'] = f'{name}'
patient['input'] = f'/workspace/data/1%/Task2/{region}/{name}/cbct_s1.nii.gz'
patient['ct'] = f'/workspace/data/1%/Task2/{region}/{name}/ct_s1.nii.gz'
patient['fov'] = f'/workspace/data/1%/Task2/{region}/{name}/fov_s1.nii.gz'
patient['output_dir'] = f'/workspace/data/1%/Task2/{region}/{name}/'
# the following keys are usually the same for all patients from the same anatomy and center
patient['task'] = 2 
patient['background'] = -1000
patient['threshold'] = 0.2

patient_list.append(patient)
## END OF MODIFICATION SECTION ##

for patient in patient_list:
    #Read Files
    input = sitk.ReadImage(patient['input'])
    ct = sitk.ReadImage(patient['ct'])
    fov = sitk.ReadImage(patient['fov'])
    
    #Generate patient outline
    mask = utils.segment_outline(input,patient['threshold'])
    mask = utils.postprocess_outline(mask,fov)
    
    #Crop images using fov mask from stage 1
    input = utils.crop_image(input,mask)
    ct = utils.crop_image(ct,mask)
    mask = utils.crop_image(mask,mask)
    fov = utils.crop_image(fov,mask)
    
    #Save cropped images
    if patient['task'] == 1:
        utils.save_image(input,patient['output_dir'] + 'mr_s2.nii.gz')
    if patient['task'] == 2:
        utils.save_image(input,patient['output_dir'] + 'cbct_s2.nii.gz')
    utils.save_image(ct,patient['output_dir'] + 'ct_s2.nii.gz')
    utils.save_image(mask,patient['output_dir'] + 'mask_s2.nii.gz')
    utils.save_image(fov,patient['output_dir'] + 'fov_s2.nii.gz')
    
    #Generate png overview
    utils.generate_overview_png(ct,input,mask,patient['output_dir'])
   