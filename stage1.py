import preprocessing_utils as utils
import os 
import SimpleITK as sitk
import logging

## set up logging to console and file
log_file = 'stage1.log'

logger = logging.getLogger()
logger.setLevel(logging.INFO) 

file_handler = logging.FileHandler('stage1.log',mode = 'a')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info('Starting stage 1 preprocessing')

patient_list = []

## THIS SECTION HAS TO BE MODIFIED FOR EACH CENTER ##

## for each patient create a dictionary with the following keys:
#   'input' :        path to input image
#   'ct' :          path to ct image
#   'output_dir' :  path to output directory
#
## the following keys are usually the same for all patients from the same anatomy and center
#   'defacing' :    boolean, if True defacing is applied
#   'reg_params' :  path to registration parameters file
#   'background' :  background value of input/MRI image (e.g. -1000)
#   'order' :       order of axes of input/MRI image (e.g. [1,0,2] in case input/MRI has different orientation than CT)
#   'flip' :        flip axes of input/MRI image (e.g. [False,False,False])
#   'task' :        1 for MRI, 2 for CBCT
#   'grid' :        grid for resampling (e.g. [1,1,1])

## EXAMPLE for a single patient, for an entire dataset use a loop to populate patient_list with a dict for each patient
patient = {}
patient['name'] = '1ABA001'
patient['input'] = '/workspace/data/Abdomen/1ABA001/mr_or.nii.gz'
patient['ct'] = '/workspace/data/Abdomen/1ABA001/ct_or.nii.gz'
patient['output_dir'] = '/workspace/data/Abdomen/1ABA001/'
# the following keys are usually the same for all patients from the same anatomy and center
patient['task'] = 1 # 1 for MRI, 2 for CBCT
patient['defacing'] = False
patient['reg_params'] = '/workspace/code/preprocessing/configs/param_rigid.txt'
patient['background'] = 0
patient['order']=[0,1,2]
patient['flip']=[False,False,False]
patient['grid'] = [1,1,3]

patient_list.append(patient)
## END OF MODIFICATION SECTION ##

for patient in patient_list:
    # log patient details
    logger.info(f'Processing patient:\n Name: {patient['name']} \n Input: {patient['input']} \n CT: {patient['ct']} \n Output: {patient['output_dir']}')
    
    # Load input (CBCT or MRI) and CT as sitk images
    input = utils.read_image(patient['input'],log=logger)
    ct = utils.read_image(patient['ct'],log=logger)
    
    # if necessary correct orientation of input/MRI (swap or flip axes)
    input = utils.correct_orientation(input,patient['order'],patient['flip'],log=logger)
    
    # calculate CBCT/MRI FOV mask
    if patient['task'] == 1:
        fov_mask = utils.get_mr_fov(input)
    if patient['task'] == 2:
        fov_mask = utils.get_cbct_fov(input,background=patient['background'],log=logger)
    
    
    # Register CBCT/MRI to CT
    parameter_file = patient['reg_params']
    input, transform = utils.rigid_registration(ct,input,parameter_file,default_value=patient['background'],log=logger)
    
    # Apply transformation to CBCT/MRI FOV mask
    fov_mask_reg = utils.apply_transform(fov_mask,transform,ct)
    
    # Deface CBCT/MRI and CT
    print(patient['defacing'])
    if patient['defacing']:
        # Get structures for defacing
        brain,skull = utils.segment_defacing(ct,log=logger)
        defacing_mask = utils.defacing(brain,skull,log=logger)
        ct = sitk.Mask(ct,defacing_mask,outsideValue=-1024,maskingValue=1)
        input = sitk.Mask(input,defacing_mask,outsideValue=patient['background'],maskingValue=1)
    
    # Resample everything
    input = utils.resample_image(input,new_spacing=patient['grid'],log=logger)
    ct = utils.resample_image(ct,new_spacing=patient['grid'],log=logger)
    fov_mask_reg = utils.resample_image(fov_mask_reg,new_spacing=patient['grid'],log=logger)
    
    # Save registered,defaced CBCT/MRI and CT, input/MRI FOV mask and transform 
    if patient['task'] == 1:
        utils.save_image(input,os.path.join(patient['output_dir'],'mr_s1.nii.gz'))
    if patient['task'] == 2:
        utils.save_image(input,os.path.join(patient['output_dir'],'cbct_s1.nii.gz'))
    utils.save_image(fov_mask_reg,os.path.join(patient['output_dir'],'fov_mask.nii.gz'))
    utils.save_image(ct,os.path.join(patient['output_dir'],'ct_s1.nii.gz'))
    sitk.WriteTransform(transform,os.path.join(patient['output_dir'],'transform.tfm'))
