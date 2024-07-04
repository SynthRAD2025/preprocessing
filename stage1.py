import preprocessing_utils as utils
import os 
import SimpleITK as sitk
import logging
import sys

if __name__ == "__main__":

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

    ## load pre-processing configuration from .csv file (sys.argv[1])
    file = sys.argv[1]
    if file.endswith('.csv'):
        patient_dict = utils.csv_to_dict(file)
    else:
        logger.error('Input file must be a csv file')
        sys.exit(1)

    for pat in patient_dict:
        patient = patient_dict[pat]
        # log patient details
        logger.info(f'Processing patient:\n Name: {pat} \n Input: {patient['input_path']} \n CT: {patient['ct_path']} \n Output: {patient['output_dir']}')
        
        # Load input (CBCT or MRI) and CT as sitk images, if dicom provide path to directory, if other file type provide path to file
        input = utils.read_image(patient['input_path'],log=logger)   
        ct = utils.read_image(patient['ct_path'],log=logger)
        
        # if necessary correct orientation of input/MRI (swap or flip axes)
        input = utils.correct_image_properties(input,patient['order'],patient['flip'],mr_overlap_correction = patient['mr_overlap_correction'],log=logger)
        
        # calculate CBCT/MRI FOV mask
        if patient['task'] == 1:
            fov_mask = utils.get_mr_fov(input)
        if patient['task'] == 2:
            fov_mask = utils.get_cbct_fov(input,background=patient['background'],log=logger)
        
        # Register CBCT/MRI to CT
        parameter_file = patient['registration']
        # input, transform = utils.translation_registration(ct, input, "/workspace/code/preprocessing/configs/param_trans_cbct_radboud.txt", mask=fov_mask, default_value=patient['background'], log=logger)
        # fov_mask = utils.apply_transform(fov_mask,transform,ct)
        input, transform = utils.rigid_registration(ct, input, parameter_file, default_value=patient['background'], log=logger)
    
        # Apply transformation to CBCT/MRI FOV mask
        fov_mask_reg = utils.apply_transform(fov_mask,transform,ct)

        # Deface CBCT/MRI and CT
        if patient['defacing']:
            # Get structures for defacing
            brain,skull = utils.segment_defacing(ct,log=logger)
            defacing_mask = utils.defacing(brain,skull,log=logger)
            ct = sitk.Mask(ct,defacing_mask,outsideValue=-1024,maskingValue=1)
            input = sitk.Mask(input,defacing_mask,outsideValue=patient['background'],maskingValue=1)
        
        # Resample everything
        input = utils.resample_image(input,new_spacing=patient['resample'],log=logger)
        ct = utils.resample_image(ct,new_spacing=patient['resample'],log=logger)
        fov_mask_reg = utils.resample_image(fov_mask_reg,new_spacing=patient['resample'],log=logger)
        if patient['defacing']:
            defacing_mask = utils.resample_image(defacing_mask,new_spacing=patient['resample'],log=logger)
        
        # generate overview png
        utils.generate_overview_stage1(ct,input,patient['output_dir'])
        
        # Save registered,defaced CBCT/MRI and CT, input/MRI FOV mask and transform 
        if not os.path.isdir(patient['output_dir']):
            os.mkdir(patient['output_dir'])
            logger.info('Creating output directory...')
        else:
            logger.warning('Output directory already exists. Overwriting existing files...')
        if patient['task'] == 1:
            utils.save_image(input,os.path.join(patient['output_dir'],'mr_s1.nii.gz'))
        if patient['task'] == 2:
            utils.save_image(input,os.path.join(patient['output_dir'],'cbct_s1.nii.gz'))
        utils.save_image(fov_mask_reg,os.path.join(patient['output_dir'],'fov_s1.nii.gz'))
        utils.save_image(ct,os.path.join(patient['output_dir'],'ct_s1.nii.gz'))
        if patient['defacing']:
            utils.save_image(defacing_mask,os.path.join(patient['output_dir'],'defacing_mask.nii.gz'))
        sitk.WriteTransform(transform,os.path.join(patient['output_dir'],'transform.tfm'))
        
        # convert rtstruct to nrrd
        if not ['struct_path']=='':
            utils.convert_rtstruct_to_nrrd(patient['struct_path'],patient['output_dir'],log=logger)
