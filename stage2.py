import preprocessing_utils as utils
import os 
import SimpleITK as sitk
import logging
import sys

# Set this to true if you want to skip already pre-processsed patients (checks if output files already exists)
skip_existing = False

if __name__ == "__main__":
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

    ## load pre-processing configuration from .csv file (sys.argv[1])
    file = sys.argv[1]
    if file.endswith('.csv'):
        patient_dict = utils.csv_to_dict(file)
    else:
        logger.error('Input file must be a csv file')
        sys.exit(1)
    for i in patient_dict:
        patient = patient_dict[i]
        # check if output files already exist and skip if flag is set
        if skip_existing:
            if patient['task'] == 1:
                if (os.path.isfile(os.path.join(patient['output_dir'],'mr_s2.nii.gz')) and 
                    os.path.isfile(os.path.join(patient['output_dir'],'ct_s2.nii.gz')) and 
                    os.path.isfile(os.path.join(patient['output_dir'],'ct_deformed_s2.nii.gz')) and 
                    os.path.isfile(os.path.join(patient['output_dir'],'mask_s2.nii.gz'))):
                    logger.info(f'Patient {i} already pre-processed. Skipping...')
                    continue
            elif patient['task'] == 2:
                if (os.path.isfile(os.path.join(patient['output_dir'],'cbct_s2.nii.gz')) and 
                    os.path.isfile(os.path.join(patient['output_dir'],'ct_s2.nii.gz')) and 
                    os.path.isfile(os.path.join(patient['output_dir'],'ct_deformed_s2.nii.gz')) and 
                    os.path.isfile(os.path.join(patient['output_dir'],'mask_s2.nii.gz'))):
                    logger.info(f'Patient {i} already pre-processed. Skipping...')
                    continue
        
        # log patient details
        logger.info(f'''Processing case {i}:
                            Output_dir: {patient['output_dir']}''')
        
        #Read Files
        if patient['task'] == 1:
            input = utils.read_image(os.path.join(patient['output_dir'],'mr_s1.nii.gz'),log=logger)
        if patient['task'] == 2:
            input = utils.read_image(os.path.join(patient['output_dir'],'cbct_s1.nii.gz'),log=logger)
        else:
            logger.error('Task not valid')
            sys.exit(1)
        ct = utils.read_image(os.path.join(patient['output_dir'],'ct_s1.nii.gz'),log=logger)    
        fov = utils.read_image(os.path.join(patient['output_dir'],'fov_s1.nii.gz'),log=logger)
        if patient['defacing_correction'] == True:
            face = utils.read_image(os.path.join(patient['output_dir'],'defacing_mask.nii.gz'),log=logger)
        
        #Generate patient outline and postprocess it
        mask = utils.segment_outline(input,patient['mask_thresh'],log=logger)
        if patient['defacing_correction']:
            defacing_correction = os.path.join(patient['output_dir'],'defacing_mask.nii.gz')
            defacing_correction = utils.read_image(defacing_correction,log=logger)
        else:
            defacing_correction = None
        if patient['IS_correction']:
            IS_correction = 10
        else:
            IS_correction = None
        mask = utils.postprocess_outline(mask,
                                         fov,
                                         defacing_correction=defacing_correction,
                                         IS_correction=IS_correction,
                                         log=logger)
        
        
        #Crop images using mask generated above
        input = utils.crop_image(input,fov)
        ct = utils.crop_image(ct,fov)
        mask = utils.crop_image(mask,fov)
        fov = utils.crop_image(fov,fov)
        
        #deform CT to match input
        ct_deformed, transform = utils.deformable_registration(input,ct,patient['parameter_def'],mask=mask,log=logger)
        #apply fov mask to deformed ct
        ct_deformed = utils.mask_image(ct_deformed,fov,-1000)
        
        #preprocess structures
        logger.info('Preprocessing and warping structures...')
        structures = os.listdir(os.path.join(patient['output_dir'],'structures'))
        structures = [struct for struct in structures if struct.endswith('.nrrd') or struct.endswith('.nii')]
        spacing = ct.GetSpacing()
        fov_s1 = utils.read_image(os.path.join(patient['output_dir'],'fov_s1.nii.gz'),log=logger)
        ct_s1 = utils.read_image(os.path.join(patient['output_dir'],'ct_s1.nii.gz'),log=logger)
        for struct in structures:
            struct_path = os.path.join(patient['output_dir'],'structures',struct)
            struct_img = utils.read_image(struct_path)
            struct_img = utils.resample_struct(struct_img,ct_s1)
            struct_img = utils.crop_image(struct_img,fov_s1)
            struct_img = utils.mask_image(struct_img,fov,0)
            struct_deformed = utils.warp_structure(struct_img,transform,log=logger)
            utils.save_image(struct_img,os.path.join(patient['output_dir'],'structures',struct.strip('.nrrd')+'_s2.nrrd'))
            utils.save_image(struct_deformed,os.path.join(patient['output_dir'],'structures',struct.strip('.nrrd')+'_s2_def.nrrd'))
        
        
        #Save cropped images and transform
        if patient['task'] == 1:
            utils.save_image(input,os.path.join(patient['output_dir'],'mr_s2.nii.gz'))
        if patient['task'] == 2:
            utils.save_image(input,os.path.join(patient['output_dir'],'cbct_s2.nii.gz'))
        utils.save_image(ct,os.path.join(patient['output_dir'],'ct_s2.nii.gz'))
        utils.save_image(mask,os.path.join(patient['output_dir'],'mask_s2.nii.gz'))
        utils.save_image(fov,os.path.join(patient['output_dir'],'fov_s2.nii.gz'))
        utils.save_image(ct_deformed,os.path.join(patient['output_dir'],'ct_s2_def.nii.gz'))
        sitk.WriteParameterFile(transform, os.path.join(patient['output_dir'],'transform_def.txt'))
        
        #Generate png overview
        utils.generate_overview_png(ct,input,mask,patient['output_dir'])
   