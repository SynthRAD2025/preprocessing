import preprocessing_utils as utils
import os 
import SimpleITK as sitk
import logging
import sys

# Set this to true if you want to skip already pre-processsed patients (checks if output files already exists)
skip_existing = True

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
                    os.path.isfile(os.path.join(patient['output_dir'],'ct_s2_def.nii.gz')) and 
                    os.path.isfile(os.path.join(patient['output_dir'],'mask_s2.nii.gz')) and
                    os.path.isfile(os.path.join(patient['output_dir'],f'{patient["ID"]}.png'))):
                    logger.info(f'Patient {i} already pre-processed. Skipping...')
                    continue
            elif patient['task'] == 2:
                if (os.path.isfile(os.path.join(patient['output_dir'],'cbct_s2.nii.gz')) and 
                    os.path.isfile(os.path.join(patient['output_dir'],'ct_s2.nii.gz')) and 
                    os.path.isfile(os.path.join(patient['output_dir'],'ct_s2_def.nii.gz')) and 
                    os.path.isfile(os.path.join(patient['output_dir'],'mask_s2.nii.gz')) and
                    os.path.isfile(os.path.join(patient['output_dir'],f'{patient["ID"]}.png'))):
                    logger.info(f'Patient {i} already pre-processed. Skipping...')
                    continue
        
        # log patient details
        logger.info(f'''Processing case {i}:
                            Output_dir: {patient['output_dir']}''')
        
        #Read Files
        if patient['task'] == 1:
            input = utils.read_image(os.path.join(patient['output_dir'],'mr_s1.nii.gz'),log=logger)
        elif patient['task'] == 2:
            input = utils.read_image(os.path.join(patient['output_dir'],'cbct_s1.nii.gz'),log=logger)
        else:
            logger.error('Task not valid')
            sys.exit(1)
        ct = utils.read_image(os.path.join(patient['output_dir'],'ct_s1.nii.gz'),log=logger)    
        fov_s1 = utils.read_image(os.path.join(patient['output_dir'],'fov_s1.nii.gz'),log=logger)
        if patient['defacing_correction'] == True:
            face = utils.read_image(os.path.join(patient['output_dir'],'defacing_mask.nii.gz'),log=logger)
        
        # Clip CT to valid HU range
        ct = utils.clip_image(ct,-1024,3071)
        
        # Perform cone correction for fov mask if task2
        if patient['task'] == 2:
            if patient['cone_correction']:
                fov_s1 = utils.cone_correction(fov_s1,log=logger)
        
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
                                         fov_s1,
                                         defacing_correction=defacing_correction,
                                         IS_correction=IS_correction,
                                         log=logger)
        
        #Crop images using mask generated above
        input = utils.crop_image(input,fov_s1)
        ct = utils.crop_image(ct,fov_s1)
        mask = utils.crop_image(mask,fov_s1)
        fov = utils.crop_image(fov_s1,fov_s1)
        
        #deform CT to match input
        ct_deformed, transform = utils.deformable_registration(input,ct,patient['parameter_def'],mask=mask,log=logger)
        sitk.WriteParameterFile(transform, os.path.join(patient['output_dir'],'transform_def.txt'))
        
        # #deform defacing mask if necessary and apply to fov
        # if patient['defacing_correction']:
        #     face_deformed = utils.warp_structure(face,transform)       
        #     fov[face_deformed == 1] = 0
             
        #apply fov mask to all images
        if patient['task'] == 1:
            mask_value = 0
        if patient['task'] == 2:
            mask_value = -1000
        ct_deformed = utils.mask_image(ct_deformed,fov,-1000)
        ct = utils.mask_image(ct,fov,-1000)
        input = utils.mask_image(input,fov,mask_value)
        mask = utils.mask_image(mask,fov,0)
        
        #preprocess structures
        logger.info('Preprocessing and warping structures...')
        rigid_reg = sitk.ReadTransform(os.path.join(patient['output_dir'],'transform.tfm'))
        ct_s1 = utils.read_image(os.path.join(patient['output_dir'],'ct_s1.nii.gz'),log=logger)
        #utils.preprocess_structures(patient,input,ct_s1,fov_s1,fov,rigid_reg,transform,mask,log=logger)
        
        # Stitch CT_def to CT_s1 for planning (structures are stitched above)
        ct_deformed_stitched = utils.stitch_image(ct_deformed, ct_s1, mask)
       
        #Save cropped images and transform
        if patient['task'] == 1:
            utils.save_image(input,os.path.join(patient['output_dir'],'mr_s2.nii.gz'),dtype='int16')
        if patient['task'] == 2:
            utils.save_image(input,os.path.join(patient['output_dir'],'cbct_s2.nii.gz'),dtype='int16')
        utils.save_image(ct,os.path.join(patient['output_dir'],'ct_s2.nii.gz'),dtype='int16')
        utils.save_image(mask,os.path.join(patient['output_dir'],'mask_s2.nii.gz'))
        utils.save_image(fov,os.path.join(patient['output_dir'],'fov_s2.nii.gz'))
        utils.save_image(ct_deformed,os.path.join(patient['output_dir'],'ct_s2_def.nii.gz'),dtype='int16')
        utils.save_image(ct_deformed_stitched, os.path.join(patient['output_dir'],'ct_s2_def_stitched.nii.gz'),dtype='int16')
        
        #Generate png overviews
        utils.generate_overview_png(ct,input,mask,patient)
        utils.generate_overview_planning(ct,input,ct_deformed,mask,patient)
   