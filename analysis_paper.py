import preprocessing_utils as utils
import os
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk

ROOT = "/workspace/data/full"

CENTERS = ['A','B','C','D','E']
TASKS = ['1','2']
REGIONS = ['HN','AB','TH']
SETS = ['train','val','test']

def outline_no_dilation(image,threshold):
    mask = utils.segment_outline(image, threshold=threshold)
    return mask

df = pd.DataFrame(columns=['Patient ID', 'Volume_mask', 'Volume_img', 'Task', 'Center', 'Region'])

for center in CENTERS:
    for task in TASKS:
        for region in REGIONS:
            for set in SETS:
                # Read patient IDs from Set file
                source = os.path.join(ROOT, center, f'Task{task}', region)
                set_file = os.path.join(source, f'{task}{region}{center}_Set.csv')
                pre_process_settings = os.path.join(source, f'stage2_config_{task}{region}{center}.csv')
                if not os.path.exists(pre_process_settings):
                    continue
                # Read pre-process settings
                pre_params = pd.read_csv(pre_process_settings)
                
                if os.path.exists(set_file):
                    with open(set_file) as f:
                        lines = f.readlines()
                    for i,line in enumerate(lines):
                        if i > 0:
                            line = line.strip()
                            line = line.strip("'")
                            if line == "":
                                continue
                            patient_id = line.split(",")[0].strip('"').strip("'")
                            patient_set = line.split(",")[1].strip('"')
                            if patient_set.lower() == set:
                                # Read mask
                                if task == '1':
                                    img_file = os.path.join(source,patient_id,'output','mr_s2.nii.gz')
                                else:
                                    img_file = os.path.join(source,patient_id,'output','cbct_s2.nii.gz')
                                if os.path.exists(img_file):
                                    print(patient_id)
                                    if not os.path.exists(os.path.join(source,patient_id,'output','mask_no_dilation.nii.gz')):
                                        print(f'{patient_id} segmenting...')
                                        img = utils.read_image(img_file)
                                        row = pre_params[pre_params['ID'] == patient_id]
                                        mask = outline_no_dilation(img, row['mask_thresh'].values[0])
                                        utils.save_image(mask, os.path.join(source,patient_id,'output','mask_no_dilation.nii.gz'))
                                    # new_row = pd.DataFrame({'Patient ID': [patient_id], 'Volume_mask': [volume], 'Volume_img': [volume_img], 'Task': [task], 'Center': [center], 'Region': [region]})
                                    # df = pd.concat([df, new_row], ignore_index=True)
                                else:
                                    print(f"Mask file not found for patient ID: {patient_id}")