import shutil
import os
import fnmatch
import SimpleITK as sitk

def nii_to_mha(nii_file, mha_file, compression:bool=True):
    image = sitk.ReadImage(nii_file)
    try:
        image = sitk.Round(image)
    except:
        print('Could not round image ...')
    if image.GetPixelIDTypeAsString() != '16-bit signed integer':
        image = sitk.Cast(image, sitk.sitkInt16)
    sitk.WriteImage(image, mha_file, useCompression=compression)

# Define export parameters
ROOT = "/workspace/data/full"
DEST = "/workspace/data/full/releases/synthRAD2025_Task1_Val_Input_D"

CENTERS = ['D']
TASK = ['1']
REGIONS = ['HN','AB','TH']
SETS = ['val']
FILES = ['overview','mask_s2.nii.gz','mr_s2.nii.gz']
#FILES = ['overview','ct_s2.nii.gz','ct_s2_def.nii.gz']

# Iterate through all centers, tasks and regions
for center in CENTERS:
    for task in TASK:
        for region in REGIONS:
            for set in SETS:
                # Read patient IDs from Set file
                source = os.path.join(ROOT, center, f'Task{task}', region)
                destination = os.path.join(DEST, f'Task{task}', region)
                set_file = os.path.join(source, f'{task}{region}{center}_Set.csv')
                if os.path.exists(set_file):
                    if not os.path.exists(destination):
                        os.makedirs(destination)
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
                            print(patient_id, patient_set)
                            if patient_set.lower() == set:
                                for file in FILES:
                                    if file == 'overview_planning':
                                        if not os.path.exists(os.path.join(destination,'overviews')):
                                            os.makedirs(os.path.join(destination,'overviews'))
                                        source_file = os.path.join(source, patient_id,'output',f'{patient_id}_planning.png')
                                        destination_file = os.path.join(destination,'overviews', f'{patient_id}_{file}.png')
                                        if os.path.exists(source_file):
                                            shutil.copy(source_file, destination_file)
                                        else:
                                            print(f'File {source_file} does not exist')
                                    elif file == "overview":
                                        if set == 'test' or set == 'val':
                                            if not os.path.exists(os.path.join(destination,'overviews')):
                                                os.makedirs(os.path.join(destination,'overviews'))
                                            source_file = os.path.join(source, patient_id,'output',f'{patient_id}_def.png')
                                            destination_file = os.path.join(destination,'overviews', f'{patient_id}_{file}.png')
                                            if os.path.exists(source_file):
                                                shutil.copy(source_file, destination_file)
                                            else:
                                                print(f'File {source_file} does not exist')
                                        else:
                                            if not os.path.exists(os.path.join(destination,'overviews')):
                                                os.makedirs(os.path.join(destination,'overviews'))
                                            source_file = os.path.join(source, patient_id,'output',f'{patient_id}.png')
                                            destination_file = os.path.join(destination,'overviews', f'{patient_id}_{file}.png')
                                            if os.path.exists(source_file):
                                                shutil.copy(source_file, destination_file)
                                            else:
                                                print(f'File {source_file} does not exist')
                                    elif file == "ct_s2_def.nii.gz":
                                        source_file = os.path.join(source, patient_id,'output',f'{file}')
                                        file_name = file.replace('_s2_def.nii.gz','_def.mha')
                                        if not os.path.exists(os.path.join(destination,patient_id)):
                                            os.makedirs(os.path.join(destination,patient_id))
                                        dest_file = os.path.join(destination,patient_id,file_name)
                                        nii_to_mha(source_file, dest_file, compression=True)
                                        
                                    else:
                                        source_file = os.path.join(source, patient_id,'output',f'{file}')
                                        file_name = file.replace('_s2.nii.gz','.mha')
                                        if not os.path.exists(os.path.join(destination,patient_id)):
                                            os.makedirs(os.path.join(destination,patient_id))
                                        dest_file = os.path.join(destination,patient_id,file_name)
                                        nii_to_mha(source_file, dest_file, compression=True)
                else:                 
                    print(f'File {set_file} does not exist, continuing...')
