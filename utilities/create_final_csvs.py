import shutil
import os
import fnmatch
import SimpleITK as sitk
import pandas as pd
import openpyxl

# Define export parameters
ROOT = "/workspace/data/full"
DEST = "/workspace/data/full/releases/synthRAD2025_Task2_Val_Input_D/"

CENTERS = ['D']
TASK = ['2']
REGIONS = ['HN','AB','TH']
SETS = ['val']
FILES = ['parameters']

# Iterate through all centers, tasks and regions
for center in CENTERS:
    for task in TASK:
        for region in REGIONS:
            for set in SETS:
                # Read patient IDs from Set file
                source = os.path.join(ROOT, center, f'Task{task}', region)
                source_file = os.path.join(source, f'{task}{region}{center}_parameters.xlsx')
                destination_file = os.path.join(DEST, f'Task{task}', region, 'overviews',f'{task}_{region}_{set}_parameters.xlsx')
                if not os.path.exists(destination_file):
                    pd.DataFrame().to_excel(destination_file, index=False)
                set_file = os.path.join(source, f'{task}{region}{center}_Set.csv')
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
                            print(patient_id, patient_set)
                            if patient_set.lower() == set:
                                df = pd.read_excel(source_file, sheet_name=None)
                                patient_data = {}
                                for sheet in df.keys():
                                    # Check if patient data exists in the source file
                                    data = df[sheet][df[sheet]['ID'] == patient_id]
                                    if len(data) > 0:
                                        patient_data[sheet] = data
                                    else:
                                        # Create a row with "-" for all columns, except patient ID
                                        empty_row = pd.DataFrame({col: ["-"] for col in df[sheet].columns})
                                        empty_row.at[0, 'ID'] = patient_id  # Set first column to patient_id
                                        patient_data[sheet] = empty_row
                                
                                # Read any existing data from the destination Excel file
                                current_data = {}
                                if os.path.exists(destination_file) and os.path.getsize(destination_file) > 0:
                                    current_data = pd.read_excel(destination_file, sheet_name=None)

                                # For each sheet in the patient data, append only new rows if they don't already exist
                                for sheet, new_data in patient_data.items():
                                    if sheet in current_data and not current_data[sheet].empty:
                                        existing_ids = current_data[sheet]['ID'].tolist()
                                        new_data_filtered = new_data[~new_data['ID'].isin(existing_ids)]
                                        combined = pd.concat([current_data[sheet], new_data_filtered], ignore_index=True)
                                    else:
                                        combined = new_data
                                    current_data[sheet] = combined

                                # Write the updated data back to the destination file
                                with pd.ExcelWriter(destination_file, engine='openpyxl') as writer:
                                    for sheet, data in current_data.items():
                                        data.to_excel(writer, sheet_name=sheet, index=False)  
                wb = openpyxl.load_workbook(destination_file)
                if "Sheet1" in wb.sheetnames:
                    std = wb["Sheet1"]
                    wb.remove(std)
                    wb.save(destination_file)
                else:                 
                    print(f'File {set_file} does not exist, continuing...')