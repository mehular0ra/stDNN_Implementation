import numpy as np
import pandas as pd
import os

import ipdb

base = '/mnt/ssd1/mehul_data/ABIDE_all_datasets/'
atlas = 'Power/'
abide_power_data = base + atlas + '50003_timeseries.txt'
phenotype_csv = base + 'Phenotypic_V1_0b_preprocessed1.csv'

# Load the data
data = np.loadtxt(abide_power_data)

# load phenotype as pd dataframe
phenotype_df = pd.read_csv(phenotype_csv)

# Step 1: Filter the phenotype DataFrame
df = phenotype_df[['SUB_ID', 'SITE_ID', 'DX_GROUP', 'AGE_AT_SCAN', 'SEX']]

# For changing DX_GROUP from 2 to 0
df.loc[df['DX_GROUP'] == 2, 'DX_GROUP'] = 0

# For changing SEX from 2 to 0
df.loc[df['SEX'] == 2, 'SEX'] = 0

# Prepare the dictionary to save the final data
fc_data = {
    "corr": [],
    "label": [],
    "site": [],
    "age": [],
    "sex": []
}

# Step 2: Iterate over the time series data files
timeseries_directory = os.path.join(base, atlas)
for filename in os.listdir(timeseries_directory):
    if filename.endswith('_timeseries.txt'):
        # Extract SUB_ID from the filename
        sub_id = int(filename.split('_')[0])
        
        # Match the SUB_ID with the phenotype data
        if sub_id in df['SUB_ID'].values:
            # Load the time series data
            filepath = os.path.join(timeseries_directory, filename)
            data = np.loadtxt(filepath)
            
            # Transpose the time series data
            data_T = data.T
            
            # Extract corresponding phenotype information
            phenotype_info = df[df['SUB_ID'] == sub_id]
            
            # Add the time series and phenotype info to the fc_data dictionary
            fc_data["corr"].append(data_T)
            fc_data["label"].append(phenotype_info['DX_GROUP'].values[0])
            fc_data['age'].append(phenotype_info['AGE_AT_SCAN'].values[0])
            fc_data['sex'].append(phenotype_info['SEX'].values[0])
            fc_data["site"].append(phenotype_info['SITE_ID'].values[0])


# Step 3: Save the fc_data dictionary
output_path = '/mnt/ssd1/mehul_data/research2/ABIDE_fc_data.npy'
np.save(output_path, fc_data)

print("Data saved successfully.")
