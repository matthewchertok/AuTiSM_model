import os
import pandas as pd
import pickle
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Convert dictionary to DataFrame and save as pickle.')
parser.add_argument('--save_name', type=str, default='rg_and_anisotropy_production_dataframe.pkl', help='Name of the output pickle file')
args = parser.parse_args()
save_name = args.save_name

# Change to the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Load the pickle file containing the dictionary
with open('rg_and_anisotropy_production.pkl', 'rb') as f:
    data_dict = pickle.load(f)

# Convert dictionary to DataFrame
df = pd.DataFrame.from_dict(data_dict, orient='index')

# Reset the index to have filenames as a column
df.reset_index(inplace=True)
df.rename(columns={'index': 'filename'}, inplace=True)

# Flatten the list of dictionaries into separate rows for each file
flattened_data = []
for index, row in df.iterrows():
    # everything but the filename
    for entry in row[1:]:
        flattened_entry = {'filename': row['filename']}
        flattened_entry.update(entry)
        flattened_data.append(flattened_entry)

# Convert the flattened data into a DataFrame
df = pd.DataFrame(flattened_data)

# Save the DataFrame to a pickle file
df.to_pickle(f'{save_name}.pkl')

# Delete the original pickle file
os.remove('rg_and_anisotropy_production.pkl')