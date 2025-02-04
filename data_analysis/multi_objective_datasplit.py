import random
from collections import defaultdict, Counter
import selfies as sf
from rdkit import Chem
import sys

def read_smiles(file_path):
    with open(file_path, 'r') as file:
        smiles = set(line.strip().split(',')[0] for line in file)
    return smiles

def write_smiles(smiles, file_path):
    with open(file_path, 'w') as file:
        for smile in smiles:
            if Chem.MolFromSmiles(smile):
                try:
                    file.write(f"{sf.encoder(smile)}\n")
                except Exception as e:
                    pass 

# Read the SMILES strings from the four files
# file0_smiles = read_smiles('/raid/home/raswanth/multiobj-rationale/data/chembl/all.txt')
file1_smiles = read_smiles('/raid/home/raswanth/multiobj-rationale/data/dual_gsk3_jnk3/actives.txt')
file2_smiles = read_smiles('/raid/home/raswanth/multiobj-rationale/data/dual_gsk3_jnk3/actives.txt')
file3_smiles = read_smiles('/raid/home/raswanth/multiobj-rationale/data/dual_gsk3_jnk3/actives.txt')

# print("file1",len(file1_smiles))
print("file2",len(file1_smiles))
print("file3",len(file2_smiles))
print("file4",len(file3_smiles))

# Combine all SMILES strings into a single set to find all unique SMILES
all_smiles = file1_smiles | file2_smiles | file3_smiles

# Create a dictionary to count occurrences of each SMILES string in each file
smiles_dict = defaultdict(lambda: [0, 0, 0])

for smile in file1_smiles:
    smiles_dict[smile][0] += 1
for smile in file2_smiles:
    smiles_dict[smile][1] += 1
for smile in file3_smiles:
    smiles_dict[smile][2] += 1

# Assign each SMILES string to a source file where it is unique
assigned_smiles = set()
unique_counts = [0, 0, 0]

print(unique_counts)

for smile in all_smiles:
    file_indices = [i for i, count in enumerate(smiles_dict[smile]) if count > 0]
    if len(file_indices) == 1:
        unique_counts[file_indices[0]] += 1
        assigned_smiles.add(smile)
    else:
        if 0 in file_indices:
            selected_index = 0
        else:
            selected_index = random.choice(file_indices)
        unique_counts[selected_index] += 1
        assigned_smiles.add(smile)

# Calculate the ratios of unique molecules
total_unique_smiles_count = sum(unique_counts)
ratios = [count / total_unique_smiles_count for count in unique_counts]

print(total_unique_smiles_count)
print(unique_counts)
print(ratios)
assert total_unique_smiles_count == len(all_smiles)
# Calculate the number of molecules for each portion
train_split = int(total_unique_smiles_count * 0.8)
val_split = int(total_unique_smiles_count * 0.1)

# Function to split SMILES strings while maintaining the ratios and ensuring no overlap
def split_smiles(smiles_dict, total_portion, ratios, used_smiles):
    portion_counts = [int(total_portion * ratio) for ratio in ratios]
    portion_smiles = []

    for smile, counts in smiles_dict.items():
        if smile in used_smiles:
            continue
        for file_index, count in enumerate(counts):
            if count > 0 and portion_counts[file_index] > 0:
                portion_smiles.append(smile)
                portion_counts[file_index] -= 1
                used_smiles.add(smile)
                break
    
    return portion_smiles

# Used SMILES tracker to ensure no overlap between splits
used_smiles = set()

# Split the SMILES strings for each portion
train_data = split_smiles(smiles_dict, train_split, ratios, used_smiles)
val_data = split_smiles(smiles_dict, val_split, ratios, used_smiles)
test_data = split_smiles(smiles_dict, val_split, ratios, used_smiles)

remaining_smiles = set(smiles_dict.keys()) - used_smiles
# negative_data = list(set(file0_smiles) - all_smiles)[:1000]
# remaining_smiles = list(remaining_smiles) + negative_data
if len(remaining_smiles):
    test_data += remaining_smiles

train_data += test_data 

write_smiles(train_data+val_data+test_data, 'datasets/dpo_train_data.txt')
write_smiles(test_data, 'datasets/dpo_test_data.txt')
sys.exit()
print(len(train_data), len(val_data), len(test_data))
# assert len(train_data) + len(val_data) + len(test_data) == total_unique_smiles_count + len(negative_data)
# Write the results to new file
write_smiles(train_data, 'datasets/train_positive_data.txt')
write_smiles(val_data, 'datasets/val_positive_data.txt')
write_smiles(test_data, 'datasets/test_positive_data.txt')
print("Files split into train_data.txt, val_data.txt, and test_data.txt")