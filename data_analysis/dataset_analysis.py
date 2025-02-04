from datasets import load_dataset
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from transformers import AutoTokenizer
import selfies as sf
from rdkit import Chem
from scipy.stats import skew
from tdc import Oracle
import torch
import seaborn as sns
import matplotlib.pyplot as plt




def plot():
# Load the .pth files (assuming they contain dictionaries)
    file1 = torch.load("multiobj_random.pth")
    file2 = torch.load("multiobj_vec.pth")
    file3 = torch.load("multiobj_zeros.pth")

    print(file1.keys())

    # Assuming "beam" is a key in these dictionaries
    beam_data1 = file1["beam"]
    beam_data2 = file2["beam"]
    beam_data3 = file3["beam"]

    sns.kdeplot(beam_data1, shade=True, color="red", label="TS_random")
    sns.kdeplot(beam_data2, shade=True, color="blue", label="TS_vec")
    sns.kdeplot(beam_data3, shade=True, color="green", label="TS_zeros")
    plt.legend()

    # Save the plot as a PNG file
    plt.savefig("diffusion_TS_all.png")

tokenizer = "zjunlp/MolGen-large"
tokenizer = AutoTokenizer.from_pretrained(tokenizer)
dataset = []
discard_pile = 0
precious_pile = 0
QED = []
qed_oracle = Oracle(name = 'QED')
SA = []
sa_oracle = Oracle(name = 'SA')
ESOL = []
GSK3B = []
gskb_oracle = Oracle(name = 'GSK3B')
JNK3 = []
jnk3_oracle = Oracle(name = 'JNK3')
gsk_score = []
jnk_score = []
sa_score = []
qed_score = []
i,j = 0,0

with open('/raid/home/raswanth/multiobj-rationale/data/chembl/actives.txt','r') as f:
    for mol in f:
        i += 1
        smiles = mol.strip().split(',')[0]
        if Chem.MolFromSmiles(smiles):
            try:
                # dataset.append(sf.encoder(smiles))
                score = gskb_oracle(smiles)
                if score >= 0.45:
                    j += 1
                gsk_score.append(score)
                jnk_score.append(jnk3_oracle(smiles))
                # sa_score.append(sa_oracle(smiles))
                # qed_score.append(qed_oracle(smiles))
            except Exception as e:
                pass

print(i,j)
print(sum(gsk_score)/len(gsk_score))
print(sum(jnk_score)/len(jnk_score))
# print(sum(sa_score)/len(sa_score))
# print(sum(qed_score)/len(qed_score))
for example in dataset:
    GSK3B.append(gskb_oracle(example))
    # JNK3.append(jnk3_oracle(example))
    # QED.append(sa_oracle(example))
    # SA.append(qed_oracle(example))

# Calculating mean, median, and mode
def MMM(numbers, plot_name):
    mean = np.mean(numbers)
    var = np.sqrt(np.var(numbers))
    n = len(numbers)
    print("mean :-", mean)
    print("var :-", var)
    median = np.median(numbers)
    print("median :-", median)

    # Plotting
    bins_doane = int(1 + np.log2(n) + np.log2(1 + abs(skew(numbers)) / np.sqrt((6 * (n - 2)) / ((n + 1) * (n + 3)))))
    plt.figure(figsize=(10, 6))
    plt.hist(numbers, bins=bins_doane, alpha=0.7, color='blue', edgecolor='black', label='Numbers')
    # plt.hist(numbers, bins=range(min(numbers), max(numbers) + 2), alpha=0.7, color='blue', edgecolor='black', label='Numbers')
    plt.axvline(mean, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {mean:.2f}')
    plt.axvline(median, color='g', linestyle='dashed', linewidth=1, label=f'Median: {median}')
    # plt.axvline(mode[0], color='y', linestyle='dashed', linewidth=1, label=f'Mode: {mode}')

    plt.legend()
    plt.title(plot_name)
    plt.xlabel('Number')
    plt.ylabel('Frequency')

    plt.savefig(f'plot_with_central_tendencies_{plot_name}.png', dpi=300)
    plt.show()

MMM(QED, "QED")
MMM(SA, "SA")
MMM(GSK3B, "GSK3B")
MMM(JNK3, "JNK3")