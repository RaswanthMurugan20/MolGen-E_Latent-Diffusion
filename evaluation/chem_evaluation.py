import selfies as sf
import sys
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from argparse import ArgumentParser
from dataset_utils.score_modules.SA_Score import sascorer
from functools import partial
from tdc import Oracle

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

def Tanimoto_Similarity(pred_mols, true_mols, gen_type = "selfies", similarity_type="morgan"):

    if gen_type == "selfies":
        true_mols = [sf.decoder(s) for s in true_mols]
        pred_mols = [sf.decoder(s) for s in pred_mols]
    
    assert len(true_mols) == len(pred_mols)
    true_mols = [Chem.MolFromSmiles(s) for s in true_mols]
    true_mols = [x for x in true_mols if x is not None]
    pred_mols = [Chem.MolFromSmiles(s) for s in pred_mols]
    pred_mols = [x for x in pred_mols if x is not None]
    if similarity_type == "morgan":
        pred_fps = [AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048) for x in pred_mols]
        true_fps = [AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048) for x in true_mols]
    elif similarity_type == "maccs":    
        true_fps = [MACCSkeys.GenMACCSKeys(x) for x in true_mols]
        pred_fps = [MACCSkeys.GenMACCSKeys(x) for x in pred_mols]
    else:
        raise NotImplementedError("this fingerprint has not been implemented")

    n = min(len(true_fps), len(pred_fps))
    similarity = 0
    for i in range(n):
        sim = DataStructs.TanimotoSimilarity(true_fps[i], pred_fps[i])
        similarity += sim
    avg_sim = similarity/n
    return avg_sim 

def validity_uniqueness(pred_mols,gen_type): 
    if gen_type == "selfies": # if text generated files contains selfies not smiles
        pred_mols = [sf.decoder(s) for s in pred_mols]
    pred_mols = [Chem.MolFromSmiles(s) for s in pred_mols]
    n = len(pred_mols)
    pred_mols = [x for x in pred_mols if x is not None]
    x = len(pred_mols)
    x_u = len(set(pred_mols))
    return x/n , x_u/x

def novelty(pred_mols, true_mols, gen_type, similarity_type="morgan"):
    if gen_type == "selfies": 
        pred_mols = [sf.decoder(s) for s in pred_mols]
    pred_mols = [Chem.MolFromSmiles(s) for s in pred_mols]
    pred_mols = [x for x in pred_mols if x is not None]

    if similarity_type == "morgan":
        pred_fps = [AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048) for x in pred_mols]
    elif similarity_type == "maccs":    
        pred_fps = [MACCSkeys.GenMACCSKeys(x) for x in pred_mols]
    else:
        raise NotImplementedError("this fingerprint has not been implemented")

    fraction_similar = 0 # computing the Tanimoto similarity
    n = len(pred_fps)

    for i in range(len(pred_fps)):
        sims = DataStructs.BulkTanimotoSimilarity(pred_fps[i], true_mols)
        if max(sims) == 1:
            fraction_similar += 1

    novelty = 1 - fraction_similar / len(pred_mols)
    return novelty # novelty score compute
    
def diversity(pred_mols, gen_type, similarity_type="morgan"):
    if gen_type == "selfies":
        pred_mols = [sf.decoder(s) for s in pred_mols]
    pred_mols = [Chem.MolFromSmiles(s) for s in pred_mols]
    pred_mols = [x for x in pred_mols if x is not None]
    if similarity_type == "morgan":
        pred_fps = [AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048) for x in pred_mols]
    elif similarity_type == "maccs":    
        pred_fps = [MACCSkeys.GenMACCSKeys(x) for x in pred_mols]
    else:
        raise NotImplementedError("this fingerprint has not been implemented")
    
    similarity = 0
    for i in range(len(pred_fps)):
        sims = DataStructs.BulkTanimotoSimilarity(pred_fps[i], pred_fps[:i])
        similarity += sum(sims)

    n = len(pred_fps) 
    n_pairs = n * (n - 1) / 2
    diversity = 1 - similarity / n_pairs
    return diversity

def sa(pred_mols,gen_type):
    sa_oracle = Oracle(name = 'SA')
    if gen_type == "selfies":
        pred_mols = [sa_oracle(sf.decoder(s)) for s in pred_mols]
    else:
        pred_mols = [sa_oracle(s) for s in pred_mols]
    return sum(pred_mols) / len(pred_mols)

def qed(pred_mols,gen_type):
    qed_oracle = Oracle(name = 'QED')
    if gen_type == "selfies":
        pred_mols = [qed_oracle(sf.decoder(s)) for s in pred_mols]
    else:
        pred_mols = [qed_oracle(s) for s in pred_mols]
    return sum(pred_mols) / len(pred_mols)

def jnk3_score(pred_mols,gen_type):
    jnk3_oracle = Oracle(name = 'JNK3')
    if gen_type == "selfies":
        pred_mols = [jnk3_oracle(sf.decoder(s)) for s in pred_mols]
    else:
        pred_mols = [jnk3_oracle(s) for s in pred_mols]
    return sum(pred_mols) / len(pred_mols)
    

def gsk3b_score(pred_mols,gen_type):
    gsk3b_oracle = Oracle(name = 'GSK3B')
    if gen_type == "selfies":
        pred_mols = [gsk3b_oracle(sf.decoder(s)) for s in pred_mols]
    else:
        pred_mols = [gsk3b_oracle(s) for s in pred_mols]
    return sum(pred_mols) / len(pred_mols)

def success_rate(pred_mols,gen_type):
    gsk3b_oracle = Oracle(name = 'GSK3B')
    jnk3_oracle = Oracle(name = 'JNK3')
    if gen_type == "selfies":
        pred_mols = [1 if jnk3_oracle(sf.decoder(s)) > 0.45 and gsk3b_oracle(sf.decoder(s)) > 0.45 else 0 for s in pred_mols]
    else:
        pred_mols = [1 if jnk3_oracle(s) > 0.45 and gsk3b_oracle(s) > 0.45 else 0 for s in pred_mols]

    return sum(pred_mols)/len(pred_mols)

def compute_average_tanimoto(original, generated, similarity_type = "morgan"):
    # Ensure both lists are the same length
    if len(original) != len(generated):
        raise ValueError("Both lists must be of the same length.")

    similarities = []

    for selfies1, selfies2 in zip(original, generated):
        mol1 = Chem.MolFromSmiles(sf.decoder(selfies1))
        mol2 = Chem.MolFromSmiles(sf.decoder(selfies2))

        if mol1 is None or mol2 is None:
            continue  # Skip pairs where conversion fails

        # Generate fingerprints for both molecules
        if similarity_type == "morgan":
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1,3,2048)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2,3,2048)
        elif similarity_type == "maccs":
            fp1 = MACCSkeys.GenMACCSKeys(mol1)
            fp2 = MACCSkeys.GenMACCSKeys(mol2)
        else:
            NotImplementedError("this fingerprint has not been implemented")

        # Compute Tanimoto similarity
        similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
        similarities.append(similarity)

    # Compute the average Tanimoto similarity
    if similarities:  # Check if the list is not empty
        average_similarity = sum(similarities) / len(similarities)
    else:
        average_similarity = 0  # In case all pairs were invalid

    return average_similarity




    




