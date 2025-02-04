from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
import torch
from torch.utils.data import DataLoader
import sys
import pickle
import selfies as sf
from rdkit import Chem
from functools import partial
# from .scores import gsk3b, jnk3, qed, sa, esol, plogp, qed_sa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import pandas as pd
from tdc import Oracle
from sklearn.preprocessing import MinMaxScaler, StandardScaler

basic_tokenizer = AutoTokenizer.from_pretrained("zjunlp/MolGen-large")

def print_dataloader(dataloader):
    for batch in dataloader:
        selfies = batch['selfies']
        input_ids = batch['input_ids']
        print(f'selfies stirng {selfies}')  # Print the first element of the batch
        print(f'tokenized input {input_ids}')
        break

def shift_tokens_right(input_ids: torch.Tensor):
        """
        Shift input ids one token to the right.
        """
        pad_token_id = 1
        decoder_start_token_id = 2
        # shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids = torch.zeros_like(input_ids)
        # shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 1:] = input_ids[:, :-1]
        shifted_input_ids[:, 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        return shifted_input_ids

def tokenize(example, tokenize=basic_tokenizer):
    col_name = "selfies"
    encoder_data = {**{"selfies":example[col_name]}, **tokenize(example[col_name], padding=True, truncation = True, return_tensors = "pt")}
    decoder_data = shift_tokens_right(encoder_data["input_ids"])
    return {**encoder_data,"decoder_input_ids":decoder_data}

# PHENOTYPIC DATA PROCESSING 

class Phenotype_CLIP:
    def __init__(self, train_path, val_path, test_path):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.basic_tokenizer = AutoTokenizer.from_pretrained("zjunlp/MolGen-large")

    def gene_data_preprocessing(self, data):
        del data['smiles']
        del data['selfies_embd']
        del data['tanimoto']
        data = Dataset.from_dict(data)
        return data
    
    def gene_tokenize(self, example):
        col_name = "selfies"
        encoder_data = self.basic_tokenizer(example[col_name], padding="max_length", truncation = True, return_tensors = "pt")
        labels = torch.where(encoder_data['input_ids'] == 1, -100, encoder_data['input_ids'])
        vec_cond = torch.tensor(example['gene_clip_embeds'][0]).unsqueeze(0).float()
        return {**encoder_data, **{"labels":labels}, **{"selfies":example[col_name]}, **{"vec_cond":vec_cond}}
    
    def gene_dataset(self, shuffle=False, train_batch_sze = 75, val_batch_sze = 75, test_batch_sze = 75):
        dataset = self.gene_data_preprocessing(torch.load(self.train_path)).train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset['train']
        test_dataset = dataset['test']
        val_dataset = test_dataset.with_transform(self.gene_tokenize)
        train_dataset = train_dataset.with_transform(self.gene_tokenize)
        test_dataset = test_dataset.with_transform(self.gene_tokenize)
        dataloader = DataLoader(train_dataset, batch_size=train_batch_sze, shuffle=shuffle)
        valloader = DataLoader(val_dataset, batch_size=val_batch_sze, shuffle=shuffle)
        testloader = DataLoader(val_dataset, batch_size=test_batch_sze, shuffle=shuffle)
        dataset = {"train":train_dataset, "val":val_dataset, "test":test_dataset}
        return dataloader, valloader, testloader, dataset
    
class Phenotype:
    def __init__(self, train_path, val_path, test_path):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.basic_tokenizer = AutoTokenizer.from_pretrained("zjunlp/MolGen-large")
    
    def csv_preprocess(self,data):
        column_names = [col for col in data.columns if 'median' in col or col == 'canonical_smiles']
        gene_names = [col for col in column_names if col != "canonical_smiles"]
        data = data[column_names]
        scaler = MinMaxScaler()
        data[gene_names] = scaler.fit_transform(data[gene_names])
        data_dict = {}
        data_dict["selfies"] = []
        data_dict["gene_expn"] = []
        for index, row in data.iterrows():
            data_dict["selfies"].append(sf.encoder(row['canonical_smiles']))
            gene_expression = [row[col] for col in column_names if col != 'canonical_smiles']
            data_dict["gene_expn"].append(gene_expression)
        return Dataset.from_dict(data_dict)
        
    def gene_data_preprocessing(self, data): 
        n = len(data['smiles'])
        new_data = {}
        new_data["selfies"], new_data["gene_expn"] = [], []
        for i in range(n):
            if data["gene_expressions"][i].shape[0] != 0:
                new_data["selfies"].append(sf.encoder(data["smiles"][i]))
                new_data["gene_expn"].append(np.median(data["gene_expressions"][i], axis=0))
        assert len(new_data["selfies"]) == len(new_data["gene_expn"])
        return Dataset.from_dict(new_data)
    
    def gene_tokenize(self, example):
        col_name = "selfies"
        encoder_data = self.basic_tokenizer(example[col_name], padding="max_length", truncation = True, return_tensors = "pt")
        labels = torch.where(encoder_data['input_ids'] == 1, -100, encoder_data['input_ids'])
        vec_cond = torch.tensor(example["gene_expn"][0]).unsqueeze(0).float()
        return {**encoder_data, **{"labels":labels}, **{"selfies":example[col_name]}, **{"vec_cond":vec_cond}}
    
    def gene_dataset(self, shuffle=False, train_batch_sze = 75, val_batch_sze = 75, test_batch_sze = 75):
        test_dataset = self.gene_data_preprocessing(torch.load(self.test_path)).with_transform(self.gene_tokenize)   
        val_dataset = self.gene_data_preprocessing(torch.load(self.val_path)).with_transform(self.gene_tokenize)
        if isinstance(self.train_path, list):
            data_lst = []
            for path in self.train_path:
                data_lst.append(pd.read_csv(path))
            data = pd.concat(data_lst)
            data = self.csv_preprocess(data)
        else:
            data = self.csv_preprocess(pd.read_csv(self.train_path))
        train_dataset = data.with_transform(self.gene_tokenize)
        dataloader = DataLoader(train_dataset, batch_size=train_batch_sze, shuffle=shuffle)
        valloader = DataLoader(val_dataset, batch_size=val_batch_sze, shuffle=shuffle)
        testloader = DataLoader(val_dataset, batch_size=test_batch_sze, shuffle=shuffle)
        dataset = {"train":train_dataset, "val":val_dataset, "test":test_dataset}
        return dataloader, valloader, testloader, dataset

# MULTI-OBJECTIVE DIRUG DISCOVERY

class MultiObjective:
    def __init__(self, dataset_path = 'datasets'):
        raw_data = []
        self.basic_tokenizer = AutoTokenizer.from_pretrained("zjunlp/MolGen-large")
        self.qed = Oracle(name = 'QED')
        self.sa = Oracle(name = 'SA')
        self.gsk3b = Oracle(name = 'GSK3B')
        self.jnk3 = Oracle(name = 'JNK3')

        train_path = dataset_path+'/train_positive_data.txt'
        train_dataset = []
        val_path = dataset_path+'/test_positive_data.txt'
        val_dataset = []
        test_path = dataset_path+'/val_positive_data.txt'
        test_dataset = []

        with open(train_path,'r') as f:
            for mol in f:
                train_dataset.append(mol.strip())
        with open(val_path,'r') as f:
            for mol in f:
                val_dataset.append(mol.strip())
        with open(test_path,'r') as f:
            for mol in f:
                test_dataset.append(mol.strip())
   
        self.dataset = {"train":Dataset.from_dict({"selfies":train_dataset}),
                          "val":Dataset.from_dict({"selfies":val_dataset}),
                          "test":Dataset.from_dict({"selfies":test_dataset})}
       
    def latent_tokenize(self, example):
        col_name = "selfies"
        encoder_data = self.basic_tokenizer(example[col_name], padding="max_length", truncation = True, return_tensors = "pt")
        labels = torch.where(encoder_data['input_ids'] == 1, -100, encoder_data['input_ids'])
        return {**encoder_data, **{"labels":labels}, **{"selfies":example[col_name]}}

    def diffusion_tokenize(self, example):
        col_name = "selfies"
        weight = 1
        encoder_data = self.basic_tokenizer(example[col_name], padding="max_length", truncation = True, return_tensors = "pt")
        sa_score = self.qed(sf.decoder(example[col_name][0]))
        qed_score = self.sa(sf.decoder(example[col_name][0]))
        gsk3b_score = self.gsk3b(sf.decoder(example[col_name][0]))
        jnk3_score = self.jnk3(sf.decoder(example[col_name][0]))
        vec_cond = torch.tensor([sa_score, qed_score, gsk3b_score, jnk3_score]).unsqueeze(0).float()

        if jnk3_score >= 0.5:
            weight += 10
        if gsk3b_score >= 0.5:
            weight += 10
        if sa_score <= 2:
            weight += 3
        if qed_score >= 0.7:
            weight += 3

        decoder_data = shift_tokens_right(encoder_data["input_ids"])
        labels = torch.where(encoder_data['input_ids'] == 1, -100, encoder_data['input_ids'])
        data = {**{"selfies":example[col_name]}, **encoder_data,"decoder_input_ids":decoder_data,**{"labels":labels},**{"vec_cond":vec_cond}, **{"weight":torch.tensor([weight]).unsqueeze(0)}}
        return data

    def multiobj_dataset(self, train_batch_sze: int = 75, val_batch_sze: int = 75, test_batch_sze: int = 75, shuffle = True, task = "encoder_training"):
        if task == "encoder_training":
            processed_train = self.dataset["train"].with_transform(self.latent_tokenize)
            processed_val = self.dataset["val"].with_transform(self.latent_tokenize)
            processed_test = self.dataset["test"].with_transform(self.latent_tokenize)
        else:
            processed_train = self.dataset["train"].with_transform(self.diffusion_tokenize)
            processed_val = self.dataset["val"].with_transform(self.diffusion_tokenize)
            processed_test = self.dataset["test"].with_transform(self.diffusion_tokenize)

        dataloader = DataLoader(processed_train, batch_size=train_batch_sze, shuffle=shuffle)
        valloader = DataLoader(processed_val, batch_size=val_batch_sze, shuffle=shuffle)
        testloader = DataLoader(processed_test, batch_size=test_batch_sze, shuffle=shuffle)

        return dataloader, valloader, testloader, self.dataset, processed_val
    

class DPO:
    def __init__(self):
        self.basic_tokenizer = AutoTokenizer.from_pretrained("zjunlp/MolGen-large")
        self.qed = Oracle(name = 'QED')
        self.sa = Oracle(name = 'SA')
        self.gsk3b = Oracle(name = 'GSK3B')
        self.jnk3 = Oracle(name = 'JNK3')

        train_path = 'datasets/dpo_train_data.txt'
        train_dataset_w = []
        train_dataset_l = []
        test_path = 'datasets/dpo_test_data.txt'
        test_dataset_w = []
        test_dataset_l = []

        with open(train_path,'r') as f:
            for mol in f:
                smiles_w, smiles_l = mol.strip().split(",")
                train_dataset_w.append(smiles_w)
                train_dataset_l.append(smiles_l)

        with open(test_path,'r') as f:
            for mol in f:
                smiles_w, smiles_l = mol.strip().split(",")
                test_dataset_w.append(smiles_w)
                test_dataset_l.append(smiles_l)
                
        self.dataset = {"train":Dataset.from_dict({"selfies_w":train_dataset_w, "selfies_l":train_dataset_l}),
                          "val":Dataset.from_dict({"selfies_w":test_dataset_w, "selfies_l":test_dataset_l})}
       
    def tokenize(self, example):
        encoder_w_data = self.basic_tokenizer(example["selfies_w"], padding="max_length", truncation = True, return_tensors = "pt")
        key_names_w = {'input_ids':'input_ids_w','attention_mask': 'attention_mask_w'}
        encoder_w_data = {key_names_w.get(k, k): v for k, v in encoder_w_data.items()}
        encoder_l_data = self.basic_tokenizer(example["selfies_l"], padding="max_length", truncation = True, return_tensors = "pt")
        key_names_l = {'input_ids': 'input_ids_l','attention_mask': 'attention_mask_l'}
        encoder_l_data = {key_names_l.get(k, k): v for k, v in encoder_l_data.items()}
        vec_cond = torch.tensor([self.qed(sf.decoder(example["selfies_w"][0])), 
                                self.sa(sf.decoder(example["selfies_w"][0])), 
                                self.gsk3b(sf.decoder(example["selfies_w"][0])), 
                                self.jnk3(sf.decoder(example["selfies_w"][0]))]).unsqueeze(0).float()
        
        return {**encoder_w_data, **encoder_l_data, **{"selfies_w":example["selfies_w"], "selfies_l":example["selfies_l"]},**{"vec_cond":vec_cond}}

    def dpo_dataset(self, train_batch_sze = 75, val_batch_sze = 75, shuffle=True):
        processed_train = self.dataset["train"].with_transform(self.tokenize)
        processed_val = self.dataset["val"].with_transform(self.tokenize)
        dataloader = DataLoader(processed_train, batch_size=train_batch_sze, shuffle=shuffle)
        valloader = DataLoader(processed_val, batch_size=val_batch_sze, shuffle=shuffle)
        return dataloader, valloader, self.dataset, processed_val