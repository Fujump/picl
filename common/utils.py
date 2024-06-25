import torch
import random
import numpy as np
import pandas as pd

from tqdm import tqdm
from transformers import AutoTokenizer
from typing import List, Union, Optional

class DatasetEncoder(torch.utils.data.Dataset):
    def __init__(self, datalist: List, model_name=None, tokenizer=None) -> None:
        self.datalist = datalist
        if model_name is None and tokenizer is None:
            raise ValueError("model_name and tokenizer could not both be None")
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.padding_side = "left"
        self.encode_dataset = []
        self.init_dataset()
        self.datalist_length = len(self.encode_dataset)

    def init_dataset(self):
        for idx, data in enumerate(self.datalist):
            tokenized_data = self.tokenizer.encode_plus(data, truncation=True, return_tensors='pt', verbose=False)
            self.encode_dataset.append({
                'input_ids': tokenized_data.input_ids[0],
                'attention_mask': tokenized_data.attention_mask[0],
                "metadata": {"id": idx, "len": len(tokenized_data.input_ids[0]),
                             "text": data}
            })

    def __len__(self):
        return self.datalist_length

    def __getitem__(self, idx):
        return self.encode_dataset[idx]

        

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    



def get_inputes(task, new_ice_idx, new_ice_target, template, template_dict, training_dataset):
    if (task == 'agnews' or task == 'sst2' or task == 'sst5' or task == 'dbpedia'):
        generated_ice_list = []
        for idx_list in range(len(new_ice_idx)):

            raw_ice = ''
            for idx in range(len(new_ice_idx[idx_list])):
                raw_ice = raw_ice + str.replace(template_dict[new_ice_target[idx_list][idx]], '</text>', str(training_dataset['text'][new_ice_idx[idx_list][idx]])) + '\n'
            generated_ice_list.append(raw_ice)
    else:
        generated_ice_list = []
        for idx_list in new_ice_idx:

            raw_ice = ''
            for idx in idx_list:
                raw_ice = raw_ice + str.replace(str.replace(template, '</text>', str(training_dataset['text'][idx])), '</answer>', str(new_ice_target[idx])) + '\n'
            generated_ice_list.append(raw_ice)

    return generated_ice_list


def collote_fn(batch_samples, tokenizer=AutoTokenizer.from_pretrained("/mnt/sharedata/ssd/users/hongfugao/model/bert-base-cased", trust_remote_code=True)):
    batch_sentence = []
    batch_label = []
    for text, target, index in batch_samples:
        batch_sentence.append(text)
        batch_label.append(target)
        
    X = tokenizer(
        batch_sentence, 
        padding=True,
        max_length = 512, 
        truncation=True, 
        return_tensors="pt"
    )
    y = torch.tensor(batch_label)
    return X, y


def get_prompt_label(task):
    if task == 'illegal' or task == 'illegal_debug':
        template = '\n </text> '
        labels = ["不违规", "违规"]
        template_dict = {
                        1: " </E> \n </text> 违规",
                        0: " </E> \n </text> 不违规" 
                    }

    elif task == 'sst2':
        template = '\n Positve or Negative New Review? \n Input: </text> \n Output:'
        labels = ["Negative", "Positve"]
        template_dict = {
                        1: " </E> \n Positve or Negative New Review? \n Input: </text> \n Output: Positive",
                        0: " </E> \n Positve or Negative New Review? \n Input: </text> \n Output: Negative" 
                    }
        
    elif task == 'sst5':
        template = '\n Very Positve, Positve, Neutral, Negative or Very Negative Movie Review? \n Input: </text> \n Output:'
        labels = ["Very Negative","Negative", "Neutral", "Positve", "Very Positve"]
        template_dict = {
                0: "</E> \n Very Positve, Positve, Neutral, Negative or Very Negative Movie Review? \n Input: </text> \n Output: Very Negative",
                1: "</E> \n Very Positve, Positve, Neutral, Negative or Very Negative Movie Review? \n Input: </text> \n Output: Negative",
                2: "</E> \n Very Positve, Positve, Neutral, Negative or Very Negative Movie Review? \n Input: </text> \n Output: Neutral",
                3: "</E> \n Very Positve, Positve, Neutral, Negative or Very Negative Movie Review? \n Input: </text> \n Output: Positve",
                4: "</E> \n Very Positve, Positve, Neutral, Negative or Very Negative Movie Review? \n Input: </text> \n Output: Very Positve"
            }

    elif task == 'agnews':
        template = '\n World, Sports, Business or Science New Topic? \n Input: </text> \n Output:'
        labels = ["World", "Sports", "Business", "Science"]
        template_dict = {
                0: "</E> \n World, Sports, Business or Science New Topic? \n Input: </text> \n Output: World",
                1: "</E> \n World, Sports, Business or Science New Topic? \n Input: </text> \n Output: Sports",
                2: "</E> \n World, Sports, Business or Science New Topic? \n Input: </text> \n Output: Business",
                3: "</E> \n World, Sports, Business or Science New Topic? \n Input: </text> \n Output: Science"
            }
        
    elif task == 'dbpedia':
        template = '\n Company, Educational Institution, Artist, Athlete, Office Holder, Mean Of Transportation, Building, Natural Place, Village, Animal, Plant, Album, Film or Written Work Topic? \n Input: </text> \n Output:'
        labels = ["Company", "Educational Institution", "Artist", "Athlete", "Office Holder", "Mean Of Transportation", "Building", "Natural Place", "Village", "Animal", "Plant", "Album", "Film", "Written Work"]
        template_dict = {
                0: "</E> \n Company, Educational Institution, Artist, Athlete, Office Holder, Mean Of Transportation, Building, Natural Place, Village, Animal, Plant, Album, Film or Written Work Topic? \n Input: </text> \n Output: Company",
                1: "</E> \n Company, Educational Institution, Artist, Athlete, Office Holder, Mean Of Transportation, Building, Natural Place, Village, Animal, Plant, Album, Film or Written Work Topic? \n Input: </text> \n Output: Educational Institution",
                2: "</E> \n Company, Educational Institution, Artist, Athlete, Office Holder, Mean Of Transportation, Building, Natural Place, Village, Animal, Plant, Album, Film or Written Work Topic? \n Input: </text> \n Output: Artist",
                3: "</E> \n Company, Educational Institution, Artist, Athlete, Office Holder, Mean Of Transportation, Building, Natural Place, Village, Animal, Plant, Album, Film or Written Work Topic? \n Input: </text> \n Output: Athlete",
                4: "</E> \n Company, Educational Institution, Artist, Athlete, Office Holder, Mean Of Transportation, Building, Natural Place, Village, Animal, Plant, Album, Film or Written Work Topic? \n Input: </text> \n Output: Office Holder",
                5: "</E> \n Company, Educational Institution, Artist, Athlete, Office Holder, Mean Of Transportation, Building, Natural Place, Village, Animal, Plant, Album, Film or Written Work Topic? \n Input: </text> \n Output: Mean Of Transportation",
                6: "</E> \n Company, Educational Institution, Artist, Athlete, Office Holder, Mean Of Transportation, Building, Natural Place, Village, Animal, Plant, Album, Film or Written Work Topic? \n Input: </text> \n Output: Building",
                7: "</E> \n Company, Educational Institution, Artist, Athlete, Office Holder, Mean Of Transportation, Building, Natural Place, Village, Animal, Plant, Album, Film or Written Work Topic? \n Input: </text> \n Output: Natural Place",
                8: "</E> \n Company, Educational Institution, Artist, Athlete, Office Holder, Mean Of Transportation, Building, Natural Place, Village, Animal, Plant, Album, Film or Written Work Topic? \n Input: </text> \n Output: Village",
                9: "</E> \n Company, Educational Institution, Artist, Athlete, Office Holder, Mean Of Transportation, Building, Natural Place, Village, Animal, Plant, Album, Film or Written Work Topic? \n Input: </text> \n Output: Animal",
                10: "</E> \n Company, Educational Institution, Artist, Athlete, Office Holder, Mean Of Transportation, Building, Natural Place, Village, Animal, Plant, Album, Film or Written Work Topic? \n Input: </text> \n Output: Plant",
                11: "</E> \n Company, Educational Institution, Artist, Athlete, Office Holder, Mean Of Transportation, Building, Natural Place, Village, Animal, Plant, Album, Film or Written Work Topic? \n Input: </text> \n Output: Album",
                12: "</E> \n Company, Educational Institution, Artist, Athlete, Office Holder, Mean Of Transportation, Building, Natural Place, Village, Animal, Plant, Album, Film or Written Work Topic? \n Input: </text> \n Output: Film",
                13: "</E> \n Company, Educational Institution, Artist, Athlete, Office Holder, Mean Of Transportation, Building, Natural Place, Village, Animal, Plant, Album, Film or Written Work Topic? \n Input: </text> \n Output: Written Work",
                }

    elif task == 'nq':
        template = '</E> \n Question: </text> \n Output: </answer>'
        labels = []
        template_dict = {0:'</E> \n Question: </text> \n Output: </answer>'}

    else:
        print('ERROR PROMPT')
    
    return template, template_dict, labels


def extract_data(dataloader, task):
    texts = []
    labels = []
    ppls = []
    if (task == 'sst2' or task == 'agnews' or task == 'sst5' or task == 'dbpedia'):
        for text, target, index in tqdm (dataloader):
            texts.append(text[0])
            labels.append(int(target[0]))
    else:
        for text, target, index in tqdm (dataloader):
            texts.append(text[0])
            labels.append(0 if target[0]=="不违规" else 1)

    data = pd.DataFrame({"text": texts, "label": labels})
    return data

def generate_label_prompt(idx, test_ds, ice, label, template):
    raw_text = str.replace(template[label], '</text>', test_ds)
    prompt = str.replace(raw_text, '</E>', ice)
    return prompt

def get_input(task, ice_idx_list, template, template_dict, training_dataset):
    if (task == 'sst2' or task == 'agnews' or task == 'sst5' or task == 'dbpedia'):
        generated_ice_list = []
        for idx_list in ice_idx_list:

            raw_ice = ''
            for idx in idx_list:
                raw_ice = raw_ice + str.replace(template_dict[training_dataset['label'][idx]], '</text>', str(training_dataset['text'][idx])) + '\n'
            generated_ice_list.append(raw_ice)
    else:
        generated_ice_list = []
        for idx_list in ice_idx_list:

            raw_ice = ''
            for idx in idx_list:
                raw_ice = raw_ice + str.replace(str.replace(template, '</text>', str(training_dataset['text'][idx])), '</answer>', str(training_dataset['label'][idx])) + '\n'
            generated_ice_list.append(raw_ice)
            
    return generated_ice_list


def delect_unavailable_word(text):
    preds = []
    
    for pred in tqdm(text):
        preds.append(str.replace(str.replace(pred.split('Question', 1)[0], '</E>', ''),'\n','').strip())
    return preds

from torch.utils.data import DataLoader
def get_dataloader(datalist: List[List], batch_size: int) -> DataLoader:
    dataloader = DataLoader(datalist, batch_size=batch_size)
    return dataloader

def calculate_precision_recall(preds, labels):
    TP = sum((pred == 1) and (label == 1) for pred, label in zip(preds, labels))
    FP = sum((pred == 1) and (label == 0) for pred, label in zip(preds, labels))
    FN = sum((pred == 0) and (label == 1) for pred, label in zip(preds, labels))

    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0

    return precision, recall

def calculate_precision_recall_neg(preds, labels):
    TP = sum((pred == 0) and (label == 0) for pred, label in zip(preds, labels))
    FP = sum((pred == 0) and (label == 1) for pred, label in zip(preds, labels))
    FN = sum((pred == 1) and (label == 0) for pred, label in zip(preds, labels))

    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0

    return precision, recall