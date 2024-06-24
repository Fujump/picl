
import torch
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from common import get_noise_label



def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

class ClassificationTasksDatasets:
    def __init__(self, dataset_type, train_test_split_ratio, task, noise_type, noise_ratio, seed):
        self.dataset_type = dataset_type
        self.train_test_split_ratio = train_test_split_ratio
        self.noise_type = noise_type
        self.noise_ratio = noise_ratio
        self.task = task
    
        setup_seed(seed)
        raw_dataset = pd.DataFrame(columns=["task_id", "text", "label", 'options'])
        if self.task == 'ag_news':
            raw_dataset = pd.read_csv("/mnt/sharedata/ssd/users/hongfugao/dataset/ag_news.csv", header=0)
        
        elif self.task == 'sst5':
            raw_dataset = pd.read_csv("/mnt/sharedata/ssd/users/hongfugao/dataset/sst5.csv", header=0)
            
        elif self.task == 'sst2':
            raw_dataset = pd.read_csv("/mnt/sharedata/ssd/users/hongfugao/dataset/sst2.csv", header=0)
        
        elif self.task == 'mnli':
            raw_dataset = pd.read_csv("/mnt/sharedata/ssd/users/hongfugao/dataset/mnli.csv", header=0)
        
        elif self.task == 'illegal':
            raw_dataset = pd.DataFrame(columns=["label_sub", "label_total", "text"])
            raw_dataset = pd.read_json("all_data.json", orient='columns', lines=True)
        elif self.task == 'illegal_debug':
            raw_dataset = pd.DataFrame(columns=["text","label_total"])
            raw_dataset = pd.read_json("data_debug.json", orient='columns', lines=True)
        elif self.task == 'cold':
            pass
        else:
            print("ERROR DATALOADER")
        
        
        if self.task!='cold':
            train_ds, test_ds = train_test_split(raw_dataset, test_size=self.train_test_split_ratio, shuffle=True, random_state=100)

        if self.dataset_type == 'train':
            if self.task == 'illegal' or self.task == 'illegal_debug':
                self.text = train_ds['text'].tolist()
                self.raw_label = pd.Series(train_ds['label_total'].tolist())
            elif self.task == 'cold':
                train_ds=pd.read_csv("/data/home/huq/picl/dataset/cold/train.csv")
                self.text = train_ds['TEXT'].tolist()
                self.raw_label = pd.Series(train_ds['label'].tolist())
            else:
                self.text = train_ds['text'].tolist()
                self.raw_label = pd.Series(train_ds['label'].tolist())
            
        else:
            if self.task == 'illegal' or self.task == 'illegal_debug':
                self.text = test_ds['text'].tolist()
                self.raw_label = pd.Series(test_ds['label_total'].tolist())
            elif self.task == 'cold':
                train_ds=pd.read_csv("/data/home/huq/picl/dataset/cold/test.csv")
                self.text = train_ds['TEXT'].tolist()
                self.raw_label = pd.Series(train_ds['label'].tolist())
            else:
                self.text = test_ds['text'].tolist()
                self.raw_label = pd.Series(test_ds['label'].tolist())
        
        classes_num = self.get_classes_num()
        self.label = get_noise_label(self.noise_type, self.raw_label, self.noise_ratio, random_state=seed, classes=classes_num).tolist()


    def __getitem__(self, index):
        text, target = self.text[index], self.label[index]

        return text, target, index

    def __len__(self):
        return len(self.text)
    
    def get_classes_num(self):
        return len(list(set(self.raw_label)))

