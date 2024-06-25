
import torch
import random
import numpy as np
import pandas as pd

from tqdm import tqdm
from datasets import load_dataset
from sklearn.model_selection import train_test_split


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
            raw_dataset = pd.read_json("dataset/all_data.json", orient='columns', lines=True)
        elif self.task == 'illegal_debug':
            raw_dataset = pd.DataFrame(columns=["text","label_total"])
            raw_dataset = pd.read_json("dataset/data_debug.json", orient='columns', lines=True)
        elif self.task == 'cold':
            pass
        else:
            print("ERROR DATALOADER")
        
        
        if self.task!='cold':
            train_ds, test_ds = train_test_split(raw_dataset, test_size=self.train_test_split_ratio, shuffle=True, random_state=100)
            print(f"test size:{len(test_ds)}")

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
        
        # classes_num = self.get_classes_num()
        self.label = self.raw_label


            
        

    def __getitem__(self, index):
        text, target = self.text[index], self.label[index]

        return text, target, index

    def __len__(self):
        return len(self.text)
    

    def get_img_num_per_cls(self, data, cls_num, imb_type, imb_factor):
        img_max = len(data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls
    

    def gen_imbalanced_data(self, text, label, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(label, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append([text[idx] for idx in selec_idx])
            new_targets.extend([the_class, ] * the_img_num)

        return sum(new_data,[]), new_targets
