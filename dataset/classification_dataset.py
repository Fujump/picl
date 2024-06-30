
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
    def __init__(self, dataset_type, task, imbalance_type, imbalance_ratio, seed):
        self.dataset_type = dataset_type
        self.imbalance_type = imbalance_type
        self.imbalance_ratio = imbalance_ratio
        self.task = task
        self.seed = seed
        setup_seed(seed)

        raw_text = []
        raw_label = []

        if self.task == 'sst2':
            if self.dataset_type == 'train':
                dat = load_dataset('SetFit/sst2', split='train')
            else:
                dat = load_dataset('SetFit/sst2', split='test')

            for datapoint in dat:
                raw_text.append(datapoint['text'])
                raw_label.append(datapoint['label'])

            class_num = 2
            raw_dat = pd.DataFrame({"text":raw_text, "label":raw_label})
        
        elif self.task == 'agnews':
            if self.dataset_type == 'train':
                raw_dat = pd.read_csv("/data/home/hongfugao/train_agnews.csv")
            else:
                raw_dat = pd.read_csv("/data/home/hongfugao/test_agnews.csv")
            class_num = 4
        
        elif self.task == 'sst5':
            if self.dataset_type == 'train':
                raw_dat = pd.read_csv("/data/home/hongfugao/train_sst5.csv")
            else:
                raw_dat = pd.read_csv("/data/home/hongfugao/test_sst5.csv")
            class_num = 5

        
        elif self.task == 'anli':
            if self.dataset_type == 'train':
                raw_dat = pd.read_csv("/data/home/hongfugao/anli_train.csv")
            else:
                raw_dat = pd.read_csv("/data/home/hongfugao/anli_test.csv")
            class_num = 3

        elif self.task == 'dbpedia':
            if self.dataset_type == 'train':
                raw_dat = pd.read_csv("/data/home/hongfugao/train_dbpedia.csv")
            else:
                raw_dat = pd.read_csv("/data/home/hongfugao/test_dbpedia.csv")
            class_num = 14


        else:
            print("ERROR DATALOADER")

        if self.dataset_type == 'train':
            img_num_list = self.get_img_num_per_cls(raw_dat, class_num, 'exp', imbalance_ratio)
            print(img_num_list)
            self.text, self.label = self.gen_imbalanced_data(raw_dat['text'], raw_dat['label'], img_num_list)

        else:
            self.text = raw_dat['text']
            self.label = raw_dat['label']



        # if (self.dataset_type == 'train') and (self.imbalance_type == 'real'):
        #     train_ds, test_ds = train_test_split(raw_dat, test_size=3000, shuffle=True, random_state=seed)
        #     self.text = test_ds['text'].tolist()
        #     self.label = test_ds['label'].tolist()

        # if (self.dataset_type == 'train') and (self.imbalance_type == 'imbalance'):
        #     data_label_0 = raw_dat[raw_dat['label'] == 0]
        #     data_label_1 = raw_dat[raw_dat['label'] == 1]

        #     p1 = imbalance_ratio/(1+imbalance_ratio)
        #     p2 = 1/(1+imbalance_ratio)

        #     train_ds, test_ds_0 = train_test_split(data_label_0, test_size=int(3000*p1), shuffle=True, random_state=seed)
        #     train_ds, test_ds_1 = train_test_split(data_label_1, test_size=int(3000*p2), shuffle=True, random_state=seed)

        #     self.text = sum([test_ds_0['text'].tolist(), test_ds_1['text'].tolist()], [])
        #     self.label = sum([test_ds_0['label'].tolist(), test_ds_1['label'].tolist()], [])
            
        # if self.dataset_type == 'test':
        #     self.text = raw_dat['text']
        #     self.label = raw_dat['label']


            
        

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