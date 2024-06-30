import pandas as pd
from sklearn.model_selection import train_test_split



class GenerationTasksDatasets:
    def __init__(self, dataset_type, task, imbalance_type, imbalance_ratio, seed):
        self.dataset_type = dataset_type
        self.imbalance_type = imbalance_type
        self.imbalance_ratio = imbalance_ratio
        self.task = task
        self.seed = seed


        raw_train_dataset = pd.read_csv("/data/home/hongfugao/train_mintaka.csv", header=0)
        raw_test_dataset = pd.read_csv("/data/home/hongfugao/test_mintaka.csv", header=0)

        
        if self.dataset_type == 'train':
            self.text_total = raw_train_dataset['question']
            self.label_total = raw_train_dataset['label']

        
        else:
            self.text_total = raw_test_dataset['question']
            self.label_total = raw_test_dataset['label']

        


    
    def __getitem__(self, index):
        text, target, = self.text_total[index], self.label_total[index]

        return text, target, index

    def __len__(self):
        return len(self.text_total)