from .generation_dataset import GenerationTasksDatasets
from .classification_dataset import ClassificationTasksDatasets

def get_dataloader(task, type, imbalance_type, imbalance_ratio, seed):

    if task == 'sst2' or task == 'agnews' or task == 'sst5' or task == 'dbpedia':
        raw_dataset = ClassificationTasksDatasets(dataset_type=type, task=task, imbalance_type=imbalance_type, imbalance_ratio=imbalance_ratio, seed=seed)

    else:
        raw_dataset = GenerationTasksDatasets(dataset_type=type, task=task, imbalance_type=imbalance_type, imbalance_ratio=imbalance_ratio, seed=seed)
        
    return raw_dataset
    