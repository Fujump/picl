from .classification_dataset import ClassificationTasksDatasets
from .generation_dataset import GenerationTasksDatasets
from .classification_dataset import ClassificationTasksDatasets

def get_dataloader(task, type, noise_type, noise_ratio, seed):
    if task == 'sst5' or task == 'sst2' or task == 'ag_news' or task == 'mnli' or task == 'illegal' or task == 'illegal_debug' or task =='cold':
        raw_dataset  = ClassificationTasksDatasets(dataset_type=type, task=task, noise_type=noise_type, noise_ratio=noise_ratio, train_test_split_ratio=0.05, seed=seed)
    
    else:
        raw_dataset = GenerationTasksDatasets(dataset_type=type, task=task, noise_type=noise_type, noise_ratio=noise_ratio, seed=seed)

    return raw_dataset
    