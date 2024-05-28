'''Random Retriever'''
import numpy as np
from tqdm import trange, tqdm

from common import get_logger
from .topk import TopkRetriever

logger = get_logger(__name__)


class RandomRetriever(TopkRetriever):
    def __init__(self, task,ice_dataloader, candidate_dataloader, noisy_model, noisy_tokenizer, device):
        super().__init__(task, ice_dataloader, candidate_dataloader, noisy_model, noisy_tokenizer, device)
        
        
    def retrieve(self, ice_num, candidate_num, noise_retriever_type, knn_num, knn_q,ranking_score,ranking):
        rtr_idx_list = []

        for entry in tqdm(self.text_forward):
            embed = np.expand_dims(entry['embed'], axis=0)
            idx_list = np.random.choice(len(self.index_ds['label']), ice_num, replace=False).tolist()
            rtr_idx_list.append(self.noise_retrieve(noise_retriever_type, embed, idx_list, ice_num, knn_num, knn_q,ranking_score,ranking))
        return rtr_idx_list
    
    
