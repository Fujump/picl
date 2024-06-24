"""Zeroshot Retriever"""

from typing import List, Union, Optional, Tuple, Dict

from .base_retrieve import BaseRetriever


class ZeroRetriever(BaseRetriever):

    def __init__(self, task,ice_dataloader, candidate_dataloader, noisy_model, noisy_tokenizer, device):
        super().__init__(task, ice_dataloader, candidate_dataloader, noisy_model, noisy_tokenizer, device)
        
        
    def retrieve(self, ice_num, candidate_num, noise_retriever_type, knn_num, knn_q, ranking_score, ranking):
        rtr_idx_list = [[] for _ in range(len(self.test_ds))]
        return rtr_idx_list
