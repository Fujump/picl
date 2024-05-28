"""DPP Retriever"""
import tqdm
import numpy as np

from common import get_logger
from .topk import TopkRetriever

logger = get_logger(__name__)


class DPPRetriever(TopkRetriever):
    model = None
    def __init__(self, task,ice_dataloader, candidate_dataloader, noisy_model, noisy_tokenizer, device):
        super().__init__(task, ice_dataloader, candidate_dataloader, noisy_model, noisy_tokenizer, device)
        self.seed = 42
        self.scale_factor = 0.1

    def dpp_search(self, ice_num, candidate_num, noise_retriever_type, knn_num, knn_q,ranking_score,ranking):

        rtr_idx_list = [[] for _ in range(len(self.text_forward))]
        logger.info("Retrieving data for test set...")

        for entry in tqdm.tqdm(self.text_forward):
            idx = entry['metadata']['id']

                # get TopK results
            embed = np.expand_dims(entry['embed'], axis=0)
            near_ids = np.array(self.base_index.search(embed, candidate_num)[1][0].tolist())

            # DPP stage
            near_reps, rel_scores, kernel_matrix = self.get_kernel(embed, near_ids.tolist())

            # MAP inference
            samples_ids = self.fast_map_dpp(kernel_matrix, ice_num)

            # ordered by relevance score
            samples_scores = np.array([rel_scores[i] for i in samples_ids])
            samples_ids = np.array(samples_ids)[(-samples_scores).argsort()].tolist()
            rtr_sub_list = [int(near_ids[i]) for i in samples_ids]

            rtr_idx_list[idx] = self.noise_retrieve(noise_retriever_type, embed, rtr_sub_list, ice_num, knn_num, knn_q,ranking_score,ranking)
        return rtr_idx_list

    def retrieve(self, ice_num, candidate_num, noise_retriever_type, knn_num, knn_q,ranking_score,ranking):
        return self.dpp_search(ice_num, candidate_num, noise_retriever_type, knn_num, knn_q,ranking_score,ranking)
