from .base_retrieve import BaseRetriever
from .rm import RandomRetriever
from .dpp import DPPRetriever
from .topk import TopkRetriever
from .zero import ZeroRetriever


def get_retriever(retriever_type, task, ice_dataloader, candidate_dataloader, noisy_model=None, noisy_tokenizer=None, device=None):
    if retriever_type == 'topk':
        print("topk")
        retriever = TopkRetriever(task, ice_dataloader, candidate_dataloader, noisy_model=noisy_model, noisy_tokenizer=noisy_tokenizer, device=device)
        
    elif retriever_type == 'random':
        print("random")
        retriever = RandomRetriever(task, ice_dataloader, candidate_dataloader, noisy_model=noisy_model, noisy_tokenizer=noisy_tokenizer, device=device)
    
    elif retriever_type == "dpp":
        print("dpp")
        retriever = DPPRetriever(task, ice_dataloader, candidate_dataloader, noisy_model=noisy_model, noisy_tokenizer=noisy_tokenizer, device=device)
    
    elif retriever_type == "zero":
        print("zero")
        retriever = ZeroRetriever(task, ice_dataloader, candidate_dataloader, noisy_model=noisy_model, noisy_tokenizer=noisy_tokenizer, device=device)
    
    else:
        print("Error Retriever")
    return retriever
    