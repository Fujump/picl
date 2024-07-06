from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from accelerate.utils import gather_object

accelerator = Accelerator()


pretrained_model_dic = {
        "gpt2":"/mnt/sharedata/ssd/users/hongfugao/model/Llama-2-7b-chat-hf",
        "bert":"/mnt/sharedata/ssd/users/huq/model/bert-base-chinese",
        "baichuan2":"/mnt/sharedata/ssd/users/huq/model/Baichuan2-7B-Chat",
        "qwen1.5":"/mnt/sharedata/ssd/users/huq/model/Qwen1.5-7B-Chat",
        "baichuan2_ft":"/mnt/sharedata/ssd/users/huq/model/Baichuan2-7B-Chat-ft-firefly-merged",
        "qwen1.5_ft":"/mnt/sharedata/ssd/users/huq/model/Qwen1.5-7B-Chat-ft-firefly-merged",
        "baichuan2_warmup":"/mnt/sharedata/ssd/users/huq/model/warmup_baichuan2/checkpoint-final-merged",
        "qwen1.5_warmup":"/mnt/sharedata/ssd/users/huq/model/warmup_qwen1.5/checkpoint-final-merged",
        "qwen1.5-14b":"/mnt/sharedata/ssd/users/huq/model/Qwen1.5-14B-Chat",
        "qwen1.5-14b-ft":"/mnt/sharedata/ssd/users/huq/model/Qwen1.5-14B-Chat-ft",
        "baichuan2_warmup_topk":"/mnt/sharedata/ssd/users/huq/model/baichuan2-warmup-topk/checkpoint-final-merged",
        "qwen1.5_warmup_topk":"/mnt/sharedata/ssd/users/huq/model/qwen1.5-warmup-topk/checkpoint-final-merged",
        "tinyllama":"/mnt/sharedata/ssd/users/huq/model/TinyLlama-1.1B-Chat-v1.0",
        "llama3":"/mnt/sharedata/ssd/users/huq/model/Llama3-ChatQA-1.5-8B",
        "llama3-c":"/mnt/sharedata/ssd/users/huq/model/Llama3-Chinese_v2",
        "chatglm3":"/mnt/sharedata/ssd/users/huq/model/chatglm3-6b",
        "internlm2-chat-7b":"/mnt/sharedata/ssd/users/huq/model/internlm2-chat-7b",
        "Yi-1.5-9B-Chat":"/mnt/sharedata/ssd/users/huq/model/Yi-1.5-9B-Chat",
        "Mistral-7B":"/mnt/sharedata/ssd/common/LLMs/Mistral-7B-Instruct-v0.3",
        "Mistral-22B":"/mnt/sharedata/ssd/common/LLMs/Mixtral-8x22B-Instruct-v0.1",
        "gemma-1.1-7b":"/mnt/sharedata/ssd/common/LLMs/gemma-1.1-7b-it",
        "GPT-J-6B":"/mnt/sharedata/ssd/common/LLMs/GPT-J-6B",
        "opt-6.7b":"/mnt/sharedata/ssd/common/LLMs/opt-6.7b",
        "opt-13b":"/mnt/sharedata/ssd/common/LLMs/opt-13b",
        "opt-30b":"/mnt/sharedata/ssd/common/LLMs/opt-30b",
        "opt-66b":"/mnt/sharedata/ssd/common/LLMs/opt-66b",
        "Qwen2-7B":"/mnt/sharedata/ssd/common/LLMs/Qwen2-7B",
        "Qwen2-72B":"/mnt/sharedata/ssd/common/LLMs/Qwen2-72B",
        "Qwen1.5-32B-Chat":"/mnt/sharedata/ssd/common/LLMs/Qwen1.5-32B-Chat",
        "Qwen1.5-72B-Chat":"/mnt/sharedata/ssd/common/LLMs/Qwen1.5-72B-Chat",
        "glm-4-9b-chat":"/mnt/sharedata/ssd/common/LLMs/glm-4-9b-chat",
        "YI-1.5-9B":"/mnt/sharedata/ssd/common/LLMs/Yi-1.5-9B",
        "YI-1.5-34B":"/mnt/sharedata/ssd/common/LLMs/Yi-1.5-34B",
        "Yi-1.5-34B-Chat":"/mnt/sharedata/ssd/common/LLMs/Yi-1.5-34B-Chat",
        "Ziya2-13B-Chat":"/mnt/sharedata/ssd/common/LLMs/Ziya2-13B-Chat",
        "deepseek-llm-7b-chat":"/mnt/sharedata/ssd/common/LLMs/deepseek-llm-7b-chat",
        "deepseek-llm-67b-chat":"/mnt/sharedata/ssd/common/LLMs/deepseek-llm-67b-chat",
        "internlm2-7b":"/mnt/sharedata/ssd/common/LLMs/internlm2-7b",
        "internlm2-chat-7b":"/mnt/sharedata/ssd/common/LLMs/internlm2-chat-7b",
        "internlm2-20b":"/mnt/sharedata/ssd/common/LLMs/internlm2-20b",
        "internlm2-chat-20b":"/mnt/sharedata/ssd/common/LLMs/internlm2-chat-20b",
        "Baichuan2-13B-Chat":"/mnt/sharedata/ssd/common/LLMs/Baichuan2-13B-Chat",
        "ShieldLM-13B-baichuan2":"/mnt/sharedata/ssd/common/LLMs/ShieldLM-13B-baichuan2",
        "ShieldLM-6B-chatglm3":"/mnt/sharedata/ssd/common/LLMs/ShieldLM-6B-chatglm3",
        "ShieldLM-7B-internlm2":"/mnt/sharedata/ssd/common/LLMs/ShieldLM-7B-internlm2",
        "ShieldLM-14B-qwen":"/mnt/sharedata/ssd/common/LLMs/ShieldLM-14B-qwen",
        "Qwen1.5-14B-Chat-ft-topk":"/mnt/sharedata/ssd/users/huq/model/Qwen1.5-14B-Chat-ft-topk/checkpoint-609-merged",
        "Qwen1.5-14B-Chat-ft-zero":"/mnt/sharedata/ssd/users/huq/model/Qwen1.5-14B-Chat-ft-zero/checkpoint-609-merged",
        "Qwen1.5-14B-Chat-ft-zero-sub":"/mnt/sharedata/ssd/users/huq/model/Qwen1.5-14B-Chat-ft-zero-sub/checkpoint-609-merged",
        }


def get_model(pretrained_model_name):
    if pretrained_model_name in pretrained_model_dic:
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_dic[pretrained_model_name], device_map="auto", trust_remote_code=True)
    else:
        print("Error: Model Type")
        
    return model


def get_tokenizer(pretrained_model_name):

    if pretrained_model_name in pretrained_model_dic:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dic[pretrained_model_name], padding_side='left', padding=True, return_tensors='pt', truncation=True, max_length=2048, trust_remote_code=True)

    else:
        print("Error: Tokenizer Type")
        
    return tokenizer