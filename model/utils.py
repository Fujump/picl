from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from accelerate.utils import gather_object

accelerator = Accelerator()


pretrained_model_dic = {
        "llama":"/mnt/sharedata/ssd/common/LLMs/Meta-Llama-3-8B",
        }


def get_model(pretrained_model_name):
    if pretrained_model_name in pretrained_model_dic:
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_dic[pretrained_model_name])
    else:
        print("Error: Model Type")
        
    return model


def get_tokenizer(pretrained_model_name):

    if pretrained_model_name in pretrained_model_dic:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dic[pretrained_model_name], padding_side='left', padding=True, return_tensors='pt', truncation=True, max_length=2048)

    else:
        print("Error: Tokenizer Type")
        
    return tokenizer