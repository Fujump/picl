from .base_inferencer import BaseInferencer
from .ppl import PPLInferencer
from .gen import GenInferencer

def get_inferencer(inferencer_type, model_name, tokenizer_name, batch_size=16):
    if inferencer_type == 'ppl':
        inferencer = PPLInferencer(model_name=model_name, tokenizer_name = tokenizer_name, batch_size=batch_size)

    elif inferencer_type == 'gen':
        inferencer = GenInferencer(model_name=model_name, tokenizer_name=tokenizer_name, batch_size=batch_size)
        
    else:
        print("Error Inferencer Type")

    return inferencer