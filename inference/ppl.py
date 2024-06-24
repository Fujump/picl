"""PPL Inferencer"""
import torch
from tqdm import trange
from accelerate import Accelerator
from typing import List, Union, Optional
from transformers import PretrainedConfig

from common.api import *
from common import generate_label_prompt, PromptTemplate, get_logger
from .base_inferencer import BaseInferencer, PPLInferencerOutputHandler

logger = get_logger(__name__)

class PPLInferencer(BaseInferencer):
    def __init__(self,
                 model_name: Optional[str] = 'gpt2-xl',
                 tokenizer_name: Optional[str] = None,
                 device =None,
                 max_model_token_num: Optional[int] = None,
                 model_config: Optional[PretrainedConfig] = None,
                 batch_size: Optional[int] = 1,
                 accelerator: Optional[Accelerator] = None,
                 output_json_filepath: Optional[str] = "./icl_inference_output",
                 output_json_filename: Optional[str] = "predictions",
                 api_name: Optional[str] = None,
                 model_parallel: Optional[bool] = False,
                 **kwargs
                 ) -> None:
        super().__init__(model_name, tokenizer_name,device, max_model_token_num, model_config, batch_size, accelerator,
                         output_json_filepath, output_json_filename, api_name, model_parallel, **kwargs)


    def inference(self,
                  task,
                  ice: Optional[List] = None,
                  candidate: Optional[List] = None,
                  labels: Optional[List] = None,
                  ice_template: Optional[PromptTemplate] = None,
                  prompt_template: Optional[PromptTemplate] = None, output_json_filepath: Optional[str] = None,
                  output_json_filename: Optional[str] = None, normalizing_str: Optional[str] = None) -> List:
        # Preparation for output logs
        output_handler = PPLInferencerOutputHandler(self.accelerator)

        sub_predictions = []
        ppl = []

        # Calculating PPL for prompts in each label's class
        for label in labels:
            index = 0
            prompt_list = []
            sub_ppl_list = []
            
            #  Generate prompts of current label and truncate
            for idx in range(len(ice)):
                prompt = generate_label_prompt(idx, candidate[idx], ice[idx], label, ice_template)
                prompt_list.append(prompt)

            #  Get PPL
            logger.info(f"Calculating PPL for prompts labeled '{label}'")
            for idx in trange(0, len(prompt_list), self.batch_size, disable=not self.is_main_process):
                sub_prompt_list = prompt_list[idx:idx + self.batch_size]
                
                with torch.no_grad():
                    sub_res = self.__get_ppl(sub_prompt_list).tolist()
                for res, prompt in zip(sub_res, sub_prompt_list):
                    sub_ppl_list.append(res)
                    output_handler.save_prompt_and_ppl(label, prompt[len(ice[idx]):], prompt, res, index)
                    index = index + 1
            ppl.append(sub_ppl_list)

        # Get lowest PPL class as predictions
        ppl = list(zip(*ppl))
        for single_ppl in ppl:
            sub_predictions.append(labels[single_ppl.index(min(single_ppl))])
        output_handler.save_predictions(sub_predictions)

        # Output
        output_handler.subprocess_write_to_json(output_json_filepath, output_json_filename)
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        output_handler.merge_to_main_process(output_json_filepath, output_json_filename)
        output_handler.write_to_json(output_json_filepath, output_json_filename)

        return [sample['prediction'] for sample in output_handler.results_dict.values()]

    def __get_ppl(self, input_texts: List[str], mask_length=None):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs = self.tokenizer(input_texts, padding=True, return_tensors='pt', truncation=True, max_length=1024)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)

        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(
            shift_labels.size())

        if mask_length is not None:
            mask = torch.zeros_like(shift_labels)  # [batch,seqlen]
            for i in range(len(mask)):
                for j in range(mask_length[i] - 1, len(mask[i])):
                    mask[i][j] = 1
            loss = loss * mask

        lens = (inputs["input_ids"] != self.tokenizer.pad_token_id).sum(-1).cpu().numpy()
        if mask_length is not None:
            lens -= np.array(mask_length)
        ce_loss = loss.sum(-1).detach().to(torch.float).cpu().numpy() / lens
        return ce_loss
