
import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import pandas as pd

import torch
import argparse
import evaluate
# from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from sklearn.metrics import accuracy_score

from dataset.utils import get_dataloader
from model import get_model, get_tokenizer
from algorithms import get_retriever
from inference import get_inferencer
from common import get_prompt_label, extract_data, setup_seed, get_input, calculate_precision_recall, calculate_precision_recall_neg, delect_unavailable_word


def main(args):
    device = torch.device("cuda:0")
    # tokenizer = get_tokenizer(args.pretrained_model_name)
    # # 使用init_empty_weights初始化模型，防止参数被卸载到CPU
    # with init_empty_weights():
    #     model = get_model(args.pretrained_model_name)

    # # 加载检查点并分派到GPU上，确保模型在加载时使用half precision
    # model = load_checkpoint_and_dispatch(
    #     model, "/mnt/sharedata/ssd/common/LLMs/Mixtral-8x22B-Instruct-v0.1", device_map="auto", dtype=torch.float16
    # )
    model, tokenizer = get_model(args.pretrained_model_name), get_tokenizer(args.pretrained_model_name)
    model.half()

    accuracys=[]
    precisions=[]
    recalls=[]
    precisions_neg=[]
    recalls_neg=[]
    for seed in [100,200,300]:
        setup_seed(seed)

        #####get data#####
        raw_ice_dataset = get_dataloader(args.task, 'train', args.ice_imbalance_type, args.imbalance_ratio, seed)
        raw_test_dataset = get_dataloader(args.task, 'test', 'real', args.imbalance_ratio, seed)
        raw_ice_dataloader = torch.utils.data.DataLoader(raw_ice_dataset, batch_size=1, shuffle=False)
        raw_test_dataloader = torch.utils.data.DataLoader(raw_test_dataset, batch_size=1, shuffle=False)

        ice_dataset = extract_data(raw_ice_dataloader, args.task)
        test_dataset = extract_data(raw_test_dataloader, args.task)


        # retriever = get_retriever(args.test_retrieving, args.task, raw_ice_dataloader, raw_test_dataloader, model, tokenizer, device)
        # ice_idx_list = retriever.retrieve(args.ice_num, args.dpp_candidate_num, args.noise_retrieving, args.knn_num, args.tau, args.similarity_score, args.ranking)
        template, template_dict, label = get_prompt_label(args.task)
        # ice = get_input(args.task, ice_idx_list, template, template_dict, ice_dataset)
        ice=['']*len(test_dataset)


        #####Inference#####
        inferencer = get_inferencer('ppl', model_name=model, tokenizer_name = tokenizer, device = device, batch_size=args.batch_size)
        test_predictions = inferencer.inference(task=args.task, ice=ice,  candidate=test_dataset['text'], labels=list(range(len(label))), ice_template=template_dict)
        
        # print(delect_unavailable_word(test_predictions))

        #####Evaluate#####
        labels=test_dataset['label'].tolist()

        precision, recall = calculate_precision_recall(test_predictions, labels)
        precisions.append(precision)
        recalls.append(recall)

        precision_neg, recall_neg = calculate_precision_recall_neg(test_predictions, labels)
        precisions_neg.append(precision_neg)
        recalls_neg.append(recall_neg)

        acc= accuracy_score(labels,test_predictions)
        accuracys.append(acc)

        dat = pd.DataFrame({"preds":test_predictions, "label":test_dataset['label'], "ice":ice, "text":test_dataset['text']})
        dat.to_csv('/data/home/huq/deepai/output_ppl_total/{model}_{ranking}_{ice_num}_{knn_num}_{task}_{dpp_candidate_num}_{test_retrieving}_{noise_retrieving}_{tau}_{seed}.csv'.format(model=args.pretrained_model_name ,ranking=args.ranking, tau=args.tau, dpp_candidate_num=args.dpp_candidate_num, ice_num=args.ice_num, task=args.task, test_retrieving=args.test_retrieving,  noise_retrieving=args.noise_retrieving, seed=seed,knn_num=args.knn_num))

    # dat = pd.DataFrame({"acc":acc})
    # dat.to_csv("/data/home/hongfugao/result/{task}_{icl}_{model}_{imbalance_ratio}.csv".format(task=args.task, icl=args.test_retrieving, model=args.pretrained_model_name, imbalance_ratio=args.imbalance_ratio))
    print(args.test_retrieving)
    print(args.noise_retrieving)
    print(f"accuracy:{accuracys}, precision:{precisions}, recall:{recalls}, precision_n:{precisions_neg}, recall_n:{recalls_neg}")
    # resul = pd.DataFrame({"em": em, "rouge":rouge})
    resul = pd.DataFrame({"accuracy":accuracys, "precision": precisions, "recall":recalls, "precision_n":precisions_neg, "recall_n":recalls_neg})
    resul.to_csv('/data/home/huq/deepai/output_ppl_total/result_{model}_{ranking}_{ice_num}_{knn_num}_{task}_{dpp_candidate_num}_{test_retrieving}_{noise_retrieving}_{tau}_{seed}.csv'.format(model=args.pretrained_model_name ,ranking=args.ranking, tau=args.tau, dpp_candidate_num=args.dpp_candidate_num, ice_num=args.ice_num, task=args.task, test_retrieving=args.test_retrieving,  noise_retrieving=args.noise_retrieving, seed=seed,knn_num=args.knn_num))
        




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    #task and prompt
    parser.add_argument('--task', type=str, choices=['nq','sciq','illegal','illegal_debug'], default='illegal', help='task.')
    
    #retriever
    parser.add_argument('--test_retrieving', type=str, choices=['random', 'topk', 'dpp', 'zero'], default='zero', help='Choose demonstration selection method.')
    parser.add_argument('--noise_retrieving', type=bool, choices=[True, False], default=False, help='Choose noise retriever.')
    parser.add_argument('--dpp_candidate_num',  type=int, default=16, help='see DPP.')
    parser.add_argument('--ice_num',  type=int, default=8)

    parser.add_argument('--tau',  type=int, choices=[25, 50, 75], default=50)
    parser.add_argument('--knn_num',  type=int, choices=[2, 4, 6, 8], default=4)
    parser.add_argument('--similarity_score', type=str, choices=['cos', "bm25"], default='cos')
    parser.add_argument('--ranking', type=str, choices=['relevent', "no"], default='relevent')

    #noise label
    parser.add_argument('--imbalance_ratio', type=int, default=0.1, help='noisy ratio.')
    parser.add_argument('--ice_imbalance_type', type=str, choices=['exp', "real"], default="exp", help='noisy type.')

    #model
    parser.add_argument('--pretrained_model_name', '-m', choices=["llamaguard",'Qwen1.5-14B-Chat-ft-zero-sub','Qwen1.5-14B-Chat-ft-zero','Qwen1.5-14B-Chat-ft-topk','Qwen1.5-32B-Chat','ShieldLM-14B-qwen','ShieldLM-7B-internlm2','ShieldLM-6B-chatglm3','ShieldLM-13B-baichuan2','internlm2-chat-20b','internlm2-chat-7b','Yi-1.5-34B-Chat',"Qwen1.5-72B-Chat",'Baichuan2-13B-Chat',"glm-4-9b-chat",'internlm2-20b','internlm2-7b','deepseek-llm-67b-chat','deepseek-llm-7b-chat','Ziya2-13B-Chat','Yi-1.5-34B',"YI-1.5-9B","glm-4-9b-chat","Qwen2-72B","Qwen2-7B","opt-66b","opt-30b","opt-13b","opt-6.7b","GPT-J-6B","gemma-1.1-7b",'Mistral-22B','Mistral-7B','llama3','Yi-1.5-9B-Chat','internlm2-chat-7b','chatglm3','llama3-c','llama','baichuan2','qwen1.5','baichuan2_ft','qwen1.5_ft','baichuan2_warmup','qwen1.5_warmup','baichuan2_warmup_topk','qwen1.5_warmup_topk','tinyllama','qwen1.5-14b','qwen1.5-14b-ft'], type=str, default='llama', help='Choose pretrained model.')
    
    #others
    parser.add_argument('--batch_size', type=int, default=2, help='Test batch size.')
    args = parser.parse_args()
    main(args)

