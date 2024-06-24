

import torch
import argparse
import evaluate


from dataset.utils import get_dataloader
from model import get_model, get_tokenizer
from algorithms import get_retriever
from inference import get_inferencer
from common import get_prompt_label, extract_data, setup_seed, get_input


def main(args):

    setup_seed(100)
    device = torch.device("cuda:0")
    model, tokenizer = get_model(args.pretrained_model_name).to(device), get_tokenizer(args.pretrained_model_name)
    model.eval()
    
    acc = []
    for seed in [100]:
        setup_seed(seed)

        #####get data#####
        raw_ice_dataset = get_dataloader(args.task, 'train', args.ice_imbalance_type, args.imbalance_ratio, seed)
        raw_test_dataset = get_dataloader(args.task, 'test', 'real', args.imbalance_ratio, seed)
        raw_ice_dataloader = torch.utils.data.DataLoader(raw_ice_dataset, batch_size=1, shuffle=False)
        raw_test_dataloader = torch.utils.data.DataLoader(raw_test_dataset, batch_size=1, shuffle=False)

        ice_dataset = extract_data(raw_ice_dataloader, args.task)
        test_dataset = extract_data(raw_test_dataloader, args.task)


        retriever = get_retriever(args.test_retrieving, args.task, raw_ice_dataloader, raw_test_dataloader, model, tokenizer, device)
        ice_idx_list = retriever.retrieve(args.ice_num, args.dpp_candidate_num, args.noise_retrieving, args.knn_num, args.tau, args.similarity_score, args.ranking)
        template, template_dict, label = get_prompt_label(args.task)
        ice = get_input(args.task, ice_idx_list, template, template_dict, ice_dataset)

        #####Inference#####
        # inferencer = get_inferencer('ppl', model=model, tokenizer = tokenizer, device=device, batch_size=args.batch_size)
        # test_predictions = inferencer.inference(task=args.task, ice=ice,  testdata=test_dataset['text'])
        
        inferencer = get_inferencer('ppl', model_name=model, tokenizer_name = tokenizer, batch_size=args.batch_size)
        test_predictions = inferencer.inference(task=args.task, ice=ice,  candidate=test_dataset['text'], labels=list(range(len(label))), ice_template=template_dict)


        #####Evaluate#####
        acc_evaluate = evaluate.load('/data/home/hongfugao/evaluate/metrics/accuracy')
        acc.append(acc_evaluate.compute(predictions=test_predictions, references=test_dataset['label']))
    print(acc)

    # dat = pd.DataFrame({"acc":acc})
    # dat.to_csv("/data/home/hongfugao/result/{task}_{icl}_{model}_{imbalance_ratio}.csv".format(task=args.task, icl=args.test_retrieving, model=args.pretrained_model_name, imbalance_ratio=args.imbalance_ratio))
        




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    #task and prompt
    parser.add_argument('--task', type=str, choices=['agnews','sst2','dbpedia'], default='sst2', help='task.')
    
    #retriever
    parser.add_argument('--test_retrieving', type=str, choices=['random', 'topk', 'dpp', 'zero'], default='zero', help='Choose demonstration selection method.')
    parser.add_argument('--noise_retrieving', type=bool, choices=[True, False], default=False, help='Choose noise retriever.')
    parser.add_argument('--dpp_candidate_num',  type=int, default=16, help='see DPP.')
    parser.add_argument('--ice_num',  type=int, default=4)

    parser.add_argument('--tau',  type=int, choices=[25, 50, 75], default=50)
    parser.add_argument('--knn_num',  type=int, choices=[2, 4, 6, 8], default=4)
    parser.add_argument('--similarity_score', type=str, choices=['cos', "bm25"], default='cos')
    parser.add_argument('--ranking', type=str, choices=['relevent', "no"], default='relevent')

    #noise label
    parser.add_argument('--imbalance_ratio', type=int, default=0.1, help='noisy ratio.')
    parser.add_argument('--ice_imbalance_type', type=str, choices=['exp', "real"], default="exp", help='noisy type.')

    #model
    parser.add_argument('--pretrained_model_name', '-m', choices=['llama','gptj','llama13','opt2b','opt6b','opt13b'], type=str, default='llama', help='Choose pretrained model.')
    
    #others
    parser.add_argument('--batch_size', type=int, default=8, help='Test batch size.')
    args = parser.parse_args()
    main(args)

