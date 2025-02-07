from dataset import get_examples_for_dataset, get_examples_with_prompt, GSMDataset, extract_answer, is_correct
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import GPT2Config, AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from calculator import sample
from dataset import json_jsonl_read, json_jsonl_write, set_seed
import re, os, torch, argparse
from multiprocessing import Process
import multiprocessing
import numpy as np


def exact_match_score(prediction, ground_truth):
    if prediction and ground_truth and prediction.strip() == ground_truth.strip():
        return 1.0
    else:
        return 0.0


def extract_answer_for_TimeQA(text):
    parts = text.split("Answer: ")
    if len(parts) > 1:
        answer = parts[1].split("<|endoftext|>")[0]
        return answer.strip()
    else:
        # return "Answer not found"
        return text.split("<|endoftext|>")[0].strip()

def generate_evaluate_result(task, dataset, augmentation_num, experiment_id, augmenter, seednum, model_name, device, sample_len=100):
    # # evaluate base_model directly on test.jsonl
    if task == "evaluate" and augmenter == "none" and int(seednum) == 0:
        if model_name == "gpt2":
            base_model_path = "gpt2"
            tokenizer = GPT2Tokenizer.from_pretrained(base_model_path)
            model = GPT2LMHeadModel.from_pretrained(base_model_path, torch_dtype=torch.float32)
            results_save_path = './save_models_gpt2/'
            
        elif model_name == "Qwen2.5-0.5B":
            base_model_path = './save_models_qwen2.5_0.5B/Qwen2.5-0.5B'
            tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float32)
            results_save_path = './save_models_qwen2.5_0.5B/'

        elif model_name == "Qwen2.5-1.5B":
            base_model_path = './save_models_qwen2.5_1.5B/Qwen2.5-1.5B'
            tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float32)
            results_save_path = './save_models_qwen2.5_1.5B/'

        elif model_name == "Qwen1.5-0.5B":
            base_model_path = './save_models_qwen1.5_0.5B/Qwen1.5-0.5B'
            tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float32)
            results_save_path = './save_models_qwen1.5_0.5B/'

        elif model_name == "Phi1.5":
            base_model_path = './save_models_phi1.5/Phi1.5'
            tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16)
            results_save_path = './save_models_phi1.5/'
        
        results_save_path += f'{dataset}/augmentation_num_{augmentation_num:02}/exp_{experiment_id:02}/none/train_subsample_000_none/evaluate_results_sample_len_{sample_len}/'

        evaluate_data_file_name = os.path.join(results_save_path, 'test.jsonl')
        acc_file_name = os.path.join(results_save_path, 'acc.json')
        evaluate_data_path = f'../datasets/{dataset}/data_jsonl/test.jsonl'

    else:
        seednum = int(seednum)
        subsample_name_ = '/train_subsample_{:03}'.format(seednum)

        if model_name == "gpt2":
            model_save_path = './save_models_gpt2/'
        elif model_name == "Qwen2.5-0.5B":
            model_save_path = './save_models_qwen2.5_0.5B/'
        elif model_name == "Qwen2.5-1.5B":
            model_save_path = './save_models_qwen2.5_1.5B/'
        elif model_name == "Qwen1.5-0.5B":
            model_save_path = './save_models_qwen1.5_0.5B/'
        elif model_name == "Phi1.5":
            model_save_path = './save_models_phi1.5/'

        model_save_path += f'{dataset}/augmentation_num_{augmentation_num:02}/exp_{experiment_id:02}/' + augmenter + subsample_name_ + f'_{augmenter}/'

        best_model_path = os.path.join(model_save_path, 'best_model')
        if model_name == "gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained(best_model_path)
            model = GPT2LMHeadModel.from_pretrained(best_model_path, torch_dtype=torch.float32)
        elif model_name in ["Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen1.5-0.5B"]:
            tokenizer = AutoTokenizer.from_pretrained(best_model_path)
            model = AutoModelForCausalLM.from_pretrained(best_model_path, torch_dtype=torch.float32)
        elif model_name == "Phi1.5":
            tokenizer = AutoTokenizer.from_pretrained(best_model_path)
            model = AutoModelForCausalLM.from_pretrained(best_model_path, torch_dtype=torch.bfloat16)

        if task == "train":
            evaluate_data_path = '../datasets/' + dataset + '/data_augmentation/' + augmenter + subsample_name_ + f'_{augmenter}.jsonl'
            results_save_path = os.path.join(model_save_path, f'train_results_sample_len_{sample_len}/')

            evaluate_data_file_name = os.path.join(results_save_path, 'train_subsample_{:03}'.format(seednum) + f'_{augmenter}.jsonl')
            acc_file_name = os.path.join(results_save_path, 'acc.json')
            
        elif task == "evaluate":
            evaluate_data_path = f'../datasets/{dataset}/data_jsonl/test.jsonl'
            results_save_path = os.path.join(model_save_path, f'evaluate_results_sample_len_{sample_len}/')

            evaluate_data_file_name = os.path.join(results_save_path, 'test.jsonl')
            acc_file_name = os.path.join(results_save_path, 'acc.json')

    evaluate_data = get_examples_for_dataset(dataset, evaluate_data_path)

    os.makedirs(results_save_path, exist_ok=True)

    model.to(device)
    print("Model Loaded")

    em_ = 0
    right_count_1 = 0
    right_count_2 = 0
    right_count_3 = 0

    len_ = len(evaluate_data)

    pbar = tqdm(range(len_))

    if dataset == "MLQA":
        for item in evaluate_data:
            question = item["question"]
            answer = item["answer"]

            response = sample(model, question, tokenizer, device, sample_len)
            item["ai_answer"] = response

            ai_answer = response[len(question):]

            '''get EM score'''
            print(ai_answer.split("<|endoftext|>")[0].strip(), "######", answer.split("<|endoftext|>")[0].strip())
            em_ += exact_match_score(ai_answer.split("<|endoftext|>")[0].strip(), answer.split("<|endoftext|>")[0].strip())
            em_score = em_/len_

            """ """
            if answer.split("<|endoftext|>")[0].strip() in ai_answer.split("<|endoftext|>")[0].strip():
                right_count_1 += 1
            acc = right_count_1/len_

            pbar.update(1)
            pbar.set_description(f"EM_score: {em_score:.4f}, acc: {acc}")

            results = {
                "EM_score": em_score,
                "acc": acc
            }

            json_jsonl_write(evaluate_data_file_name, evaluate_data)
            json_jsonl_write(acc_file_name, results)
        return em_/len_, right_count_1/len_


def generate_evaluate_result_by_prompt(task, dataset, augmenter, seednum, model_name="Qwen2.5-0.5B", device=None, sample_len=100):
    seednum = int(seednum)
    subsample_name_ = '/train_subsample_{:03}'.format(seednum)

    model_save_path = './save_models_qwen2.5/' + dataset + '/' + augmenter + subsample_name_ + f'_{ augmenter}/'

    if task == "train":
        evaluate_data_path = '../datasets/' + dataset + '/data_augmentation/' + augmenter + subsample_name_ + f'_{augmenter}.jsonl'
        results_save_path = os.path.join(model_save_path, f'train_results_sample_len_{sample_len}_by_prompt/')

        evaluate_data_file_name = os.path.join(results_save_path, 'train_subsample_{:03}'.format(seednum) + f'_{augmenter}.jsonl')
        acc_file_name = os.path.join(results_save_path, 'acc.json')
        
    elif task == "evaluate":
        evaluate_data_path = '../datasets/' + dataset + '/data_jsonl/test.jsonl'
        results_save_path = os.path.join(model_save_path, f'evaluate_results_sample_len_{sample_len}_by_prompt/')

        evaluate_data_file_name = os.path.join(results_save_path, 'test.jsonl')
        acc_file_name = os.path.join(results_save_path, 'acc.json')


    os.makedirs(results_save_path, exist_ok=True)

    best_model_path = os.path.join(model_save_path, 'best_model')
    tokenizer = AutoTokenizer.from_pretrained(best_model_path)
    model = AutoModelForCausalLM.from_pretrained(best_model_path)

    model.to(device)
    print("Model Loaded")

    evaluate_data = get_examples(evaluate_data_path)

    right_count_1 = 0
    right_count_2 = 0
    right_count_3 = 0

    len_ = len(evaluate_data)

    pbar = tqdm(range(len_))
    for i, item in enumerate(evaluate_data):
        question = item["question"]
        answer = item["answer"]

        single_turn_prompt = \
        f"Q: {evaluate_data[i-1]['question']}" +\
        f"A: {evaluate_data[i-1]['answer']}\n\n" +\
        f"Q: {question}" +\
        f"A: "
        # single_turn_prompt = f"{question}"

        messages = [{"role": "user", "content": single_turn_prompt}]

        print(single_turn_prompt)

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=sample_len, pad_token_id=model.config.eos_token_id)

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(response)

        item["ai_answer"] = response

        # ai_answer = response[len(question):]
        ai_answer = response

        if is_correct(ai_answer, item):
            right_count_1 += 1
        
        acc_1 = right_count_1/len_

        '''get acc by matching'''
        pattern_ = r'[-+]?\d*\.\d+|\d+,\d+|\d+'

        answer_ = re.findall(pattern_, extract_answer(answer))[0].replace(',', '')  # 845,640
        ai_answer_ = [float(num.replace(',', '')) for num in re.findall(pattern_, ai_answer)]

        if float(answer_) in ai_answer_[-1:]:
            right_count_2 += 1

        acc_2 = right_count_2/len_

        if float(answer_) in ai_answer_:
            right_count_3 += 1
        acc_3 = right_count_3/len_

        pbar.update(1)
        pbar.set_description(f"acc_1: {acc_1:.4f}  acc_2: {acc_2:.4f}")

        acc = {
            "acc_1": right_count_1/len_,
            "acc_2": right_count_2/len_,
            "acc_3": right_count_3/len_
        }

        json_jsonl_write(evaluate_data_file_name, evaluate_data)
        json_jsonl_write(acc_file_name, acc)


def get_acc_from_evaluate_result(dataset, augmenter, seednum, model_name):
    seednum = int(seednum)
    subsample_name_ = '/train_subsample_{:03}'.format(seednum)

    if model_name == "gpt2":
        save_path = './save_models_gpt2/' + dataset + '/' + augmenter + subsample_name_ + f'_{ augmenter}/'
    elif model_name == "Qwen2.5-0.5B":
        save_path = './save_models_qwen2.5/' + dataset + '/' + augmenter + subsample_name_ + f'_{ augmenter}/'
    elif model_name == "Qwen1.5-0.5B":
        save_path = './save_models_qwen1.5/' + dataset + '/' + augmenter + subsample_name_ + f'_{ augmenter}/'

    # results_save_path = os.path.join(save_path, 'evaluate_results/')
    # results_save_path = os.path.join(save_path, 'evaluate_results_sample_len_500/')
    results_save_path = os.path.join(save_path, 'evaluate_results_sample_len_300/')
    evaluate_results_path = os.path.join(results_save_path, 'test.jsonl')

    # if 

    # results_save_path = os.path.join(save_path, 'train_results_sample_len_300/')
    # results_save_path = os.path.join(save_path, 'train_results_sample_len_300_by_prompt/')
    # evaluate_results_path = os.path.join(results_save_path, 'train_subsample_{:03}'.format(seednum) + f'_{ augmenter}.jsonl')

    evaluate_data = json_jsonl_read(evaluate_results_path)

    em_ = 0
    right_count_1 = 0
    right_count_2 = 0
    right_count_3 = 0

    len_ = len(evaluate_data)
    # len_ = 1319


    if dataset == "TimeQA":
        for item in evaluate_data:
            question = item["question"]
            answer = item["answer"]
            ai_answer = item["ai_answer"][len(question):]

            '''get EM score'''
            print("--"*20)
            print(extract_answer_for_TimeQA(ai_answer), ";", extract_answer_for_TimeQA(answer))
            em_ += exact_match_score(extract_answer_for_TimeQA(ai_answer), extract_answer_for_TimeQA(answer))
            em_score = em_/len_

            """ """
            if extract_answer_for_TimeQA(answer) in ai_answer:
                right_count_1 += 1
            acc = right_count_1/len_

            em_score_ = {
                "EM_score": em_/len_,
                "acc": acc
            }
        json_jsonl_write(os.path.join(results_save_path, 'acc.json'), em_score_)
    
    else:
        for i, item in enumerate(evaluate_data):
            if "ai_answer" not in item:
                print(evaluate_results_path)


            question = item["question"]
            answer = item["answer"]
            ai_answer = item["ai_answer"][len(question):]

            '''get acc by GSM8K'''
            # # print(extract_answer(answer), extract_answer(ai_answer))
            if is_correct(ai_answer, item):
                right_count_1 += 1
            
            acc_1 = right_count_1/len_

            '''get acc by matching'''
            pattern_ = r'[-+]?\d*\.\d+|\d+,\d+|\d+'

            answer_ = re.findall(pattern_, extract_answer(answer))[0].replace(',', '')  # 845,640
            ai_answer_ = [float(num.replace(',', '')) for num in re.findall(pattern_, ai_answer)]

            # print(extract_answer(answer), float(answer_), ai_answer_)
            if float(answer_) in ai_answer_[-1:]:
                right_count_2 += 1

            acc_2 = right_count_2/len_

            if float(answer_) in ai_answer_:
                right_count_3 += 1
            acc_3 = right_count_3/len_

            # if i == 1219:
            #     print(f"{i}, acc_1:{acc_1}, acc_2:{acc_2}, acc_3:{acc_3}")


    acc = {
        "acc_1": right_count_1/len_,
        "acc_2": right_count_2/len_,
        "acc_3": right_count_3/len_,
    }

    # if os.path.isfile(os.path.join(results_save_path, 'acc.json')):
    #     return

    # json_jsonl_write(os.path.join(results_save_path, 'acc.json'), acc)


def show_result():
    # AugGPT_path = './save_models_qwen2.5/GSM8K_11/AugGPT/train_subsample_100_AugGPT/evaluate_results_sample_len_300/test.jsonl'
    # llmda_path = './save_models_qwen2.5/GSM8K_11/llmda/train_subsample_100_llmda/evaluate_results_sample_len_300/test.jsonl'
    none_path = './save_models_qwen2.5/GSM8K_11/none/train_subsample_000_none/evaluate_results_sample_len_300/test.jsonl'

    # auggpt = json_jsonl_read(AugGPT_path)
    # llmda = json_jsonl_read(llmda_path)
    none = json_jsonl_read(none_path)

    '''sample len'''
    # ai_answer_len = [[], [], []]
    # auggpt_count, llmda_count, none_count = 0, 0, 0
    # for i in range(len(auggpt)):
    #     question = auggpt[i]["question"]
    #     answer = auggpt[i]["answer"]

    #     auggpt_answer = auggpt[i]["ai_answer"][len(question):]
    #     llmda_answer = llmda[i]["ai_answer"][len(question):]
    #     none_answer = none[i]["ai_answer"][len(question):]
    #     # print(i, extract_answer(none_answer), extract_answer(auggpt_answer), extract_answer(llmda_answer), extract_answer(answer))

    #     auggpt_answer_len = len(auggpt_answer.split(' '))
    #     llmda_answer_len = len(llmda_answer.split(' '))
    #     none_answer_len = len(none_answer.split(' '))

    #     ai_answer_len[0].append(auggpt_answer_len)
    #     ai_answer_len[1].append(llmda_answer_len)
    #     ai_answer_len[2].append(none_answer_len)

    #     if auggpt_answer_len >= 200:
    #         auggpt_count += 1
    #     if llmda_answer_len >= 200:
    #         llmda_count += 1
    #     if none_answer_len >= 200:
    #         none_count += 1

    #     # print(i, auggpt_answer_len, llmda_answer_len, none_answer_len)

    # for i in range(len(ai_answer_len)):
    #     print(min(ai_answer_len[i]), max(ai_answer_len[i]), np.mean(ai_answer_len[i]))
    # print(auggpt_count, llmda_count, none_count)


    '''acc compare'''
    evaluate_data = none[0:10]
    for i, item in enumerate(evaluate_data):
        right_count_1 = 0
        right_count_2 = 0
        right_count_3 = 0

        question = item["question"]
        answer = item["answer"]
        ai_answer = item["ai_answer"][len(question):]

        '''get acc by GSM8K'''
        if is_correct(ai_answer, item):
            right_count_1 = 1
        
        '''get acc by matching'''
        pattern_ = r'[-+]?\d*\.\d+|\d+,\d+|\d+'

        answer_ = re.findall(pattern_, extract_answer(answer))[0].replace(',', '')  # 845,640
        ai_answer_ = [float(num.replace(',', '')) for num in re.findall(pattern_, ai_answer)]

        if float(answer_) in ai_answer_[-1:]:
            right_count_2 = 1

        if float(answer_) in ai_answer_:
            right_count_3 = 1

        # if (right_count_1 != right_count_2) or (right_count_1 != right_count_3) or (right_count_2 != right_count_3):
        if right_count_1 != right_count_2:
            print("-"*50)
            print(f"Index: {i + 1}\n")
            print(f"Question: {question}")
            print(f"Answer: {answer}\n")
            print(f"AI_answer: {ai_answer}\n")
            print("-"*10)
            print(extract_answer(answer), extract_answer(ai_answer))
            print("-"*10)
            print(float(answer_), ai_answer_)



if __name__ == "__main__":
    # sample_len = 300
    sample_len = 50

    """ generate evaluate result"""
    parser = argparse.ArgumentParser(description='QA Task')

    parser.add_argument('--cuda', default=1, type=str)
    parser.add_argument('--task', default="evaluate", type=str, choices=("train", "evaluate"))
    parser.add_argument("--augmenter", default="none", type=str, choices=("none", "MoreData", "BackTranslation", "EDA", "MixText", "GPT3Mix", "AugGPT", "LLM2LLM", "llmda", "llmda_", "Copy"))
    parser.add_argument("--seednum", default=0, type=float)#, choices=(50, 100, 200))
    parser.add_argument("--model_name", default="gpt2", type=str)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    cuda = True if torch.cuda.is_available() else False
    devices = torch.device('cuda' if cuda else 'cpu')

    # # Set seed
    set_seed(42)  # set_seed(args.seed)  # set_seed(1234)
    # generate_evaluate_result(args.task, "GSM8K", args.augmenter, args.seednum, args.model_name, devices, sample_len=sample_len)
    # generate_evaluate_result(args.task, "SVAMP", args.augmenter, args.seednum, args.model_name, devices, sample_len=sample_len)
    # generate_evaluate_result(args.task, "TimeQA", args.augmenter, args.seednum, args.model_name, devices, sample_len=sample_len)
    generate_evaluate_result(args.task, "MLQA", args.augmenter, args.seednum, args.model_name, devices, sample_len=sample_len)



    # # # generate_evaluate_result_by_prompt(args.task, "GSM8K", args.augmenter, args.seednum, args.model_name, devices, sample_len=sample_len)


    """Multi Processing"""
    # print("主进程执行中>>> pid={0}".format(os.getpid()))
    # multiprocessing.set_start_method('spawn')

    # def generate_sub_process(sub_process_id, augmenter, model_name):
    #     print("子进程执行中>>> pid={},ppid={},编号{}".format(os.getpid(), os.getppid(), sub_process_id))
    #     generate_evaluate_result(dataset="GSM8K_1", augmenter=augmenter, seednum=100, 
    #                 model_name=model_name, device=devices, sample_len=sample_len)
    #     print("子进程终止>>> pid={},ppid={},编号{}".format(os.getpid(), os.getppid(), sub_process_id))

    # sub_processes = []
    # index = 0

    # for augmenter in ["none", "AugGPT", "llmda"]:
    #     for model_name in ["gpt2", "Qwen2.5-0.5B"]:
    #         print("generation %02i" % index)
    #         sub_process = Process(target=generate_sub_process, name="worker" + str(index), args=(index, augmenter, model_name))
    #         sub_processes.append(sub_process)
    #         index += 1

    # for i in range(len(sub_processes)):  # 开启进程
    #     sub_processes[i].start()

    # print("主进程终止")


    """ get_acc_from_evaluate_result """
    # for dataset in ["GSM8K_00", "GSM8K_01"]:
    #     for augmenter in ["none", "AugGPT", "llmda"]:
    #         seednum = 100
    #         # for model_name in ["gpt2", "Qwen2.5-0.5B"]:
    #         get_acc_from_evaluate_result(dataset, augmenter, seednum, model_name="Qwen2.5-0.5B")

    # for dataset in ["GSM8K_01"]:
    #     for augmenter in ["none", "AugGPT", "llmda"]:
    #         seednum = 100
    #         get_acc_from_evaluate_result(dataset, augmenter, seednum, model_name="Qwen2.5-0.5B")

    # for dataset in ["GSM8K"]:
    #     for augmenter in ["none", "MoreData", "AugGPT", "llmda"]:  # 
    #         for seednum in [100, 150, 200]:  # 000, 
    #             get_acc_from_evaluate_result(dataset, augmenter, seednum, model_name="Qwen2.5-0.5B")

    # for dataset in ["GSM8K_11"]:
    #     for augmenter in ["llmda"]:  # 
    #         for seednum in [100, 150, 200]:  # 000, 
    #             get_acc_from_evaluate_result(dataset, augmenter, seednum, model_name="Qwen2.5-0.5B")


    # for dataset in ["TimeQA"]:
    #     for augmenter in ["none", "MoreData", "AugGPT", "llmda"]:  # 
    #         for seednum in [50, 100]:  # 000, 
    #             get_acc_from_evaluate_result(dataset, augmenter, seednum, model_name="Qwen1.5-0.5B")

    # show_result()
