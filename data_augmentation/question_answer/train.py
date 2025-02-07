from dataset import get_examples_for_dataset, GSMDataset
from dataset import json_jsonl_read, json_jsonl_write, set_seed
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AdamW, get_scheduler
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm.auto import tqdm
from calculator import sample
from evaluate import generate_evaluate_result
import argparse, os, torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter


def main_train_fixed_epochs(dataset_name, args, device):
    augmentation_num = args.aug_num
    experiment_id = args.exp_id
    augmenter = args.augmenter
    seednum = args.seednum
    model_name = args.model_name

    subsample_name = f'train_subsample_{seednum:03}'

    """Model selection and initialization"""
    if model_name == "gpt2":
        base_model_path = 'gpt2'
        save_path = f'./save_models_gpt2/{dataset_name}/augmentation_num_{augmentation_num:02}/exp_{experiment_id:02}/' + f"{augmenter}/{subsample_name}_{augmenter}/"
        
        tokenizer = GPT2Tokenizer.from_pretrained(base_model_path)
        config = GPT2Config.from_pretrained(base_model_path)
        model = GPT2LMHeadModel.from_pretrained(base_model_path, config=config, torch_dtype=torch.float32)

    elif model_name in ["Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen1.5-0.5B", "Phi1.5"]:
        if model_name == "Qwen2.5-0.5B":
            base_model_path = f'./save_models_qwen2.5_0.5B/Qwen2.5-0.5B'
            save_path = './save_models_qwen2.5_0.5B/'
        if model_name == "Qwen2.5-1.5B":
            base_model_path = f'./save_models_qwen2.5_1.5B/Qwen2.5-1.5B'
            save_path = './save_models_qwen2.5_1.5B/' 
        elif model_name == "Qwen1.5-0.5B":
            base_model_path = f'./save_models_qwen1.5_0.5B/Qwen1.5-0.5B'
            # base_model_path = "./save_models_qwen1.5_0.5B/TimeQA/llmda/train_subsample_050_llmda/best_model"
            # base_model_path = "./save_models_qwen1.5_0.5B/TimeQA/AugGPT/train_subsample_050_AugGPT/best_model"

            save_path = './save_models_qwen1.5_0.5B/'
        else:
            base_model_path = f'./save_models_phi1.5/Phi1.5'
            save_path = './save_models_phi1.5/'

        save_path += f"{dataset_name}/augmentation_num_{augmentation_num:02}/exp_{experiment_id:02}/" + f"{augmenter}/{subsample_name}_{augmenter}/"
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        config = AutoConfig.from_pretrained(base_model_path)
        model = AutoModelForCausalLM.from_pretrained(base_model_path, config=config, torch_dtype=torch.bfloat16)#, device_map="auto")

    model.to(device)
    model.train()
    os.makedirs(save_path, exist_ok=True)

    # if os.path.isfile(os.path.join(save_path, 'best_model')):
    #     return

    """Data splitting"""
    base_path = f'../datasets/{dataset_name}/data_augmentation/augmentation_num_{augmentation_num:02}/exp_{experiment_id:02}/'
    train_data_path = base_path + f"{augmenter}/{subsample_name}_{augmenter}.jsonl"
    train_examples = get_examples_for_dataset(train_data_path)

    # print(train_examples[0])
    # return
    # print(train_data_path)

    train_size = int(0.8 * len(train_examples))
    eval_size = len(train_examples) - train_size
    training_data, eval_data = random_split(train_examples, [train_size, eval_size])

    train_dset = GSMDataset(tokenizer, training_data)
    val_dset = GSMDataset(tokenizer, eval_data)

    """DataLoader and optimizer setup"""
    batch_size = 8  # 16 8 
    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True) # len(train_dset)/batch_size
    val_loader = DataLoader(val_dset, batch_size=batch_size, shuffle=False)
    optim = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)  # 添加权重衰减

    """Training setup"""
    # num_epochs = 20
    # num_epochs = 3
    # num_epochs = 1000  # 1-5
    # num_epochs = 3
    num_epochs = 8
    
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optim,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # print(num_epochs, len(train_loader), num_training_steps)  # 20, 13, 260

    """Early stopping threshold"""
    best_loss = float('inf')
    best_model_path = os.path.join(save_path, 'best_model')

    pbar = tqdm(range(num_training_steps))
    early_stopping_patience = 3
    no_improve_counter = 0

    """Training loop"""
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            optim.zero_grad()
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs.loss
            loss.backward()

            """Gradient Accumulation (if needed)"""
            optim.step()
            lr_scheduler.step()
            pbar.update(1)
            pbar.set_description(f"train_loss: {loss.item():.5f}")
        
        """Evaluation"""
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(model.device) for k, v in batch.items()}
                outputs = model(**batch, labels=batch["input_ids"])
                val_loss += outputs.loss.item()

        val_loss /= len(val_loader)
        pbar.set_description(f"val_loss: {val_loss:.5f}")

        if val_loss < best_loss:
            best_loss = val_loss
            print(f"******* epoch:{epoch} Model Save *******")
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            no_improve_counter = 0
        else:
            no_improve_counter += 1

        if val_loss > best_loss * 1.05:  # 如果验证集损失增加超过5%，则提前停止
            print("Early stopping!")
            break
        
        if no_improve_counter >= early_stopping_patience:
            print("Early stopping due to no improvement in validation loss.")
            break

    # model.save_pretrained(best_model_path)
    # tokenizer.save_pretrained(best_model_path)

    ''' generate_evaluate_result '''
    # generate_evaluate_result("evaluate", dataset, augmenter, seednum, model_name, device, sample_len=50)


def main_train_epochs_test(dataset_name, args, device):
    augmentation_num = args.aug_num
    experiment_id = args.exp_id
    augmenter = args.augmenter
    seednum = args.seednum
    model_name = args.model_name

    subsample_name = f'train_subsample_{seednum:03}'

    """Model selection and initialization"""
    if model_name == "Qwen2.5-0.5B":
        base_model_path = './save_models_qwen2.5_0.5B/Qwen2.5-0.5B'
        save_path = './save_models_qwen2.5_0.5B/'
    if model_name == "Qwen2.5-1.5B":
        base_model_path = './save_models_qwen2.5_1.5B/Qwen2.5-1.5B'
        save_path = './save_models_qwen2.5_1.5B/'
    elif model_name == "Qwen1.5-0.5B":
        base_model_path = './save_models_qwen1.5_0.5B/Qwen1.5-0.5B'
        save_path = './save_models_qwen1.5_0.5B/'
    else:
        base_model_path = './save_models_phi1.5/Phi1.5'
        save_path = './save_models_phi1.5/'

    save_path += f"{dataset_name}/augmentation_num_{augmentation_num:02}/exp_{experiment_id:02}/" + f"{augmenter}/{subsample_name}_{augmenter}/"
    os.makedirs(save_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    config = AutoConfig.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(base_model_path, config=config, torch_dtype=torch.bfloat16)#, device_map="auto")
    model.to(device)
    model.train()

    # if os.path.isfile(os.path.join(save_path, 'best_model')):
    #     return

    """Data splitting"""
    base_path = f'../datasets/{dataset_name}/data_augmentation/augmentation_num_{augmentation_num:02}/exp_{experiment_id:02}/'
    train_data_path = base_path + f"{augmenter}/{subsample_name}_{augmenter}.jsonl"

    train_examples = get_examples_for_dataset(dataset_name, train_data_path)

    train_size = int(0.8 * len(train_examples))
    eval_size = len(train_examples) - train_size
    training_data, eval_data = random_split(train_examples, [train_size, eval_size])

    train_dset = GSMDataset(tokenizer, training_data)
    val_dset = GSMDataset(tokenizer, eval_data)

    """DataLoader and optimizer setup"""
    batch_size = 8  # 16 8 
    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True) # len(train_dset)/batch_size
    val_loader = DataLoader(val_dset, batch_size=batch_size, shuffle=False)
    optim = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)  # 添加权重衰减

    """Training setup"""
    # num_epochs = 1000
    num_epochs = 100

    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optim,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    """Early stopping threshold"""
    best_loss = float('inf')
    best_model_path = os.path.join(save_path, 'best_model')
    log_dir = os.path.join(save_path, 'training_record')

    pbar = tqdm(range(num_training_steps))
    early_stopping_patience = 3
    no_improve_counter = 0

    epochs_performace_record = []
    writer = SummaryWriter(log_dir=log_dir)
    """Training loop"""
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_loader):
            optim.zero_grad()
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs.loss
            loss.backward()

            """Gradient Accumulation (if needed)"""
            optim.step()
            lr_scheduler.step()
            pbar.update(1)
            pbar.set_description(f"train_loss: {loss.item():.5f}")
            
            writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + i + 1)

            total_loss += loss.item()
        
        average_train_loss = total_loss / len(train_loader)
        writer.add_scalar('Average Training Loss', average_train_loss, epoch + 1)

        """Evaluation"""
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for j, batch in enumerate(val_loader):
                batch = {k: v.to(model.device) for k, v in batch.items()}
                outputs = model(**batch, labels=batch["input_ids"])
                loss = outputs.loss.item()
                
                pbar.set_description(f"val_loss: {loss:.5f}")
                writer.add_scalar('Validation Loss', loss, epoch * len(val_loader) + j + 1)

                val_loss += loss

        average_val_loss = val_loss / len(val_loader)
        writer.add_scalar('Average Validation Loss', average_val_loss, epoch + 1)

        # # save and evaluate model every epoch
        if epoch < 50:
            print(f"******* epoch:{epoch} Model Save *******")
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            em_score, acc = generate_evaluate_result("evaluate", dataset_name, augmentation_num, experiment_id, augmenter, seednum, model_name, device, sample_len=50)
            # em_score, acc = 1, 2

            epochs_performace_record.append({
                "epoch": epoch + 1,
                "em_score": em_score,
                "acc": acc
            })

            writer.add_scalar('EM Score', em_score, epoch + 1)
            json_jsonl_write(os.path.join(save_path, 'epochs_performace/record.jsonl'), epochs_performace_record)
        

        """Early stopping"""
        # if average_val_loss < best_loss:
        #     best_loss = average_val_loss
        #     print(f"******* epoch:{epoch} Model Save *******")
        #     model.save_pretrained(best_model_path)
        #     tokenizer.save_pretrained(best_model_path)
        #     no_improve_counter = 0
        # else:
        #     no_improve_counter += 1

        # if average_val_loss > best_loss * 1.05:  # 如果验证集损失增加超过5%，则提前停止
        #     print("Early stopping!")
        #     break
        
        # if no_improve_counter >= early_stopping_patience:
        #     print("Early stopping due to no improvement in validation loss.")
        #     break
    
    writer.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='QA Task')

    parser.add_argument('--cuda', default=0, type=str)
    # parser.add_argument('--task', default="train", type=str, choices=("train", "evaluate"))
    parser.add_argument('--dataset_name', type=str)  # choices=["GSM8K"])
    parser.add_argument("--aug_num", default=0, type=int)
    parser.add_argument("--exp_id", default=0, type=int)
    parser.add_argument("--augmenter", default="none", type=str, choices=("none", "MoreData", "BackTranslation", "EDA", "MixText", "GPT3Mix", "AugGPT", "LLM2LLM", "llmda", "Copy"))
    parser.add_argument("--seednum", default=100, type=int)
    parser.add_argument("--model_name", default="gpt2", type=str)
    # parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    args = parser.parse_args()

    """Devices"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    cuda = True if torch.cuda.is_available() else False
    devices = torch.device('cuda' if cuda else 'cpu')

    """Set seed"""
    set_seed(42)  # set_seed(args.seed)  # set_seed(1234)
    # main_train_fixed_epochs("GSM8K", args, devices)
    # main_train_fixed_epochs("SVAMP", args, devices)
    # main_train_fixed_epochs("TimeQA", args, devices)
    # main_train_fixed_epochs("MLQA", args, devices)

    main_train_epochs_test(args.dataset_name, args, devices)


