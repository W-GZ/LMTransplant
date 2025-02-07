from contextlib import contextmanager
import signal
import torch


# taken from
# https://stackoverflow.com/questions/492519/timeout-on-a-function-call
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)


def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            return eval(formula)
    except Exception as e:
        signal.alarm(0)
        print(f"Warning: Failed to eval {formula}, exception: {e}")
        return None


def use_calculator(sample):
    if "<<" not in sample:
        return None

    parts = sample.split("<<")
    remaining = parts[-1]

    if ">>" in remaining:
        return None
    if "=" not in remaining:
        return None
    
    lhs = remaining.split("=")[0]
    lhs = lhs.replace(",", "")

    if any([x not in "0123456789*+-/.()" for x in lhs]):
        return None
    
    return eval_with_timeout(lhs)


def check_repetition(string, leng_threshold=5, repeat_num=5):
    seen = {}
    words = string.split()

    for i in range(len(words)):
        current_phrase = ' '.join(words[i:i + leng_threshold])
        if current_phrase in seen:
            seen[current_phrase] += 1
            if seen[current_phrase] >= repeat_num:
                return True
        else:
            seen[current_phrase] = 1

    return False


def sample(model, question, tokenizer, device, sample_len):
    # Inefficient version of calculator sampling -- no batches, doesn't
    # cache activations from previous tokens

    question_length = len(question)

    for _ in range(sample_len):
        with torch.no_grad():
            toks = tokenizer([question], padding=False, return_tensors="pt").to(device)

            out = model.generate(**toks, max_new_tokens=1, pad_token_id=model.config.eos_token_id)  # qwen2.5
            text = tokenizer.batch_decode(out)[0]

            if text.endswith("="):
                answer = use_calculator(text)
                if answer is not None:
                    # print("Triggered calculator, answer", answer)
                    text = text + str(answer) + ">>"

            question = text
            # print(question)

            """early stop"""
            if question.endswith("<|endoftext|>"):
                break

            if check_repetition(question[question_length:]):
                break

            # # Qwen
            if model.config.model_type == "qwen2":
                if question.endswith("!!!!!"):
                    break
                if question.count("#### ") >= 4:
                    break

    return question



def sample_batch(model, questions, tokenizer, device, sample_len):
    inputs = tokenizer(questions, padding=True, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    for _ in range(sample_len):
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                pad_token_id=model.config.eos_token_id
            )

            new_tokens = out[:, -1:]

            input_ids = torch.cat([input_ids, new_tokens], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(new_tokens)], dim=1)

            for i, text in enumerate(tokenizer.batch_decode(input_ids, skip_special_tokens=True)):
                if text.endswith("="):
                    answer = use_calculator(text)
                    if answer is not None:
                        text = text + str(answer) + ">>"
                        # input_ids[i] = tokenizer.encode(text, return_tensors="pt")[0].to(device)

                        encoded_text = tokenizer.encode_plus(
                            text,
                            return_tensors='pt',
                            max_length=input_ids.shape[1],
                            padding='max_length',
                            truncation=True
                        ).to(device)
                        input_ids[i] = encoded_text['input_ids'].flatten()
                        attention_mask[i] = encoded_text['attention_mask'].flatten()

                # if text.endswith("<|endoftext|>"):
                #     input_ids[i] = input_ids[i][:-1]
                #     attention_mask[i] = attention_mask[i][:-1]
                # elif model.config.model_type == "gpt2" and check_repetition(text):
                #     input_ids[i] = input_ids[i][:-1]
                #     attention_mask[i] = attention_mask[i][:-1]
                # elif model.config.model_type == "qwen2" and (text.endswith("!!!!!") or text.count("#### ") >= 4):
                #     input_ids[i] = input_ids[i][:-1]
                #     attention_mask[i] = attention_mask[i][:-1]

    return tokenizer.batch_decode(input_ids, skip_special_tokens=True)

