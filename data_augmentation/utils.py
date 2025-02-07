import json
import os

try:
    from openai import OpenAI
    from nltk.tokenize import sent_tokenize
except ModuleNotFoundError:
    print("Module is not installed.")

######### dataset2text_label #########
dataset2text_label = {
    'SNIPS': {
        'text':'text', 
        'label':'label'
        },
    'SST2': {
        'text':'text', 
        'label':'label'
        },
    'TREC': {
        'text':'text', 
        'label':'label'
        },
    "MLQA": {
        'text':'question', # content, question, answer
        'label':'answer'
        }
}


######### DATASET_METATYPES #########
snips_label2id = {'AddToPlaylist': 0, 'BookRestaurant': 1, 'GetWeather': 2, 'PlayMusic': 3, 'RateBook': 4, 'SearchCreativeWork': 5, 'SearchScreeningEvent': 6}
snips_id2label = {v:k for (k,v) in snips_label2id.items()}

terc_coarse_label2id = {'Abbreviation': 0, 'Entity': 1, 'Description': 2, 'Human': 3, 'Location': 4, 'Numeric': 5}
terc_coarse_id2label = {v:k for (k,v) in terc_coarse_label2id.items()}

sst2_label2id = {'Negative': 0, 'Positive': 1}
sst2_id2label = {v:k for (k,v) in sst2_label2id.items()}

agnews_label2id = {'World': 0, 'Sports': 1, 'Business': 2, 'Technology': 3}
agnews_id2label = {v:k for (k,v) in agnews_label2id.items()}

DATASET_METATYPES = {
    "SNIPS": {
        "task_type": "classification",
        "text_type": "instruction",
        "label_type": "intent",
        "label2id": snips_label2id,
        "id2label": snips_id2label,
        "label_set": set(snips_label2id.keys())
    },
    "SST2": {
        "task_type": "classification",
        "text_type": "movie review",
        "label_type": "sentiment",
        "label2id": sst2_label2id,
        "id2label": sst2_id2label,
        "label_set": set(sst2_label2id.keys())
    },
    "TREC": {
        "task_type": "classification",
        "text_type": "question",
        "label_type": "topic",
        "label2id": terc_coarse_label2id,
        "id2label": terc_coarse_id2label,
        "label_set": set(terc_coarse_label2id.keys())
    },
    "MLQA": {
        "task_type": "qa",
        "text_type": "question",
    }
}



def construct_messages_for_get_new_label(input_text, text_type, label_set):
    single_turn_prompt = f"You are a classification expert, and you need to classify the {text_type} you receive into one of the following {len(label_set)} categories:\n"

    for label_ in label_set:
        single_turn_prompt += f"{label_}\n"

    single_turn_prompt += f"\nPlease output the category name directly and only the category name.\n"
    single_turn_prompt += f"{text_type.lower().capitalize()}:{input_text}\n"
    single_turn_prompt += f"Category:"

    single_turn_dialogue = [{'role': 'user', 'content': single_turn_prompt}]

    return single_turn_dialogue


######### sentences_split_into_parts #########
def sentences_split_into_parts(input_text, num_parts):
    sentences = sent_tokenize(input_text)  # only for English

    if len(sentences) < num_parts:
        words = input_text.split()
        part_len = len(words) // num_parts

        # parts = [' '.join(words[i * part_len:(i + 1) * part_len]) for i in range(num_parts)]

        parts = []
        for i in range(num_parts):
            if i != num_parts-1:
                parts.append(' '.join(words[i * part_len:(i + 1) * part_len]))
            else:
                parts.append(' '.join(words[i * part_len:]))

    else:
        part_len = len(sentences) // num_parts
        # parts = [' '.join(sentences[i * part_len:(i + 1) * part_len]) for i in range(num_parts)]

        parts = []
        for i in range(num_parts):
            if i != num_parts-1:
                parts.append(' '.join(sentences[i * part_len:(i + 1) * part_len]))
            else:
                parts.append(' '.join(sentences[i * part_len:]))

    return parts




def json_jsonl_read(input_path):
    if input_path.endswith('.json'):
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]

    return data


def json_jsonl_write(output_path, data):
    content_before_last_slash = output_path.rsplit('/', 1)[0]
    if '.json' not in content_before_last_slash:
        os.makedirs(content_before_last_slash, exist_ok=True)

    if output_path.endswith('.json'):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    else:
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')


######### ChatGPT #########

class ChatGPT():
    # import the OpenAI Python library for calling the OpenAI API
    # https://platform.openai.com/docs/api-reference/chat/create
    def __init__(self, key="Your key", url="API url"):
        self.key = key
        self.url = url

        self.client = OpenAI(api_key=self.key, base_url=self.url)

    def completions_create(self, model="gpt-3.5-turbo", messages=None, max_tokens=512, temperature=1.0):
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return response.choices[0].message.content



if __name__ == "__main__":
    client = ChatGPT()

    messages = [{'role': 'user', 'content': "hello!"},]
    response = client.completions_create(messages=messages)
    print(response)

    # response = client.completions_create(model="gpt-3.5-turbo", messages=messages, max_tokens=64, temperature=0.1)
    # print(response)



