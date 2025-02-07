# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou

import random
from random import shuffle
import os
random.seed(1)


# stop words list
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 
            'ours', 'ourselves', 'you', 'your', 'yours', 
            'yourself', 'yourselves', 'he', 'him', 'his', 
            'himself', 'she', 'her', 'hers', 'herself', 
            'it', 'its', 'itself', 'they', 'them', 'their', 
            'theirs', 'themselves', 'what', 'which', 'who', 
            'whom', 'this', 'that', 'these', 'those', 'am', 
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 
            'have', 'has', 'had', 'having', 'do', 'does', 'did',
            'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
            'because', 'as', 'until', 'while', 'of', 'at', 
            'by', 'for', 'with', 'about', 'against', 'between',
            'into', 'through', 'during', 'before', 'after', 
            'above', 'below', 'to', 'from', 'up', 'down', 'in',
            'out', 'on', 'off', 'over', 'under', 'again', 
            'further', 'then', 'once', 'here', 'there', 'when', 
            'where', 'why', 'how', 'all', 'any', 'both', 'each', 
            'few', 'more', 'most', 'other', 'some', 'such', 'no', 
            'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
            'very', 's', 't', 'can', 'will', 'just', 'don', 
            'should', 'now', '']


# cleaning up text
import re
def get_only_chars(line):

    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ") # replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +',' ',clean_line) # delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line

########################################################################
# Synonym replacement
# Replace n words in the sentence with synonyms from wordnet
########################################################################

# # for the first time you use wordnet
# import nltk
# nltk.download('wordnet')
from nltk.corpus import wordnet 

def synonym_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)

    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            # print("replaced", random_word, "with", synonym)
            num_replaced += 1
        if num_replaced >= n: # only replace up to n words
            break

    # this is stupid but we need it, trust me
    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')

    return new_words

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word): 
        for l in syn.lemmas(): 
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym) 
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################

def random_deletion(words, p):

    # obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return words

    # randomly delete words with probability p
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    # if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]

    return new_words

########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################

def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words

def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
    return new_words

########################################################################
# Random insertion
# Randomly insert n words into the sentence
########################################################################

def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words

def add_word(new_words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words)-1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)

########################################################################
# main data augmentation function
########################################################################

def eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):
    
    sentence = get_only_chars(sentence)
    words = sentence.split(' ')
    words = [word for word in words if word != '']
    num_words = len(words)
    
    augmented_sentences = []
    num_new_per_technique = int(num_aug/4)+1

    # sr
    if (alpha_sr > 0):
        n_sr = max(1, int(alpha_sr*num_words))
        for _ in range(num_new_per_technique):
            a_words = synonym_replacement(words, n_sr)
            augmented_sentences.append(' '.join(a_words))

    # ri
    if (alpha_ri > 0):
        n_ri = max(1, int(alpha_ri*num_words))
        for _ in range(num_new_per_technique):
            a_words = random_insertion(words, n_ri)
            augmented_sentences.append(' '.join(a_words))

    # rs
    if (alpha_rs > 0):
        n_rs = max(1, int(alpha_rs*num_words))
        for _ in range(num_new_per_technique):
            a_words = random_swap(words, n_rs)
            augmented_sentences.append(' '.join(a_words))

    # rd
    if (p_rd > 0):
        for _ in range(num_new_per_technique):
            a_words = random_deletion(words, p_rd)
            augmented_sentences.append(' '.join(a_words))

    augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
    shuffle(augmented_sentences)

    # trim so that we have the desired number of augmented sentences
    if num_aug >= 1:
        augmented_sentences = augmented_sentences[:num_aug]
    else:
        keep_prob = num_aug / len(augmented_sentences)
        augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

    # # append the original sentence
    # augmented_sentences.append(sentence)

    return augmented_sentences




########################################################################
# augment.py + eda.py
########################################################################
import os
from tqdm import tqdm
import argparse
from multiprocessing import Process
from utils import json_jsonl_read, json_jsonl_write, ChatGPT, sentences_split_into_parts, DATASET_METATYPES
from utils import construct_messages_for_get_new_label


# generate more data with standard augmentation
def eda_aug(dataset_name, input_path, output_file, alpha_sr, alpha_ri, alpha_rs, alpha_rd, num_aug=9):
    data = json_jsonl_read(input_path)

    data_augmentation = []
    for item in tqdm(data, desc='Processing', unit='item'):
        if dataset_name in ["SNIPS", "SST2", "TREC", "AGNews"]:
            input_text = item['text']
            input_label = item['label']

            data_augmentation.append({
                'text': input_text,
                'label': input_label
            })

            aug_sentences = eda(input_text, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=alpha_rd, num_aug=num_aug)

            for aug_sentence in aug_sentences:
                data_augmentation.append({
                    'text': aug_sentence,
                    'label': input_label,
                    "aug_sample_flag": 1
                })

            json_jsonl_write(output_file, data_augmentation)
        

def main(dataset_name, augmentation_num, exp_id, subsample_num, augmenter, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, alpha_rd=0.1):
    task_name = DATASET_METATYPES[dataset_name]["task_type"]
    random.seed(exp_id)

    # the output file
    base_path = f'./datasets/{task_name}/{dataset_name}/data_augmentation/augmentation_num_{augmentation_num:02}/exp_{exp_id:02}/'
    input_path = base_path + f'none/train_subsample_{int(subsample_num):03}_none.jsonl'
    output_path = base_path + f'{augmenter}/train_subsample_{int(subsample_num):03}_{augmenter}.jsonl'

    # generate augmented sentences and output into a new file
    eda_aug(dataset_name, input_path, output_path, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, alpha_rd=alpha_rd, num_aug=augmentation_num)


if __name__ == "__main__":
    augmentation_num = 3
    augmenter = 'EDA'

    '''Test'''
    # sentence = "Generate augmented sentences and output into a new file"
    # aug_sentences = eda(sentence, num_aug=4)
    # print(aug_sentences)


    '''
    Multi Processing
    python EDA.py
    '''
    print("主进程执行中>>> pid={0}".format(os.getpid()))

    def generate_sub_process(sub_process_id, dataset_name, exp_id, subsample_num):
        print("子进程执行中>>> pid={},ppid={},编号{}".format(os.getpid(), os.getppid(), sub_process_id))
        main(dataset_name, augmentation_num, exp_id, subsample_num, augmenter)
        print("子进程终止>>> pid={},ppid={},编号{}".format(os.getpid(), os.getppid(), sub_process_id))

    all_datasets = ['SST2']

    subsample_nums = {
        'SST2': [20],
    }

    sub_processes = []
    index = 0
    for dataset_name in all_datasets:
        for exp_id in range(10):
            
            for subsample_num in subsample_nums[dataset_name]:
                print("generation %02i" % index)

                sub_process = Process(target=generate_sub_process, name="worker" + str(index), args=(index, dataset_name, exp_id, subsample_num))
                sub_processes.append(sub_process)
                
                index += 1

    for i in range(len(sub_processes)):
        sub_processes[i].start()
    
    for i in range(len(sub_processes)):
        sub_processes[i].join()    

    print("主进程终止")

