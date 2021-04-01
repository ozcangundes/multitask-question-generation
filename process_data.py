# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 11:23:04 2021

@author: ozcan
"""

import json
import logging
import os

import nltk
import requests

_URL = "https://raw.githubusercontent.com/okanvk/Turkish-Reading-Comprehension-Question-Answering-Dataset/master/data/2018%20%2B%202020%20veri%20k%C3%BCmesi/"
_DEV_FILE = "final_dev_data_v2.json"
_TRAINING_FILE = "final_train_data_v2.json"



def get_correct_alignment(context, answer):
    gold_text = answer['text']
    start_idx = answer['answer_start']
    end_idx = start_idx + len(gold_text)
    #if context[start_idx:end_idx] == gold_text:
    return start_idx, end_idx       # When the gold label position is good
    #elif context[start_idx-1:end_idx-1] == gold_text:
        #return start_idx-1, end_idx-1   # When the gold label is off by one character
    #elif context[start_idx-2:end_idx-2] == gold_text:
        #return start_idx-2, end_idx-2   # When the gold label is off by two character
    #else:
        #raise ValueError()

def process_ans_ext(paragraph):
       context = paragraph['context'].strip()
   
       # split into sentences
       sents = nltk.sent_tokenize(context)

       # get positions of the sentences
       positions = []
       for i, sent in enumerate(sents):
           if i == 0:
               start, end = 0, len(sent)
           else:
               start, end = (prev_end + 1), (prev_end + len(sent) + 1)
           prev_end = end
           positions.append({'start': start, 'end': end})
       
       # get answers
       answers = [qa['answers'][0] for qa in paragraph['qas']]

       # get list of answers for each sentence
       sent_answers = []
       for pos, sent in zip(positions, sents):
           target_answers = []
           for ans in answers:
               if ans['answer_start'] in range(pos['start'], pos['end']):
                   target_answers.append(ans['text'].strip())
           sent_answers.append(target_answers)

       # build inputs and targets
       examples = []
       for i, ans in enumerate(sent_answers):
           context = "extract answers:"
           if len(ans) == 0: continue
           ans = list(set(ans))
           for j, sent in enumerate(sents):
               if i == j:
                   sent = "{hl_token} %s {hl_token}" % sent
               context = "%s %s" % (context, sent)
               context = context.strip()
           input_text = context
           target_text = " {sep_token} ".join(ans) + " {sep_token}"

           examples.append({'source_text': input_text, "target_text": target_text, "task": "ans_ext"})
       
       return examples
   
def process_qa_text(context, question, answer):
        ans_gen_input = f"question: {question}  context: {context}"
        ans_gen_target = f"{answer}"
        return {"source_text": ans_gen_input, "target_text": ans_gen_target, "task": "qa"}

def process_qg_text(context, question, answer):
    answer_text = answer['text'].strip()

    start_pos, end_pos = get_correct_alignment(context, answer)
    que_gen_input = f"generate question: {context[:start_pos]} {{hl_token}} {answer_text} {{hl_token}} {context[end_pos:]}"

    
    que_gen_target = f"{question}"
    return {"source_text": que_gen_input, "target_text": que_gen_target, "task": "qg"}

def generate_data(mode="train"):
    if mode=="train":
        train=os.path.join(_URL, _TRAINING_FILE)
        resp = requests.get(train)
        files = json.loads(resp.text)
        
    else:
        dev=os.path.join(_URL, _DEV_FILE)
        resp1 = requests.get(dev)
        files = json.loads(resp1.text)
 
    final=[]
    tasks = ['qa', 'qg', 'ans_ext']
    for article in files["data"]:
        title = article.get("title", "").strip()
        for paragraph in article["paragraphs"]:
            context = paragraph["context"].strip()
            
            if 'ans_ext' in tasks:
                ans_ext_examples = process_ans_ext(paragraph)
                for example in ans_ext_examples:
                        final.append(example)
            for qa in paragraph["qas"]:
                question = qa["question"].strip()
                id_ = qa["id"]
    
                answers = [answer["text"].strip() for answer in qa["answers"]]
                for task in tasks:
                    if task == 'qa':
                        final.append(process_qa_text(context, question, answers[0]))
                   
                    if task == 'qg':
                        final.append(process_qg_text(context, question, qa["answers"][0]))
    return final

