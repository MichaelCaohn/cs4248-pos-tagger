# python3.5 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys
import datetime
import numpy as np
from collections import defaultdict
import json

START = '<s>'
END = '<\s>'
UNK = '<UNK>'
ALL_UPPER = 'ALL_UPPER'
UPPER = 'UPPER'
LOWER = 'LOWER'
SYMBOL = 'SYMBOL'

def train_model(train_file, model_file):
    reader = open(train_file)
    out_lines = reader.readlines()
    reader.close()

    tag_counts = defaultdict(lambda:0)
    tag_transitions = defaultdict(lambda:defaultdict(lambda:0))
    word_emissions = defaultdict(lambda:defaultdict(lambda:0))
    vocab = defaultdict(lambda:0)
    tag_caps = defaultdict(lambda:defaultdict(lambda:0))
    tag_suffixes = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:0)))

    for i in range(0, len(out_lines)):
        cur_out_line = out_lines[i].strip()
        cur_out_tokens = cur_out_line.split(' ')
        # total_count += len(cur_out_tokens)

        prev_tag = START
        tag_counts[START] += 1

        for j in range(0, len(cur_out_tokens)):
            word, _,  tag = cur_out_tokens[j].rpartition('/')
            # word = word.lower()
            
            tag_counts[tag] += 1
            tag_transitions[prev_tag][tag] += 1
            word_emissions[tag][word] += 1
            prev_tag = tag
            vocab[word] += 1
            if word.isupper():
                tag_caps[tag][ALL_UPPER] += 1
            elif word[0].isupper():
                tag_caps[tag][UPPER] += 1
            elif word[0].islower():
                tag_caps[tag][LOWER] += 1
            else:
                tag_caps[tag][SYMBOL] += 1
            
            for k in range(1, 5):
                if len(word) < k: break
                if not word[-k:].islower(): break
                tag_suffixes[k][word[-k:]][tag] += 1

        tag = END
        tag_counts[END] += 1
        tag_transitions[prev_tag][tag] += 1

    for tag in tag_caps.values():
        if tag[ALL_UPPER] == 0: tag[ALL_UPPER] = 1
        if tag[LOWER] == 0: tag[LOWER] = 1
        if tag[UPPER] == 0: tag[UPPER] = 1
        if tag[SYMBOL] == 0: tag[SYMBOL] = 1

    for tag in tag_counts.keys():
        unknown_count = len(word_emissions[tag].keys()) # witten bell assumption
        tag_counts[tag] += unknown_count
        word_emissions[tag][UNK] = unknown_count

    # for tag, d in tag_suffixes.items():
    #     for k in range(1, 5):
    #         unknown_count = len(d[k].keys()) # witten bell assumption
    
    
    file = open(model_file, 'w')
    file.write(json.dumps({
        'tag_counts':tag_counts, 
        'tag_transitions':tag_transitions,
        'word_emissions':word_emissions,
        'tag_caps':tag_caps,
        'tag_suffixes':tag_suffixes,
        'vocab':vocab,
    }, indent=2, sort_keys=True))
    file.close()

    print('Finished...')

if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
