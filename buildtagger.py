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

def train_model(train_file, model_file):
    reader = open(train_file)
    out_lines = reader.readlines()
    reader.close()

    tag_counts = defaultdict(lambda:0)
    tag_transitions = defaultdict(lambda:defaultdict(lambda:0))
    word_emissions = defaultdict(lambda:defaultdict(lambda:0))

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

        tag = END
        tag_counts[END] += 1
        tag_transitions[prev_tag][tag] += 1
    
    
    file = open(model_file, 'w')
    file.write(json.dumps({
        'tag_counts':tag_counts, 
        'tag_transitions':tag_transitions,
        'word_emissions':word_emissions
    }, indent=4, sort_keys=True))
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