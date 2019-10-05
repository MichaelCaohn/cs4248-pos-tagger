# python3.5 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import math
import sys
import datetime
import numpy as np
import json
from collections import OrderedDict
from collections import defaultdict

START = '<s>'
END = '<\s>'

# TODO: Make the number of tags constant

def viterbi(words, tag_counts, tag_transitions, start_transition, word_emissions):
    tags = list(tag_transitions.keys())

    matrix = [[None] * len(words) for i in range(len(tags))]
    backpointer = [[None] * len(words) for i in range(len(tags))]
    

    word = words[0]
    for s in range(len(tags)):
        tag = tags[s]
        if tag == START or tag == END: raise Exception
        matrix[s][0] = start_transition[tag]/tag_counts[START] * word_emissions[tag][word]
        backpointer[s][0] = 0

    for t in range(1, len(words)):
        word = words[t]
        for s in range(len(tags)):
            tag = tags[s]
            emission = word_emissions[tag][word]/tag_counts[tag]
            max = -1
            argmax = None
            for s2 in range(len(tags)):
                tag2 = tags[s2]
                p = matrix[s2][t-1] * tag_transitions[tag2][tag]/tag_counts[tag2] * emission
                if p > max: 
                    max = p
                    argmax = s2
            matrix[s][t] = max
            backpointer[s][t] = argmax
    

    t = len(words)-1
    max = -1
    argmax = None
    for s in range(len(tags)):
        p = matrix[s][t]*tag_transitions[tags[s]][END]/tag_counts[tags[s]]
        if p > max:
            max = p
            argmax = s

    s = argmax
    tagged_words = [(word, None) for word in words]
    for t in range(len(words)-1, -1, -1):
        word = tagged_words[t][0]
        tagged_words[t] = (word, tags[s])
        s = backpointer[s][t]
    
    return tagged_words



def tag_sentence(test_file, model_file, out_file):
    reader = open(model_file)  
    json_data = json.load(reader)
    reader.close()

    tag_counts = defaultdict(lambda:1, json_data['tag_counts'])
    tag_transitions = {key:defaultdict(lambda:1,value) for (key,value) in json_data['tag_transitions'].items()}
    start_transition = defaultdict(lambda:1, tag_transitions.pop(START))
    word_emissions = {key:defaultdict(lambda:1,value) for (key,value) in json_data['word_emissions'].items()}

    reader = open(test_file)
    test_lines = reader.readlines()
    reader.close()

    file = open(out_file, "w")

    
    for i in range(0, len(test_lines)):
        cur_out_line = test_lines[i].strip()
        words = cur_out_line.split(' ')

        tagged_words = viterbi(words, tag_counts, tag_transitions, start_transition, word_emissions)
        # print(tagged_words)
        string = ""
        for word, tag in tagged_words:
            string += word + "/" + tag + " "
        string += "\n"
        file.write(string)

    file.close()

    print('Finished...')

if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    start_time = datetime.datetime.now()
    tag_sentence(test_file, model_file, out_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
