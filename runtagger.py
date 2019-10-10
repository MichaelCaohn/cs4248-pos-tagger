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
UNK = '<UNK>'
ALL_UPPER = 'ALL_UPPER'
UPPER = 'UPPER'
LOWER = 'LOWER'
SYMBOL = 'SYMBOL'
HYPHEN = 'HYPHEN'
NO_HYPHEN = 'NO_HYPHEN'
DIGIT = 'DIGIT'
NO_DIGIT = 'NO_DIGIT'

# TODO: Make the number of tags constant

def viterbi(
        words, tag_counts, tag_transitions, start_transition, word_emissions, 
        tag_caps, tag_hyphen, tag_digit, tag_suffixes, vocab, vocab_suffix_count
    ):
    tags = list(tag_transitions.keys())
    vocab_size = len(vocab.keys())

    matrix = [[None] * len(words) for i in range(len(tags))]
    backpointer = [[None] * len(words) for i in range(len(tags))]

    word = words[0]
    for s in range(len(tags)):
        unseen = False
        tag = tags[s]
        if word not in vocab:
            emission = math.log2(word_emissions[tag][UNK]/(tag_counts[tag] * (vocab_size-len(tag_transitions[tag].keys()))))
            unseen = True
        elif word not in word_emissions[tag]:
            emission = math.log2(1/10000000000)
            unseen = True
        else:
            emission = math.log2(word_emissions[tag][word]/tag_counts[tag])
        
        cap_count = sum(tag_caps[tag].values())
        if word.isupper():
            capitalisation = math.log2(tag_caps[tag][ALL_UPPER]/cap_count)
        elif word[0].isupper():
            capitalisation = math.log2(tag_caps[tag][UPPER]/cap_count)
        elif word[0].islower():
            capitalisation = math.log2(tag_caps[tag][LOWER]/cap_count)
        else:
            capitalisation = math.log2(tag_caps[tag][SYMBOL]/cap_count)

        hyphen_count = sum(tag_hyphen[tag].values())
        if '-' in word:
            hyphen = math.log2(tag_hyphen[tag][HYPHEN]/hyphen_count)
        else:
            hyphen = math.log2(tag_hyphen[tag][NO_HYPHEN]/hyphen_count)

        digit_count = sum(tag_digit[tag].values())
        if any(char.isdigit() for char in word):
            digit = math.log2(tag_digit[tag][DIGIT]/digit_count)
        else:
            digit = math.log2(tag_digit[tag][NO_DIGIT]/digit_count)
        
        suffix = 0
        for k in range(4, 5):
            # TODO: try backoff instead of interpolation
            if len(word) < k: break
            # if not word[-k:].islower(): break
            suffixes = tag_suffixes[k][tag]
            T = len(suffixes.keys())
            C = sum(suffixes.values())
            if T == 0:
                suffix += math.log2(1/10000000000)
            else: 
                suf = word[-k:]
                if suf not in suffixes:
                    pst = math.log2(T/ ((vocab_suffix_count[k] - T) * (C + T)))
                else:
                    pst = math.log2(suffixes[suf]/(sum(suffixes.values()) + len(suffixes.keys())))

                suffix += pst
        
        # suffix = math.tanh(suffix)

        if unseen:
            matrix[s][0] = (
                math.log2(start_transition[tag]/tag_counts[START])
                + emission
                + capitalisation
                + hyphen
                + suffix
                + digit
            )
        else:
            matrix[s][0] = (
                math.log2(start_transition[tag]/tag_counts[START])
                + emission
                # + capitalisation
                # + hyphen
                # + suffix
            )
        backpointer[s][0] = 0

    for t in range(1, len(words)):
        word = words[t]
        for s in range(len(tags)):
            unseen = False
            tag = tags[s]
            # print(word, tag)
            # print(word_emissions[tag][word])
            # print(tag_counts[tag])

            if word not in vocab:
                emission = math.log2(word_emissions[tag][UNK]/(tag_counts[tag] * (vocab_size-len(tag_transitions[tag].keys()))))
                unseen = True
            elif word not in word_emissions[tag]:
                emission = math.log2(1/10000000000)
                unseen = True
            else:
                emission = math.log2(word_emissions[tag][word]/tag_counts[tag])

            cap_count = sum(tag_caps[tag].values())
            if word.isupper():
                capitalisation = math.log2(tag_caps[tag][ALL_UPPER]/cap_count)
            elif word[0].isupper():
                capitalisation = math.log2(tag_caps[tag][UPPER]/cap_count)
            elif word[0].islower():
                capitalisation = math.log2(tag_caps[tag][LOWER]/cap_count)
            else:
                capitalisation = math.log2(tag_caps[tag][SYMBOL]/cap_count)
            
            hyphen_count = sum(tag_hyphen[tag].values())
            if '-' in word:
                hyphen = math.log2(tag_hyphen[tag][HYPHEN]/hyphen_count)
            else:
                hyphen = math.log2(tag_hyphen[tag][NO_HYPHEN]/hyphen_count)
            
            digit_count = sum(tag_digit[tag].values())
            if any(char.isdigit() for char in word):
                digit = math.log2(tag_digit[tag][DIGIT]/digit_count)
            else:
                digit = math.log2(tag_digit[tag][NO_DIGIT]/digit_count)
                
            suffix = 0
            for k in range(4, 5):
                # TODO: try backoff instead of interpolation
                if len(word) < k: break
                # if not word[-k:].islower(): break
                
                suffixes = tag_suffixes[k][tag]
                T = len(suffixes.keys())
                C = sum(suffixes.values())
                if T == 0:
                    suffix += math.log2(1/10000000000)
                else: 
                    suf = word[-k:]
                    if suf not in suffixes:
                        pst = math.log2(T/ ((vocab_suffix_count[k] - T) * (C + T)))
                    else:
                        pst = math.log2(suffixes[suf]/(sum(suffixes.values()) + len(suffixes.keys())))

                    suffix += pst
            
            # suffix = math.tanh(suffix)
            
            max = None
            argmax = None
            for s2 in range(len(tags)):
                tag2 = tags[s2]
                if unseen:    
                    p = (
                        matrix[s2][t-1]
                        + math.log2(tag_transitions[tag2][tag]/tag_counts[tag2])
                        + emission
                        + capitalisation
                        + hyphen
                        + suffix
                        + digit
                    )
                else:
                    p = (
                        matrix[s2][t-1]
                        + math.log2(tag_transitions[tag2][tag]/tag_counts[tag2])
                        + emission
                        # + capitalisation
                        # + hyphen
                        # + suffix
                    )
                if max is None or p > max: 
                    max = p
                    argmax = s2
            matrix[s][t] = max
            backpointer[s][t] = argmax
    

    t = len(words)-1
    max = None
    argmax = None
    for s in range(len(tags)):
        p = (
            matrix[s][t]
            + math.log2(tag_transitions[tags[s]][END]/tag_counts[tags[s]])
        )
        if max is None or p > max:
            max = p
            argmax = s

    s = argmax
    tagged_words = [(word, None) for word in words]
    for t in range(len(words)-1, -1, -1):
        word = tagged_words[t][0]
        tagged_words[t] = (word, tags[s])
        # if tags[s] == 'NN' and word[-1] == 's' and word[:-1] in vocab.keys():
        #     # print(word)
        #     tagged_words[t] = (word, 'NNS')
        s = backpointer[s][t]
    
    return tagged_words



def tag_sentence(test_file, model_file, out_file):
    reader = open(model_file)  
    json_data = json.load(reader)
    reader.close()

    tag_counts = defaultdict(lambda:1, json_data['tag_counts'])
    tag_transitions = {key:defaultdict(lambda:1,value) for (key,value) in json_data['tag_transitions'].items()}
    start_transition = defaultdict(lambda:1, tag_transitions.pop(START))
    word_emissions = json_data['word_emissions']
    tag_caps = json_data['tag_caps']
    tag_hyphen = json_data['tag_hyphen']
    tag_digit = json_data['tag_digit']
    tag_suffixes = {int(key): value for key, value in json_data['tag_suffixes'].items()}
    for k in range(4, 5):
        tag_suffixes[k] = defaultdict(lambda:defaultdict(lambda:1), tag_suffixes[k]) # TODO: account for tags that won't have k-length suffix
    vocab = json_data['vocab']

    vocab_suffix_count = {int(k):len(v.keys()) for k, v in json_data['vocab_suffix'].items()}
    reader = open(test_file)
    test_lines = reader.readlines()
    reader.close()

    file = open(out_file, "w")

    
    for i in range(0, len(test_lines)):
        cur_out_line = test_lines[i].strip()
        words = cur_out_line.split(' ')

        tagged_words = viterbi(words, tag_counts, tag_transitions, start_transition, word_emissions, tag_caps, tag_hyphen, tag_digit, tag_suffixes, vocab, vocab_suffix_count)
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
