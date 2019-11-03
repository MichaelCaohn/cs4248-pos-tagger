# python3.5 runtagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import math
import sys
import torch
from buildtagger import POSModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

TAG_TO_IX = {
  '``' : 0,
  '#' : 1, 
  '$' : 2, 
  "''" : 3, 
  ',' : 4, 
  '-LRB-' : 5, 
  '-RRB-' : 6, 
  '.' : 7, 
  ':' : 8,
  'CC' : 9,
  'CD' : 10,
  'DT' : 11,
  'EX' : 12,
  'FW' : 13,
  'IN' : 14,
  'JJ' : 15,
  'JJR' : 16,
  'JJS' : 17,
  'LS' : 18,
  'MD' : 19,
  'NN' : 20,
  'NNP' : 21,
  'NNPS' : 22,
  'NNS' : 23,
  'PDT' : 24,
  'POS' : 25,
  'PRP' : 26,
  'PRP$' : 27,
  'RB' : 28,
  'RBR' : 29,
  'RBS' : 30,
  'RP' : 31,
  'SYM' : 32,
  'TO' : 33,
  'UH' : 34,
  'VB' : 35,
  'VBD' : 36,
  'VBG' : 37,
  'VBN' : 38,
  'VBP' : 39,
  'VBZ' : 40,
  'WDT' : 41,
  'WP' : 42,
  'WP$' : 43,
  'WRB' : 44,
}

IX_TO_TAG = {ix:tag for tag, ix in TAG_TO_IX.items()}

from collections import defaultdict

def tag_sentence(test_file, model_file, out_file):

    word_vocab, char_vocab, model_state_dict = torch.load(model_file)
    word_vocab, char_vocab = defaultdict(lambda:len(word_vocab), word_vocab), defaultdict(lambda:len(char_vocab), char_vocab)
    model = POSModel(word_vocab, char_vocab)
    model.load_state_dict(model_state_dict)
    
    reader = open(test_file)
    test_lines = reader.readlines()
    reader.close()

    file = open(out_file, "w")

    for i in range(0, len(test_lines)):
        cur_out_line = test_lines[i].strip()
        words = cur_out_line.split(' ')

        output = model(words)
        output = torch.argmax(output, dim=1)
        tags = [IX_TO_TAG[idx] for idx in output.tolist()]
       
        string = ""
        for word, tag in zip(words, tags):
            string += word + "/" + tag + " "
        string += "\n"
        file.write(string)

    file.close()


    # write your code here. You can add functions as well.
		# use torch library to load model_file
    print('Finished...')

if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    tag_sentence(test_file, model_file, out_file)
