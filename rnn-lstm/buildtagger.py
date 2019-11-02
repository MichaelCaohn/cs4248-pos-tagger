# python3.5 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
from collections import defaultdict

## HELPER FUNCTIONS ##
def rand_generator(dimensions):
  yield random.uniform(-math.sqrt(6)/math.sqrt(dimensions), math.sqrt(6)/math.sqrt(dimensions))

def bound_generator(dimensions):
  return math.sqrt(6)/math.sqrt(dimensions)

def random_init(generator, dimensions):
  """
  Returns array of dimensions with values initialized between
  [-sqrt(6)/sqrt(d), sqrt(6)/sqrt(d)]
  """
  return [generator() for i in range(dimensions)]

TAG_TO_IX = {
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
  '``' : 45,
}

IX_TO_TAG = {ix:tag for tag, ix in TAG_TO_IX.items()}

## HYPERPARAMETERS ##
RAND_GENERATOR = rand_generator
BOUND_GENERATOR = bound_generator
WORD_VEC_DIM = 128
CHAR_VEC_DIM = 16
CNN_WINDOW_K = 3
CNN_FILTERS_L = 5

LSTM_FEATURES = 10
LSTM_LAYERS = 2

EPOCH = 2
LEARNING_RATE = 0.1

## POTENTIAL STRATS ##
# CNN: 
#   padding,
#   layers, 
#   stride length, 
#   combine instead of max pool, 
#   activation function
# Effiency:
#   parallelise,
# DROPOUT

## MODEL
class POSModel(nn.Module):
    
  def __init__(self, word_vocab, char_vocab):
    super(POSModel, self).__init__()

    self.word_vocab = word_vocab
    self.char_vocab = char_vocab
    self.word_embeddings = nn.Embedding(len(word_vocab), WORD_VEC_DIM)
    self.char_embeddings = nn.Embedding(len(char_vocab), CHAR_VEC_DIM)
    self.word_embeddings.weight.data.uniform_(-BOUND_GENERATOR(WORD_VEC_DIM), BOUND_GENERATOR(WORD_VEC_DIM))
    self.char_embeddings.weight.data.uniform_(-BOUND_GENERATOR(CHAR_VEC_DIM), BOUND_GENERATOR(CHAR_VEC_DIM))

    self.conv1d = nn.Conv1d(in_channels=CHAR_VEC_DIM, out_channels=CNN_FILTERS_L, kernel_size=CNN_WINDOW_K, stride=1, padding=(CNN_WINDOW_K-1)//2, bias=True)
    self.pool = nn.AdaptiveMaxPool1d(1)

    self.lstm = nn.LSTM(
      input_size=WORD_VEC_DIM+CNN_FILTERS_L,
      hidden_size=LSTM_FEATURES,
      num_layers=LSTM_LAYERS,
      dropout=0,
      bidirectional=True
    )
  
  def init_hidden(self, batch_size):
    weight = next(self.parameters()).data
    return (weight.new(LSTM_LAYERS, batch_size, WORD_VEC_DIM+CNN_FILTERS_L).zero_(),
            weight.new(LSTM_LAYERS, batch_size, WORD_VEC_DIM+CNN_FILTERS_L).zero_())

  def forward(self, word_input, char_input, hidden):
    # TODO: Handle unknown characters

    # input: len(sentence) * 1
    # output: len(sentence) * WORD_VEC_DIM
    word_embeds = self.word_embeddings(word_input).transpose(0, 1).unsqueeze(0)
    print(word_embeds.size())

    # input: len(sentence) * len(word)
    # output: len(sentence) * CNN_FILTERS_L
    char_cnn_out = []
    for char_sequence in char_input:
      char_embeds = self.char_embeddings(char_sequence).transpose(0, 1).unsqueeze(0)
      # input: len(word) * CHAR_VEC_DIM
      x = self.conv1d(char_embeds)

      # input: (len(word)) * CNN_FILTERS_L
      # output: CNN_FILTERS_L * 1
      x = self.pool(x)

      char_cnn_out.append(x)
    char_cnn_out = torch.cat(char_cnn_out, dim=2)
    print(char_cnn_out.size())
    
    word_rep = torch.cat((word_embeds, char_cnn_out), dim=1)
    print(word_rep.size())

    output, hidden = self.lstm(word_rep, hidden)
    print(output.size())
    print(hidden.size())
    print("here")

    return(word_rep)


# word_embeddings = defaultdict(lambda: random_init(RAND_GENERATOR, WORD_VEC_DIM))
# char_embeddings = defaultdict(lambda: random_init(RAND_GENERATOR, CHAR_VEC_DIM))

def train_model(train_file, model_file):
  # use torch library to save model parameters, hyperparameters, etc. to model_file
  reader = open(train_file)
  out_lines = reader.readlines()
  pairs_sentence_lines = []
  reader.close()

  word_vocab = {}
  char_vocab = {}

  for i in range(0, len(out_lines)):
    cur_out_line = out_lines[i].strip()
    cur_out_tokens = cur_out_line.split(' ')
    pairs_sentence = []

    for j in range(0, len(cur_out_tokens)):
      word, _, tag = cur_out_tokens[j].rpartition('/')
      if word not in word_vocab:
        word_vocab[word] = len(word_vocab)
      for char in word:
        if char not in char_vocab:
          char_vocab[char] = len(char_vocab)

      # convert tags to index at this point
      pairs_sentence.append((word, TAG_TO_IX[tag]))
    
    pairs_sentence_lines.append(pairs_sentence)
  
  losses = []
  loss_function = nn.CrossEntropyLoss()
  model = POSModel(word_vocab, char_vocab)
  optimizer = optim.SGD(params=model.parameters(), lr=LEARNING_RATE)

  for _ in range(EPOCH):
    total_loss = 0
    for pair_sentence in pairs_sentence_lines:
      
      word_sentence, tag_sentence = zip(*pair_sentence)
      
      # construct list of tensors representing word indices
      word_input = torch.LongTensor([word_vocab[word] for word in word_sentence])

      # construct list of list of tensors representing char indices
      char_input = [torch.LongTensor([char_vocab[char] for char in word]) for word in word_sentence]

      model.zero_grad()
      # optimizer.zero_grad() https://algorithmia.com/blog/convolutional-neural-nets-in-pytorch

      hidden = model.init_hidden(len(word_sentence))

      output = model(word_input, char_input, hidden)

      loss = loss_function(output, torch.LongTensor(tag_sentence))

      loss.backward()
      optimizer.step()

      total_loss += loss.item()

    losses.append(total_loss)

  print(losses)

        

    # 1. Construct separate sentence embedding
    # word_embedding = word_embeddings[sentence]

    # 2. Construct character embeddings via CNN
    # char_embedding = [char_embeddings[char] for char in sentence]

    # 3. Concat to get input representation

    # 4. Construct bi-directional LSTM

    # 5. Linear projection

    # 6. Softmax

  print('Finished...')
		
if __name__ == "__main__":
  # make no changes here
  train_file = sys.argv[1]
  model_file = sys.argv[2]
  train_model(train_file, model_file)
