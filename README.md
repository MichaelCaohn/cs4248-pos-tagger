# CS4248 Natural Language Processing
**Programming Assignment 1: Viterbi Algorithm (Hidden Markov Models)**  
**Programming Assignment 2: RNN with Bi-directional LSTM**  
## Training
```
python3 buildtagger.py ../sents.train model.txt
```
## Executing
```
python3 runtagger.py ../sents.test model.txt sents.out
```
## Scoring
```
python3 ../eval.py sents.out ../sents.answer
```