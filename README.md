# CS4248 Natural Language Processing

---

The code in this repo was written haphazardly as I was trying to optimise for accuracy in a short time period. I might come back to clean this up, I might not.

---

### Programming Assignment 1: Viterbi Algorithm (Hidden Markov Model)
Test Set Accuracy: **96.33%**  
Blind Test Set Accuracy: **95.82%**  
### Programming Assignment 2: CNN with Bi-directional LSTM  
Test Set Accuracy (Codecrunch): **96.51%**  
Blind Test Set Accuracy (Codecrunch): **96.16%**  
  
_Note: The trainer executes for 9 minutes regardless of progress, and as such the amount of training performed (and in turn accuracy) is highly dependent on hardware resources._
## Usage
Enter the directory of the model you would like to execute. Options: `rnn-lstm` or `viterbi`
```
$ cd rnn-lstm
```
### Train
```
$ python3 buildtagger.py ../sents.train model.txt
```
### Execute
```
$ python3 runtagger.py ../sents.test model.txt sents.out
```
### Score
```
$ python3 ../eval.py sents.out ../sents.answer
```