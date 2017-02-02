Multi Feed-Fordward Networks for Natural Language Inference
===========================================================

This code is a Tensorflow implementation of the model described in `A Decomposable Attention Model for Natural Language Inference`__

.. __: https://arxiv.org/abs/1606.01933

It is composed of feedforward neural networks that model alignments between the two sentences and combine then. No recurrent architectures such as LSTM or GRU are used, but it achieved 86.8% accuracy on the Stanford SNLI dataset, one of the best values so far reported.

The repository also features a variant using LSTMs to transform the sentences, inspired by `Enhancing and Combining Sequential and Tree LSTM for Natural Language Inference`__, which has more parameters but achieves slightly superior results.

.. __: https://arxiv.org/abs/1609.06038

Usage
-----

Training
^^^^^^^^

Run `train.py -h` to see an explanation of its usage. A lot of hyperparameter customization is possible; but as a reference, using the MLP model on SNLI, great results can be obtained with 200 units, 0.8 dropout keep probability (i.e., 0.2 dropout), 0 l2 loss, a batch size of 32, an initial learning rate of 0.05 and Adagrad. 

The train and validation data should be in the JSONL format used in the SNLI corpus. The embeddings can be given in two different ways:

    1) A text file where each line has a word and its vector with values separated by whitespace or tabs
    
    2) **(faster!)** A numpy file with the saved embedding matrix and an extra text file with the vocabulary, such that its *i*-th line corresponds to the *i*-th row in the matrix.
    
The code can be run on either GPU or CPU transparently; it only depends on the tensorflow installation.


Running a trained model
^^^^^^^^^^^^^^^^^^^^^^^

In order to run a trained model interactively in the command line, use `interactive-eval.py`:

::

    $ python src/interactive-eval.py saved-model/ glove-42B.npy --vocab glove-42B-vocabulary.txt
    Reading model
    Type sentence 1: The man is eating spaghetti with sauce.
    Type sentence 2: The man is having a meal.
    Model answer: entailment
    
    Type sentence 1: The man is eating spaghetti with sauce.
    Type sentence 2: The man is running in the park.
    Model answer: contradiction
    
    Type sentence 1: The man is eating spaghetti with sauce.
    Type sentence 2: The man is eating in a restaurant.
    Model answer: neutral



