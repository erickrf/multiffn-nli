Multi Feed-Fordward Networks for Natural Language Inference
===========================================================

This code is a Tensorflow implementation of the model described in `A Decomposable Attention Model for Natural Language Inference`__

.. __: https://arxiv.org/abs/1606.01933

It is composed of feedforward neural networks that model alignments between the two sentences and combine then. No recurrent architectures such as LSTM or GRU are used, but it achieved 86.8% accuracy on the Stanford SNLI dataset, one of the best values so far reported.


Usage
-----

Training
^^^^^^^^

Run `train.py -h` to see an explanation of its usage:

::

    usage: train.py [-h] [--vocab VOCAB] [-e NUM_EPOCHS] [-b BATCH_SIZE]
                    [-u NUM_UNITS] [-d DROPOUT] [-c CLIP_NORM] [-r RATE]
                    [--use-intra] [--l2 L2] [--report REPORT] [-v]
                    embeddings train validation save logs

    positional arguments:
      embeddings       Text or numpy file with word embeddings
      train            JSONL or TSV file with training corpus
      validation       JSONL or TSV file with validation corpus
      save             Directory to save the model files
      logs             Log directory to save summaries

    optional arguments:
      -h, --help       show this help message and exit
      --vocab VOCAB    Vocabulary file (only needed if numpy embedding file is
                       given)
      -e NUM_EPOCHS    Number of epochs
      -b BATCH_SIZE    Batch size
      -u NUM_UNITS     Number of hidden units
      -d DROPOUT       Dropout keep probability
      -c CLIP_NORM     Norm to clip training gradients
      -r RATE          Learning rate
      --use-intra      Use intra-sentence attention
      --l2 L2          L2 normalization constant
      --report REPORT  Number of batches between performance reports
      -v               Verbose


The train and validation data should be in the JSONL format used in the SNLI corpus. The embeddings can be given in two different ways:

    1) A text file where each line has a word and its vector with values separated by whitespace or tabs
    
    2) **(faster!)** A numpy file with the saved embedding matrix and an extra text file with the vocabulary, such that its $i$-th line corresponds to the $i$-th row in the matrix.

After training a model, it is possible to analyze its logs using `tensorboard`. 

Running a trained model
^^^^^^^^^^^^^^^^^^^^^^^

In order to run a trained model interactively in the command line, use `interactive-eval.py`:

::

    $ python src/interactive-eval.py saved-model/ glove-42B.npy --vocab glove-42B-vocabulary.txt
    Reading model
    Type sentence 1: The man is eating spaghetti with pasta.
    Type sentence 2: The man is having a meal.
    Model answer: entailment
    
    Type sentence 1: The man is eating spaghetti with pasta.
    Type sentence 2: The man is running in the park.
    Model answer: contradiction
    
    Type sentence 1: The man is eating spaghetti with pasta.
    Type sentence 2: The man is eating in a restaurant.
    Model answer: neutral

So far, I have been able to achieve 85% accuracy on the development set with this code. Higher values are probably possible with more hyperparameter optimization.

If you want to call a trained model inside your code, check `interactive-eval.py`.

