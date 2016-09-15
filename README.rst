Multi Feed-Fordward Networks for Natural Language Inference
===========================================================

This code is a Tensorflow implementation of the model described in `A Decomposable Attention Model for Natural Language Inference`__

.. __: https://arxiv.org/abs/1606.01933

It is composed of feedforward neural networks that model alignments between the two sentences and combine then. No recurrent architectures such as LSTM or GRU are used, but it achieved 86.8% accuracy on the Stanford SNLI dataset, one of the best values so far reported.


Usage
-----

Run `train.py -h` to see an explanation of its usage:

::

    usage: train.py [-h] [-e NUM_EPOCHS] [-b BATCH_SIZE] [-u NUM_UNITS]
                [-d DROPOUT] [-c CLIP_NORM] [-r RATE] [--use-intra] [--l2 L2]
                [--report REPORT]
                embeddings train validation save logs
    
    positional arguments:
    embeddings       Text file with word embeddings
    train            JSONL file with training corpus
    validation       JSONL file with validation corpus
    save             Directory to save the model files
    logs             Log directory to save summaries
    
    optional arguments:
    -h, --help       show this help message and exit
    -e NUM_EPOCHS    Number of epochs
    -b BATCH_SIZE    Batch size
    -u NUM_UNITS     Number of hidden units
    -d DROPOUT       Dropout probability
    -c CLIP_NORM     Norm to clip training gradients
    -r RATE          Learning rate
    --use-intra      Use intra-sentence attention
    --l2 L2          L2 normalization constant
    --report REPORT  Number of batches between performance reports

The train and validation data should be in the JSONL format used in the SNLI corpus. The embeddings should be given in a file where each line has a word and its vector with values separated by whitespace or tabs.

After training a model, it is possible to analyze its logs using `tensorboard`. In order to run a trained model interactively in the command line, use `interactive-eval.py`.

**Disclaimer:** I have not been able to achieve the exact same results reported by the authors of the original model, probably because of slightly different experimental conditions and hyperparameters. That said, it is possible to achieve over 80% accuracy on the dev set after a few epochs.
