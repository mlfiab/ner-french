"""import csv
import os
# import pyconll
from io import BytesIO
from itertools import compress
from pathlib import Path
from typing import Union, List, Dict
from urllib.request import urlopen
from zipfile import ZipFile
import os
import csv
from itertools import compress
# from models import NERDA
import torch.nn as nn
import torch.optim as optim
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
import nltk
import re"""


import os
os.environ['TRANSFORMERS_CACHE'] = 'tmp/cache/'
import torch
import sys
from model import FLAUNER
import numpy as np
from sklearn.model_selection import train_test_split
from adatasets import get_conll_data
import warnings
import nni



def nbOfSentencesGreaterThan(length, sentences):
    i = 0
    for sentence in sentences:
        if len(sentence) > length:
            i = i + 1

    return i


def main(args, transformer):

    EPOCHS = args['epochs']
    MAX_LENGTH = args['max_lenght']
    BATCH_SIZE = args['train_batch_size']
    DROPOUT = 0.1
    LEARNING_RATE = 0.0001
    WARMUP_STEPS = 500

    print("TRANSFORMER=", transformer)

    # For wikiner corpus
    print("loading data")
    data = get_conll_data(file_path="./train.txt")
    data_eval = get_conll_data(file_path="./valid.txt")
    print("data loaded")

    print("Nb of sentences: ", len(data['tags']))
    print("Max length: ", max(len(x) for x in data['sentences']))
    print("Nb of sentences >", MAX_LENGTH, ":", nbOfSentencesGreaterThan(MAX_LENGTH, data['sentences']))

    sentences = np.array(data['sentences'], dtype=object)
    tags = np.array(data['tags'], dtype=object)
    sentences_eval = np.array(data_eval['sentences'], dtype=object)
    tags_eval = np.array(data_eval['tags'], dtype=object)

    # sentences = sentences[:100]
    # tags = tags[:100]
    # sentences_eval = sentences_eval[:10]
    # tags_eval = tags_eval[:10]

    sentences_train, sentences_test, tags_train, tags_test = train_test_split(sentences, tags, test_size=0.1)

    print("Length of train set (sentences): ", len(sentences_train))
    print("Length of train set (tags): ", len(tags_train))
    print("Length of test set (sentences): ", len(sentences_test))
    print("Length of test set (tags): ", len(tags_test))

    # TODO: nb of sentences with a least one item
    training = {'sentences': sentences_train, 'tags': tags_train}
    validation = {'sentences': sentences_test, 'tags': tags_test}
    evaluation = {'sentences': sentences_eval, 'tags': tags_eval}

    # tag_scheme = ['I-PER','I-ORG','I-LOC','I-MISC', 'I-DAT', 'I-ID', 'Email'] # for the small dataset
    tag_scheme = ['I-PER', 'I-ORG', 'I-LOC', 'I-MISC', '']  # for wikiner
    # tag_scheme = ['PROC', 'ANAT', 'DISO','PHYS', 'PHEN', 'CHEM', 'DEVI', 'LIVB', 'OBJC'] # for medline
    # tag_scheme = ['PROC']  # for medline

    # transformer = 'bert-base-multilingual-uncased'
    # transformer = 'flaubert/flaubert_base_uncased'

    dropout = DROPOUT
    training_hyperparameters = {
        'epochs': EPOCHS,
        'warmup_steps': WARMUP_STEPS,
        'train_batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE
    }

    model = FLAUNER(
        dataset_training=training,
        max_len=MAX_LENGTH,
        dataset_validation=validation,
        tag_scheme=tag_scheme,
        tag_outside='O',
        transformer=transformer,
        dropout=dropout,
        hyperparameters=training_hyperparameters
    )

    warnings.filterwarnings('ignore')

    model.train()

    evaluation['sentences'] = evaluation['sentences'].tolist()
    print("=====================================")
    print("EPOCHS=", EPOCHS)
    print("MAX_LENGTH=", MAX_LENGTH)
    print("BATCH_SIZE=", BATCH_SIZE)
    print("DROP_OUT=", DROPOUT)
    print("=====================================")
    print(model.evaluate_performance(evaluation))

    # Specify a path
    PATH = "./test1_flaubert_uncased_256_v4.pt"

    # Save
    torch.save(model, PATH)


if __name__ == "__main__":
    params = nni.get_next_parameter()
    #params = {'epochs': 10, 'train_batch_size': 32, 'max_lenght': 128}
    main(params, sys.argv[1])
