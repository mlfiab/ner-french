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

import torch
import sys
from model import FLAUNER
import numpy as np
from sklearn.model_selection import train_test_split
from adatasets import get_conll_data
import warnings
import pandas as pd
import argparse

torch.cuda.empty_cache()



def nbOfSentencesGreaterThan(length, sentences):
    i = 0
    for sentence in sentences:
        if len(sentence) > length:
            i = i + 1

    return i

def parse_args():
    parser = argparse.ArgumentParser(description="Named Entity Recognition System")
    
    parser.add_argument(
        "--train_file", type=str, default=None, help="Text training file in CONLL format"
    )
    
    parser.add_argument(
        "--eval_file", type=str, default=None, help="Text evaluation file in CONLL format"
    )
    
    parser.add_argument(
        "--transformer", type=str, default=None, help="Transformer model"
    )
    arg = parser.parse_args()
    return arg


def main(args):
    arg = parse_args()
    EPOCHS = args['epochs']
    MAX_LENGTH = args['max_lenght']
    BATCH_SIZE = args['train_batch_size']
    DROPOUT = 0.1
    LEARNING_RATE = 0.0001
    WARMUP_STEPS = 500

    print("TRANSFORMER=", arg.transformer)

    # For wikiner corpus
    print("loading data")
    data = get_conll_data(file_path=arg.train_file)
    data_eval = get_conll_data(file_path=arg.eval_file)
    print("data loaded")

    print("Nb of sentences: ", len(data['tags']))
    print("Max length: ", max(len(x) for x in data['sentences']))
    print("Nb of sentences >", MAX_LENGTH, ":", nbOfSentencesGreaterThan(MAX_LENGTH, data['sentences']))

    sentences = np.array(data['sentences'], dtype=object)
    tags = np.array(data['tags'], dtype=object)
    sentences_eval = np.array(data_eval['sentences'], dtype=object)
    tags_eval = np.array(data_eval['tags'], dtype=object)

    sentences_train, sentences_test, tags_train, tags_test = train_test_split(sentences, tags, test_size=0.1)

    print("Length of train set (sentences): ", len(sentences_train))
    print("Length of train set (tags): ", len(tags_train))
    print("Length of test set (sentences): ", len(sentences_test))
    print("Length of test set (tags): ", len(tags_test))

    # TODO: nb of sentences with a least one item
    training = {'sentences': sentences_train, 'tags': tags_train}
    validation = {'sentences': sentences_test, 'tags': tags_test}
    evaluation = {'sentences': sentences_eval, 'tags': tags_eval}
    
    print(f"Length of train set (sentences): {len(training['sentences'])}")
    print(f"Length of test set (sentences): {len(validation['sentences'])}")
    print(f"Length of Validation set (sentences): {len(evaluation['sentences'])}")
    
    # ['AGE' 'DATE' 'Email' 'I-REF' 'LOC' 'O' 'ORG' 'PER' 'QID' 'TEL' 'URL']
    # tag_scheme = ['I-PER','I-ORG','I-LOC','I-MISC', 'I-DAT', 'I-ID', 'Email'] # for the small dataset
    tag_scheme = ['PER', 'ORG', 'LOC', 'AGE', 'DATE', 'TEL', 'URL', 'Email', 'REF', 'QID']  # for personnal dataset
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
        transformer=arg.transformer,
        dropout=dropout,
        hyperparameters=training_hyperparameters
    )

    warnings.filterwarnings('ignore')

    model.train()

    evaluation['sentences'] = evaluation['sentences'].tolist()
    df_result = model.evaluate_performance(evaluation)
    file = "output_"+str(EPOCHS)+"_"+str(MAX_LENGTH)+"_"+str(BATCH_SIZE)+".csv"
    #with open(file, 'a') as f:
    print("=====================================")
    print("EPOCHS=", EPOCHS)
    print("MAX_LENGTH=", MAX_LENGTH)
    print("BATCH_SIZE=", BATCH_SIZE)
    print("DROP_OUT=", DROPOUT)
    print("=====================================")
    print(df_result)
    
    df_result.to_csv(file, sep=";", index=False)
    

    # Specify a path
    PATH = "./trained_flaubert_uncased_" + "e_" +str(EPOCHS) + "max_" + str(MAX_LENGTH) + "b_" + str(BATCH_SIZE) +  ".pt"

    # Save
    torch.save(model, PATH)


if __name__ == "__main__":
    #params = nni.get_next_parameter()
    params = {'epochs': 1, 'train_batch_size': 32, 'max_lenght': 128}
    main(params)
