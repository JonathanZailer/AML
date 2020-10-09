from __future__ import print_function, division, unicode_literals
from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchMoji.torchmoji.global_variables import VOCAB_PATH
import pandas as pd
import json
import numpy as np
import torch

maxlen = 30
batch_size = 32
with open(VOCAB_PATH, 'r') as f:
    vocabulary = json.load(f)
st = SentenceTokenizer(vocabulary, maxlen)


the_model = torch.load('data/model')
test_file = '2018-E-c-Es-test.txt'
df = pd.read_csv('data/{}'.format(test_file), sep="\t", header=0)
# I need the translation!!!!!!
with open("data/{}_eng.json".format('test'), "r") as f:
    translated = json.load(f)

tokenized, _, _ = st.tokenize_sentences(translated)
results = the_model(tokenized)
myfunc_vec = np.vectorize(lambda x: int(x > 0))
result = myfunc_vec(results)
df.values[:, 2:] = result[:, :-1]
df.to_csv('E-C_es_pred.txt', sep="\t", index=False)