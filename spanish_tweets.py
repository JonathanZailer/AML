from __future__ import print_function, division, unicode_literals
from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchMoji.torchmoji.model_def import torchmoji_transfer
from torchMoji.torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH, ROOT_PATH
from torchmoji.class_avg_finetuning import class_avg_finetune
from torchMoji.torchmoji.finetuning import (
     load_benchmark,
     finetune)
import emoji
import pandas as pd
from googletrans import Translator
import json
import time
import numpy as np
import torch


train_file = '2018-E-c-Es-train.txt'
test_file = '2018-E-c-Es-test.txt'
dev_file = '2018-E-c-Es-dev.txt'
tables_meaning = ['train', 'dev', 'test']
tables_names = [train_file, dev_file, test_file]
tables = []
for table in tables_names:
    df = pd.read_csv('data/{}'.format(table), sep="\t", header=None)
    new_header = df.iloc[0]  # grab the first row for the header
    df = df[1:]  # take the data less the header row
    df.columns = new_header  # set the header row as the df header
    tables.append(df)
    print(len(df))

# creating translated tables
translator = Translator()


def translate(tweet):
    tweet = ' ' + tweet + ' '
    tweet = tweet.replace('  ', ' ')
    new_tweet = ''
    emojis = []
    for i, c in enumerate(tweet):
        start_emoji = ''
        end_emoji = ''
        if c in emoji.UNICODE_EMOJI:
            start_emoji = ' ' if tweet[i-1] != ' ' else ''
            end_emoji = ' ' if tweet[i+1] != ' ' else ''
            emojis.append((new_tweet[1:].count(' ') + len(start_emoji), c))
        new_tweet += start_emoji + c + end_emoji
    new_tweet = new_tweet[1:-1]
    new_tweet = new_tweet.replace('  ', ' ')
    sp_sentence = new_tweet.encode('ascii', 'ignore').decode('ascii')
    sp_sentence = ' ' + sp_sentence + ' '
    sp_sentence = sp_sentence.replace('  ', ' ')
    sp_sentence = sp_sentence[1:-1]
    en_sentence = translator.translate(sp_sentence)
    text = en_sentence.text
    text = ' ' + text + ' '
    text = text.replace('  ', ' ')
    for (word_num, em) in emojis:
        poss = [pos for pos, char in enumerate(text) if char == ' ']
        if len(poss)-1 < word_num:
            text += em
        else:
            place_to_inject = poss[word_num]
            text = text[:place_to_inject] + ' ' + em + text[place_to_inject:]
    return text

# to create translated files
for i, df in enumerate(tables):
    print(tables_meaning[i])
    try:
        with open("data/{}_eng.json".format(tables_meaning[i]), "r") as f:
            translated = json.load(f)
    except:
        translated = []

    while len(translated) != len(df['Tweet']):
        ix = len(translated)
        if ix == len(df['Tweet']):
            continue
        t = df['Tweet'].iloc[ix]
        # batch = "\n\n\n".join(df['Tweet'].iloc[ix:ix + 10])
        print(len(translated))
        i_t = 0
        while True:
            try:
                translated.append(translate(t))
                # tr_batch = translate(batch)
                # splited = tr_batch.split('\n\n\n')
                # if len(splited) != 10:
                #     while True:
                #         print('error!!!')
                # translated.extend(splited)

            except Exception as e:
                print(e)
                print('not working for'+str(i_t))
                json.dump(translated, open("data/{}_eng.json".format(tables_meaning[i]), "w"))
                time.sleep(2)
                i_t += 1
                continue
            break
        json.dump(translated, open("data/{}_eng.json".format(tables_meaning[i]), "w"))
    json.dump(translated, open("data/{}_eng.json".format(tables_meaning[i]), "w"))

# change to data

maxlen = 30
batch_size = 32
with open(VOCAB_PATH, 'r') as f:
    vocabulary = json.load(f)
st = SentenceTokenizer(vocabulary, maxlen)
# Use torchMoji to encode texts into emotional feature vectors.
data = {'texts': [], 'batch_size': batch_size, 'labels': [], 'maxlen': maxlen, 'added':0}
for i, df in enumerate(tables):
    print(tables_meaning[i])
    with open("data/{}_eng.json".format(tables_meaning[i]), "r") as f:
        translated = json.load(f)
        tokenized, _, _ = st.tokenize_sentences(translated)
        data['texts'].append(tokenized)

# add labels
feelings = ['anger', 'anticipation', 'disgust', 'fear', 'joy',
            'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']
for i, df in enumerate(tables):
    tmp_data = data['texts'][i]
    new_data = []
    labels = []
    for ri in range(len(df)):
        # labels.append([df.iloc[ri][f] == '1' for fi, f in enumerate(feelings)])
        not_find = True
        for fi, f in enumerate(feelings):
            if df.iloc[ri][f] == '1':
                new_data.append(tmp_data[ri])
                labels.append(fi)
                not_find = False
                break
        if not_find:
            new_data.append(tmp_data[ri])
            labels.append(len(feelings))

    if len(new_data) != len(labels):
        raise ValueError
    data['labels'].append(labels)
    data['texts'][i] = new_data

for i in range(3):
    data['labels'][i] = np.array(data['labels'][i])
    data['texts'][i] = np.array(data['texts'][i])

# train and get results
nb_classes = 11+1

# Set up model and finetune
model = torchmoji_transfer(nb_classes, PRETRAINED_PATH)

model, acc = finetune(model, data['texts'], data['labels'], nb_classes, data['batch_size'], method='last')

torch.save(model, 'data/model')
# the_model = torch.load(PATH)
# print(model)
# model, acc = finetune(model, data['texts'], data['labels'], nb_classes, data['batch_size'], method='last')
# print('Acc: {}'.format(acc))



