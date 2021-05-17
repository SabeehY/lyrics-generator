#import statements
import pandas as pd
from google.colab import files
from textgenrnn import textgenrnn
import os

#model configuration
model_cfg = {
  'rnn_size': 256,
  'rnn_layers': 2,
  'rnn_bidirectional': True,
  'max_length': 10,
  'max_words': 10000,
  'dim_embeddings': 100, #this is default; dimensionality of word embeddings
  'word_level': True, #we may want to change this after running once
}

train_cfg = {
  'line_delimited': True,
  'num_epochs': 5,
  'gen_epochs': 3,
  'batch_size': 256,
  'train_size': 0.8,
  'dropout': 0.2, #mess with this one; should be low though
  'max_gen_length': 300,
  'validation': True,
  'is_csv': True
}

# 50 most popular artists (i.e with most songs)
top_50 = df['artist'].value_counts()[:50].keys()
top_50 = df.loc[df['artist'].isin(top_50)]
top_50['Lyric'].to_csv('top_50.csv', index=False)

#Instantiates model and names it
model_name = 'rocklyrics'
textgen = textgenrnn(name=model_name)

#model configuration
model_cfg = {
    'rnn_size': 256,
    'rnn_layers': 2,
    'rnn_bidirectional': True,
    'max_length': 10,
    'max_words': 10000,
    'dim_embeddings': 100, #this is default; dimensionality of word embeddings
    'word_level': True, #we may want to change this after running once
}

train_cfg = {
    'line_delimited': True,
    'num_epochs': 5,
    'gen_epochs': 3,
    'batch_size': 256,
    'train_size': 0.8,
    'dropout': 0.2, #mess with this one; should be low though
    'max_gen_length': 300,
    'validation': True,
    'is_csv': True
}

##### Train model on top 50 artist songs #####
textgen.train_from_file('top_50.csv', **train_cfg)

# Pink Floyd lyrics only
pf = df[df['artist'] == 'Pink Floyd']
pf['Lyric'].to_csv('pink_floyd.csv', index=False)

## Training on Pink Floyd with pretrained weights
model_name = 'pink_floyd_pre'
textgen_pf = textgenrnn('rocklyrics_weights.hdf5', name=model_name)

pf_cfg = {
  "new_model": False,
  "num_epochs": 30,
  "gen_epochs": 10,
}
final_cfg = {**train_cfg, **pf_cfg}
textgen_pf.train_from_file('pink_floyd.csv', **final_cfg)

## Training on Pink Floyd with pretrained weights and dropout 0.4
model_name = 'pink_floyd_pre_dropout_04'
textgen_pf = textgenrnn('rocklyrics_weights.hdf5', name=model_name)

pf_cfg = {
  "new_model": False,
  "num_epochs": 30,
  "gen_epochs": 10,
  "dropout": 0.4
}
final_cfg = {**train_cfg, **pf_cfg}
textgen_pf.train_from_file('pink_floyd.csv', **final_cfg)

## Training on Pink Floyd with "new_model" = True
model_name = 'pink_floyd_new'
textgen_pf_new = textgenrnn(name=model_name)

pf_cfg = {
  "new_model": True,
  "num_epochs": 30,
  "gen_epochs": 10,
}
final_cfg = {**train_cfg, **pf_cfg}
textgen_pf_new.train_from_file('pink_floyd.csv', **final_cfg)

## Training on Pink Floyd with "new_model" = True and dropout 0.4
model_name = 'pink_floyd_new_dropout_04'
textgen_pf_new = textgenrnn(name=model_name)

pf_cfg = {
  "new_model": True,
  "num_epochs": 30,
  "gen_epochs": 10,
  "dropout": 0.4
}
final_cfg = {**train_cfg, **pf_cfg}
textgen_pf_new.train_from_file('pink_floyd.csv', **final_cfg)

## Removing punctuation and making everything lowercase
top_50_np = top_50['Lyric'].str.replace('[^a-zA-z0-9\s]', '',)
top_50_lower = top_50_np.str.lower()
top_50_lower.to_csv('top_50_lower_nopun.csv', index=False)

## Training on the cleaned lowercased dataset
model_name = 'top_50_lower_no_pun_pre'
textgen_t50 = textgenrnn(name=model_name)
t50_cfg = {
    "new_model": False,
    "num_epochs": 10,
    "gen_epochs": 10,
    "is_csv": True,
    'line_delimited': True,
    'batch_size': 256,
    'train_size': 0.8,
    'dropout': 0.2,
    'max_gen_length': 300,
    'validation': True,
}
texts = pd.read_csv('top_50_lower_nopun.csv')['Lyric'].to_list()
textgen_t50.train_on_texts(texts, **t50_cfg)

## Training on the cleaned lowercased dataset with dropout 0.4
model_name = 'top_50_lower_no_pun_pre_dropout_04'
textgen_t50 = textgenrnn(name=model_name)
t50_cfg = {
    "new_model": False,
    "num_epochs": 10,
    "gen_epochs": 10,
    "is_csv": True,
    'line_delimited': True,
    'batch_size': 256,
    'train_size': 0.8,
    'dropout': 0.4,
    'max_gen_length': 300,
    'validation': True,
}
texts = pd.read_csv('top_50_lower_nopun.csv')['Lyric'].to_list()
textgen_t50.train_on_texts(texts, **t50_cfg)

