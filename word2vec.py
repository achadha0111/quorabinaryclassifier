import pandas as pd
import numpy as np
import gensim.models.word2vec as w2v
import multiprocessing
import re
import os

quora_dataset = pd.read_csv("questions.csv", header=0)

for col in quora_dataset:
    col, quora_dataset[col].dtypes

text_corpus = []


def create_text_corpus(columname):
	for row in quora_dataset[columname]:
		words = str(row).split()
		text_corpus.append(words)

create_text_corpus('question1')
create_text_corpus('question2')


num_features = 300

min_word_count = 1

num_workers = multiprocessing.cpu_count()

context_size = 7

downsampling = 1e-5

seed = 1

questions2vec = w2v.Word2Vec(
	sg=1,
	seed=seed,
	workers=num_workers,
	size=num_features,
	min_count=min_word_count,
	window=context_size,
	sample=downsampling
)

questions2vec.build_vocab(text_corpus)

import time
start_time = time.time()

questions2vec.train(text_corpus)

if not os.path.exists("trained"):
	os.makedirs("trained")

questions2vec.save(os.path.join("trained", "questions2vec.w2v"))

print ("Time taken: %s seconds" % (time.time() - start_time))

