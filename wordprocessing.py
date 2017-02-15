import gensim.models.word2vec as w2v
import os
import pandas as numpy
import pandas as pd
import sklearn

#quora_word_embeddings = w2v.Word2Vec.load(os.path.join("trained", "questions2vec.w2v"))

question_pairs = pd.read_csv("questions.csv", header=0)
print (question_pairs.head())

def questionVector(row):
	vector_sum = 0
	words = row.lower().split()
	for word in words:
		vector_sum = vector_sum + quora_word_embeddings[word]

	vector_sum = vector_sum.reshape(1,-1)
	normalised_vector_sum = sklearn.preprocessing.normalise(vector_sum)
	return normalised_vector_sum

import time
start_time = time.time()
question_pairs['question1_vector'] = question_pairs['question1'].apply(questionVector)
question_pairs['question2_vector'] = question_pairs[question2] 