from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nltk.stem.porter import *
from nltk.corpus import stopwords
from getCandidateKeyValueIndices import getTopKCandidateKeyValueIndices
import os
import csv
import re
import numpy as np

# TODO : See whether you would want to keep the punctuations or not
# Get the arg1, pred, arg2 tuples from the knowledge base
def load_knowledge_base(knowledge_base_file, writeTuplestoFile=False):
	tuples = []
	if writeTuplestoFile == True:
		knowledgeBaseFileForMeta = open('aristo_kb/aristo_kb.dat', 'w')
		kb2 = csv.writer(knowledgeBaseFileForMeta, delimiter=' ')
	with open(knowledge_base_file) as f:
		next(f)
		kb = csv.reader(f, delimiter='\t')
		for row in kb:
			arg1, pred, arg2 = row[2:5]
			if(writeTuplestoFile == True):
				# kb2.writerow([arg1, pred, arg2])
				# Write only keys
				kb2.writerow([arg1, pred])
			arg1 = tokenize(arg1)
			pred = tokenize(pred)
			arg2 = tokenize(arg2)
			tuples.append((arg1, pred, arg2))	
		return tuples

# Get the Training and Test Data for Questions dataset
def load_questions_dataset(data_dir):
	files = os.listdir(data_dir)
	files = [os.path.join(data_dir, f) for f in files]
	train_file = [f for f in files if 'Train' in f][0]
	test_file = [f for f in files if 'Test' in f][0]
	train_data = parse_questions_data(train_file)
	test_data = parse_questions_data(test_file)
	return train_data, test_data

def parse_questions_data(file):
	data = []
	questions = []
	answers = []
	with open(file) as f:
		next(f)
		questions_data = csv.reader(f, delimiter=',')
		for row in questions_data:
			mcq_question = row[9]
			mcq_answer = row[3]
			question, answer_options = parse_question_answer(mcq_question)
			answer = answer_options[ord(mcq_answer)- ord('A')]
			question = tokenize(question)
			answer = tokenize(answer)
			#For now just using the correct answer and ignoring the answer options
			#TODO : Think about how answer options can be used
			data.append((question, answer))
		return data
			
def parse_question_answer(mcq_question):
	extractedTokens = [answer.strip() for answer in re.split('\([A-Z]\)', mcq_question) if answer.strip()]
	question = extractedTokens[0]
	answer_options = extractedTokens[1:]
	return question, answer_options

def tokenize(sent):
	'''Return the stemmed tokens of a sentence or a phrase including punctuation.
	>>> tokenize('Bob dropped the apple. Where is the apple?')
	['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
	In this case it would be only phrases since we are using a knowledge base
	'''
	return [ x.lower().strip() for x in re.split('(\W+)?', sent) if x.strip() and x.isalpha()]


def tokenize_and_stem(sent):
	'''Return the stemmed tokens of a sentence or a phrase including punctuation.
	>>> tokenize('Bob dropped the apple. Where is the apple?')
	['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
	In this case it would be only phrases since we are using a knowledge base
	'''
	stemmer = PorterStemmer()
	return [stemmer.stem(x.lower().strip()) for x in re.split('(\W+)?', sent) if x.strip() and x.isalpha()]

'''
	word_index is a dictionary from each word to corresponding index
	word_inverted_index is a dictionary from each index to its corresponding word
'''
def build_vocab(kb_data, question_data, usePreloaded=True):
	if usePreloaded == True:
		word_index = dict()
		word_index_inverted = dict()
		with open('vocab_dict.csv') as f:
			vocab_dict = csv.reader(f, delimiter=',')
			for row in vocab_dict:
				word_index[row[0]] = int(row[1])
				word_index_inverted[int(row[1])] = row[0]
	else :
		word_index = dict()
		word_index_inverted = dict()
		vocab_questions = reduce(lambda x,y: x | y, (set( q + a ) for q, a in question_data))
		vocab_kb = reduce(lambda x,y: x | y, (set( arg1 + pred + arg2 ) for arg1, pred, arg2 in kb_data))
		vocab = sorted(vocab_questions | vocab_kb)
		with open('vocab_dict.csv', 'w') as f:
			for i, c in enumerate(vocab):
				word_index[c] = i+1
				word_index_inverted[i+1] = c
				f.write('%s,%d\n' %(c, i+1))
	return word_index, word_index_inverted

def vectorize_questions_data(data, word_idx, query_size):
	"""
	Vectorize queries and answers
# 
	If a query length < query_size, the query will be padded with 0's.

	The answer array is returned as a one-hot encoding.
	"""
	Q = []
	A = []
	for query, answer in data:
		lq = max(0, query_size - len(query))
		q = [word_idx[w] for w in query] + [0] * lq

		y = np.zeros(len(word_idx) + 1) # 0 is reserved for nil word
		for a in answer:
			y[int(word_idx[a])] = 1

		Q.append(q)
		A.append(y)
	return np.array(Q), np.array(A)


def vectorize_kb_data(keys, values, word_idx, key_size, value_size):
	"""
	Vectorize the keys and values that will stored as memories
	Key would be the concatenation of arg1 and pred (matched against the question)
	Value would be the arg2 (matched against the answer)

	Padding with 0's if length is less than maximum lenegth for either key or value
	"""

	K = []
	V = []

	stemmer = PorterStemmer()
	stopwordsEnglish = [stemmer.stem(word) for word in stopwords.words('english')]

	# f = open('word_key_index', 'w')
	# TODO: Select candidate keys only based on entity match and not all words match in query and key
	# word_key_index = dict()
	for index, key in enumerate(keys):
		lk = max(0, key_size - len(key))
		k = []
		for word in key:
			k.append(word_idx[word])
			# Key Hashing with Large Knowledge base (Otherwise just crashes the memory of the system)
			# TODO: A better way would be to look at entities data in this dataset and index based on only entities 
			# if word not in stopwordsEnglish:
			# 	if  word_idx[word] not in word_key_index:
			# 		word_key_index[word_idx[word]] = set([index])
			# 	else:
			# 		word_key_index[word_idx[word]] = word_key_index[word_idx[word]] | set([index])
				
		k = k + [0] * lk
		K.append(k)

	for value in values:
		lv = max(0, value_size - len(value))
		v = [word_idx[w] for w in value] + [0] * lv
		V.append(v)

	# for word_index in word_key_index:
	# 	f.write("%s\t" %(word_index))
	# 	f.write(",".join(str(elem) for elem in word_key_index[word_index]) )
	# 	f.write('\n')

	return np.array(K), np.array(V)

def getCandidateKeys(queries, query_size, keys, key_size, values, value_size, word_index_inverted):
	queryKeys = []
	queryValues = []

	queryBatchKeys = []
	queryBatchValues = []

	for i in range(0, queries.shape[0]):
		keyIndices = set()
		query = []
		for j in range(query_size):
			if queries[i,j] == 0:
				break
			query.append(word_index_inverted[queries[i,j]])
		
		query = " ".join(query)
		print(query)
		keyIndices = getTopKCandidateKeyValueIndices(query, 100)
		keyIndices = np.array(list(keyIndices))
		# print(keyIndices.shape)
		# print(type(keyIndices[0]))
		candidateKeys = keys[keyIndices]
		candidateKeyValues = values[keyIndices]

		queryKeys.append(candidateKeys)
		queryValues.append(candidateKeyValues)

		queryBatchKeys.append(queryKeys[i])
		queryBatchValues.append(queryValues[i])

	queryBatchKeys = np.array(queryBatchKeys)
	queryBatchValues = np.array(queryBatchValues)

	print(queryBatchKeys.shape)
	print(queryBatchValues.shape)

	return queryBatchKeys, queryBatchValues

# def getCandidateKeys(queries, query_size, keys, key_size, values, value_size, word_index_inverted):
# 	queryKeys = []
# 	queryValues = []

# 	queryBatchKeys = []
# 	queryBatchValues = []

# 	# maxKeyMemorySize = -1
# 	for i in range(0, queries.shape[0]):
# 		keyIndices = set()
# 		query = []
# 		for j in range(query_size):
# 			if queries[i,j] == 0:
# 				break
# 			query.append(word_index_inverted[queries[i,j]])
		
# 		query = " ".join(query)
# 		keyIndices = getTopKCandidateKeyValueIndices(100, query)
# 			# if(queries[i,j] in  keyIndex):
# 			# 	keyIndices = keyIndices | keyIndex[queries[i,j]]
# 		# countOfCandidateKeys = len(keyIndices)
# 		# if(countOfCandidateKeys > maxKeyMemorySize):
# 		# 	maxKeyMemorySize = countOfCandidateKeys
# 		keyIndices = np.array(list(keyIndices))
# 		# print(keyIndices.shape)
# 		# print(type(keyIndices[0]))
# 		candidateKeys = keys[keyIndices]
# 		candidateKeyValues = values[keyIndices]

# 		# candidateKeys = np.expand_dims(candidateKeys, axis=0)
# 		# candidateKeyValues = np.expand_dims(candidateKeyValues, axis=0)

# 		queryKeys.append(candidateKeys)
# 		queryValues.append(candidateKeyValues)

# 		# if(queryKeys.size == 0):
# 		# 	queryKeys = candidateKeys
# 		# 	queryValues = candidateKeyValues
# 		# else:
# 		# 	queryKeys = np.append(queryKeys, candidateKeys, axis=0)
# 		# 	queryValues = np.append(queryValues, candidateKeyValues, axis=0)


# 	# print(queryKeys.shape)
# 	# Append 0 memory keys
# 	# TODO: Integrate with META, so that I can retrieve top n memories for each question. 
# 	for i in range(len(queryKeys)):
# 		lqueryKey = max(0, maxKeyMemorySize - queryKeys[i].shape[0])	
# 		print('Padding length required : %d'%(lqueryKey))
# 		zeroPaddingKeys = [0] * key_size
# 		zeroPaddingValues = [0] * value_size

# 		# zeroPaddingKeys = np.expand_dims(np.tile(zeroPaddingKeys, (lqueryKey,1)), axis=0)
# 		# zeroPaddingValues = np.expand_dims(np.tile(zeroPaddingValues, (lqueryKey,1)), axis=0)

# 		zeroPaddingKeys = np.tile(zeroPaddingKeys, (lqueryKey,1))
# 		zeroPaddingValues = np.tile(zeroPaddingValues, (lqueryKey,1))

# 		print('Before concatenation')
# 		print(queryKeys[i].shape)
# 		queryKeys[i] = np.concatenate((queryKeys[i], zeroPaddingKeys), axis=0)
# 		queryValues[i] = np.concatenate((queryValues[i], zeroPaddingValues), axis=0)
# 		print('After concatenation')
# 		print(queryKeys[i].shape)
# 		queryBatchKeys.append(queryKeys[i])
# 		queryBatchValues.append(queryValues[i])

# 	queryBatchKeys = np.array(queryBatchKeys)
# 	queryBatchValues = np.array(queryBatchValues)
# 	print(queryBatchKeys.shape)
# 	print(queryBatchValues.shape)
# 	print(maxKeyMemorySize)

# 	return maxKeyMemorySize, queryBatchKeys, queryBatchValues

if __name__ == "__main__":
	tuples = load_knowledge_base("./aristo_kb/COMBINED-KB.tsv", True)
	# train_data, test_data = load_questions_dataset('aristo_questions/AI2-Elementary-NDMC-v1')
	# build_vocab(tuples, train_data + test_data, False)
