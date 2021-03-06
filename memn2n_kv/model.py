"""Key Value Memory Networks with GRU reader.
The implementation is based on https://arxiv.org/abs/1606.03126
The implementation is based on http://arxiv.org/abs/1503.08895 [1]
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from six.moves import range
import numpy as np
# from attention_reader import Attention_Reader

def position_encoding(sentence_size, embedding_size):
	"""
	Position Encoding described in section 4.1 [1]
	"""
	encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
	ls = sentence_size+1
	le = embedding_size+1
	for i in range(1, le):
		for j in range(1, ls):
			encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
	encoding = 1 + 4 * encoding / embedding_size / sentence_size
	return np.transpose(encoding)

def add_gradient_noise(t, stddev=1e-3, name=None):
	"""
	Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].

	The input Tensor `t` should be a gradient.

	The output will be `t` + gaussian noise.

	0.001 was said to be a good fixed value for memory networks [2].
	"""
	with tf.name_scope(name, "add_gradient_noise", [t, stddev]) as name:
		t = tf.convert_to_tensor(t, name="t")
		gn = tf.random_normal(tf.shape(t), stddev=stddev)
		return tf.add(t, gn, name=name)

def zero_nil_slot(t, name=None):
	"""
	Overwrites the nil_slot (first row) of the input Tensor with zeros.
	The nil_slot is a dummy slot and should not be trained and influence
	the training algorithm.
	"""
	with tf.name_scope(name, "zero_nil_slot", [t]) as name:
		t = tf.convert_to_tensor(t, name="t")
		s = tf.shape(t)[1]
		z = tf.zeros(tf.pack([1, s]))
		return tf.concat(0, [z, tf.slice(t, [1, 0], [-1, -1])], name=name)

class MemN2N_KV(object):
	"""Key Value Memory Network."""
	def __init__(self, batch_size, vocab_size, memory_size,
				 query_size, key_size, value_size, embedding_size,
				 feature_size=30,
				 hops=3,
				 l2_lambda=0.2,
				 name='KeyValueMemN2N'):
		"""Creates an Key Value Memory Network

		Args:
		batch_size: The size of the batch.

		vocab_size: The size of the vocabulary (should include the nil word). The nil word one-hot encoding should be 0.

		query_size: largest number of words in question

		key_size: largest number of words in key

		value_size: largest number of words in value

		embedding_size: The size of the word embedding.

		memory_key_size: the size of memory slots for keys
		memory_value_size: the size of memory slots for values
		
		feature_size: dimension of features extraced from word embedding

		hops: The number of hops. A hop consists of reading and addressing a memory slot.

		debug_mode: If true, print some debug info about tensors
		name: Name of the End-To-End Memory Network.\
		Defaults to `KeyValueMemN2N`.
		"""
		self._key_size = key_size
		self._value_size = value_size
		self._batch_size = batch_size
		self._vocab_size = vocab_size
		self._memory_key_size = memory_size
		self._memory_value_size = memory_size
		self._query_size = query_size
		#self._wiki_sentence_size = doc_size
		self._embedding_size = embedding_size
		self._hops = hops
		self._l2_lambda = l2_lambda
		self._name = name
		self._key_encoding = tf.constant(position_encoding(self._key_size, self._embedding_size), name="encoding")
		self._value_encoding = tf.constant(position_encoding(self._value_size, self._embedding_size), name="encoding")
		self._query_encoding = tf.constant(position_encoding(self._query_size, self._embedding_size), name="encoding")
		self._build_inputs()

		
		d = feature_size
		self._feature_size = feature_size
		self._build_graph()
		
	def _build_graph(self):

		# trainable variables
		self.reader_feature_size = self._embedding_size
		
		self.A = tf.get_variable('A', shape=[self._feature_size, self._embedding_size],
								 initializer=tf.contrib.layers.xavier_initializer())
		
		self.TK = tf.get_variable('TK', shape=[self._memory_key_size, self._embedding_size],
								  initializer=tf.contrib.layers.xavier_initializer())
		print(self.TK)
		self.TV = tf.get_variable('TV', shape=[self._memory_value_size, self._embedding_size],
								  initializer=tf.contrib.layers.xavier_initializer())
		print(self.TV)

		# Embedding layer
		with tf.name_scope("embedding"):
			nil_word_slot = tf.zeros([1, self._embedding_size])
			self.W = tf.concat(axis=0, values=[nil_word_slot, tf.get_variable('W', shape=[self._vocab_size-1, self._embedding_size],
																  initializer=tf.contrib.layers.xavier_initializer())])
			print(self.W)
			self.W_memory = tf.concat(axis=0, values=[nil_word_slot, tf.get_variable('W_memory', shape=[self._vocab_size-1, self._embedding_size],
																		 initializer=tf.contrib.layers.xavier_initializer())])
			# self.W_memory = self.W
			self._nil_vars = set([self.W.name, self.W_memory.name])
			# shape: [batch_size, query_size, embedding_size]
			self.embedded_query = tf.nn.embedding_lookup(self.W, self._query)
			print(self.embedded_query)

			print('Printing the shape of memory_key tensor')
			print(self._memory_key)
			# # [1, memory_size, key_size]
			# self._memory_key = tf.expand_dims(self._memory_key, [0])
			# # [batch_size, memory_key_size, key_size]
			# self._memory_key = tf.tile(self._memory_key, [batch_size, 1, 1])
			# shape: [batch_size, memory_key_size, key_size, embedding_size]
			self.embedded_mkeys = tf.nn.embedding_lookup(self.W_memory, self._memory_key)
			print(self.embedded_mkeys)

			# # [1, memory_size, memory_value_size]
			# self._memory_value = tf.expand_dims(self._memory_value, [0])
			# # [batch_size, memory_size, memory_value_size]
			# self._memory_value = tf.tile(self._memory_value, [batch_size, 1, 1])
			# shape: [batch_size, memory_size, memory_value_size, embedding_size]
			self.embedded_mvalues = tf.nn.embedding_lookup(self.W_memory, self._memory_value)
			print(self.embedded_mvalues)

		# TODO : Why does it encode query using position encoding?? to ensure order of words is preserved
		q_r = tf.reduce_sum(self.embedded_query*self._query_encoding, 1)
		key_r = tf.reduce_sum(self.embedded_mkeys*self._key_encoding, 2)
		value_r = tf.reduce_sum(self.embedded_mvalues*self._value_encoding, 2)
		
		r_list = []
		for _ in range(self._hops):
			# define R for variables
			R = tf.get_variable('R{}'.format(_), shape=[self._feature_size, self._feature_size],
								initializer=tf.contrib.layers.xavier_initializer())
			r_list.append(R)

		o = self._key_addressing(key_r, value_r, q_r, r_list)
		o = tf.transpose(o)
		self.B = self.A
		
		#logits_bias = tf.get_variable('logits_bias', [self._vocab_size])
		y_tmp = tf.matmul(self.B, self.W_memory, transpose_b=True)
		with tf.name_scope("prediction"):
			logits = tf.matmul(o, y_tmp)# + logits_bias
			#logits = tf.nn.dropout(tf.matmul(o, self.B) + logits_bias, self.keep_prob)
			probs = tf.nn.softmax(tf.cast(logits, tf.float32))
			probs = tf.Print(probs, [probs])
			print(probs)
			cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.cast(self._labels, tf.float32), name='cross_entropy')
			cross_entropy_sum = tf.reduce_sum(cross_entropy, name="cross_entropy_sum")

			# loss op
			vars = tf.trainable_variables()
			print('Trainable Variables')
			names = [v.name for v in vars]
			print(names)
			lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars])
			loss_op = cross_entropy_sum + self._l2_lambda*lossL2

			tf.summary.scalar("cross_entropy", loss_op)
			summary_op = tf.summary.merge_all()

			# predict ops
			predict_op = tf.argmax(probs, 1, name="predict_op")

			# assign ops
			self.loss_op = loss_op
			self.predict_op = predict_op
			self.probs = probs
			self.summary_op = summary_op


	def _build_inputs(self):
		with tf.name_scope("input"):
			self._memory_key = tf.placeholder(tf.int32, [None, self._memory_key_size, self._key_size], name='memory_key')
			self._query = tf.placeholder(tf.int32, [None, self._query_size], name='question')

			self._memory_value = tf.placeholder(tf.int32, [None, self._memory_value_size, self._value_size], name='memory_value')

			self._labels = tf.placeholder(tf.float32, [None, self._vocab_size], name='answer')
			self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

	'''
	mkeys: the vector representation for keys in memory
	-- shape of each mkeys: [1, embedding_size]
	mvalues: the vector representation for values in memory
	-- shape of each mvalues: [1, embedding_size]
	questions: the vector representation for the question
	-- shape of questions: [1, embedding_size]
	-- shape of R: [feature_size, feature_size]
	-- shape of self.A: [feature_size, embedding_size]
	-- shape of self.B: [feature_size, embedding_size]
	self.A, self.B and R are the parameters to learn
	'''
	def _key_addressing(self, mkeys, mvalues, questions, r_list):
		
		with tf.variable_scope(self._name):
			# [feature_size, batch_size]
			u = tf.matmul(self.A, questions, transpose_b=True)
			print(u)
			u = [u]
			for _ in range(self._hops):
				R = r_list[_]
				u_temp = u[-1]
				print(mkeys)
				print(self.TK)
				mk_temp = mkeys + self.TK
				# [embedding_size, batch_size x memory_size]
				k_temp = tf.reshape(tf.transpose(mk_temp, [2, 0, 1]), [self._embedding_size, -1])
				# [feature_size, batch_size x memory_size]
				a_k_temp = tf.matmul(self.A, k_temp)
				# [batch_size, memory_size, feature_size]
				a_k = tf.reshape(tf.transpose(a_k_temp), [-1, self._memory_key_size, self._feature_size])
				# [batch_size, 1, feature_size]
				u_expanded = tf.expand_dims(tf.transpose(u_temp), [1])
				# [batch_size, memory_size]
				dotted = tf.reduce_sum(a_k*u_expanded, 2)
				print(dotted)

				# Calculate probabilities
				# [batch_size, memory_size]
				probs = tf.nn.softmax(dotted)
				# [batch_size, memory_size, 1]
				probs_expand = tf.expand_dims(probs, -1)
				mv_temp = mvalues + self.TV
				# [embedding_size, batch_size x memory_size]
				v_temp = tf.reshape(tf.transpose(mv_temp, [2, 0, 1]), [self._embedding_size, -1])
				# [feature_size, batch_size x memory_size]
				a_v_temp = tf.matmul(self.A, v_temp)
				# [batch_size, memory_size, feature_size]
				a_v = tf.reshape(tf.transpose(a_v_temp), [-1, self._memory_key_size, self._feature_size])
				# [batch_size, feature_size]
				o_k = tf.reduce_sum(probs_expand*a_v, 1)
				# [feature_size, batch_size]
				o_k = tf.transpose(o_k)
				# [feature_size, batch_size]
				u_k = tf.matmul(R, u[-1]+o_k)

				u.append(u_k)

			return u[-1]

	