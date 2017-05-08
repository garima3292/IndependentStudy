from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from data_utils import *
from model import MemN2N_KV, add_gradient_noise, zero_nil_slot
from sklearn import cross_validation, metrics
from pprint import pprint

import tensorflow as tf
import numpy as np


tf.flags.DEFINE_string("data_dir", "aristo_questions/AI2-Elementary-NDMC-v1", "Directory containing bAbI tasks")
tf.flags.DEFINE_string("kb_path", "aristo_kb/COMBINED-KB.tsv", "File containing Aristo Knowledge Base Tuples")
tf.flags.DEFINE_string("log_dir", "./logs", "Path at which logs will be stored")

tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")

# Model Parameters
tf.flags.DEFINE_float("epsilon", 0.1, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("l2_lambda", 0.1, "Lambda for l2 loss.")
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
tf.flags.DEFINE_float("max_grad_norm", 20.0, "Clip gradients to this norm.")
tf.flags.DEFINE_float("keep_prob", 1.0, "Keep probability for dropout")
tf.flags.DEFINE_integer("evaluation_interval", 50, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("feature_size", 40, "Feature size")
tf.flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 100, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 30, "Embedding size for embedding matrices.")
tf.flags.DEFINE_string("reader", "bow", "Reader for the model (bow, simple_gru)")

# ConfigProto Options
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("allow_growth", True, "Allow GPU memory region growth starting from small and grwoing as needed")

FLAGS = tf.flags.FLAGS

train, test = load_questions_dataset(FLAGS.data_dir)
data = train + test
kb_tuples = load_knowledge_base(FLAGS.kb_path)
print('%d tuples in knowledge base' %(len(kb_tuples)))

# Use Pre Loaded index
word_idx, word_idx_inverted = build_vocab(kb_tuples, train + test)
print("Vocab size : %d" %(len(word_idx)))

query_size = max(map(len, (q for q, _ in data)))
max_memory_size = len(kb_tuples)
keys = [arg1 + pred for arg1, pred, _ in kb_tuples]
values = [arg2 for _, _, arg2 in kb_tuples]
key_size = max(map(len, (key for key in keys)))
value_size = max(map(len, (value for value in values)))
vocab_size = len(word_idx) + 1 # +1 for nil word

Q, A = vectorize_questions_data(train, word_idx, query_size)
trainQ, valQ, trainA, valA = cross_validation.train_test_split(Q, A, test_size=.1)
testQ, testA = vectorize_questions_data(test, word_idx, query_size)
keys, values = vectorize_kb_data(keys, values, word_idx, key_size, value_size)


# Transform the keys and values to be of shape batch_size, memory_size, (key_size/value_size)
# keys = np.tile(np.expand_dims(keys, axis=0), [FLAGS.batch_size, 1, 1])
# values = np.tile(np.expand_dims(values, axis=0), [FLAGS.batch_size, 1, 1])

# params
n_train = trainQ.shape[0]
n_test = testQ.shape[0]
n_val = valQ.shape[0]

print("Training Size", n_train)
print("Validation Size", n_val)
print("Testing Size", n_test)

print("trainA.shape")
print(trainA.shape)

train_labels = np.argmax(trainA, axis=1)
test_labels = np.argmax(testA, axis=1)
val_labels = np.argmax(valA, axis=1)

print('Train Labels.shape')
print(train_labels.shape)

batch_size = FLAGS.batch_size
batches = zip(range(0, n_train-batch_size, batch_size), range(batch_size, n_train, batch_size))


# np.random.shuffle(batches)
# train_preds = []
# start = 0
# end = start + 2
# q = trainQ[start:end]
# a = trainA[start:end]
# print(q)
# print(a)
# maxMemorySize, k, v = getCandidateKeys(q, query_size, keys, key_size, values, value_size, word_key_idx)
			
with tf.Graph().as_default():
	session_conf = tf.ConfigProto(
		allow_soft_placement=FLAGS.allow_soft_placement,
		log_device_placement=FLAGS.log_device_placement)

	session_conf.gpu_options.allow_growth=FLAGS.allow_growth

	global_step = tf.Variable(0, name="global_step", trainable=False)
	
	# decay learning rate
	starter_learning_rate = FLAGS.learning_rate
	learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 20000, 0.96, staircase=True)

	# TODO : Experiment with other optimizers
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=FLAGS.epsilon)

	with tf.Session() as sess:
		model = MemN2N_KV(batch_size=batch_size, vocab_size=vocab_size, memory_size=100, 
						  query_size=query_size, key_size=key_size, value_size=value_size,
						  feature_size=FLAGS.feature_size, embedding_size=FLAGS.embedding_size, 
						  hops=FLAGS.hops, l2_lambda=FLAGS.l2_lambda)
		
		grads_and_vars = optimizer.compute_gradients(model.loss_op)

		grads_and_vars = [(tf.clip_by_norm(g, FLAGS.max_grad_norm), v)
						  for g, v in grads_and_vars if g is not None]
		grads_and_vars = [(add_gradient_noise(g), v) for g, v in grads_and_vars]
		nil_grads_and_vars = []
		for g, v in grads_and_vars:
			if v.name in model._nil_vars:
				nil_grads_and_vars.append((zero_nil_slot(g), v))
			else:
				nil_grads_and_vars.append((g, v))

		train_op = optimizer.apply_gradients(nil_grads_and_vars, name="train_op", global_step=global_step)
		sess.run(tf.global_variables_initializer())

		def train_step(k, v, q, a):
			feed_dict = {
				model._memory_value: v,
				model._query: q,
				model._memory_key: k,
				model._labels: a,
				model.keep_prob: FLAGS.keep_prob
			}
			_, step, predict_op, probs, summary_op = sess.run([train_op, global_step, model.predict_op, model.probs, model.summary_op], feed_dict)
			return predict_op, probs, summary_op

		def test_step(k, v, q):
			feed_dict = {
				model._query: q,
				model._memory_key: k,
				model._memory_value: v,
				model.keep_prob: 1
			}
			preds = sess.run(model.predict_op, feed_dict)
			return preds


		# for t in range(1, FLAGS.epochs+1):
		writer = tf.summary.FileWriter(FLAGS.log_dir, graph=tf.get_default_graph())
		for t in range(1, FLAGS.epochs+1):
			np.random.shuffle(batches)
			train_preds = []
			train_probs = []
			batch_count = 0
			for start in range(0, n_train, batch_size):
				batch_count=batch_count+1
				end = start + batch_size
				q = trainQ[start:end]
				a = trainA[start:end]
				k, v = getCandidateKeys(q, query_size, keys, key_size, values, value_size, word_idx_inverted)
				print(k.shape)
				print(v.shape)
				predict_op, probs, summary = train_step(k, v, q, a)
				train_preds += list(predict_op)
				train_probs += list(probs)
				writer.add_summary(summary, t * FLAGS.batch_size + batch_count)

				# total_cost += cost_t
			print('Printing softmax distribution of answers over vocab for %d answers : '%(len(train_preds)))
			train_probs = np.array(train_probs)
			for i in range(train_probs.shape[0]):
				print(train_probs[i,:])
				indices = np.where(train_probs[i,:] > 0.1)
				print('Printing indices where more than 0')
				print(indices)
				# len_i = len(probs[i])
				# print('Print Answer : %d, len : %d'%(i, len(probs[i])))
				# for j in range(len_i):
				# 	print(" %d"%(probs[i][j]))
				# print('\n')

			train_acc = metrics.accuracy_score(np.array(train_preds), train_labels)
			print('-----------------------')
			print('Epoch', t)
			print('Training Accuracy: {0:.2f}'.format(train_acc))
			print('-----------------------')
				
			if t % FLAGS.evaluation_interval == 0:
				k, v = getCandidateKeys(valQ, keys, values, word_idx_inverted)
				model.load_memory_size(maxMemorySize)
				val_preds = test_step(k, v, valQ)
				val_acc = metrics.accuracy_score(np.array(val_preds), val_labels)
				print (val_preds)
				print('-----------------------')
				print('Epoch', t)
				print('Validation Accuracy:', val_acc)
				print('-----------------------')
		# test on train dataset
		# train_preds = test_step(np.tile(np.expand_dims(keys, axis=0), [trainQ.shape[0], 1, 1]), np.tile(np.expand_dims(values, axis=0), [trainQ.shape[0], 1, 1]), trainQ)
		# train_acc = metrics.accuracy_score(train_labels, train_preds)
		# train_acc = '{0:.2f}'.format(train_acc)
		# # eval dataset
		# val_preds = test_step(np.tile(np.expand_dims(keys, axis=0), [valQ.shape[0], 1, 1]), np.tile(np.expand_dims(values, axis=0), [valQ.shape[0], 1, 1]), valQ)
		# val_acc = metrics.accuracy_score(val_labels, val_preds)
		# val_acc = '{0:.2f}'.format(val_acc)
		# # testing dataset
		# test_preds = test_step(np.tile(np.expand_dims(keys, axis=0), [testQ.shape[0], 1, 1]), np.tile(np.expand_dims(keys, axis=0), [testQ.shape[0], 1, 1]), testQ)
		# test_acc = metrics.accuracy_score(test_labels, test_preds)
		# test_acc = '{0:.2f}'.format(test_acc)
		# print("Testing Accuracy: {}".format(test_acc))
		# print('Writing final results to {}'.format(FLAGS.output_file))
		# with open(FLAGS.output_file, 'a') as f:
		# 	f.write('{}, {}, {}, {}\n'.format(FLAGS.task_id, test_acc, train_acc, val_acc))
