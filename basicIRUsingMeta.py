import sys
import time
import metapy
import pytoml

def load_ranker(cfg_file):
	"""
	Use this function to return the Ranker object to evaluate.

	The parameter to this function, cfg_file, is the path to a
	configuration file used to load the index. You can ignore this, unless
	you need to load a ForwardIndex for some reason...
	"""
	# return PL2Ranker(c=0.75)
	return metapy.index.OkapiBM25()
	# return metapy.index.JelinekMercer(0.8)

def evaluate(generatedAnswers, correctAnswers):
	print len(generatedAnswers)
	print len(correctAnswers)

	correctCount = 0
	for index, answer in enumerate(generatedAnswers):
		print '%s %s' %(answer, correctAnswers[index])
		if(answer == correctAnswers[index]):
			correctCount=correctCount+1

	print correctCount
	print float(correctCount)/len(generatedAnswers)


if __name__ == '__main__':
	if len(sys.argv) != 2:
		print("Usage: {} config.toml".format(sys.argv[0]))
		sys.exit(1)

	cfg = sys.argv[1]
	print('Building or loading index...')
	idx = metapy.index.make_inverted_index(cfg)
	print idx.num_docs()
	ranker = load_ranker(cfg)

	with open(cfg, 'r') as fin:
		cfg_d = pytoml.load(fin)

	query_cfg = cfg_d['query-runner']
	if query_cfg is None:
		print("query-runner table needed in {}".format(cfg))
		sys.exit(1)

	start_time = time.time()
	top_k = 1
	query_path = query_cfg.get('query-path', 'queries.txt')
	query_start = query_cfg.get('query-id-start', 0)

	query = metapy.index.Document()
	print('Running queries')
	answerKey = open('aristo-mini/correctAnswers_Dev.txt', 'r').readlines()
	questionCount = 0
	correctAnswers = []
	generatedAnswers = []
	with open(query_path) as query_file:
		scores = []
		answerKeyLine = answerKey[questionCount].split(' ')
		correctAnswers.append(answerKeyLine[0])
		choices = int(answerKeyLine[1])
		queryCount = 0
		for query_num, line in enumerate(query_file):
			queryCount=queryCount+1
			query = metapy.index.Document()
			query.content(line.strip())
			results = ranker.score(idx, query, top_k);
			if(results == []):
				scores.append(0)
			else:
				scores.append(results[0][1])
			if(queryCount == choices):
				# print 'Inside if'
				maxScore = max(scores)
				maxIndex = scores.index(maxScore)
				answer = chr(ord('A') + maxIndex)
				generatedAnswers.append(answer)
				scores = []
				questionCount=questionCount+1
				if(questionCount == len(answerKey)):
					break
				answerKeyLine = answerKey[questionCount].split(' ')
				correctAnswers.append(answerKeyLine[0])
				choices = int(answerKeyLine[1])
				queryCount = 0
				# print("{}. {}...\n".format(num + 1, content[0:250]))
			

	evaluate(generatedAnswers, correctAnswers)
				

			
			
