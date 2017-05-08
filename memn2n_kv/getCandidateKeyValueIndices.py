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


def getTopKCandidateKeyValueIndices(query, top_k=100):

	candidateKeyValueIndices = []
	cfg = 'config.toml'

	# print('Building or loading index...')
	idx = metapy.index.make_inverted_index(cfg)

	# print idx.num_docs()
	ranker = load_ranker(cfg)

	queryDoc = metapy.index.Document()
	queryDoc.content(query.strip())
	results = ranker.score(idx, queryDoc, top_k);

	for num, (d_id, _) in enumerate(results):
		candidateKeyValueIndices.append(d_id)
		# d_content = idx.metadata(d_id).get('content')
		# print("{}. {}\n".format(d_id, d_content))
	return sorted(candidateKeyValueIndices)

if __name__ == '__main__':
	getCandidateKVPairsPerQuery('aardvark')