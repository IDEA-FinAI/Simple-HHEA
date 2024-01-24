'''
Reference implementation of node2vec. 
We modified the implementation to support Entity-Relation Random Walks.
'''

import argparse
import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec
import json

def parse_args():
	'''
	Parses the node2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run node2vec.")

	parser.add_argument('--input', nargs='?', default="data/icews_wiki/deepwalk.data",
	                    help='Input graph path')

	parser.add_argument('--output', nargs='?', default='data/icews_wiki/longterm.vec',
	                    help='Embeddings path')

	parser.add_argument('--dimensions', type=int, default=64,
	                    help='Number of dimensions. Default is 128.')

	parser.add_argument('--walk-length', type=int, default=80,
	                    help='Length of walk per source. Default is 80.')

	parser.add_argument('--num-walks', type=int, default=10,
	                    help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=10,
                    	help='Context size for optimization. Default is 10.')

	parser.add_argument('--iter', default=5, type=int,
                      help='Number of epochs in SGD')

	parser.add_argument('--workers', type=int, default=12,
	                    help='Number of parallel workers. Default is 8.')

	parser.add_argument('--p', type=float, default=1e-100,
	                    help='Return hyperparameter. Default is 1.')

	parser.add_argument('--q', type=float, default=1,
	                    help='Inout hyperparameter. Default is 1.')

	parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)

	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)

	parser.add_argument('--node2rel', default='data/icews_wiki/node2rel',
                      help='for[h,r,t], node2rel is a dict (h+t)->r')

	return parser.parse_args()

def read_graph():
	'''
	Reads the input network in networkx.
	'''
	if args.weighted:
		G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
	else:
		G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1

	if not args.directed:
		G = G.to_undirected()

	return G

def learn_embeddings(walks):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	walks = [list(map(str, walk)) for walk in walks]
	model = Word2Vec(walks, vector_size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, epochs=args.iter)
	model.wv.save_word2vec_format(args.output)
	
	return

def main(args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	print('start read graph...\n')
	nx_G = read_graph()
	print('read graph finished...\n')
	G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
	print('node2vec finished...\n')
	G.preprocess_transition_probs()
	print('preprocess transition probs finished...\n')
	walks = G.simulate_walks(args.num_walks, args.walk_length)
	print('simulate walks finished...\n')
	node2rel_path = args.node2rel

	if node2rel_path != 'none':
		print("node2rel:",node2rel_path)
		with open(node2rel_path,'r') as f:
			node2rel = json.load(f) 
		print("Insert Relation...")
		temp = []
		for line in walks:
			new_line = [line[0]]
			for i in range(len(line)-1):
				h = str(line[i])
				t = str(line[i+1])
				r = node2rel[h+'+'+t]
				new_line += [ r, t]
			temp.append(new_line)
      
		print(len(walks))
		walks = temp
		print(len(walks))	
	
	learn_embeddings(walks)

if __name__ == "__main__":
	args = parse_args()
	main(args)
