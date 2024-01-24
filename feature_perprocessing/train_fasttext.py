import scipy.sparse as sp
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
import json
import string
from gensim.models import FastText, Word2Vec
import random
import gensim
from copy import deepcopy
random.seed(1)
import argparse 

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', type=str)
	parser.add_argument('--output', type=str)
	parser.add_argument('--size', type=int, default=64)
	args = parser.parse_args()

	
	input_path = args.input
	out_path = args.output
	size = args.size

	sentences = []
	with open(input_path,'r') as f:
		for line in f:
			sentences.append(line.strip().split())

	model = FastText(sentences,
					vector_size=size,
					window=5,
					negative=5,
					min_count=1,
					sg=0,
					cbow_mean=1,
					epochs=10,
					min_n=2,
					max_n = 3,
					word_ngrams = 1,
					workers=-1)

	model.wv.save_word2vec_format(out_path+'fast.vec')