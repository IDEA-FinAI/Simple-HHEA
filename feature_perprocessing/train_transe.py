import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import argparse 
import torch

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', type=str)
	parser.add_argument('--output', type=str)
	parser.add_argument('--size', type=int, default=64)
	args = parser.parse_args()

	input_path = args.input
	out_path = args.output
	size = args.size

	# dataloader for training
	train_dataloader = TrainDataLoader(
		in_path = input_path, 
		nbatches = 100,
		threads = 8, 
		sampling_mode = "normal", 
		bern_flag = 1, 
		filter_flag = 1, 
		neg_ent = 25,
		neg_rel = 0)

	# define the model
	transe = TransE(
		ent_tot = train_dataloader.get_ent_tot(),
		rel_tot = train_dataloader.get_rel_tot(),
		dim = size, 
		p_norm = 1, 
		norm_flag = True)


	# define the loss function
	model = NegativeSampling(
		model = transe, 
		loss = MarginLoss(margin = 5.0),
		batch_size = train_dataloader.get_batch_size()
	)

	# train the model
	trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1000, 
						alpha = 0.01, opt_method = 'adam', checkpoint_dir=out_path+'transe.ckpt', use_gpu = True)
	# trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1000,
	# 					alpha = 0.01, opt_method = 'adam', patience = 100, checkpoint_dir=out_path+'transe.ckpt', use_gpu = True)
	trainer.run()
	
	transe.save_checkpoint(out_path+'transe.ckpt')
	# transe.load_checkpoint(out_path+'transe.ckpt')

	embedding = transe.ent_embeddings.cpu()(torch.LongTensor([i for i in range(train_dataloader.get_ent_tot())]))
	relation_embedding = transe.rel_embeddings.cpu()(torch.LongTensor([i for i in range(train_dataloader.get_rel_tot())]))
	shape = embedding.shape
	embedding = embedding.tolist()
	rel_shape = relation_embedding.shape
	relation_embedding = relation_embedding.tolist()
	with open(out_path+'transe.vec','w') as f:
		f.write('%d %d\n'%tuple(shape))
		for i,line in enumerate(embedding):
			f.write('%d %s\n'%(i,' '.join([str(j) for j in line])))
	with open(out_path+'transe_rel.vec','w') as f:
		f.write('%d %d\n'%tuple(rel_shape))
		for i,line in enumerate(relation_embedding):
			f.write('%d %s\n'%(i,' '.join([str(j) for j in line])))