
import os
import random
import torch
import torch.optim as optim
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torcheval.metrics.aggregation.auc import AUC
from tqdm import tqdm
from dataset import KeplerDataset
from transformer import ViT
import seaborn as sns
import matplotlib.pyplot as plt  

os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
size = 1

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '32123'

base_lr = 8e-4
num_classes = 2
batch_size = 1
max_epoch = 100
save_interval = int(max_epoch / 4)


snapshot_path = '/big-data/BrainProject/ryan/models/ts_xformer/out/'

def worker_init_fn(worker_id):
	random.seed(1234 + worker_id)

def acc():
	pass

def test(rank, size):
	dist.init_process_group("gloo", rank=rank, world_size=size)
	#path = snapshot_path + 'epoch_24.pth'
	#checkpoint = torch.load(path, map_location=torch.device('cuda'))
	model = ViT(num_classes=1, depth=12, heads=12, mlp_dim=3072).to(rank)
	model = DDP(model, device_ids=[rank], find_unused_parameters=True)
	model.load_state_dict(torch.load(snapshot_path + "epoch_99.pth"))

	test_data = KeplerDataset(path='/big-data/BrainProject/ryan/models/ts_xformer/data/exoTest.csv')
	print("The length of test set is: {}".format(len(test_data)))

	sampler = DistributedSampler(test_data)
	testloader = DataLoader(test_data, batch_size=batch_size, sampler=sampler, num_workers=8, pin_memory=True,
								worker_init_fn=worker_init_fn)

	iter_num = 0
	iterator = tqdm(range(max_epoch), ncols=70)
	model.eval()
	acc = 0
	tp = 0
	fp = 0
	tn = 0
	fn = 0
	for i_batch, sampled_batch in enumerate(testloader):
		data_batch, label_batch = sampled_batch
		data_batch, label_batch = data_batch.to(rank), label_batch.to(rank)
		outputs = model(data_batch)[:, 0]

		label_batch = label_batch.detach().cpu().item()
		outputs = outputs.detach().cpu().item()
		outputs = 1 if outputs >= 0 else 0
		if outputs == 1 and label_batch == 1:
			tp += 1
		elif outputs == 0 and label_batch == 0:
			tn += 1
		elif outputs == 0 and label_batch == 1:
			fn += 1
		else:
			fp += 1
		acc += outputs == label_batch
		
		iter_num = iter_num + 1
		if rank == 0:
			print('iteration %d' % (iter_num))

	print("Acc", (tp + tn) / len(test_data))
	print("NegAcc", tn / (tn + fn))
	print("Precision", tp / (tp + fp))
	print("Recall", tp / (tp + fn))
	print(tp + fn)

	a = [[tp, fp],
	  	 [fn, tn]]

	ax = plt.subplot()
	sns.heatmap(a, annot=True, fmt='g', ax=ax)  #annot=True to annotate cells, ftm='g' to disable scientific notation

	# labels, title and ticks
	ax.set_xlabel('Predicted labels')
	ax.set_ylabel('True labels')
	ax.set_title('Confusion Matrix')
	ax.xaxis.set_ticklabels(['Exoplanet', 'No Exoplanet'])
	ax.yaxis.set_ticklabels(['Exoplanet', 'No Exoplanet'])

	fig = ax.get_figure()
	fig.savefig("confusionmatrix.png") 
	
	dist.destroy_process_group()

if __name__ == "__main__":
	#mp.spawn(train, args=(size,), nprocs=size, join=True)
	
	children = []
	for i in range(size):
		subproc = mp.Process(target=test, args=(i, size))
		children.append(subproc)
		subproc.start()

	for i in range(size):
		children[i].join()
	