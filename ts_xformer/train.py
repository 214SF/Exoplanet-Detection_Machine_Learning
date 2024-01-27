
import os
import random
import torch
import torch.optim as optim
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from dataset import KeplerDataset
from transformer import ViT

os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
size = 3

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '32123'

base_lr = 8e-4
num_classes = 2
batch_size = 4
max_epoch = 100
save_interval = int(max_epoch / 4)


snapshot_path = '/big-data/BrainProject/ryan/models/ts_xformer/out/'

def worker_init_fn(worker_id):
	random.seed(1234 + worker_id)

def train(rank, size):
	dist.init_process_group("gloo", rank=rank, world_size=size)
	#path = snapshot_path + 'epoch_24.pth'
	#checkpoint = torch.load(path, map_location=torch.device('cuda'))
	model = ViT(num_classes=1, depth=12, heads=12, mlp_dim=3072).to(rank)
	model = DDP(model, device_ids=[rank], find_unused_parameters=True)
	#model.load_state_dict(checkpoint)

	train_data = KeplerDataset(path='/big-data/BrainProject/ryan/models/ts_xformer/data/exoTrain.csv')
	print("The length of train set is: {}".format(len(train_data)))

	sampler = DistributedSampler(train_data)
	trainloader = DataLoader(train_data, batch_size=batch_size, sampler=sampler, num_workers=8, pin_memory=True,
								worker_init_fn=worker_init_fn)

	loss_bce = BCEWithLogitsLoss()
	optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
	iter_num = 0

	max_iterations = max_epoch * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
	print("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
	best_performance = 0.0
	iterator = tqdm(range(max_epoch), ncols=70)
	model.float()
	model.train()
	for epoch_num in iterator:
		sampler.set_epoch(epoch_num)
		for i_batch, sampled_batch in enumerate(trainloader):
			data_batch, label_batch = sampled_batch
			data_batch, label_batch = data_batch.to(rank), label_batch.to(rank)
			optimizer.zero_grad()
			outputs = model(data_batch)[:, 0]
			loss = loss_bce(outputs, label_batch)
			loss.backward()
			optimizer.step()
			lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
			for param_group in optimizer.param_groups:
				param_group['lr'] = lr_

			iter_num = iter_num + 1

			if rank == 0:
				print('iteration %d : loss : %f' % (iter_num, loss.item()))

		if rank == 0:
			if (epoch_num + 1) % save_interval == 0 and epoch_num >= max_epoch - 100:
				save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
				torch.save(model.state_dict(), save_mode_path)
				print("save model to {}".format(save_mode_path))

			if epoch_num >= max_epoch - 1:
				save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
				torch.save(model.state_dict(), save_mode_path)
				print("save model to {}".format(save_mode_path))
				iterator.close()
				break

	dist.destroy_process_group()

if __name__ == "__main__":
	#mp.spawn(train, args=(size,), nprocs=size, join=True)
	
	children = []
	for i in range(size):
		subproc = mp.Process(target=train, args=(i, size))
		children.append(subproc)
		subproc.start()

	for i in range(size):
		children[i].join()
	