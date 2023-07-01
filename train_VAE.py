from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset, ModelNetDataset
from pointnet.model import VariationalAutoencoder, Decoder
import torch.nn.functional as F
from torch.nn.functional import normalize
from tqdm import tqdm
import numpy as np
from pytorch3d.loss import chamfer_distance


parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.dataset_type == 'shapenet':
    dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        npoints=opt.num_points)

    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=True,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
elif opt.dataset_type == 'modelnet40':
    dataset = ModelNetDataset(
        root=opt.dataset,
        npoints=opt.num_points,
        split='trainval')

    test_dataset = ModelNetDataset(
        root=opt.dataset,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
else:
    exit('wrong dataset type')


dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = len(dataset.classes)
print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

vae = VariationalAutoencoder(latent_dims=num_classes)
decoder = Decoder(k=num_classes)

if opt.model != '':
    vae.load_state_dict(torch.load(opt.model))

lr = 1e-8
optimizer = optim.Adam(vae.parameters(), lr=lr, weight_decay=1e-6)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
vae.cuda()

num_batch = len(dataset) / opt.batchSize

vae.train()
train_loss = 0.0
test_loss = 0.0


for epoch in range(opt.nepoch):
    scheduler.step()
    for i, data in enumerate(dataloader, 0):
        points, target = data
        target = target[:, 0]
        #for x in range(points.size(0)):
            #points[x] = normalize(points[x], p=2.0, dim = 1)
        
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        points_hat = vae(points)
        #points_hat = points_hat.transpose(2, 1)
        #loss = ((points - points_hat)**2).sum() + vae.encoder.kl
        loss, _ = chamfer_distance(sample_sphere, sample_test)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
        print('[%d: %d/%d] train loss: %f' % (epoch, i, num_batch, loss.item() ))
        #print(i)

        if i % 10 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target = data
            target = target[:, 0]
            #for x in range(points.size(0)):
               # points[x] = normalize(points[x], p=2.0, dim = 0)
            
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            points_hat = vae(points)
            #loss = ((points - points_hat)**2).sum() + vae.encoder.kl
            loss, _ = chamfer_distance(sample_sphere, sample_test)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            test_loss+=loss.item()
            print('[%d: %d/%d] train loss: %f' % (epoch, i, num_batch, loss.item() ))

    #torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))

vae.eval()
val_loss = 0.0
for i,data in tqdm(enumerate(testdataloader, 0)):
    points, target = data
    target = target[:, 0]
    #for x in range(points.size(0)):
        #points[x] = normalize(points[x], p=2.0, dim = 1)
    
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    points_hat = vae(points)
    #loss = ((points - points_hat)**2).sum() + vae.encoder.kl
    loss, _ = chamfer_distance(sample_sphere, sample_test)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    test_loss+=loss.item()
    print('[%d/%d] test loss: %f' % (i, num_batch, loss.item() ))

#print("final accuracy {}".format(total_correct / float(total_testset)))

print("train loss: ", train_loss / len(dataloader.dataset))
print("test loss: ", test_loss / len(testdataloader.dataset))


array = torch.normal(0, 1, size=(32, 16))
data = decoder(array)

x = data[0, :, 0].detach().numpy()
y = data[0, :, 1].detach().numpy()
z = data[0, :, 2].detach().numpy()

pts = data[0].detach().numpy()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z)
plt.show()
plt.savefig("img.jpg")

