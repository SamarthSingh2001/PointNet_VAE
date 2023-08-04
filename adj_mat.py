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
parser.add_argument('--class_choice', type=str, default='Chair', help="class_choice")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=False,
    class_choice=[opt.class_choice])
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

test_dataset = ShapeNetDataset(
    root=opt.dataset,
    classification=False,
    class_choice=[opt.class_choice],
    split='test',
    data_augmentation=False)
testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

directory = "Adj_matrix"
path = os.path.join(opt.dataset, directory)
os.makedirs(path)

dir_points = "points"
path_points = os.path.join(opt.dataset, dir_points)

for data, filename in enumerate(opt.dataset, os.listdir(path_points)):
    Adj_mat = np.zeros(2500,2500)
    points = data
    for x in range(2500):
        for y in range(x+1, 2500):
            d = (points[x][0]-points[y][0])**2 + (points[x][1]-points[y][1])**2 + (points[x][2]-points[y][2])**2
            d = d**1/2
            if (d > 0.05):
                Adj_mat[x][y] = 1
                Adj_mat[y][x] = 1
    np.savetxt("{str(filename)}.txt", Adj_mat)
    print("done")
    print(Adj_mat)
