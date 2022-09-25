'''
Training script to for models running various kinds of soft labels (CIFAR-10H vs. CIFAR-10S), as well as the original one-hot labels
Note, the base structure of this code, and some auxillary functions, are modified from the original mixup repo: https://github.com/facebookresearch/mixup-cifar10
And the runtime parameters are also from Uma et al: https://github.com/AlexandraUma/dali-learning-with-disagreement
CIFAR-10H data is from: https://github.com/jcpeterson/cifar-10h
'''
import argparse
import csv
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F

from models import resnet_uma as resnet
import models
from torch.distributions import Categorical
import random

import utils

from data import CIFAR10_SHO

# following Uma parameters: https://github.com/AlexandraUma/dali-learning-with-disagreement/blob/main/iccifar10/soft_loss.py
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--model', default="VGG", type=str)
parser.add_argument('--seed', default=0, type=int, help='random seed for alg')
parser.add_argument('--split_seed', default=7, type=int,
                    help='seed for the validation split')
parser.add_argument('--batch-size', default=32, type=int, help='batch size')
parser.add_argument('--epoch', default=65, type=int,
                    help='total epochs to run')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='use standard augmentation (default: True)')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--download', action="store_true")
parser.add_argument('--checkpoint_dir', default='./sim_annotator_uncertainty/')
parser.add_argument('--training_stats_dir',
                    default='./sim_annotator_training_stats/')
parser.add_argument('--label_method', default='cifar10h',
                    type=str, help='type of relabeling')
parser.add_argument('--use_all_cifar10h', action="store_true")
parser.add_argument('--use_early_stopping', action="store_true")
parser.add_argument('--original_cifar10_trainset', action="store_true",
                    help="run training over original cifar10 set instead, not peterson")
parser.add_argument('--num_annotators_sample', default=-1, type=int,
                    help='subsample number of annotators. -1 means use all the annotators (no downsampling)')
parser.add_argument('--annotator_subsample_seed', default=0, type=int,
                    help='for reproducibilty of annotator subsampling in label space construction')
parser.add_argument('--redist_level', default=0.05, type=float)
parser.add_argument('--holdout_some_ours', action="store_true")
parser.add_argument('--data_split_seed', default=7, type=int,
                    help='seed for heldout set of our labels')
parser.add_argument('--num_examples_holdout', default=100,
                    type=int, help='num of the soft labels to holdout')
parser.add_argument('--ls_smooth_amt', default=0.01, type=float)

args = parser.parse_args()

use_cuda = torch.cuda.is_available()

if use_cuda:
    device = "cuda"
else:
    device = "cpu"

print("use cuda: ", use_cuda)

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

checkpoint_dir = args.checkpoint_dir
training_stats_dir = args.training_stats_dir

# Data
print('==> Preparing data..')
# same as mixup work: https://github.com/facebookresearch/mixup-cifar10
norm_mean = (0.4914, 0.4822, 0.4465)
norm_std = (0.2023, 0.1994, 0.2010)
if args.augment:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean,
                             norm_std),
    ])
else:
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean,
                             norm_std),
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(norm_mean,
                         norm_std),
])

use_all_cifar10h = args.use_all_cifar10h
full_label_method = args.label_method
use_early_stopping = args.use_early_stopping
annotator_subsample_seed = args.annotator_subsample_seed
num_annotators_sample = args.num_annotators_sample
redist_level = args.redist_level
data_split_seed = args.data_split_seed
num_classes = 10

if "PerAnnotator" in full_label_method:
    use_per_annotator = True
    label_method = full_label_method.split("PerAnnotator")[0]
else:
    use_per_annotator = False
    label_method = full_label_method

ls_smooth_amt = args.ls_smooth_amt
num_examples_holdout = args.num_examples_holdout

if args.holdout_some_ours:
    split = "trainSub"
else:
    split = "train"

trainset = CIFAR10_SHO(transform=transform_train,  # transform_test,
                       split=split,
                       use_all_cifar10h=use_all_cifar10h,
                       label_method=label_method,
                       num_annotators_sample=num_annotators_sample,
                       annotator_subsample_seed=annotator_subsample_seed,
                       redist_level=redist_level, data_split_seed=data_split_seed,
                       num_examples_holdout=num_examples_holdout,
                       use_per_annotator=use_per_annotator,
                       ls_smooth_amt=ls_smooth_amt,
                       )
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=args.batch_size,
                                          shuffle=True, num_workers=8)

# because cifar-10h uses the test set as training, use orig cifar10 train set for val and test portions
# conduct split help from: https://discuss.pytorch.org/t/how-to-split-dataset-into-test-and-validation-sets/33987
eval_cifar10_dataset = datasets.CIFAR10(root='~/data', train=True, download=args.download,
                                        transform=transform_test)
n_tot_eval = len(eval_cifar10_dataset)
val_prop = 0.3
val_size = int(n_tot_eval * val_prop)
# set seed help from: https://stackoverflow.com/questions/55820303/fixing-the-seed-for-torch-random-split
split_seed = args.split_seed
random.seed(split_seed)
np.random.seed(split_seed)
torch.manual_seed(split_seed)
torch.cuda.manual_seed(split_seed)
valset, _ = torch.utils.data.random_split(
    eval_cifar10_dataset, [val_size, n_tot_eval - val_size])
valloader = torch.utils.data.DataLoader(valset, batch_size=100,
                                        shuffle=False, num_workers=8)


# somewhat messy, but if we want to just train on the full original train set
# but using the same training infrastructure...
# then override with the (near full) train set
# but holdout a part for val like before
if args.original_cifar10_trainset:
    orig_cifar10_set = datasets.CIFAR10(root='~/data', train=True, download=args.download,
                                        transform=transform_train)  #  apply data aug
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True, num_workers=8)
    split_seed = args.split_seed
    random.seed(split_seed)
    np.random.seed(split_seed)
    torch.manual_seed(split_seed)
    torch.cuda.manual_seed(split_seed)
    n_tot = len(orig_cifar10_set)
    val_prop = 0.1
    val_size = int(n_tot * val_prop)
    train_size = n_tot - val_size
    trainset, val_size = torch.utils.data.random_split(
        orig_cifar10_set, [train_size, val_size])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=8)
    valloader = torch.utils.data.DataLoader(val_size, batch_size=100,
                                            shuffle=False, num_workers=8)

# set seed for model init and training
alg_seed = args.seed
random.seed(alg_seed)
np.random.seed(alg_seed)
torch.manual_seed(alg_seed)
torch.cuda.manual_seed(alg_seed)
# create model
net = models.__dict__[args.model]()

if not os.path.isdir(training_stats_dir):
    os.mkdir(training_stats_dir)
logname = (f'{training_stats_dir}/log_' + net.__class__.__name__ + '_' + args.name + '_'
           + str(args.seed) + '.csv')

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net)
    print(torch.cuda.device_count())
    cudnn.benchmark = True
    print('Using CUDA..')

criterion = utils.cross_entropy_loss
test_criterion = criterion  #  both use CE for now

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
                      weight_decay=args.decay)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    reg_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets = map(Variable, (inputs, targets))

        if len(targets.shape) == 1:  #  if scalar index -> one-hot encode
            targets = F.one_hot(targets, num_classes=num_classes)

        outputs = net(inputs)
        loss = criterion(outputs, targets, num_classes=num_classes)

        train_loss += loss.data

        _, predicted = torch.max(outputs.data, 1)

        _, max_likely = torch.max(targets.data, 1)
        total += targets.size(0)

        correct += predicted.eq(max_likely).cpu().sum().float()

        optimizer.zero_grad()
        loss.backward()
        # grad clip help from: https://stackoverflow.com/questions/66648432/pytorch-test-loss-becoming-nan-after-some-iteration
        # and Uma: https://github.com/AlexandraUma/dali-learning-with-disagreement
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        print(batch_idx, len(trainloader),
              'Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
              % (train_loss/(batch_idx+1), reg_loss/(batch_idx+1),
                 100.*correct/total, correct, total))

    return (train_loss/batch_idx, reg_loss/batch_idx, 100.*correct/total)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(valloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)

        loss = test_criterion(outputs, targets, num_classes=num_classes)

        # test_loss += loss.data[0]
        test_loss += loss.data

        _, predicted = torch.max(outputs.data, 1)
        _, max_likely = torch.max(targets.data, 1)

        total += targets.size(0)
        correct += predicted.eq(max_likely).cpu().sum().float()

        print(batch_idx, len(valloader),
              'Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (test_loss/(batch_idx+1), 100.*correct/total,
                 correct, total))
    acc = 100.*correct/total
    if epoch == start_epoch + args.epoch - 1 or acc > best_acc:
        checkpoint(acc, epoch)
    if acc > best_acc:
        best_acc = acc
    return (test_loss/batch_idx, 100.*correct/total)


def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    torch.save(state, checkpoint_dir + 'ckpt.' + args.name + '_'
               + str(args.seed))


# apply the same lr transform as Uma et al
def adjust_learning_rate(optimizer, epoch, lr_drop_schedule=[50, 55]):
    """
    Decrease the learning rate at various epochs
    Note, original code is from mixup codebase
    Modified for Uma's lr scheduler: https://github.com/AlexandraUma/dali-learning-with-disagreement/blob/main/iccifar10/soft_loss.py
    """
    lr = args.lr
    if epoch >= lr_drop_schedule[0]:
        lr /= 10
    if epoch >= lr_drop_schedule[1]:
        lr /= 10  #  decrease twice
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train loss', 'reg loss', 'train acc',
                            'test loss', 'test acc'])

# implement early stopping
# help from: https://pythonguides.com/pytorch-early-stopping/
# and: https://stackoverflow.com/questions/68929471/implementing-early-stopping-in-pytorch-without-torchsample
max_patience = 5
best_val_acc = 0
patience = 0
use_early_stopping = args.use_early_stopping

for epoch in range(start_epoch, args.epoch):
    train_loss, reg_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)

    adjust_learning_rate(optimizer, epoch)
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss, reg_loss, train_acc, test_loss,
                            test_acc])

    if use_early_stopping:
        if test_acc > best_val_acc:
            print("New acc: ", test_acc, " old acc: ", best_val_acc)
            best_val_acc = test_acc
            patience = 0
        else:
            print("Still waiting... current: ", test_acc,
                  " old: ", best_val_acc, " patience: ", patience)
            if patience >= max_patience:
                break
            else:
                patience += 1
