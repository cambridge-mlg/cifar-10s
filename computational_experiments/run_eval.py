'''
Load in model and run eval suite
Uses some of mixup codebase structure as backbone
https://github.com/facebookresearch/mixup-cifar10 (with many modifications)
'''

import random
import json
import utils
from data import CIFAR10_SHO
from torch.distributions import Categorical
import models
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch
import numpy as np
from scipy.stats import wasserstein_distance
import argparse
import csv
import os


parser = argparse.ArgumentParser(description='Run evaluation')
parser.add_argument('--run_name', default='0', type=str, help='name of run')
parser.add_argument('--split_seed', default=7, type=int,
                    help='seed for the validation split')
parser.add_argument('--checkpoint_dir', default='./checkpoints/',
                    help='directory where checkpoints were saved during training')
parser.add_argument('--eval_stats_dir', default='./eval_stats/')
parser.add_argument('--eval_set', default='val', help='dataset to evaluate on')
parser.add_argument('--n_eval_adv_robustness', default=1000, type=int,
                    help='number of examples over to run for adversarial robustness checks')
parser.add_argument('--no_adv', action="store_true",
                    help="do not run any adversarial attacks")
parser.add_argument('--save_detailed_stats', action="store_true",
                    help="whether to save out detailed more detailed intermediate eval info")
parser.add_argument('--data_split_seed', default=7, type=int,
                    help='seed for heldout set of our labels')
parser.add_argument('--num_examples_holdout', default=100,
                    type=int, help='seed for heldout set of our labels')

'''
Set-up and extraction of key set-up parameters
'''
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
eval_stats_dir = args.eval_stats_dir

no_adv = args.no_adv

if not os.path.exists(eval_stats_dir):
    os.makedirs(eval_stats_dir)

'''
Handle data loading
And split for eval
'''
# same as mixup work: https://github.com/facebookresearch/mixup-cifar10
norm_mean = (0.4914, 0.4822, 0.4465)
norm_std = (0.2023, 0.1994, 0.2010)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(norm_mean,
                         norm_std),
])

# we need to *not* normalize for pgd
pgd_transform = transforms.Compose([
    transforms.ToTensor()
])

batch_size = 1
n_workers = 0

eval_set = args.eval_set
data_split_seed = args.data_split_seed
save_detailed_stats = args.save_detailed_stats


if eval_set == "cifar10hHoldout":
    split = "test"
    label_method = "cifar10h"
    num_annotators_sample = -1
    use_all_cifar10h = True  #  use CIFAR-10H labels for all that we've held out
    num_examples_holdout = -1
else:
    split = "testSub"
    # we use Top 2 clamp as the heldout eval type
    label_method = "oursTop2Clamp"
    num_annotators_sample = 6
    use_all_cifar10h = False
    num_examples_holdout = args.num_examples_holdout  #  eval on 100

redist_level = 0.1  #  to match the HCOMP paper
# peterson soft label eval w/ held-out 30%
hsoft_heldout_set = CIFAR10_SHO(transform=transform,
                                split=split,
                                label_method=label_method,
                                num_annotators_sample=num_annotators_sample,
                                redist_level=redist_level,
                                use_all_cifar10h=use_all_cifar10h,
                                data_split_seed=data_split_seed,
                                num_examples_holdout=num_examples_holdout,
                                )
hsoft_heldout_loader = torch.utils.data.DataLoader(hsoft_heldout_set,
                                                   batch_size=batch_size,
                                                   shuffle=True, num_workers=n_workers)

'''
Optionally use the same validation set we'd run during training
Using original CIFAR-10 labels (i.e., hard labels)
'''
# because cifar-10h uses the test set as training, use orig cifar10 train set for val and test portions
# conduct split help from: https://discuss.pytorch.org/t/how-to-split-dataset-into-test-and-validation-sets/33987
eval_cifar10_dataset = datasets.CIFAR10(root='~/data', train=True, download=False,
                                        transform=transform)
n_tot_eval = len(eval_cifar10_dataset)
val_prop = 0.3
val_size = int(n_tot_eval * val_prop)
test_size = n_tot_eval - val_size
# set seed help from: https://stackoverflow.com/questions/55820303/fixing-the-seed-for-torch-random-split
# and: https://stackoverflow.com/questions/67728748/pytorch-data-random-split-doesnt-split-randomly
split_seed = args.split_seed  # use same as during training val
random.seed(split_seed)
np.random.seed(split_seed)
torch.manual_seed(split_seed)
torch.cuda.manual_seed(split_seed)
valset, _ = torch.utils.data.random_split(
    eval_cifar10_dataset, [val_size, n_tot_eval - val_size])
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                        shuffle=False, num_workers=n_workers)

# mapping from idx to class label
# {v: k.capitalize() for k,v in dataset.get_class_per_idx.items()}
idx2class = hsoft_heldout_set.get_class_per_idx()
num_classes = hsoft_heldout_set.num_classes
class_labels = [class_name.capitalize()
                for class_name in hsoft_heldout_set.classes]

'''
Create a second copy of the dataset without normalization applied
We need to do this b/c adversarial attack checks can't have normalization
Note: this part of the code is currently messy/repetive...
'''
if eval_set == "val":
    dataset = valset
    dataloader = valloader

    pgd_eval_cifar10_dataset = datasets.CIFAR10(root='~/data', train=True, download=False,
                                                transform=pgd_transform)
    # note: need to resplit, same method as earlier
    random.seed(split_seed)
    np.random.seed(split_seed)
    torch.manual_seed(split_seed)
    torch.cuda.manual_seed(split_seed)
    adv_attack_valset, _ = torch.utils.data.random_split(
        pgd_eval_cifar10_dataset, [val_size, n_tot_eval - val_size])
    adv_attack_valloader = torch.utils.data.DataLoader(adv_attack_valset, batch_size=batch_size,
                                                       shuffle=False, num_workers=n_workers)

    adv_attack_dataloader = adv_attack_valloader

elif eval_set == "cifar10hHoldout" or eval_set == "testSub":
    dataset = hsoft_heldout_set
    dataloader = hsoft_heldout_loader

    # soft label eval w/ held-out 30%
    adv_attack_hsoft_heldout_set = CIFAR10_SHO(transform=pgd_transform,
                                               split=split,
                                               label_method=label_method,
                                               num_annotators_sample=num_annotators_sample,
                                               redist_level=redist_level
                                               )
    adv_attack_peterson_loader = torch.utils.data.DataLoader(adv_attack_hsoft_heldout_set,
                                                             batch_size=batch_size,
                                                             shuffle=True, num_workers=n_workers)

    adv_attack_dataloader = adv_attack_peterson_loader


'''
Load current checkpoint for eval
'''
# load the checkpoint and update device as needed
checkpoint_tag = "ckpt."
checkpoint_name = args.run_name
checkpoint_pth = f"{checkpoint_dir}{checkpoint_tag}{checkpoint_name}"
checkpoint = torch.load(checkpoint_pth, map_location=torch.device(device))
net = checkpoint['net']

# cpu conversion help from:
# https://stackoverflow.com/questions/68551032/is-there-a-way-to-use-torch-nn-dataparallel-with-cpu
net.eval()

if not use_cuda:
    net = net.module.to("cpu")
    net.eval()
else:
    net.eval()

torch.cuda.empty_cache()

'''
Run suite of tasks
'''
# evaluate predictions on eval set
# store the data per point indexed by order for investigation later
# stores the original targets, model preds, and the predicted cross entropy
model_preds = utils.get_model_preds(net, dataloader, num_classes, device)

'''
Get distances between model preds and targets
'''


def get_wassertstein(v1, v2):
    return wasserstein_distance(v1, v2)


l1_dists = []
wass_dists = []
for example_idx, example_data in model_preds.items():
    trgt = example_data["target"].cpu().numpy()[0]
    pred = utils.get_dist(example_data["preds"].cpu().numpy()[0])
    wass_dists.append(get_wassertstein(pred, trgt))
    l1_dists.append(np.sum(np.abs(pred - trgt)))

# for json
l1_dists = [str(x) for x in l1_dists]
wass_dists = [str(x) for x in wass_dists]


'''
Extract per group accuracy, as well as cross-entropy (e.g., Peterson)
'''
per_class_perf = utils.get_per_class_performance(
    model_preds, class_labels, idx2class)

class_accs = {}
class_crossents = {}
class_top2_accs = {}

worst_acc = 1.0
worst_acc_class = None

tot_ce = 0

for class_name, class_data in per_class_perf.items():
    scores = class_data["scores"]
    n_examples = len(scores)
    acc = np.sum(scores) / n_examples
    # print(f"{class_name} Acc: {acc}")
    class_accs[class_name] = (scores, n_examples)

    crossents = class_data["ce"]
    # avg_ce = np.sum(crossents)/n_examples
    tot_ce += np.sum(crossents)

    class_crossents[class_name] = (crossents, n_examples)

    class_top2_accs[class_name] = (class_data["top2scores"], n_examples)

    if acc < worst_acc:
        worst_acc = acc
        worst_acc_class = class_name

print("Worst acc: ", worst_acc_class, " Acc: ", worst_acc)

if save_detailed_stats:
    outfile = f"{eval_stats_dir}class_accs_{checkpoint_name}_{eval_set}.json"
    with open(outfile, "w") as f:
        json.dump(class_accs, f)

    outfile = f"{eval_stats_dir}class_ce_{checkpoint_name}_{eval_set}.json"
    with open(outfile, "w") as f:
        json.dump(class_crossents, f)

print("Avg CE: ", tot_ce/len(model_preds.keys()))

'''
Optionally run adversarial attacks
'''
if not no_adv:
    '''
    Run PGD robustness
    '''
    # PGD attack code and loop modified from: https://github.com/Harry24k/PGD-pytorch/blob/master/PGD.ipynb
    # And torchattacks
    agg_losses = utils.run_pgd_attack_loop(checkpoint['net'], adv_attack_dataloader, idx2class,
                                           n_eval_adv_robustness=args.n_eval_adv_robustness, device=device, num_classes=num_classes)

    if save_detailed_stats:
        outfile = f"{eval_stats_dir}adv_losses_{checkpoint_name}_{eval_set}.npy"
        np.save(outfile, np.array(agg_losses))

    '''
    Run FGSM robustness
    '''
    # FGSM attack code modified from torchattacks repo: https://github.com/Harry24k/PGD-pytorch/blob/master/PGD.ipynb

    fgsm_losses = utils.run_fgsm_attack(checkpoint['net'], adv_attack_dataloader, idx2class,
                                        n_eval_adv_robustness=args.n_eval_adv_robustness, device=device, num_classes=num_classes)
else:
    agg_losses = []
    fgsm_losses = []

'''
Get calibration
Inspired by PixMix and using their calib_err function
'''
if eval_set == "testSub":
    n_bins = 10
else:
    n_bins = 100
calibration_score = utils.compute_calibration(model_preds, n_bins=n_bins)

'''
Get correlation of entropy
Inspired by Uma et al, but custom written here
'''
entropy_corr = utils.compute_entropy_correlation(model_preds)
'''
Save out stats in a single main files
'''
main_stats = {"PGD Loss": agg_losses, "FGSM Loss": fgsm_losses, "Worst Class Acc": [worst_acc, worst_acc_class],
              "CE": class_crossents, "Acc": class_accs, "Calibration": calibration_score, "Entropy Correlation": entropy_corr, "L1 Dist": l1_dists, "Wasserstein Dist": wass_dists,
              "Top-2 Acc": class_top2_accs}
outfile = f"{eval_stats_dir}main_stats_{checkpoint_name}_{eval_set}.json"
with open(outfile, "w") as f:
    json.dump(main_stats, f)
