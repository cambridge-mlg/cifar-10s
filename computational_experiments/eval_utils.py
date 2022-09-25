'''
Store util functions for evaluation
'''

from ossaudiodev import SNDCTL_COPR_RESET

import numpy as np
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pandas as pd
from netrc import netrc
import os
import torch
from data import CIFARMixHILL
import models
import matplotlib.pylab as plt
import torch.nn.functional as F
import json
from scipy.stats import entropy
from scipy.stats import spearmanr, pearsonr


def get_dist(z):
    '''
    Convert unnormalized vec (z) to prob dist
    Softmax transform
    '''
    if torch.is_tensor(z):
        z = z.detach().cpu().numpy()

    return np.exp(z) / np.sum(np.exp(z))


def get_acc(model_preds):
    correct_counts = 0
    tot = 0
    for _, example_data in model_preds.items():
        correct_counts += bool(torch.argmax(
            example_data["target"]) == torch.argmax(example_data["preds"]))
        tot += 1
    return correct_counts/tot


def get_avg_ce(model_preds):
    ce_sum = 0
    tot = 0
    for _, example_data in model_preds.items():
        ce_sum += float(example_data["ce"])
        tot += 1
    return ce_sum/tot


def cross_entropy(preds, trgts):
    '''
    Cross-entropy over vectors
        preds and trgts of size: [batch_size, num_classes]
    From Peterson: https://github.com/jcpeterson/cifar10-human-experiments/blob/efadada2c1adc6bcdc8d86aca7e542564ff2980e/pytorch_image_classification/utils.py
    Slight modifications for clarity
    Similar to: https://discuss.pytorch.org/t/soft-cross-entropy-loss-tf-has-it-does-pytorch-have-it/69501/2
    And Uma:
    '''

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = "cuda"
    else:
        device = "cpu"

    preds = preds.to(device)
    trgts = trgts.to(device)

    batch_size, num_classes = preds.shape
    if len(trgts.shape) == 1:
        #  https://pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html
        trgts = F.one_hot(trgts, num_classes=num_classes)
        #trgts = onehot(trgts, num_classes, device)

    #  log, then softmax -- see: https://pytorch.org/docs/stable/generated/torch.nn.functional.log_softmax.html
    preds = F.log_softmax(preds, dim=1)
    loss = -torch.sum(preds * trgts)
    return loss / batch_size


# norm from: https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/demos/White%20Box%20Attack%20(ImageNet).ipynb
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std


def get_model_preds(model, dataloader, num_classes=10, device="cuda", clear_cache_step=1000, select_indices=None, max_eval=None, pred_latents=False):
    '''
    Extract model predictions for each example
    Returns the original dataset targets, model preds, and resulting cross-entropy
    Mainly for easy access
    '''
    model_preds = {}
    for idx, (inputs, targets) in enumerate(dataloader):

        # if idx > 20: break
        if max_eval is not None:
            if idx > max_eval:
                continue

        if select_indices is not None and idx not in select_indices:
            continue

        if idx % clear_cache_step == 0:  #  for memory reasons
            torch.cuda.empty_cache()
            # print(f"Currently evaluating example: {idx}")

        # ensure all labels are in K class vector form if not already
        if len(targets.shape) == 1:  #  index tensor(s), e.g., just batch size
            #  https://pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html
            targets = F.one_hot(targets, num_classes=num_classes)

        inputs, targets = Variable(inputs), Variable(targets)
        preds = model(inputs)

        if pred_latents:
            preds, latents = model(inputs)
        else:
            preds = model(inputs)

        loss_val = cross_entropy(preds, targets).item()

        preds = preds.detach()
        targets = targets.detach()

        model_preds[idx] = {"target": targets, "preds": preds, "ce": loss_val}
    return model_preds


def get_per_class_performance(model_preds, class_labels, idx2class):
    '''
    Extract subset of metrics per class
    '''

    # store per-class performance metrics
    per_class_perf = {class_name: {
        "scores": [], "top2scores": [], "ce": []} for class_name in class_labels}

    # get avg acc
    correct = 0
    tot = 0

    top2_correct = 0
    for idx, (example_idx, example_data) in enumerate(model_preds.items()):

        # if idx > 20: break

        trgt = example_data["target"]
        pred = example_data["preds"]

        # print("ce: ", example_data["ce"])
        ce_val = example_data["ce"]  # .cpu().detach().numpy()

        batch_size = trgt.shape[0]

        _, max_pred = torch.max(pred.data, 1)
        _, trgt = torch.max(trgt.data, 1)

        max_prob_class_trgt = trgt.numpy()[0]

        trgt_name = idx2class[max_prob_class_trgt]

        trgt = trgt.cpu()
        max_pred = max_pred.cpu()

        score = max_pred.eq(trgt.data).sum().cpu().detach().item()

        per_class_perf[trgt_name]["scores"].append(score)

        correct += score
        tot += 1  #  assumes batch size 1

        _, pred_class_idxs = torch.topk(pred.data, k=2)
        top2_preds = {x for x in pred_class_idxs.cpu().detach().numpy()[0]}
        if max_prob_class_trgt in top2_preds:
            top2_score = 1
        else:
            top2_score = 0
        top2_correct += top2_score
        per_class_perf[trgt_name]["top2scores"].append(top2_score)

        per_class_perf[trgt_name]["ce"].append(ce_val)

    return per_class_perf


'''
PGD Attack code from the torchattacks repo
https://adversarial-attacks-pytorch.readthedocs.io/en/latest/_modules/torchattacks/attacks/pgd.html#PGD (w/ some modifications)
Modified to return the loss so we can analyze like Peterson's Fig 4 and Table 2
Also considered earlier version: https://github.com/Harry24k/PGD-pytorch/blob/master/PGD.ipynb
'''


def pgd_attack(model, images, labels, eps=0.3, alpha=2/255, iters=10, loss_func=None, random_start=True, device="cuda"):
    print("PGD DEVICE: ", device)
    images = images.to(device)
    labels = labels.to(device)
    if loss_func is None:
        loss_func = nn.CrossEntropyLoss()

    losses = []

    adv_images = images.clone().detach()

    if random_start:
        # Starting at a uniformly random point
        adv_images = adv_images + \
            torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for i in range(iters):
        adv_images.requires_grad = True
        # adv_images = adv_images.to(device)
        outputs = model(adv_images)

        model.zero_grad()
        cost = loss_func(outputs, labels).to(device)
        # cost.backward()

        # print(cost.cpu().detach().data.numpy())
        losses.append(cost.cpu().detach().data.numpy())

        # adv_images = images + alpha*images.grad.sign()
        # eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        # images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

        # Update adversarial images
        grad = torch.autograd.grad(cost, adv_images,
                                   retain_graph=False, create_graph=False)[0]

        adv_images = adv_images.detach() + alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    return images, losses


'''
FGSM Attack code from the torchattacks repo
https://adversarial-attacks-pytorch.readthedocs.io/en/latest/_modules/torchattacks/attacks/fgsm.html (w/ some modifications)
'''


def fgsm_attack(model, images, labels, eps=4/255, loss_func=None, device="cuda"):

    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    if loss_func is None:
        loss_func = nn.CrossEntropyLoss()

    images.requires_grad = True
    outputs = model(images)

    cost = loss_func(outputs, labels).to(device)
    acc_score = torch.eq(outputs, labels)

    # Update adversarial images
    grad = torch.autograd.grad(cost, images,
                               retain_graph=False, create_graph=False)[0]

    adv_images = images + eps*grad.sign()
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    # now run model on generated adv imgs
    outputs = model(adv_images)
    adv_cost = loss_func(outputs, labels).to(device)
    adv_loss = adv_cost.cpu().detach().data.item()
    acc_score = torch.eq(outputs, labels)
    return adv_images, adv_loss


def run_pgd_attack_loop(model, dataloader, idx2class,
                        eps=4/255, pgd_iters=10,
                        n_eval_adv_robustness=1000,
                        device="cuda", num_classes=10):
    # PGD attack code and loop modified from: https://github.com/Harry24k/PGD-pytorch/blob/master/PGD.ipynb
    correct = 0
    total = 0

    # pre norm from: https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/demos/White%20Box%20Attack%20(ImageNet).ipynb
    norm_layer = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
                           0.2023, 0.1994, 0.2010])

    if device == "cpu":
        model = model.module.to("cpu")

    model = nn.Sequential(
        norm_layer,
        model
    ).to(device)

    model.eval()

    agg_losses = []
    criterion = cross_entropy

    for idx, (orig_images, labels) in enumerate(dataloader):
        if idx >= n_eval_adv_robustness:
            break

        batch_size = labels.shape[0]

        # ensure all labels are in K class vector form if not already
        if len(labels.shape) == 1:  #  index tensor(s), e.g., just batch size
            #  https://pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html
            labels = F.one_hot(labels, num_classes=num_classes)

        adv_images, losses = pgd_attack(
            model, orig_images, labels, loss_func=criterion, eps=eps, iters=pgd_iters, device=device)
        labels = labels.to(device)
        outputs = model(adv_images)

        # print("pred shape: ", outputs.data.shape, " labels: ", labels.shape)

        _, pred = torch.max(outputs.data, 1)
        _, labels = torch.max(labels.data, 1)  #  b/c one-hot

        total += batch_size  #  num in batch
        correct += (pred == labels).sum()

        orig = idx2class[labels.cpu().detach().numpy()[0]]
        pred = idx2class[pred.cpu().detach().numpy()[0]]

        #  b/c had been saved as numpy arrays
        agg_losses.append([float(l) for l in losses])

    return agg_losses


def run_fgsm_attack(model, dataloader, idx2class,
                    eps=4/255,
                    n_eval_adv_robustness=1000,
                    device="cuda", num_classes=10):
    correct = 0
    total = 0

    # pre norm from: https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/demos/White%20Box%20Attack%20(ImageNet).ipynb
    norm_layer = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
                           0.2023, 0.1994, 0.2010])

    if device == "cpu":
        model = model.module.to("cpu")

    model = nn.Sequential(
        norm_layer,
        model
    ).to(device)

    model.eval()

    agg_losses = []
    criterion = cross_entropy

    for idx, (orig_images, labels) in enumerate(dataloader):
        if idx >= n_eval_adv_robustness:
            break

        batch_size = labels.shape[0]

        # ensure all labels are in K class vector form if not already
        if len(labels.shape) == 1:  #  index tensor(s), e.g., just batch size
            #  https://pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html
            labels = F.one_hot(labels, num_classes=num_classes)

        adv_images, loss = fgsm_attack(
            model, orig_images, labels, loss_func=criterion, eps=eps, device=device)
        labels = labels.to(device)
        outputs = model(adv_images)

        # print("pred shape: ", outputs.data.shape, " labels: ", labels.shape)

        _, pred = torch.max(outputs.data, 1)
        _, labels = torch.max(labels.data, 1)  #  b/c one-hot

        total += batch_size  #  num in batch
        correct += (pred == labels).sum()

        orig = idx2class[labels.cpu().detach().numpy()[0]]
        pred = idx2class[pred.cpu().detach().numpy()[0]]

        agg_losses.append(loss)  #  b/c had been saved as numpy arrays

    return agg_losses


def compute_calibration(model_preds, p="2", n_bins=100):
    '''
    Measure calibration, inspired by PixMix (see citation for their calib_err function)
    p is the norm they use -- see their paper for details
    Modified for our model pred structure
    '''
    # get the confidence assigned to the true class
    confidence = []
    correct = []  #  0/1
    for idx, pred_info in model_preds.items():
        trgt = pred_info["target"]
        preds = pred_info["preds"]

        _, max_pred = torch.max(preds.data, 1)
        _, trgt = torch.max(trgt.data, 1)

        trgt_name = model_preds[trgt.numpy()[0]]

        trgt = trgt.cpu()
        max_pred = max_pred.cpu()

        score = max_pred.eq(trgt.data).sum().cpu(
        ).detach().item()  #  1 if correct, 0 if wrong

        # convert to dist to get pred prob
        pred_probs = get_dist(preds.data)

        pred_prob = pred_probs[0][max_pred]
        confidence.append(pred_prob)
        correct.append(score)

    confidence = np.array(confidence)
    correct = np.array(correct)

    return calib_err(confidence, correct, p=p, beta=n_bins)


def compute_entropy_correlation(model_preds):
    '''
    Get the correlation of predicted model entropy to label entropy
    Inspired by Uma et al, but custom-written here
    Note, this really only makes sense with soft targets
    '''
    pred_ents = []
    trgt_ents = []  #  0/1
    for idx, pred_info in model_preds.items():
        trgt = pred_info["target"][0].cpu().detach().numpy()
        preds = pred_info["preds"][0].cpu().detach().numpy()
        pred_dist = get_dist(preds)

        pred_ent = entropy(pred_dist)
        trgt_ent = entropy(trgt)

        pred_ents.append(pred_ent)
        trgt_ents.append(trgt_ent)

        # if idx == 0: print(pred_dist, preds, pred_ent, trgt_ent)

    # print(pred_ents, trgt_ents)
    return pearsonr(pred_ents, trgt_ents)


def calib_err(confidence, correct, p='2', beta=100):
    '''
    Calibration error function used in PixMix
    From: https://github.com/andyzoujm/pixmix/blob/ee8af2d53c3c8ec99219c1fb1f76def0d3e9a6d3/calibration_tools.py
    '''
    # beta is target bin size
    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]
    bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
    bins[-1] = [bins[-1][0], len(confidence)]

    cerr = 0
    total_examples = len(confidence)
    for i in range(len(bins) - 1):
        bin_confidence = confidence[bins[i][0]:bins[i][1]]
        bin_correct = correct[bins[i][0]:bins[i][1]]
        num_examples_in_bin = len(bin_confidence)

        if num_examples_in_bin > 0:
            difference = np.abs(np.nanmean(
                bin_confidence) - np.nanmean(bin_correct))

            if p == '2':
                cerr += num_examples_in_bin / \
                    total_examples * np.square(difference)
            elif p == '1':
                cerr += num_examples_in_bin / total_examples * difference
            elif p == 'infty' or p == 'infinity' or p == 'max':
                cerr = np.maximum(cerr, difference)
            else:
                assert False, "p must be '1', '2', or 'infty'"

    if p == '2':
        cerr = np.sqrt(cerr)

    return cerr
