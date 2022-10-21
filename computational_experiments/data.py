

from scipy.stats import entropy, beta
import random
import pickle
import torch.nn.functional as F
import json
from torch.autograd import Variable
import torchvision.datasets as datasets
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import numpy as np
from typing import List
from turtle import home
from shutil import unregister_archive_format
from cProfile import label
# path modification help from: https://stackoverflow.com/questions/24868733/how-to-access-a-module-from-outside-your-file-folder-in-python
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..', 'cifar10s')))
from cifar10s_data.label_construction_utils import construct_elicited_soft_label, create_cifar10h_sim2, create_smoothed_label


class CIFAR10_SHO(Dataset):
    """
    Data loader that can be used for CIFAR-10S, CIFAR-10H, or the original CIFAR-10 (O)
    CIFAR-10H data from Peterson et al: https://github.com/jcpeterson/cifar-10h
    modified dataloader from example: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __init__(self, cifar10h_data_pth="../other_data/cifar10h-counts.npy", img_dir='../other_data',
                 transform=None, split_idx_pth="../other_data/", split="train",
                 label_method="cifar10h", cifar10s_data_pth="../cifar10s_data/human_soft_labels_data.json",
                 use_all_cifar10h=False, num_annotators_sample=-1, annotator_subsample_seed=0,
                 redist_level=0.1, data_split_seed=7, num_examples_holdout=100,
                 use_per_annotator=False, ls_smooth_amt=0.01
                 ):
        """
        Args:
            cifar10h_data_pth (string): Path to file with CIFAR-10H data from Peterson/Battleday et al
            cifar10s_data_pth (string): Path to file with CIFAR-10S data from our 2022 HCOMP paper
            img_dir (string): Pth to directory with CIFAR-10 (test) images
            transform (callable): Optional transform to be applied
                on image.
            split_idx_pth (string): Path to where csvs are saved with train/test pre-split indices
            use_all_cifar10h (bool): set to True when we want to use the CIFAR-10H labels for the entire select
                i.e., not just for the examples we have CIFAR-10S labels on (for parity)
            num_annotators_sample (int): number of annotators to subsample
                i.e., if there are M total annotators for an example, keep at most num_annotators_sample of them
            annotator_subsample_seed (int): seed for the M annotator subsampling
            data_split_seed (int): seed for selecting soft labels to holdout
            split (string): what subset of the data to use
                if "Sub" in split, holdout some of the soft labels for use, e.g., as a CIFAR-10S test set
            redist_level (float): how to spread any leftover mass (see our 2022 HCOMP paper for details)
            use_per_annotator (bool): whether to sample a single annotators' label per batch
                if False, (naively) aggregate across annotators to form label
            num_examples_holdout (int): if subsampling ("Sub" in split), number of examples to hold out
            ls_smooth_amt (float): how much smoothing to apply, if in label smoothing (LS) mode
            label_method (string): the kind of labeling to use
                for instance, oursTop2Clamp = CIFAR-10S Top 2 w/ prob + clamp
                and cifar10h = use CIFAR-10H
            modified from example: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        """
        # shape = [num_data_points, num_classes], e.g. [10,000, 10]
        self.human_label_counts = np.load(cifar10h_data_pth)

        self.use_per_annotator = use_per_annotator

        # read in our soft labels and use to replace any of the one hot
        with open(cifar10s_data_pth, "r") as f:
            self.elicitation_data = json.load(f)

        # extract from our elicitation data the indexes of the images queried
        self.relabed_idxs = set([int(example_idx)
                                for example_idx in self.elicitation_data.keys()])
        self.img_dir = img_dir
        if img_dir is None:
            # re-download cifar10 test set
            download_orig_cifar10 = True
            img_dir = "~/data"  # save to
        else:
            download_orig_cifar10 = False

        self.img_dataset = datasets.CIFAR10(root=img_dir, train=False, download=download_orig_cifar10,
                                            transform=transform)

        data_size = len(self.img_dataset)
        # modified from Uma et al: https://github.com/AlexandraUma/dali-learning-with-disagreement/blob/main/iccifar10/soft_loss.py
        self.imgs = [self.img_dataset[i][0] for i in range(data_size)]
        self.orig_hard_labels = [self.img_dataset[i][1]
                                 for i in range(data_size)]

        # extract names of classes
        self.classes = self.get_classes()
        # map from names of classes to indices
        self.class2idx = {class_name: idx for idx,
                          class_name in enumerate(self.classes)}
        self.idx2class = {idx: class_name for idx,
                          class_name in enumerate(self.classes)}
        self.num_classes = len(self.classes)

        np.random.seed(annotator_subsample_seed)
        random.seed(annotator_subsample_seed)

        self.labels = self.orig_hard_labels

        self.transform = transform

        # keep track of individ annotator labels if using on our the humna-based soft label methods
        self.individ_annotator_labels = {}

        self.label_method = label_method

        if "ours" in label_method:

            # extract properties of transform from label method
            if "Top2" in label_method:
                include_top_2 = True
            else:
                include_top_2 = False

            #  the tag of the label / redist method should be all lowercase
            redist = label_method.lower()

            for example_idx, elicited_info in self.elicitation_data.items():
                example_idx = int(example_idx)

                if example_idx not in self.relabed_idxs:
                    continue

                n_tot_annotators = len(elicited_info)
                # possibly downsample number of annotators
                # if num_annotators_sample != -1 or num_annotators_sample < n_tot_annotators:
                if num_annotators_sample != -1 and num_annotators_sample < n_tot_annotators:
                    elicited_annotator_info = np.array(elicited_info)[np.random.choice(
                        list(range(n_tot_annotators)), num_annotators_sample, replace=False)]
                else:
                    elicited_annotator_info = np.array(
                        elicited_info)  # take all
                # transform into labels, and take average to aggregate (for now) if M > 1
                annotator_labels = []
                for single_annotator_info in elicited_annotator_info:
                    if not "SimSelect2" in label_method:
                        annotator_label = construct_elicited_soft_label(single_annotator_info, self.class2idx, self.idx2class, include_top_2=include_top_2, redist=redist,
                                                                        redist_factor=redist_level)
                    else:
                        # run the top 2 simulated cifar-10h select
                        annotator_label = create_cifar10h_sim2(
                            single_annotator_info, self.class2idx)

                    if "Smooth" in label_method:
                        annotator_label = annotator_label * \
                            (1-ls_smooth_amt) + \
                            np.ones([self.num_classes]) * ((1/self.num_classes) * ls_smooth_amt)
                    annotator_labels.append(annotator_label)
                annotator_labels = np.array(annotator_labels)
                label = np.mean(annotator_labels, axis=0)
                self.labels[example_idx] = label
                self.individ_annotator_labels[example_idx] = annotator_labels

        elif "cifar10h" in label_method:
            # convert counts into labels -- optionally subsample annotators
            for example_idx, counts in enumerate(self.human_label_counts):
                if example_idx in self.relabed_idxs or use_all_cifar10h:  # if use all, just replace for every label
                    n_tot_annotators = np.sum(counts)
                    # extract the "original" annotators' labels (we know that they're just the hard labels)
                    # then sample from this
                    annotator_labels = []
                    for label_id, num_ann_pred in enumerate(counts):
                        for _ in range(num_ann_pred):
                            annotator_label = np.zeros([self.num_classes])
                            annotator_label[label_id] = 1.0
                            annotator_labels.append(annotator_label)
                    # check if we're subsampling (if -1, use all annotations)
                    if num_annotators_sample != -1:
                        # subsample and optionally aggregate if num_annotators_sample > 1
                        sampled_annotator_idxs = np.random.choice(
                            list(range(len(annotator_labels))), num_annotators_sample, replace=False)
                        sampled_annotators = np.array(annotator_labels)[
                                                      sampled_annotator_idxs]
                        label = np.mean(sampled_annotators, axis=0)
                    else:
                        # use all annotations
                        # label = counts / n_tot_annotators
                        sampled_annotators = np.array(
                            annotator_labels)  # just all annotators
                        label = np.mean(sampled_annotators, axis=0)

                    if "Smooth" in label_method:
                        label = label * (1-ls_smooth_amt) + \
                                         np.ones([self.num_classes]) * \
                                                 ls_smooth_amt
                    self.labels[example_idx] = label
                    self.individ_annotator_labels[example_idx] = sampled_annotators

        elif label_method == "uniform":
            for example_idx, elicited_info in self.elicitation_data.items():
                example_idx = int(example_idx)
                self.labels[example_idx] = np.ones(
                    [self.num_classes]) * 1/self.num_classes

        elif label_method == "random":
            for example_idx, elicited_info in self.elicitation_data.items():
                example_idx = int(example_idx)
                # help from: https://stackoverflow.com/questions/18659858/generating-a-list-of-random-numbers-summing-to-1
                rand_vec = np.array([random.random()
                                    for i in range(self.num_classes)])
                rand_vec = rand_vec/np.sum(rand_vec)
                self.labels[example_idx] = rand_vec

        elif label_method == "labelSmooth":
            for example_idx, elicited_info in self.elicitation_data.items():
                example_idx = int(example_idx)
                hard_label_class = self.orig_hard_labels[example_idx]
                smoothed_label = create_smoothed_label(
                    hard_label_class, num_classes=self.num_classes, smoothing_factor=ls_smooth_amt)
                self.labels[example_idx] = smoothed_label

        elif label_method == "baseline":
            for example_idx, elicited_info in self.elicitation_data.items():
                example_idx = int(example_idx)
                self.labels[example_idx] = self.img_dataset[example_idx][1]

        '''
        optionally use our train/test split, like Uma et al
            https://github.com/AlexandraUma/dali-learning-with-disagreement/blob/main/iccifar10/soft_loss.py
        with the ability to hold out examples
            we want to hold out examples to be able to evaluate on some of CIFAR10S
            if "Sub" is in the split, we hold out some ("Sub" for "Subset")
        '''

        if split_idx_pth is not None and split is not None:
            train_indices = np.load(
                f"{split_idx_pth}/train_indices.npy").tolist()
            test_indices = np.load(f"{split_idx_pth}/test_indices.npy")

            if "Sub" in split:
                # set seed for reproducibility of split
                np.random.seed(data_split_seed)
                random.seed(data_split_seed)
                holdout = random.sample(
                    list(self.relabed_idxs), num_examples_holdout)
                if "train" in split:
                    indices = [
                        idx for idx in train_indices if idx not in holdout]
                else:
                    # just use the heldout, e.g., as a CIFAR-10S test set
                    indices = list(holdout)
            elif split == "train":
                indices = train_indices
            else:
                indices = test_indices  # the full 3k test set atm

            # non-numpy filtering help from:
            # self.imgs = [i for (i, v) in zip(list_a, filter) if v]
            self.imgs = list(np.array(self.imgs)[indices])
            self.labels = list(np.array(self.labels)[indices])
            self.indices = indices
        else:
            self.indices = list(range(0, len(self.imgs)))

        # map from the sampled indices back to the "original" indices
        self.index_converter = {
            new_idx: orig_idx for new_idx, orig_idx in enumerate(indices)}

    def __len__(self):
        '''
        Return num data points
        '''
        return len(self.imgs)  # num data points

    def get_class_per_idx(self):
        '''
        Get semantically meaningful class labels associated with each example index
        Returns a map of {CIFAR-10/H example idx: associated label (txt), ...}
        '''
        idx2class = {v: k.capitalize()
                     for k, v in self.img_dataset.class_to_idx.items()}
        return idx2class

    def get_classes(self):
        '''
        Return semantically meaningful class labels
        As an (ordered) list
        '''
        return [class_name.capitalize() for class_name in self.img_dataset.classes]

    def __getitem__(self, idx):
        '''
        Combine the imgs from cifar10 loader with human generated soft labels
        Assumes these images have transform applied (per cifar10-dataloader init)
        Returns img with target
        '''

        img = self.imgs[idx]

        if self.use_per_annotator:
            # sample an annotator label and use
            # need to convert from this new index to the "original"
            # in order to extract the matched label (todo: refactor)
            converted_idx = self.index_converter[idx]
            if converted_idx not in self.individ_annotator_labels:
                # this means it is one of the examples we don't have soft labels for, so use hard
                label = torch.tensor(self.labels[idx])
            else:
                # sample an annotator's label
                individ_labels = self.individ_annotator_labels[converted_idx]
                label = torch.tensor(random.choice(individ_labels))
        else:
            # otherwise, use the agg label
            label = torch.tensor(self.labels[idx])

        # ensure all labels are in K class vector form if not already
        if len(label.shape) == 0:  # index tensor
            # https://pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html
            label = F.one_hot(label, num_classes=self.num_classes)

        # handle type conversions
        # help from: https://stackoverflow.com/questions/56741087/how-to-fix-runtimeerror-expected-object-of-scalar-type-float-but-got-scalar-typ
        # note: we used np.float32 for the results reported in the paper --- but change in the release to float64 for stability
        label = torch.FloatTensor(label.detach().numpy().astype(np.float64))

        return (img, label)
        return (img, label)
