'''
Helper functions for processing elicitation data to construct labels
'''

from asyncio import current_task
import os
import json
import numpy as np
import pandas as pd
import random

# same keys as used in constructing the elicitation data obj
# for ease of access and readibility

most_prob_class_txt = "Most Probable Class"
second_prob_class_txt = "Second Most Probable Class"
imposs_txt = "Impossible Class(es)"

most_prob_txt = f"{most_prob_class_txt} Prob"
second_prob_txt = f"{second_prob_class_txt} Prob"

none_option = "No"  # selected if someone specifies no second most prob class


def redist_via_clamp(current_label, poss_classes, class2idx, redist_factor=0.1, smooth_existing=False):
    '''
    Spread mass over the "possible" classes (those not in the clamp)
    Apply this to a work-in-progress, or already constructed, label


    Note, if the tot mass allocated is already >= 1, and some classes are leftover as possible
    Then still spread some mass over these, according to redist factor

    E.g., if someone places 80% prob on deer, and 20% on horse, but leaves dog and cat as also possible
    Then it's sensible that the dog and cat categories are also mentally plausible
    Redist_factor * 100 prob mass is spread across those "possible" categories
    '''

    all_classes = set(class2idx.keys())
    num_classes = len(all_classes)
    # see how many labels to spread over (1 or 2 in current set-up)
    n_current_non_zero = np.sum(current_label > 0.0)
    redist_label = np.zeros([num_classes])
    if len(poss_classes) - n_current_non_zero != 0:
        mass_per = redist_factor / (len(poss_classes) - n_current_non_zero)
        for poss_label in poss_classes:
            # only place if already zero prob (o.w. added)
            if current_label[class2idx[poss_label]] == 0:
                redist_label[class2idx[poss_label]] = mass_per

        # if we're also smoothing our existing labels (e.g., if redist_facstor + amt of mass already > 1), then adjust current label
        if smooth_existing:
            adjusted_prob_labels = np.zeros([num_classes])
            for class_idx, current_label_val in enumerate(current_label):
                if current_label_val != 0:
                    adjusted_prob_labels[class_idx] = current_label_val * \
                        (1-redist_factor)
        else:
            adjusted_prob_labels = current_label  # no change
        adjusted_label = adjusted_prob_labels + \
            redist_label  # add the equiv of a redist masker
        return adjusted_label
    else:
        return current_label  # no spreading, since all others deemed impossible


def apply_uniform_redist(current_label, redist_factor=0.1):
    '''
    Redist any leftover probability to all remaining categories uniformly
    Assumes no access to Clamp information
    '''
    # check how many classes already have mass on them (e.g., are non-zero)
    n_current_non_zero = np.sum(current_label > 0.0)
    num_classes = len(current_label)

    if num_classes == n_current_non_zero:
        # all mass has already been assigned
        # this isn't classical label smoothing, just for *redistribution* of any missing mass
        # so just return label -- no need to redist any extra mass b/c all accounted for
        return current_label
    else:
        # spread any extra mass uniformly over all categories that have no prob currently assigned
        unif_spread = redist_factor/(num_classes-n_current_non_zero)

        smoothed_label = np.zeros([num_classes])

        for label_id, current_prob in enumerate(current_label):
            # if no mass currently, place the uniform spread-ed amt of mass on it
            if current_prob == 0:
                smoothed_label[label_id] = unif_spread
            else:
                smoothed_label[label_id] = current_prob
        return smoothed_label


def construct_elicited_soft_label(elicited_data, class2idx, idx2class, include_top_2=True, redist="clamp", redist_factor=0.1):
    '''
    Takes in a participant's elicitation data (in the form of a dict)
    Creates a labeling either using both most prob and second most prob (if "include_top_2" is True)
        O.w., just use most prob info
    With optional redistribution
    If "clamp" -- spread over any labels not listed in "impossible"
    If "uniform" -- spread uniformly over the remaining labels
    If "none" -- no smoothing
    If smoothing is applied, smooth based on redist_factor
    But, only apply this redistribution based on any leftover mass is "smooth_existing" is False
    If true, also dampen the mass on the top 2
    Note, class2idx specifies a mapping of all possible class names
        to their ordered id for label construction
    '''
    # save imposs classes
    imposs_classes = elicited_data[imposs_txt]

    # form label
    all_classes = set(class2idx.keys())
    num_classes = len(all_classes)
    subj_label = np.zeros([num_classes])
    most_prob_class = elicited_data[most_prob_class_txt]
    prob = elicited_data[most_prob_txt]

    subj_label[class2idx[most_prob_class]] = prob

    if include_top_2:
        second_prob_class = elicited_data[second_prob_class_txt]
        second_prob = elicited_data[second_prob_txt]

        if second_prob_class is not none_option and second_prob is not None:
            subj_label[class2idx[second_prob_class]] = second_prob

    subj_label = subj_label/100  # b/c probs were specified \in [0,100]

    # determine which classes are leftover as "possible"
    all_classes = set(class2idx.keys())
    poss_classes = all_classes.difference(imposs_classes)

    # see how much mass has been allocated so far
    '''
    Determine how much mass has been allocated already
    If an annotator has only specified X amt of mass (out of 1), set "redist_factor" to the remaining
        this is used to determine how much mass to spread either over the categories deemed probable (in the "clamp" category)
        or over all other categories unifomly ("uniform" category)
    If an annotator has spread mass that is >= 1,
        then we only still allot some mass to possible labels in the "clamp" case
        here, "redist_factor" uses the value passed to this function (which could have been tuned)
        note, only applies in the "clamp" case. in uniform, no spreading done if mass >= 1
            b/c we don't have access to any other "poss"
    '''
    allocated_mass = np.sum(subj_label)
    if allocated_mass < 1:
        redist_factor = 1-allocated_mass
        smooth_existing = False
    else:
        # make sure sums to 1 (some people specified "probs" that sum > 1)
        subj_label = subj_label/np.sum(subj_label)
        # only consider still doing any form of smoothing in this case if we're in "clamp"
        # and have some "possible" labels left -- which suggests some prob on those
        # in other settings, we assume no access to said imposs class annotations
        # see how many labels to spread over (1 or 2 in current set-up)
        n_leftover_spread = len(poss_classes) - np.sum(subj_label > 0.0)
        if ("clamp" in redist or "Clamp" in redist) and n_leftover_spread != 0:
            redist_factor = redist_factor
            smooth_existing = True
        else:
            # b/c we have no info on other classes, or no other classes to spread over, e.g., in uniform case
            redist_factor = 0
            smooth_existing = False

    if redist == "clamp":
        # spread mass over any classes not in the imposs class
        subj_label = redist_via_clamp(subj_label, poss_classes, class2idx,
                                      redist_factor=redist_factor, smooth_existing=smooth_existing)
    elif redist == "uniform":
        # spread mass uniformly over all other classes
        subj_label = apply_uniform_redist(
            subj_label, redist_factor=redist_factor)

    # ensure we sum to 1 in case we do not at this stage
    subj_label = subj_label/np.sum(subj_label)

    return subj_label


def create_cifar10h_sim2(elicited_data, class2idx):
    '''
    Simulate a CIFAR-10H label, but as if the annotator could have selected the top 2
    But with no associated probs
    E.g., spread mass over both labels (50/50)
    For now, we do this without any smoothing to most simulate the per-annotator "hard" labels
    '''
    # form label
    poss_classes = set(class2idx.keys())
    num_classes = len(poss_classes)
    most_prob_class = elicited_data[most_prob_class_txt]
    second_prob_class = elicited_data[second_prob_class_txt]

    # create a label that places mass equally on top 2 to simulate if the annotator could just "select"
    # e.g., no probs
    top2_select_subj_label = np.zeros([num_classes])
    if second_prob_class in class2idx:
        top2_select_subj_label[class2idx[second_prob_class]] = 0.5
        top2_select_subj_label[class2idx[most_prob_class]] = 0.5
    else:
        top2_select_subj_label[class2idx[most_prob_class]] = 1.0
    return top2_select_subj_label


def create_smoothed_label(hard_label_class, num_classes=10, smoothing_factor=0.1):
    '''
    Construct a smoothed label from a provided hard label class
    '''
    hard_label = np.zeros([num_classes])
    hard_label[hard_label_class] = 1.0
    # smoothed_label = np.ones([self.num_classes]) * (ls_alpha/(self.num_classes - 1)) # apply smoother to all classes which are not the hard label class
    # smoothed_label[hard_label] = 1.0 * (1-ls_alpha)
    smoothed_label = hard_label * \
        (1-smoothing_factor) + np.ones([num_classes]) * smoothing_factor
    return smoothed_label
