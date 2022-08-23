# CIFAR-10S 

We introduce `CIFAR-10S`, a dataset of soft labels elicited from individual annotators over the [`CIFAR-10`](https://www.cs.toronto.edu/~kriz/cifar.html) test set (the same images used in [`CIFAR-10H`](https://github.com/jcpeterson/cifar-10h/blob/master/README.md)). At present, we have soft labels collected for 1000 images. Each image have been annotated by at least 6 annotators. This structure of this README is based on the form used by [Peterson et al](https://github.com/jcpeterson/cifar-10h/blob/master/README.md).

## Repository Contents

* `human_soft_label_data.json`: .... 
* `raw_data.zip`: de-anonymized raw annotation information collected during crowdsourcing on [Prolific](https://app.prolific.co/). [Pavlovia](https://pavlovia.org/) was used as a backend. Details on column information are included below. 
* `soft_labels_redist_01.npy`: ..... redist 0.1
* `label_construction_utils.py`: .... 
* We will include a custom dataloader shortly. For the time being, we recommend... 

## Constructing Soft Labels

We recommend...... 

## Raw Data Format

The columns in our raw_data represent: 
* subject: unique id randomly generated for a given annotator.
* response: annotations provided for a given image (most prob class w/ prob, second prob class w/ optional prob, any impossible classes). note, the final page shown to each annotator was a debrief questionarre; for this page, you can see comments to the questions included below. 
* img_id: integer into the original CIFAR-10 ordered test set for the image show.
* label: category assigned to the image according to the [CIFAR-10 test set](https://www.cs.toronto.edu/~kriz/cifar.html).
* filename: readable tag for image shown: "cifar10_train_{img_id}_{img_label}.png" note, these are from the "test" set. we called these "train" because we were training on the labels, but will soon change this tag, and downstream code which uses the filename, to avoid confusion. 
* rt: time spent (msec) on a given page, by an annotator.
* time_elapsed: total time (msec) an annotator has taken on the experiment so far.
* task: indicates the type of the screen shown to the annotator. "spec conf" is the soft label elicitation; "rerun_spec_conf" are repeat trials of earlier soft label elicitations (same screen type). 
* trial_index: the order of the trials/pages the annotator saw.
* condition: batch of images annotator was allocated to.
* trial_type: the [jsPsych](https://www.jspsych.org/6.3/) screen-type shown on a given page.
* view_history: meta-data on annotator viewing instructions.



## Citing

If you use our data, please consider the following bibtex entry: 

```
@article{softLabelElicitingLearning2022,
  title={Eliciting and Learning with Soft Labels from Every Annotator},
  author={Collins, Katherine M and Bhatt, Umang and Weller, Adrian},
  journal={arXiv preprint arXiv:2207.00810},
  year={2022}
}
```
