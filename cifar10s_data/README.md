# CIFAR-10S 

We introduce `CIFAR-10S`, a dataset of soft labels elicited from individual annotators over the [`CIFAR-10`](https://www.cs.toronto.edu/~kriz/cifar.html) test set (the same images used in [`CIFAR-10H`](https://github.com/jcpeterson/cifar-10h/blob/master/README.md)). At present, we have soft labels collected for 1000 images. Each image have been annotated by at least 6 annotators. This structure of this README is based on the form used by [Peterson et al](https://github.com/jcpeterson/cifar-10h/blob/master/README.md).

## Repository Contents

* `human_soft_label_data.json`: .... 
* `raw_data.zip`: de-anonymized raw annotation information collected during crowdsourcing on [Prolific](). [Pavlovia]() was used as a backend. Details on column information are included below. 
* `soft_labels_redist_01.npy`: ..... redist 0.1
* `label_construction_utils.py`: .... 
* We will include a custom dataloader shortly. For the time being, we recommend... 

## Constructing Soft Labels

We recommend...... 

## Raw Data Format

The columns in our raw_data represent: 



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
