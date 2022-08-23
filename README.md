README
This is the official codebase for our 2022 HCOMP paper:

# Eliciting and Learning with Soft Labels from Every Annotator

Katherine M. Collins*, Umang Bhatt*, and Adrian Weller

*Contributed equally

Paper: https://arxiv.org/abs/2207.00810 

Project Page: TBD

## Abstract

The labels used to train machine learning (ML) models are of paramount importance. Typically for ML classification tasks, datasets contain hard labels, yet learning using soft labels has been shown to yield benefits for model generalization, robustness, and calibration. Earlier work found success in forming soft labels from multiple annotators' hard labels; however, this approach may not converge to the best labels and necessitates many annotators, which can be expensive and inefficient. We focus on efficiently eliciting soft labels from individual annotators. We collect and release a dataset of soft labels for CIFAR-10 via a crowdsourcing study (N=248). We demonstrate that learning with our labels achieves comparable model performance to prior approaches while requiring far fewer annotators. Our elicitation methodology therefore shows promise towards enabling practitioners to enjoy the benefits of improved model performance and reliability with fewer annotators, and serves as a guide for future dataset curators on the benefits of leveraging richer information, such as categorical uncertainty, from individual annotators.

## Repository Details 

### Data

The `CIFAR-10S` dataset is included in the `cifar10s_data` directory. Details are included in the README in the directory.

We download the `CIFAR-10H` data from the authors' fantastic [repository](https://github.com/jcpeterson/cifar-10h). 

### Experimental Data

Code to run the experiments included in the paper will be added shortly.

## Citing

If citing us, please consider the following bibtex entry: 

```
@article{softLabelElicitingLearning2022,
  title={Eliciting and Learning with Soft Labels from Every Annotator},
  author={Collins, Katherine M and Bhatt, Umang and Weller, Adrian},
  journal={arXiv preprint arXiv:2207.00810},
  year={2022}
}
```


