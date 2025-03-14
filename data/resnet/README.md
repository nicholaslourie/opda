ResNet Hyperparameter Tuning Data
=================================
*Hyperparameter tuning data for ResNet models*

*Optimal Design Analysis* (OPDA) offers a framework for better designing
and analyzing deep learning experiments. This directory contains results
from tuning the hyperparameters of
[ResNet](https://arxiv.org/abs/1512.03385) for
[ImageNet](https://arxiv.org/abs/1409.0575) via random search. These
results can be used to apply optimal design analysis to these
problems. For more information, see the code repository,
[`opda`](https://github.com/nicholaslourie/opda).


Data Creation
-------------
The data represent the results from tuning the hyperparameters of
ResNet18 for ImageNet via random search.

For the *tuning* experiment (`resnet18_tuning.results.jsonl`), the
hyperparameters were sampled randomly from the following distribution:

    epochs ~ DiscreteUniform({20, 21, ..., 100})
    batch_size ~ DiscreteUniform({128, 256, 512, 1024})
    lr ~ LogUniform([5e-3, 5e1])
    lr_peak_epoch = floor(proportion * epochs), proportion ~ Uniform([0.0, 0.8])
    momentum ~ Uniform([0.7, 1.0])
    weight_decay ~ LogUniform([1e-6, 1e-3])
    label_smoothing ~ Uniform([0.0, 0.5])
    use_blurpool ~ DiscreteUniform({0, 1})

1,024 hyperparameter configurations were sampled. After each epoch,
the model was evaluated on ImageNet's validation set. These learning
curves are then reported in the data.

For the *residual* experiment (`resnet18_residual.results.jsonl`), the
results were generated from those of the *tuning* experiment as
follows. First, all the hyperparameter configurations from the *tuning*
experiment were sorted by the best top 1 validation accuracy of any of
the training run's checkpoints (with an accuracy of 0 when the training
loss NaN'd before the first checkpoint). Next, the hyperparameter
configurations at the 12.5th, 25th, 37.5th, 50th, 62.5th, 75th, 87.5th,
and 100th quantiles of accuracy (ascending) were selected. Finally, each
of these 8 configurations was then retrained 128 times with different
random seeds, for a total of 1,024 training runs.

For the *ablation* experiment (`resnet18_ablation.results.jsonl`), the
results from the *tuning* experiment along with a few additional trial
runs were used to design a hyperparameter search space that efficiently
captures the loss surface in the neighborhood around the optimum. Some
of the hyperparameters were sampled while others were fixed. At the
start, `epochs`, `batch_size`, `momentum` and `use_blurpool` were fixed:

    epochs = 85
    batch_size = 512
    momentum = 0.85
    use_blurpool = 1

while `lr`, `lr_peak_epoch`, `weight_decay` and `label_smoothing` were
sampled from the following distribution:

    lr ~ LogUniform([2e-1, 2e1])
    lr_peak_epoch = DiscreteUniform({0, 1, 2, ..., 30})
    weight_decay ~ LogUniform([5e-7, 5e-5])
    label_smoothing ~ Uniform([0.0, 0.3])

The sampled hyperparameters were then sequentially fixed (i.e.,
ablated). From iteration 1 to 256 none of the four hyperparameters was
fixed. At iteration 257, `label_smoothing` was fixed:

    label_smoothing = 0.15

At iteration 513, `weight_decay` was fixed:

    weight_decay = 5e-6

At iteration 769, `lr_peak_epoch` was fixed:

    lr_peak_epoch = 15

The `lr` hyperparameter was never fixed. Thus, the first 256 iterations
vary 4 hyperparameters, the next 256 vary 3, the next 256 vary 2, and
the last 256 vary 1.


Data Structure
--------------
The data are encoded in [JSON Lines](https://jsonlines.org)
format. Each line is a JSON object with the following keys and values:

  - **iteration**: the iteration of random search
  - **nan_train_loss**: whether or not the loss became NaN during
    training
  - **epochs**: the number of epochs used for training
  - **batch_size**: the training batch size
  - **lr**: the peak learning rate
  - **lr_peak_epoch**: the epoch (0-indexed) at which the learning
    rate reaches its maximum
  - **momentum**: the momentum for SGD
  - **weight_decay**: the weight decay
  - **label_smoothing**: the mixing weight for label smoothing
  - **use_blurpool**: whether or not to use blur-pooling in place of
    max-pooling
  - **learning_curve**: a list of dictionaries recording metrics from
    after each epoch during training
    - **time**: the number of seconds since the start of training
    - **epoch**: the epoch (0-indexed) after which the measurements
      were taken
    - **top_1**: the top-1 accuracy on the validation set
    - **top_5**: the top-5 accuracy on the validation set
    - **train_loss**: the training loss (cross-entropy)

Some of the files have important additional structure.

In `resnet18_residual.results.jsonl`, each successive block of 128
lines (i.e., lines 1 through 128, 129 through 256, and so on)
represents the same hyperparameter configuration trained with
different random seeds. Moreover, these hyperparameter configurations
are in increasing order of performance. See *Data Creation* for how
the configurations were created.

In `resnet18_ablation.results.jsonl`, each successive block of 256
lines (i.e., lines 1 through 256, 257 through 512, and so on) fixes an
additional hyperparameter, ablating it from the search distribution. See
*Data Creation* for how the configurations were created.


Files
-----
This directory should contain the following files:

  - **LICENSE**: the license for using this data
  - **README.md**: this README file
  - **resnet18_tuning.results.jsonl**: the hyperparameter tuning
    results for ResNet18
  - **resnet18_residual.results.jsonl**: the results from retraining
    ResNet18 hyperparameter configurations with different random seeds
  - **resnet18_ablation.results.jsonl**: the results from ablating
    hyperparameters from the search distribution for ResNet18

See *Data Creation* and *Data Structure* for detailed descriptions of
the results within each file.


Contact
-------
For more information, see the code
repository, [`opda`](https://github.com/nicholaslourie/opda). Questions
and comments may be addressed to Nicholas Lourie.
