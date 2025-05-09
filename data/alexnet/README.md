AlexNet Hyperparameter Tuning Data
==================================
*Hyperparameter tuning data for AlexNet models*

*Optimal Design Analysis* (OPDA) offers a framework for better designing
and analyzing deep learning experiments. This directory contains results
from tuning the hyperparameters of
[AlexNet](https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) for
[ImageNet](https://arxiv.org/abs/1409.0575) via random search. These
results can be used to apply optimal design analysis to these
problems. For more information, see the code repository,
[`opda`](https://github.com/nicholaslourie/opda).


Data Creation
-------------
The data represent the results from tuning the hyperparameters of
AlexNet for ImageNet via random search.

For the *tuning* experiment (`alexnet_tuning.results.jsonl`), the
hyperparameters were sampled randomly from the following distribution:

    epochs ~ DiscreteUniform({20, 21, ..., 100})
    batch_size ~ DiscreteUniform({128, 256, 512, 1024})
    lr ~ LogUniform([5e-3, 5e1])
    lr_peak_epoch = floor(proportion * epochs), proportion ~ Uniform([0.0, 0.8])
    momentum ~ Uniform([0.7, 1.0])
    weight_decay ~ LogUniform([1e-6, 1e-3])
    label_smoothing ~ Uniform([0.0, 0.5])

512 hyperparameter configurations were sampled. After each epoch, the
model was evaluated on ImageNet's validation set. These learning curves
are then reported in the data.


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
  - **learning_curve**: a list of dictionaries recording metrics from
    after each epoch during training
    - **time**: the number of seconds since the start of training
    - **epoch**: the epoch (0-indexed) after which the measurements
      were taken
    - **top_1**: the top-1 accuracy on the validation set
    - **top_5**: the top-5 accuracy on the validation set
    - **train_loss**: the training loss (cross-entropy)


Files
-----
This directory should contain the following files:

  - **LICENSE**: the license for using this data
  - **README.md**: this README file
  - **alexnet_tuning.results.jsonl**: the hyperparameter tuning
    results for AlexNet

See *Data Creation* and *Data Structure* for detailed descriptions of
the results within each file.


Contact
-------
For more information, see the code
repository, [`opda`](https://github.com/nicholaslourie/opda). Questions
and comments may be addressed to Nicholas Lourie.
