Llama Hyperparameter Tuning Data
================================
*Hyperparameter tuning data for Llama models*

*Optimal Design Analysis* (OPDA) offers a framework for better designing
and analyzing deep learning experiments. This directory contains results
from tuning the hyperparameters of small
[Llama](https://arxiv.org/abs/2302.13971) models on a subset of
[SlimPajama](https://www.cerebras.net/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama) via
random search. These results can be used to apply optimal design
analysis to these problems. For more information, see the code
repository, [`opda`](https://github.com/nicholaslourie/opda).


Data Creation
-------------
The data represent the results from tuning the hyperparameters of small
Llama models for SlimPajama via random search.

All Llama models used a vocabulary size of 50,304 and a sequence
length of 512. Other scaling parameters were set based on the desired
overall size:

| Parameters | `n_embd` | `n_layer` | `n_head` |
| ---------- | -------- | --------- | -------- |
|        33M |      384 |         8 |        6 |
|        53M |      512 |         8 |        8 |
|        60M |      512 |        10 |        8 |
|        93M |      640 |        12 |       10 |
|       124M |      768 |        12 |       12 |
|       151M |      768 |        16 |       12 |
|       210M |      768 |        24 |       12 |
|       360M |     1024 |        24 |       16 |

In addition, metrics are reported for both the model and the model with
the weights averaged every 5th step over the last 500 steps.

For the *tuning* experiment (`llama-33m_tuning.results.jsonl`), the
models all had 33M parameters, used a WSD learning rate schedule that
linearly decayed the learning rate to 0 over the last 20% of training,
and were trained for 10,000 steps. The hyperparameters were sampled
randomly from the following distribution:

    lr ~ LogUniform([1e-5, 1e-1])
    beta1 ~ Uniform([0.7, 1.0])
    beta2 ~ Uniform([0.8, 1.0])
    warmup_steps ~ DiscreteUniform({0, 1, ..., 3000})
    weight_decay ~ LogUniform([1e-4, 1e0])
    dropout ~ Uniform([0.0, 0.1])

1,024 hyperparameter configurations were sampled. The model was
evaluated on separate validation and test sets every 500 gradient
updates. These learning curves are then reported in the data.

For the *residual* experiment (`llama-33m_residual.results.jsonl`), the
results were generated from those of the *tuning* experiment as
follows. First, all the hyperparameter configurations from the *tuning*
experiment were sorted by the best raw validation loss (`val_raw_loss`)
logged at any point during training (with any NaNs or missing losses
being replaced by infinity). Next, the hyperparameter configurations at
the 12.5th, 25th, 37.5th, 50th, 62.5th, 75th, 87.5th, and 100th
quantiles of loss (descending) were selected. Finally, each of these 8
configurations was then retrained 128 times with different random seeds,
for a total of 1,024 training runs.


Data Structure
--------------
The data are encoded in [JSON Lines](https://jsonlines.org)
format. Each line is a JSON object with the following keys and values:

  - **iteration**: the iteration of random search
  - **nan_train_loss**: whether or not the loss became NaN during
    training
  - **hyperparameters**: the hyperparameters for that training run
    - **n_embd**: the hidden dimension for the transformer MLPs
    - **n_layer**: the number of transformer layers
    - **n_head**: the number of attention heads
    - **sequence_length**: the sequence length
    - **total_steps**: the total number of training steps
    - **warmup_steps**: the number of training steps for learning rate
       warmup
    - **lr**: the (maximum) learning rate
    - **beta1**: the beta1 hyperparameter for the adam optimizer
    - **beta2**: the beta2 hyperparameter for the adam optimizer
    - **weight_decay**: the weight decay
    - **dropout**: the dropout rate
  - **learning_curve**: a list of dictionaries recording metrics after
    every 500 steps during training
    - **step**: the (1-indexed) training step
    - **tokens**: the number of tokens seen in training so far
    - **lr**: the current learning rate
    - **train_end_time**: the number of seconds since the start of
      training when all the training updates ended for this step
    - **eval_end_time**: the number of seconds since the start of
      training when all evaluation ended for this step
    - **train_raw_loss**: the per-token cross-entropy on the training
      set (estimated with the most recent batch)
    - **val_raw_loss**: the per-token cross-entropy on the validation set
    - **val_raw_accuracy**: the per-token accuracy on the validation set
    - **val_weight_averaged_loss**: the per-token cross-entropy of the
      weight averaged model on the validation set
    - **val_weight_averaged_accuracy**: the per-token accuracy of the
      weight averaged model on the validation set
    - **test_raw_loss**: the per-token cross-entropy on the test set
    - **test_raw_accuracy**: the per-token accuracy on the test set
    - **test_weight_averaged_loss**: the per-token cross-entropy of the
      weight averaged model on the test set
    - **test_weight_averaged_accuracy**: the per-token accuracy of the
      weight averaged model on the test set

Some of the files have important additional structure.

In `llama-33m_residual.results.jsonl`, each successive block of 128
lines (i.e., lines 1 through 128, 129 through 256, and so on) represents
the same hyperparameter configuration trained with different random
seeds. Moreover, these hyperparameter configurations are in decreasing
order of the loss. See *Data Creation* for how the configurations were
created.


Files
-----
This directory should contain the following files:

  - **LICENSE**: the license for using this data
  - **README.md**: this README file
  - **llama-33m_tuning.results.jsonl**: the hyperparameter tuning
    results for Llama 33M
  - **llama-33m_residual.results.jsonl**: the results from retraining
    Llama hyperparameter configurations with different random seeds

See *Data Creation* and *Data Structure* for detailed descriptions of
the results within each file.


Contact
-------
For more information, see the code
repository, [`opda`](https://github.com/nicholaslourie/opda). Questions
and comments may be addressed to Nicholas Lourie.
