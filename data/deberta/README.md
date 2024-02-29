DeBERTa Hyperparameter Tuning Data
==================================
*Hyperparameter tuning data for DeBERTa and DeBERTaV3*

*Optimal Design Analysis* (OPDA) offers a framework for better designing
and analyzing deep learning experiments. This directory contains results
from tuning the hyperparameters of [DeBERTa](https://arxiv.org/abs/2006.03654)
and [DeBERTaV3](https://arxiv.org/abs/2111.09543)
for [MultiNLI](https://arxiv.org/abs/1704.05426) via random
search. These results can be used to apply optimal design analysis to
these problem. For more information, see the code
repository, [`opda`](https://github.com/nicholaslourie/opda), or the
paper that introduced this data,
["Show Your Work with Confidence: Confidence Bands for Tuning Curves"](https://arxiv.org/abs/2311.09480).

To cite this data or other aspects of the paper, please use:

    @misc{lourie2023work,
        title={Show Your Work with Confidence: Confidence Bands for Tuning Curves},
        author={Nicholas Lourie and Kyunghyun Cho and He He},
        year={2023},
        eprint={2311.09480},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }


Data Creation
-------------
The data represent the results from tuning the hyperparameters of
DeBERTa and DeBERTaV3 for MultiNLI via random search. For both models,
hyperparameters were sampled randomly from the following distribution:

    num_train_epochs ~ DiscreteUniform({1, 2, 3, 4})
    train_batch_size ~ DiscreteUniform({16, 17, ..., 64})
    learning_rate ~ LogUniform([10^-6, 10^-3])
    warmup_proportion ~ Uniform([0.0, 0.6])
    cls_drop_out ~ Uniform([0.0, 0.3])

Where `warmup_proportion` is the proportion of the first epoch to use
for learning rate warmup. In the released data, this hyperparameter is
instead expressed as `warmup_steps`, or the number of optimization
steps used for warmup.

For each model, 1,024 hyperparameter configurations were
sampled. We evaluated the model on both of MultiNLI's validation sets
(matched and mismatched) every 1,000 training steps and at the end of
training. These learning curves are then reported in the data.


Data Structure
--------------
The data are encoded in [JSON Lines](https://jsonlines.org)
format. Each line is a JSON object with the following keys and values:

  - **iteration**: the iteration of random search
  - **num_train_epochs**: the number of epochs used for training
  - **train_batch_size**: the training batch size
  - **total_model_steps**: the total number of optimization steps
  - **learning_rate**: the learning rate
  - **warmup_steps**: the number of training steps for learning rate warmup
  - **cls_drop_out**: the dropout rate
  - **learning_curves**: a dictionary mapping MultiNLI's validation sets to the
    model's learning curves on them
    - **matched**: a list of [step, accuracy] pairs recorded on MultiNLI's
      matched validation set during training
    - **mismatched**: a list of [step, accuracy] pairs recorded on MultiNLI's
      mismatched validation set during training


Files
-----
This directory should contain the following files:

  - **LICENSE**: the license for using this data
  - **README.md**: this README file
  - **deberta-base.results.jsonl**: the hyperparameter tuning results for
    DeBERTa base
  - **deberta-v3-base.results.jsonl**: the hyperparameter tuning results for
    DeBERTaV3 base


Citation
--------
If you use the DeBERTa Hyperparameter Tuning Data, please cite:

    @misc{lourie2023work,
        title={Show Your Work with Confidence: Confidence Bands for Tuning Curves},
        author={Nicholas Lourie and Kyunghyun Cho and He He},
        year={2023},
        eprint={2311.09480},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }


Contact
-------
For more information, see the code
repository, [`opda`](https://github.com/nicholaslourie/opda), or the
paper:
["Show Your Work with Confidence: Confidence Bands for Tuning Curves"](https://arxiv.org/abs/2311.09480). Questions
and comments may be addressed to Nicholas Lourie.
