========
Examples
========
The following examples show how to solve common tasks using opda.

To dive deeper, checkout :doc:`Usage </tutorial/usage>` or the :doc:`API
Reference </reference/opda>`.


Compare Models
==============
Let's compare two models while accounting for hyperparameter tuning
effort. We could compare a new model against a baseline or perhaps
against an ablation of some component. Either way, the comparison works
pretty much the same:

.. plot::
   :format: doctest
   :context:

   >>> from matplotlib import pyplot as plt
   >>> import numpy as np
   >>>
   >>> from opda.nonparametric import EmpiricalDistribution
   >>>
   >>> generator = np.random.default_rng(0)  # Set the random seed.
   >>> uniform = generator.uniform
   >>>
   >>> # Simulate results from random search.
   >>> ys0 = uniform(0.70, 0.90, size=48)  # scores from model 0
   >>> ys1 = uniform(0.75, 0.95, size=48)  # scores from model 1
   >>>
   >>> # Plot tuning curves and confidence bands for both models in one figure.
   >>> fig, ax = plt.subplots()
   >>> ns = np.linspace(1, 10, num=1_000)
   >>> for name, ys in [("baseline", ys0), ("model", ys1)]:
   ...   # Construct the confidence bands.
   ...   dist_lo, dist_pt, dist_hi = EmpiricalDistribution.confidence_bands(
   ...     ys=ys,            # accuracy results from random search
   ...     confidence=0.80,  # confidence level
   ...     a=0.,             # (optional) lower bound on accuracy
   ...     b=1.,             # (optional) upper bound on accuracy
   ...   )
   ...   # Plot the tuning curve.
   ...   ax.plot(ns, dist_pt.quantile_tuning_curve(ns), label=name)
   ...   # Plot the confidence bands.
   ...   ax.fill_between(
   ...     ns,
   ...     dist_hi.quantile_tuning_curve(ns),
   ...     dist_lo.quantile_tuning_curve(ns),
   ...     alpha=0.275,
   ...     label="80% confidence",
   ...   )
   ...   # Format the plot.
   ...   ax.set_xlabel("search iterations")
   ...   ax.set_ylabel("accuracy")
   ...   ax.set_title("Model Comparison")
   ...   ax.legend(loc="lower right")
   [<matplotlib...
   >>> # plt.show() or fig.savefig(...)

Breaking down this example, first we construct confidence bands for each
model's CDF using
:py:class:`~opda.nonparametric.EmpiricalDistribution.confidence_bands`:

.. code-block:: python

   >>> for name, ys in [("baseline", ys0), ("model", ys1)]:
   ...   # Construct the confidence bands.
   ...   dist_lo, dist_pt, dist_hi = EmpiricalDistribution.confidence_bands(
   ...     ys=ys,            # accuracy results from random search
   ...     confidence=0.80,  # confidence level
   ...     a=0.,             # (optional) lower bound on accuracy
   ...     b=1.,             # (optional) upper bound on accuracy
   ...   )

Then, we compute the tuning curves via the
:py:meth:`~opda.nonparametric.EmpiricalDistribution.quantile_tuning_curve`
method. The *lower CDF* band gives the *upper tuning curve* band, and
the *upper CDF* band gives the *lower tuning curve* band. In this way,
you can plot the tuning curve with confidence bands:

.. code-block:: python

   ...   # Plot the tuning curve.
   ...   ax.plot(ns, dist_pt.quantile_tuning_curve(ns), label=name)
   ...   # Plot the confidence bands.
   ...   ax.fill_between(
   ...     ns,
   ...     dist_hi.quantile_tuning_curve(ns),
   ...     dist_lo.quantile_tuning_curve(ns),
   ...     alpha=0.275,
   ...     label="80% confidence",
   ...   )

The rest just makes the plot look pretty, then shows it or saves it to
disk.

To learn more, checkout
:py:class:`~opda.nonparametric.EmpiricalDistribution` in the reference
documentation or get interactive help in a Python REPL by running
``help(EmpiricalDistribution)``.
