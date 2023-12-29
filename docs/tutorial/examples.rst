========
Examples
========
The following examples show how to solve common tasks using opda.

Compare Models' Tuning Curves
-----------------------------
Let's compare two models while accounting for hyperparameter tuning
effort:

.. plot::
   :format: doctest
   :context:

   >>> from matplotlib import pyplot as plt
   >>> import numpy as np
   >>>
   >>> from opda.nonparametric import EmpiricalDistribution
   >>>
   >>> generator = np.random.default_rng(0)  # Set the random seed.
   >>>
   >>> # Simulate results from random search.
   >>> ys0 = generator.uniform(0.70, 0.90, size=48)  # scores from model 0
   >>> ys1 = generator.uniform(0.75, 0.95, size=48)  # scores from model 1
   >>>
   >>> # Plot tuning curves and confidence bands for both models in one figure.
   >>> fig, ax = plt.subplots()
   >>> ns = np.linspace(1, 10, num=1_000)
   >>> for i, ys in enumerate([ys0, ys1]):
   ...   # Construct the confidence bands.
   ...   dist_lo, dist_pt, dist_hi = EmpiricalDistribution.confidence_bands(
   ...     ys=ys,            # accuracy results from random search
   ...     confidence=0.80,  # confidence level
   ...     a=0.,             # (optional) lower bound on accuracy
   ...     b=1.,             # (optional) upper bound on accuracy
   ...   )
   ...   # Plot the tuning curve.
   ...   ax.plot(ns, dist_pt.quantile_tuning_curve(ns), label=f"model {i}")
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
   ...   ax.legend(loc="lower right")
   [<matplotlib...
   >>> # plt.show() or fig.savefig(...)

Breaking down the above example, first it constructs confidence bands
for each model's CDF using
:py:class:`~opda.nonparametric.EmpiricalDistribution.confidence_bands`:

.. code-block:: python

   >>> for i, ys in enumerate([ys0, ys1]):  # doctest: +SKIP
   ...   # Construct the confidence bands.
   ...   dist_lo, dist_pt, dist_hi = EmpiricalDistribution.confidence_bands(
   ...     ys=ys,            # accuracy results from random search
   ...     confidence=0.80,  # confidence level
   ...     a=0.,             # (optional) lower bound on accuracy
   ...     b=1.,             # (optional) upper bound on accuracy
   ...   )

You can compute the tuning curve for each of these distributions using
:py:meth:`~opda.nonparametric.EmpiricalDistribution.quantile_tuning_curve`.
The *lower CDF* band gives the *upper tuning curve* band, and the *upper
CDF* band gives the *lower tuning curve* band. In this way, you can plot
the tuning curve with confidence bands:

.. code-block:: python

   ...   # Plot the tuning curve.
   ...   ax.plot(ns, dist_pt.quantile_tuning_curve(ns), label=f"model {i}")
   ...   # Plot the confidence bands.
   ...   ax.fill_between(
   ...     ns,
   ...     dist_hi.quantile_tuning_curve(ns),
   ...     dist_lo.quantile_tuning_curve(ns),
   ...     alpha=0.275,
   ...     label="80% confidence",
   ...   )  # doctest: -SKIP

The rest is just formatting to make the plot look pretty, and then
showing it or saving it to disk.

To dive deeper, checkout :doc:`Usage </tutorial/usage>`, the reference
docs for :py:class:`~opda.nonparametric.EmpiricalDistribution`, or run
``help(EmpiricalDistribution)`` in a Python REPL for interactive help.
