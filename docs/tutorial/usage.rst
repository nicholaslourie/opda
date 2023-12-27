=====
Usage
=====
Opda helps you compare machine learning methods while accounting for
their hyperparameters. This guide reviews opda's basic functionality,
while :doc:`Examples </tutorial/examples>` walks through how to run such
an analysis.


Working with the Empirical Distribution
=======================================
Many analyses in opda are designed around `random search
<https://jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf>`_. Each
round of random search independently samples a choice of hyperparameters
and then evaluates it, producing a score (like accuracy). The
probability distribution over these scores determines how good the model
is, and how quickly performance goes up with more hyperparameter tuning
effort. Given a bunch of scores from random search, you can estimate
this distribution with the `empirical distribution
<https://en.wikipedia.org/wiki/Empirical_distribution_function>`_ of the
sample.

If ``ys`` is an array of scores from random search, you can represent
the empirical distribution with the
:py:class:`~opda.nonparametric.EmpiricalDistribution` class:

.. code-block:: python

   >>> from opda.nonparametric import EmpiricalDistribution
   >>> import opda.random
   >>>
   >>> opda.random.set_seed(0)
   >>>
   >>> dist = EmpiricalDistribution(
   ...   ys=[.72, .83, .92, .68, .75],  # scores from random search
   ...   ws=None,                       # (optional) sample weights
   ...   a=0.,                          # (optional) lower bound on the score
   ...   b=1.,                          # (optional) upper bound on the score
   ... )

Using the class, you can do many things you might expect with a
probability distribution. For example, you can sample from it:

.. code-block:: python

   >>> dist.sample()
   0.75
   >>> dist.sample(2)
   array([0.68, 0.92])
   >>> dist.sample(size=(2, 3))
   array([[0.83, 0.83, 0.72],
          [0.72, 0.72, 0.72]])

You can compute the `PMF (probability mass function)
<https://en.wikipedia.org/wiki/Probability_mass_function>`_:

.. code-block:: python

   >>> dist.pmf(0.5)
   0.0
   >>> dist.pmf(0.75)
   0.2
   >>> dist.pmf([0.5, 0.75])
   array([0. , 0.2])

the `CDF (cumulative distribution function)
<https://en.wikipedia.org/wiki/Cumulative_distribution_function>`_:

.. code-block:: python

   >>> dist.cdf(0.5)
   0.0
   >>> dist.cdf(0.72)
   0.4
   >>> dist.cdf([0.5, 0.72])
   array([0. , 0.4])

and even the `quantile function
<https://en.wikipedia.org/wiki/Quantile_function>`_, also known as the
percentile-point function or PPF:

.. code-block:: python

   >>> dist.ppf(0.0)
   0.0
   >>> dist.ppf(0.4)
   0.72
   >>> dist.ppf([0.0, 0.4])
   array([0.  , 0.72])

The :py:class:`~opda.nonparametric.EmpiricalDistribution` class also
prints nicely and supports equality checks:

.. code-block:: python

   >>> EmpiricalDistribution(ys=[0.5, 1.], a=-1., b=1.)
   EmpiricalDistribution(ys=array([0.5, 1. ]), ws=None, a=-1.0, b=1.0)
   >>> EmpiricalDistribution([1.]) == EmpiricalDistribution([1.])
   True


Estimating Tuning Curves
========================
:py:class:`~opda.nonparametric.EmpiricalDistribution` provides several
methods for estimating tuning curves. *Tuning curves* plot the median
score as a function of hyperparameter tuning effort. Thus, tuning curves
capture the cost-benefit trade-off offered by tuning a model's
hyperparameters.

.. plot::
   :format: python
   :include-source: false
   :caption: The median tuning curve plots the score as a function of
             tuning effort.

   from matplotlib import pyplot as plt
   import numpy as np

   from opda.nonparametric import EmpiricalDistribution

   generator = np.random.default_rng(0)

   ns = np.linspace(1, 10, num=1_000)
   bounds = (0.70, 0.90)
   pt = EmpiricalDistribution(
       ys=generator.uniform(*bounds, size=(48,)),
       a=0.,
       b=1.,
   )

   plt.plot(ns, pt.quantile_tuning_curve(ns))

   plt.xlabel("search iterations")
   plt.ylabel("accuracy")
   plt.show()

You can compute the median tuning curve using
:py:meth:`~opda.nonparametric.EmpiricalDistribution.quantile_tuning_curve`:

.. code-block:: python

   >>> dist.quantile_tuning_curve(1)
   0.75
   >>> dist.quantile_tuning_curve([1, 2, 4])
   array([0.75, 0.83, 0.92])

:py:meth:`~opda.nonparametric.EmpiricalDistribution.quantile_tuning_curve`
computes the median tuning curve by default, but you can also explicitly
pass the ``q`` argument to compute the 20th, 95th, or any other quantile
of the tuning curve:

.. code-block:: python

   >>> dist.quantile_tuning_curve([1, 2, 4], q=0.2)
   array([0.68, 0.75, 0.83])

In this way, you can describe the *entire distribution* of scores from
successive rounds of random search.

Depending on your metric, you may want to minimize a loss rather than
maximize a utility. Control minimization versus maximization using the
`minimize` argument:

.. code-block:: python

   >>> dist.quantile_tuning_curve([1, 2, 4], minimize=True)
   array([0.75, 0.72, 0.68])

While `the median tuning curve is recommended
<https://arxiv.org/abs/2311.09480>`_, you can also compute the expected
tuning curve with
:py:meth:`~opda.nonparametric.EmpiricalDistribution.average_tuning_curve`:

.. code-block:: python

   >>> dist.average_tuning_curve([1, 2, 4])
   array([0.78    , 0.8272  , 0.871936])


Constructing Confidence Bands
=============================
In theory, tuning curves answer any question you might ask about
performance as a function of tuning effort. In practice, tuning curves
must be estimated from data. Thus, we need to know how reliable such
estimates are. `Confidence bands
<https://en.wikipedia.org/wiki/Confidence_and_prediction_bands>`_
quantify an estimate's reliability.

.. plot::
   :format: python
   :include-source: false
   :caption: Confidence bands quantify uncertainty when comparing models.

   from matplotlib import pyplot as plt
   import numpy as np

   from opda.nonparametric import EmpiricalDistribution

   generator = np.random.default_rng(0)

   ns = np.linspace(1, 10, num=1_000)
   for i, bounds in enumerate([(0.70, 0.90), (0.75, 0.95)]):
       lo, pt, hi = EmpiricalDistribution.confidence_bands(
           ys=generator.uniform(*bounds, size=(48,)),
           confidence=0.80,
           a=0.,
           b=1.,
       )

       plt.plot(ns, pt.quantile_tuning_curve(ns), label=f"model {i+1}")
       plt.fill_between(
           ns,
           hi.quantile_tuning_curve(ns),
           lo.quantile_tuning_curve(ns),
           alpha=0.275,
           label="80% confidence",
       )

   plt.xlabel("search iterations")
   plt.ylabel("accuracy")
   plt.legend(loc="lower right")
   plt.show()

Opda offers `simultaneous
<https://en.wikipedia.org/wiki/Confidence_and_prediction_bands#Pointwise_and_simultaneous_confidence_bands>`_
confidence bands that contain the entire curve at once. You can compute
simultaneous confidence bands for the CDF and then translate these to
the tuning curve.

Cumulative Distribution Functions
---------------------------------
Compute simultaneous confidence bands for the CDF using the
:py:meth:`~opda.nonparametric.EmpiricalDistribution.confidence_bands`
class method:

.. code-block:: python

   >>> dist_lo, dist_pt, dist_hi = EmpiricalDistribution.confidence_bands(
   ...   ys=[.72, .83, .92, .68, .75],  # scores from random search
   ...   confidence=0.80,               # the confidence level
   ...   a=0.,                          # (optional) lower bound on the score
   ...   b=1.,                          # (optional) upper bound on the score
   ... )

The returned objects, ``dist_lo``, ``dist_pt``, and ``dist_hi``, are
:py:class:`~opda.nonparametric.EmpiricalDistribution` instances
respectively representing the lower band, point estimate, and upper band
for the empirical distribution.

Thus, you can use ``dist_pt`` to estimate the CDF:

.. code-block:: python

   >>> dist_pt == EmpiricalDistribution([.72, .83, .92, .68, .75], a=0., b=1.)
   True
   >>> dist_pt.cdf([0., 0.72, 1.])
   array([0. , 0.4, 1. ])

While ``dist_lo`` and ``dist_hi`` bound the CDF:

.. code-block:: python

   >>> all(dist_lo.cdf([0., 0.72, 1.]) <= dist_pt.cdf([0., 0.72, 1.]))
   True
   >>> all(dist_pt.cdf([0., 0.72, 1.]) <= dist_hi.cdf([0., 0.72, 1.]))
   True

The bounds from
:py:meth:`~opda.nonparametric.EmpiricalDistribution.confidence_bands`
are guaranteed to contain the true CDF with probability equal to
``confidence``, as long as the underlying distribution is continuous and
the scores are independent.

Tuning Curves
-------------
When comparing models, we don't care primarily about the CDF but rather
the tuning curve. Given confidence bounds on the CDF, we can easily
convert these into bounds on the tuning curve using the
:py:meth:`~opda.nonparametric.EmpiricalDistribution.quantile_tuning_curve`
method. In particular, the *lower* CDF bound gives the *upper* tuning
curve bound, and the *upper* CDF bound gives the *lower* tuning curve
bound:

.. code-block:: python

   >>> ns = range(1, 11)
   >>> tuning_curve_lo = dist_hi.quantile_tuning_curve(ns)
   >>> tuning_curve_pt = dist_pt.quantile_tuning_curve(ns)
   >>> tuning_curve_hi = dist_lo.quantile_tuning_curve(ns)
   >>>
   >>> all(tuning_curve_lo <= tuning_curve_pt)
   True
   >>> all(tuning_curve_pt <= tuning_curve_hi)
   True

This reversal occurs because the CDF maps *from* the scores (e.g.,
accuracy numbers), while the tuning curve maps *to* the scores. Swapping
the direction of the mapping also swaps the inequalities on the bounds.


Running the notebooks
=====================
Several :source-dir:`notebooks <nbs/>` are available in opda's source
repository. These notebooks offer experiments and develop the theory
behind opda.

To run the notebooks, make sure you've installed the necessary
:ref:`optional dependencies <tutorial/setup:Optional Dependencies>`.
Then, run the notebook server:

.. code-block:: console

   $ cd nbs/ && jupyter notebook

And open up the URL printed to the console.


Getting More Help
=================
More documentation is available for exploring opda even further.

Interactive Help
----------------
Opda is self-documenting. Use :py:func:`help` to view the documentation
for a function or class in the REPL:

.. code-block:: python

   >>> from opda.nonparametric import EmpiricalDistribution
   >>> help(EmpiricalDistribution)  # doctest: +NORMALIZE_WHITESPACE
   Help on class EmpiricalDistribution in module opda.nonparametric:
   <BLANKLINE>
   class EmpiricalDistribution(builtins.object)
    |  EmpiricalDistribution(ys, ws=None, a=-inf, b=inf)
    |
    |  The empirical distribution.
    |
    |  Parameters
    |  ----------
    |  ys : 1D array of floats, required
    |      The sample for which to create an empirical distribution.
   ...

Interactive shell and REPL sessions are a great way to explore opda!

API Reference
-------------
The :doc:`API Reference </reference/opda>` documents every aspect of
opda.

The :py:mod:`~opda.parametric` and :py:mod:`~opda.nonparametric` modules
implement much of the core functionality.
