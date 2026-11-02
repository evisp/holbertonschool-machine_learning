# Probability

Data is noisy and models are never certain. Probability is the language for saying how noisy, and how certain. In this project you build four classic distributions as Python classes, without any maths library, so you see exactly what each formula does.

## Goal

By the end of this project you can pick the distribution that fits a situation, estimate its parameters from data, and compute the probability of an outcome from first principles.

## Learning objectives

After this project you can explain, without looking anything up:

- what a probability distribution is
- the difference between discrete (PMF) and continuous (PDF) distributions
- what a CDF measures, and how it relates to the PMF or PDF
- what the mean and standard deviation of a distribution tell you
- what a z-score is, and why it makes different scales comparable
- how to estimate a distribution's parameters from observed data
- why the normal distribution appears so often

## Choosing a distribution

| Distribution | Models | Type | Parameters | Estimated from data |
|--------------|--------|------|------------|---------------------|
| Poisson | Number of events in a fixed interval | Discrete | `lambtha` (average rate) | mean |
| Exponential | Waiting time until the next event | Continuous | `lambtha` (rate) | 1 / mean |
| Normal | Values that cluster around an average | Continuous | `mean`, `stddev` | mean, standard deviation |
| Binomial | Number of successes in `n` trials | Discrete | `n`, `p` | from mean and variance |

Poisson and Exponential are two views of the same process: Poisson counts the events, Exponential measures the time between them.

## Files

| File | Class | Methods |
|------|-------|---------|
| [poisson.py](./poisson.py) | `Poisson` | `pmf`, `cdf` |
| [exponential.py](./exponential.py) | `Exponential` | `pdf`, `cdf` |
| [normal.py](./normal.py) | `Normal` | `z_score`, `x_value`, `pdf`, `cdf` |
| [binomial.py](./binomial.py) | `Binomial` | `pmf`, `cdf` |

Every class is created in one of two ways: from parameters, or from a list of data from which it estimates the parameters.

## Example

```python
#!/usr/bin/env python3
from normal import Normal

heights = [162, 170, 175, 168, 181, 177, 169, 172]
n = Normal(heights)

print(n.mean, n.stddev)
print(n.z_score(181))   # how unusual is a height of 181?
print(n.cdf(172))       # probability of being 172 or shorter
```

## Key ideas

- **PMF and PDF.** For a discrete variable, the PMF gives the probability of each exact value. For a continuous variable, the PDF gives a density: probabilities come from areas under the curve, not from single points.
- **CDF.** The probability that the variable is at most a given value. It always rises from 0 to 1.
- **Estimating parameters.** Each class can learn its parameters from data using the fact that a distribution's mean (and variance) are tied to them. For example `mean = n * p` and `variance = n * p * (1 - p)` for the binomial.
- **z-score.** `(x - mean) / stddev` counts how many standard deviations a value is from the mean.

## Implementation notes

- Only plain Python is used: `e` and `pi` are hard-coded constants and factorials are computed with loops.
- The `Normal.cdf` uses a truncated series for the error function. It is accurate near the mean and drifts in the far tails.
- Methods that take a count (`k`) truncate non-integers to an integer.
- Invalid input raises `ValueError` or `TypeError` with a clear message.
