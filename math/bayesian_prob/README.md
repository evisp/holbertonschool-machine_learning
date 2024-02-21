# README

Bayesian probability is a statistical framework that involves **interpreting probability as a measure of belief or degree of certainty**, and **updating that belief as new evidence or data becomes available**. In Bayesian probability, prior knowledge and assumptions are combined with observed data to compute a posterior probability, which represents the updated belief given the data.

## What is Bayesian Probability?

Bayesian probability is a way of measuring the degree of belief or certainty in a hypothesis or statement, based on prior knowledge and observed evidence. It involves using **Bayes' theorem** to calculate the probability of a hypothesis given the data, by combining prior probability with the likelihood of the data given the hypothesis. Mathematically, it can be expressed as:

$$
P(hypothesis | data) = \frac{P(data | hypothesis) * P(hypothesis)}{P(data)}
$$

where $P(hypothesis | data)$ is the **posterior probability** of the hypothesis given the data, $P(data | hypothesis)$ is the **likelihood** of the data given the hypothesis, $P(hypothesis)$ is the **prior** probability of the hypothesis, and $P(data)$ is the probability of the data.

## What is Bayes’ rule and how do you use it?

Bayes' rule is a fundamental theorem in Bayesian probability theory that describes how to update probabilities in light of new evidence. Mathematically, it can be written as:

$$
P(A | B) = \frac{P(B | A) * P(A)}{P(B)}
$$

where $P(A | B)$ is the posterior probability of hypothesis $A$ given evidence $B$, $P(B | A)$ is the likelihood of evidence $B$ given hypothesis $A$, $P(A)$ is the prior probability of hypothesis $A$, and $P(B)$ is the probability of evidence $B$.

## Key concepts in a nutshell

- **Base Rate:** Frequency of an event occurring in general.

- **Prior:** Initial belief or probability before new evidence.

- **Posterior:** Updated belief or probability after considering evidence.

- **Likelihood:** Probability of evidence given a hypothesis.