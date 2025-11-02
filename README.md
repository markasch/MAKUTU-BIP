# MAKUTU-BIP

*Training on Bayesian Inverse Problems.*

The objective of this course is to introduce and develop Bayesian approaches for solving statistical inverse problems. 

The prerequisites are:

- probability and measure theory,
- ordinary and partial differential equations,
- statistical simulation (using python).

All the required results will be presented at the start of the course, making it accessible to anyone with reasonable undegraduate training. The course will consist of lectures (mornings) and practical sessions (afternoons). The practical sessions will cover theoretical examples and coding exercises in python. All necessary environments and tools are explained [here](./02practicals/00_setup.pdf).

--- 

## Day 1

### Lecture

[Introduction to inverse problems and data assimilation](./01lectures/10_IP_DA_intro.pdf): 

- overview, 
- setting, 
- history, 
- definitions, 
- examples.

### Practical

Theoretical and coding [exercises](./02practicals/11_IP_DA_intro_prac.pdf).

- ill-posedness
- constant coefficient deterministic inversion
- variable coefficient deterministic inversion
- PDE deterministic inversion

--- 

## Day 2

### Lecture

[Bayesian inverse problems (BIP)](./01lectures/20_BIP.pdf):

 1. Bayesian inference.
 2. Bayesian/Statistical inversion theory.
 3. Full wave inversion example.
 4. Point and interval estimates.


### Practical

Theoretical and coding [exercises](./02practicals/21_BIP_prac.pdf):

1. Point estimators (MAP, CM).
2. Interval estimators (BCI).
3. Bayesian inverse problem for finite-dimensional, Gaussian case.

--- 

## Day 3


### Lecture

[Posterior Estimation methods](./01lectures/30_McMC.pdf):  
  
  1. Monte Carlo methods. 
  2. Rejection Sampling. Importance Sampling.
  3. Markov chain Monte Carlo (McMC) and variants for posterior estimation.
  4. Metropolis Hastings, Gibbs and Hamiltonian McMC.
  5. Introduction to Variational Inference (VI) for posterior estimation.

### Practical

Theoretical and coding [exercises](./02practicals/31_McMC_prac.pdf):

1. Monte Carlo intgeration.
2. Markov chains.
3. Markov chain Monte Carlo (McMC).
4. Hamiltonian Monte Carlo.
5. Variational Inference.
6. Probabilistic programming language (`pymc`).

--- 

## Day 4

This is an optional, advanced treatment of the Ensemble Kalman approach for solving the BIP. A very complete presentation can be found [here](https://markasch.github.io/kfBIPq/).

### Lecture

### Practical