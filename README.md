# MAKUTU-BIP

*Training on Bayesian Inverse Problems.*

The objective of this course is to introduce and develop Bayesian approaches for solving statistical inverse problems. 

The prerequisites are:

- probability and measure theory,
- partial differential equations,
- statistical simulation (using python).

All the above will be presented at the start of the course, making it accessible to anyone with reasonable undegraduate training. The course will consist of lectures (mornings) and practical sessions (afternoons). The practical sessions will cover theoretical examples and coding exercises in python. All necessary environments and tools will be explained.

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


### Practical

Theoretical and coding [exercises](./02practicals/21_BIP_prac.pdf):

1. Point estimators (MAP, CM).
2. Interval estimators (BCI).
3. Bayesian inverse problem for finite-dimensional, Gaussian case.

--- 

## Day 3


### Lecture

[Posterior Estimation methods](./01lectures/30_McMC.pdf):  
  
  1. Point and interval estimates.
  2. Markov chain Monte Carlo (McMC) and variants for posterior estimation.
  3. Variationa Inference (VI) for posterior estimation.

### Practical


1. Monte Carlo intgeration.
2. Markov chains.
3. Markov chain Monte Carlo (McMC).
4. Hamiltonian Monte Carlo.
5. Variational Inference.

--- 

## Day 4

This is an optional, advanced treatment of the Ensemble Kalman approach for solving the BIP. A very complete presentation can be found [here](https://markasch.github.io/kfBIPq/).

### Lecture

### Practical