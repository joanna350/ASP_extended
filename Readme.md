### Project Scope
- Gaussian Process is nonparametric regression model with two main challenging features.
- One is the complexity of kernel fit. In the spirit of Automated Statician Project, the project offers kernel search through linear combination.
- Another is computational cost. It derives from taking the inverse of covariance, O(n^3). This is a field of active research in GP domain.
- This project provides an automated solution to the combined challenge.
- Similar effort was precedent, and was referred as the [baseline](http://proceedings.mlr.press/v64/kim_scalable_2016.pdf) of the result comparison. By deploying Stochastic Variational Inference, the code achieves reduction to O(m^3), seen in result.
- Evaluation metric for kernel fit is Bayesian information criteria

#### Environment:

- Mac OS
- python 3.5+
  
```
pip install GPy
```

- Install climin pre-0.1 (For Mac users)
```
git clone https://github.com/BRML/climin.git /path/to/climin
cd /path/to/climin
pip install -e .
```

#### Modify the optimization library

- Parameters to control on Base.py
```
SVI: 1, 0
opt: 'sgd', 'adadelta', 'rprop'
max_iters: 1000 - 2500
savedir: the path to save plots
line 102 - 112: parameters for optimizer
```

#### Run

```
python3 ABCD.py -depth __depth__ -rtol __tolerance__
```
- input parameters: depth and tolerance
