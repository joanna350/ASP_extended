### Project Scope
- Gaussian Process has two main challenging features
- One is the complexity of kernel fitting > Automated Statician Project provides an end-to-end process
- Another is computational cost to derivave inverse of covariance > Researches ongoing
- This project combines the two and offers an automated solution
- Drastic time reduction from Stochastic Variational Inference (SVI) and results on par with the [baseline](http://proceedings.mlr.press/v64/kim_scalable_2016.pdf)
- Evaluation metric is Bayesian information criteria

#### Environment:

- Mac OS
- python 3.5+ (say alias python3)
  
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

- PARAMETERS TO CONTROL on Base.py

- SVI: 1, 0
- opt: 'sgd', 'adadelta', 'rprop'
- max_iters: 1000 - 2500
- savedir: to where you want to save plots
- line 102 - 112: parameters for optimizer

#### Run

```
python3 ABCD.py -depth __depth__ -rtol __tolerance__
```
- input parameters: depth and tolerance
