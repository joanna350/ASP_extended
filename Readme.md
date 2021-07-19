#### ENVIRONMENT:

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

#### MODIFY the optimization library

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
