import GPy
import numpy as np
import datetime
import os
import climin
from scipy.cluster.vq import kmeans
from Optimizer import GradientDescent, RProp
from ABCD import explain_

# restarts tend to optimize performance (for standard GPR model)
REST = 2
# SVI would require some hpp tuning
SVI = 1
# number of inducing points
M = 800
# choose optimizer, recommended to use SGD or Adadelta
opt = 'sgd'

max_iters = 2500

import matplotlib
import matplotlib.pyplot as plt

# for recreation of results
np.random.seed(42)


class KernelComb():
    '''
     BIC to compare the results with existing approaches

        subset: either for demonstration purpose or to make feasible operation (randomized)
        Z: inducing points for SVI

    initializes the instance to handle {GPR/SVGP model fitting, optimizing and plotting}
    **careful HPP tuning required for different optimizer for SVI
    '''
    def __init__(self, subset, inputs, new_kern=None):

        self.x_values, self.y_values = inputs['x_values'], inputs['y_values']
        self.train_x, self.train_y = self.x_values, self.y_values
        self.N = inputs['N']
        self.N_train = self.N
        self.name = "[" + inputs['name'][:10] + "]"
        self.subset = subset
        self.Z = 800
        # optimizer_array are 7 params, length 33023 parameter_names
        # [Z,
        # sum.rbf.variance, sub.rbf.lengthscale, sum.white.var, Gaussian.noise.variance,
        #  q_u_mean, q_u_chol]

        self.passed_dir = self.set_params_dir(subset)

        self.new_kern = new_kern

        self.kernel_structure = self.kernel_compute_(new_kern)

        if subset:
            idx = np.random.choice(self.N, subset, replace=False)
            print('\nsubsetting', subset)
            self.train_x, self.train_y = self.train_x[idx], self.train_y[idx]

        if SVI:
            # inducing points based on k-means cluster
            # k x N array of k centroids
            induce = kmeans(self.train_x, M)
            self.Z = induce[0]
            # basic structure
            self.gp = GPy.core.SVGP(self.train_x, self.train_y, self.Z,
                                   kernel =  self.kernel_structure + GPy.kern.White(1, name="white"),
                                   batchsize = self.batchsiz,
                                   likelihood= GPy.likelihoods.Gaussian())
            # enhances numerical stability -- non-positive diagonal elements
            self.gp.kern.white.variance = 1.e-6
            self.gp.kern.white.fix()
        else:
            # standard model
            self.gp = GPy.models.GPRegression(self.train_x, self.train_y, self.kernel_structure)

        # checking the kernel in use/printing in operation
        kernel_str = explain_(new_kern)
        print('\n' + self.name.ljust(20) + kernel_str + '\n')

    def set_params_dir(self, subset):
        '''
        setup distinguishable directory name in automation
        (more during development for optimization)
        '''

        savedir = '/Users/hsl/PycharmProjects/diss/CTL/'

        if subset:
            savedir += str(subset)

        if SVI:
            self.batchsiz = 1000
            self.niter = 1000
            savedir += 'SVI_X' + str(subset) + '_Z' + str(M) + '_batch' + str(self.batchsiz)
            if opt == 'adadelta':
                self.d = 0.7
                self.o = 1e-6
                self.stepsiz = 1e-5
                self.m = 0.9
                savedir += '_stepsize' + str(self.stepsiz) + '_momentum'+str(self.m) + '_offset'+str(self.o)\
                                +'_decay'+str(self.d)
            elif opt == 'sgd':
                self.stepsiz = 1e-5
                self.m = 0.9
                savedir += '_stepsize' + str(self.stepsiz) + '_momentum' + str(self.m)

        if not os.path.exists(savedir):
            os.makedirs(savedir)

        return savedir

    def optimize(self):
        '''
        The main engine
        Returns:
             kernel_in_use: to compare among the nodes in a tree
             BIC: loss parameter to base comparison between nodes "
        modification necessary:
            must. GPy/core/svgp.py to reflect batchsize update for every optimizer in use
            optional. climin/gd.py if to reflect iteration rate in step learning rate
        '''
        print('kernel check\n', self.kernel_structure)
        print("\n$-----------WAIT FOR IT(OPTIMIZING up to {} iterations)------------$\n".format(
            str(max_iters)))

        try:
            if SVI:
                if 'sgd':
                    GD = GradientDescent(max_iters=max_iters)
                    self.gp.optimize(GD, messages = True)
                elif 'adadelta':
                    opt = climin.Adadelta(self.model.optimizer_array, self.model.stochastic_grad,
                                          decay=self.d, offset=self.o,
                                          step_rate=self.stepsiz, momentum=self.m)
                    opt.minimize_until(lambda i: True if i['n_iter'] > self.niter else False)
                elif 'rprop':
                    RPP = RProp(max_iters = max_iters)
                    self.gp.optimize(RPP, messages = True)
            else:

                self.gp.optimize(max_iters=max_iters, messages = True)
                self.gp.optimize_restarts(num_restarts=REST, parallel=False)

        except Exception as e:
            print("Evaluation error")
            print(e)
            return self.new_kern, np.inf

        # keeping record for view
        print('\n objective', self.gp.objective_function())

        # passing on the variables for printing and node comparison use
        titled, bicnum = self.plot()

        print("\n$-------Evaluation Result-------$\n{}\n$-------------------------------$\n".format(titled))

        # keep track of current time
        print(datetime.datetime.now())

        return self.new_kern, bicnum

    def predict_measures(self):
        '''
        measures required to make title for plots and printing use
        '''

        pred_mean, pred_var = self.gp.predict(self.x_values)

        # so that I only have to call once
        bicnum, bicstr = self.bic()

        MSE_Train = sum((pred_mean[:self.N_train] - self.y_values) ** 2)/self.N

        return MSE_Train[0], pred_mean, pred_var, bicnum, bicstr

    def plot(self):
        '''
        plot data points with 98 % confidence interval, mean +/- 1.96 * stdev
        add metric like mean-squared-error
        add inducing points when there are
        format, add informative title
        '''
        MSE_Train, pred_m, pred_v, bicnum, bicstr = self.predict_measures()

        titled = str(self.name) + explain_(self.new_kern) + " | BIC=" + bicstr

        # test the case of negative covariance..
        # https://github.com/SheffieldML/GPy/issues/253
        # print(np.diag(self.gp.kern.K(self.gp.X)))
        # print(np.linalg.eigvals(self.gp.kern.K(self.gp.X)).min())

        plt.close('all')

        # main stream plot
        plt.figure(figsize=(18,6))
        plt.plot(self.x_values, pred_m, '-k', label="Prediction")
        plt.fill_between(np.squeeze(self.x_values), np.squeeze(pred_m - 1.96 * pred_v ** 0.5),
                         np.squeeze(pred_m + 1.96 * pred_v ** 0.5), alpha=0.2)

        # plots the shade
        plt.plot(self.train_x, self.train_y, ',', label= "loss = " +
                              str(round(-1 * float(self.gp._log_marginal_likelihood), 3)) +
                              ", MSE = " + str(round(MSE_Train, 3)))

        if SVI:
            plt.plot(self.Z, [plt.ylim()[0] for _ in self.Z], 'k^', mew=3, label= 'Inducing points')

        plt.legend(fontsize="large", markerscale=3, loc ='lower left')
        plt.xlim(min(self.x_values), max(self.x_values))
        plt.ylim(-4, 4)
        plt.title(titled)

        # plot every fit
        plotted = self.passed_dir + "/" + explain_(self.new_kern) + ".png"
        plt.savefig(plotted)
        print('$----------------------------------saving to::\n', plotted)

        return titled, bicnum

    def kernelchoice(self, kernel):
        '''
        uiltity function to select kernel by string
        '''
        
        if kernel == 'RBF':
            kern_module = GPy.kern.RBF(1)
        elif kernel == 'PERIODIC':
            kern_module = GPy.kern.StdPeriodic(1)
        elif kernel == 'LIN':
            kern_module = GPy.kern.Linear(1) + GPy.kern.Bias(1)

        return kern_module

    def kernel_compute_(self, kernlist):
        '''
        combine the kernel based on notation of the following
        [[X,Y],Z] -> (X+Y)*Z
        '''
        reverse = lambda l: (reverse (l[1:]) + l[:1] if l else [])

        kern_comp = None
        for ele in reverse(kernlist):
            if type(ele) is str:
                kern_comp = self.kernelchoice(ele) if kern_comp is None else kern_comp + self.kernelchoice(ele)
            elif type(ele) is list:
                kern_comp = self.kernel_compute_(ele) if kern_comp is None else kern_comp * self.kernel_compute_(ele)

        return kern_comp.copy()


    def bic(self):
        '''
        BIC is composed by sum of marginal log likelihood /KL-divergence
        '''
        # number of model parameters
        params = [p for p in self.gp]
        k = len(params)

        # number of observations
        n = np.shape(self.train_x)[0]

        # mll incorporates -KL
        mll = float(self.gp._log_marginal_likelihood)

        # BIC for high-dimensional model
        bicnum = float(1 * np.log(n)*k - 2*mll)
        bicstr = "{0:.3f}".format(bicnum)

        # return different types for each use
        return bicnum, bicstr

