import Base
import argparse, datetime
import numpy as np
from scipy import io
from anytree import NodeMixin

np.random.seed(42)

def create(dir='/Users/hsl/PycharmProjects/diss/CTL/', file = None, dim = 0):
    '''
    initializing all data for model 
    '''

    #load
    data = io.loadmat(dir + file)
    
    #file's variable names
    x_values = np.squeeze(data['times'])
    y_values = np.squeeze(data['loads'])[:, dim] #(38070,20 originally)

    #preprocess
    x_values = preprocess(x_values)
    y_values = preprocess(y_values)

    N = (x_values.shape)[0]
    N_train = int(N * 0.8)
    N_val = int(N * 0.1) #split 8/1/1
    N_test = N - N_train - N_val

    train_x = x_values[:N_train][:,np.newaxis]
    validation_x = x_values[N_train:(N_train+N_val)][:,np.newaxis]
    test_x = x_values[(N_train+N_val):][:,np.newaxis]

    train_y = y_values[:N_train][:,np.newaxis]
    validation_y = y_values[N_train:(N_train+N_val)][:,np.newaxis]
    test_y = y_values[(N_train+N_val):][:,np.newaxis]

    x_values = x_values[:,np.newaxis]
    y_values = y_values[:,np.newaxis]

    #output parameters
    inputs = {}
    inputs['x_values'], inputs['y_values'] = x_values, y_values
    inputs['train_x'], inputs['train_y'] = train_x, train_y
    inputs['val_x'], inputs['val_y'] = validation_x, validation_y
    inputs['test_x'], inputs['test_y'] = test_x, test_y
    inputs['N'], inputs['N_train'], inputs['N_val'], inputs['N_test'] = N, N_train, N_val, N_test
    inputs['name'] = file + " "

    return inputs

def preprocess(data):
    '''
    normalize data
    '''
    return (data - np.mean(data))/np.std(data)

def explain_(kernlist):
    '''
    Viewing purpose, used in both scripts
    returns a kernel in human-readable pattern
    '''
    kern_str = None
    for idx, ele in enumerate(kernlist):
        if type(ele) is str:
            if idx == len(kernlist) - 1:
                kern_str = ele if kern_str is None else kern_str + ele
            else:
                addn = ele + "+"
                kern_str = addn if kern_str is None else kern_str + addn

        elif type(ele) is list:
            addn = "(" + explain_(ele) + ")"
            kern_str = addn if kern_str is None else kern_str + addn
    return kern_str


class KernelTree(NodeMixin):
    '''
    tree
    ----
        parent: nodes from previous depth
        children: nodes emanating from parent node
    ----
        kernls: explainable form of kern
        bic: for loss comparison
        minl: distinguish minimum loss of current depth (determines whether to continue search)
    '''
    def __init__(self, subset, kernel=None):
        self.parent = None

        if kernel == None:
            self.kernls = []
            # so that comparison will work
            self.bic = np.inf

        else:
            #instantiate the node of GPR & optimize to create loss to base performance
            self.kernls, self.bic = Base.KernelComb(subset, input, kernel).optimize()

        #purely for tree search purpose - used in branch_node function
        self.minl = False

        #following the standards from Auto.Stats.proj
        self.basekern = ['RBF', 'PERIODIC', 'LIN']


    def combined(self):
        '''
        combination of addition and multiplication given the kernels
        supported for next-depth exploration
        '''
        if self.kernls == []:
            return [[kern] for kern in self.basekern]

        nodes_ = []
        for new_kern in self.basekern:
            nodes_.extend([ self.kernls + [new_kern], [self.kernls, new_kern] ])
        return nodes_

    def list_nodes(self):
        '''
        Given the kernels, generate nodes that contain the model
        which will calculate loss for comparison based on given kernel
        '''
        nodes = []

        for kernel in self.combined():
            nodes.append(KernelTree(subset, kernel))
        self.children = nodes

    def branch_node(self):
        '''
        Recursive function to exhaustively search the node with the least BIC
        based on which more kernel combinations will be expanded in the next depth
        If not better than the previous depth, the search stops.
        node parameters:
            .parent
            .bic
            .minl
        '''
        curr = self.get_length(self.kernls)
        print("$----------------------------------Depth " + str(curr + 1) + "/" + str(depth))

        self.list_nodes()

        comparison = [node.bic for node in self.children]

        min_node = self.children[comparison.index(min(comparison))]

        if min_node.bic + np.abs(min_node.bic) * float(rtol) < min_node.parent.bic:
            min_node.minl = True
            print("LOCAL OPTIMAL------------------------------------$")
            print(explain_(min_node.kernls), str(round(min_node.bic, 3)))
        else:
            min_node.minl = False
            print("CANNOT BE WORSE----------------------------------%")

        # branch out of the best node in a depth
        if (self.get_length(min_node.kernls) < depth) and min_node.minl:
            min_node.branch_node()
        else:
            print("DEPTH SEARCH DONE--------------------------------%")

    def get_length(self, ls):
        '''the length'''
        if type(ls) == list:
            return sum(self.get_length(l) for l in ls)
        return 1


if __name__ == "__main__":
    '''
    takes parameters from stdin 
    '''
    parser = argparse.ArgumentParser(description='Depth, loss value comparison for kernel tree') #dimension choice at discretion
    parser.add_argument('-depth', action="store", type=int)
    parser.add_argument('-rtol', action="store", type=float) #recommended 1e-5 for SVI, ow ~1e-2
    # parser.add_argument('-dim', action="store", type=float)

    args = vars(parser.parse_args())
    #check
    print(args)
    print('SVI', Base.SVI, '\nInducing points', Base.M)

    #global variables
    depth = args['depth']
    rtol = args['a']
    # dimension = args['dim']

    data = 'gefcom.mat'
    input = create(file = data)

    #to track record of the run
    print(datetime.datetime.now())

    subsets = [10000]
    for subset in subsets:
        # print('subset check', subset)
        KernelTree(subset).branch_node()
