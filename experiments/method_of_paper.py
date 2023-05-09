# Method of paper
# This version prints out separate csv file for tables and plots
# this version implements a default test set size of 2/9 times the total data-set size
# It also uses the 1/sqrt(d) factor when applying 'prewhitening' processing of x vals in order to be compatible with the setup used for SVGP and distributed rtuns.
 
import torch
import math
import numpy as np
import gpytorch
import os
import time
from sklearn.neighbors import NearestNeighbors
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
 
    # Input and output file args
    parser.add_argument('-xyinfl', '--xy_input_file',
                        help='xy vals input file',
                        default='xy_input_file')
    parser.add_argument('-resfl', '--results_file',
                        help='Results output file',
                        default='results_output_file2')
    parser.add_argument('-resflcsv', '--results_file_csv',
                        help='Results csv output file',
                        default='csv_results_output_file2')

# type of x preprocsseing to be applied
    parser.add_argument('-xprep', '--x_preprocess',
                        help='type of x preprocessing to be applied',
                        default='whiten', choices=['axis_rescale', 'whiten','none'])

# Num test and recalibration points args
    parser.add_argument('-n_recal', '--recal-data-size',
                        help='Size of recalibration data set',
                        default=1000, type=int)
    parser.add_argument('-n_test', '--test-data-size',
                        help='Size of test data set',
                        default=-999, type=int) # if not specified then the -999 default value will trigger use of n_test = 2/9 total data-set size
    parser.add_argument('-n_test_cap', '--test-data-size_cap',
                        help='Cap on size of test data set',
                        default= 100000000, type=int)
 
    # Assumed kernel family
    parser.add_argument('-aker', '--assum-kernel-type',
                        help='Assumed kernel type',
                        default='RBF', choices=['RBF', 'Matern', 'Exp'])
    #some estimation and prediction parameters
    parser.add_argument('-ssize', '--subset-size',
                        help='Size of subsets used in param-estimation',
                        default=300,type = int )
    parser.add_argument('-maxns', '--max-nsubsets',
                        help='Max num subsets used in param-estimation',
                        default=10,type = int )
    parser.add_argument('-numnn', '--number-nn',
                        help='Number of nearest neighbours used for prediction',
                        default=400,type = int )
    # Miscellaneous Args
    parser.add_argument('-seed', '--random-seed',
                        help='Random seed to initialise data split, random subset selection  etc',
                        default=42, type=int)
    return parser.parse_args()

def make_nn_predictions2(x_predict,predict_data_size,x_train,y_train,input_dim,assum_ls,assum_ks,assum_nv,neigh,assum_kern):
    nn_pred_mean = np.zeros([predict_data_size],dtype=np.float64)
    nn_pred_sd = np.zeros([predict_data_size],dtype=np.float64)
    tot_duration = 0.0
    for i_trial in range(predict_data_size):
        np_predict_x = x_predict[np.newaxis, i_trial, :]
        predict_x = torch.from_numpy(np_predict_x)
        # neigh_list = neigh.kneighbors(np_predict_x, return_distance = False)
        #neigh_list = neigh.query(np_predict_x, k=num_near)[0]
        neigh_list = neigh.kneighbors(np_predict_x, return_distance = False)
        nearest_x = torch.from_numpy(x_train[neigh_list.flatten(),:])
        nearest_y = torch.from_numpy(y_train[neigh_list.flatten()])
        #set up nn model with the assumed (possibly mispecified) kernel family and parameters and put in eval mode ready to make predictions
        tic = time.perf_counter()
        if (assum_kern.lower() == 'rbf'):
            nn_model = ExactGP_RBF(nearest_x, nearest_y,  likelihood)
        if (assum_kern.lower() == 'matern'):
            nn_model = ExactGP_Matern(nearest_x, nearest_y,  likelihood)
        if (assum_kern.lower() == 'exp'):
            nn_model = ExactGP_Exp(nearest_x, nearest_y,  likelihood)
        nn_model.double()
        nn_model.covar_module.base_kernel.lengthscale = assum_ls
        nn_model.likelihood.noise = assum_nv
        nn_model.covar_module.outputscale = assum_ks
        nn_model.eval()
        #NOTE: to speed up
        output_nn_model = nn_model(predict_x)
        # extract nn_model predictive mean and sd values corresponding to location predict_x
        nn_pred_mean[i_trial] = output_nn_model.mean
        nn_predf_var = output_nn_model.variance #doesn't account for noise
        nn_pred_var = nn_predf_var + assum_nv  
        nn_pred_sd[i_trial] = nn_pred_var**0.5
        toc = time.perf_counter()
        tot_duration += toc - tic
    #print('total time for GP component (ie. exclude nn gather) of predictions = %f' %(tot_duration))
    ans = nn_pred_mean, nn_pred_sd
    return(ans)
                       
def get_param_estimates2(all_subset_xvals,all_subset_yvals,input_dim,subset_size,num_subsets,num_adam_iters,assum_kern):
    #  for random selection might want to change to choose permuted postions so that different calls use different subsets<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # this version applies latched param estimation (i.e. with block diag kernel defined by subsets).
    # If num_subsets  = 1 then this redcudes to standard SoD estimation
    if (assum_kern.lower() == 'rbf'):
        model = ExactGP_RBF(torch.from_numpy(all_subset_xvals), torch.from_numpy(all_subset_yvals),  likelihood)
    if (assum_kern.lower() == 'matern'):
        model = ExactGP_Matern(torch.from_numpy(all_subset_xvals), torch.from_numpy(all_subset_yvals),  likelihood)
    if (assum_kern.lower() == 'exp'):
        model = ExactGP_Exp(torch.from_numpy(all_subset_xvals), torch.from_numpy(all_subset_yvals),  likelihood)
    model.double()
    likelihood.double()
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    # XXXXXXXXXXX perform descent iterations to estimate kernel params using this training subset  XXXXXXXXXXXXXXXXXX
    smoke_test = ('CI' in os.environ)
    training_iter = 2 if smoke_test else num_adam_iters
    loss = 0.0
    for i in range(training_iter):
        if i == training_iter - 1 or num_subsets ==1:
            graph_params = {}
        else:
            graph_params = {"retain_graph": True}
        optimizer.zero_grad()
        train_xx = torch.from_numpy(all_subset_xvals[(0*subset_size):(1*subset_size),:])
        train_yy = torch.from_numpy(all_subset_yvals[(0*subset_size):(1*subset_size)])
        with gpytorch.settings.debug(False):
            output = model(train_xx)
            loss = -mll(output, train_yy)
            for i_subset in range(1,num_subsets):
                train_xx = torch.from_numpy(all_subset_xvals[(i_subset*subset_size):((i_subset+1)*subset_size),:])
                train_yy = torch.from_numpy(all_subset_yvals[(i_subset*subset_size):((i_subset+1)*subset_size)])
                output = model(train_xx)
                loss -= mll(output, train_yy)
        loss.backward(**graph_params)
        optimizer.step()
    # Optimiser iterations complete so pull out hyper-param estimates
    est_lenscale = model.covar_module.base_kernel.lengthscale.item()
    est_noise_var = model.likelihood.noise.item()
    est_kernelscale = model.covar_module.outputscale.item()
    ans = est_lenscale, est_noise_var, est_kernelscale
    return(ans)

def evaluate_nn_predictions2(nn_pred_mean, nn_pred_sd, predict_data_size, y_true):
    mse_nn = nll_nn = mscal_nn = 0.0
    const = (2.0*math.pi)**0.5
    ave = 0.0 # to remove<<<<<<<<<<<<<<<<<<<<<<<
    for i_trial in range(predict_data_size):  
        #update performance stats
        nll_nn +=  math.log(1.0/(nn_pred_sd[i_trial]*const))
        nll_nn -=  0.5 * ((y_true[i_trial] - nn_pred_mean[i_trial])/nn_pred_sd[i_trial]) ** 2  
        mse_nn += (nn_pred_mean[i_trial] - y_true[i_trial])** 2
        cal = (y_true[i_trial] - nn_pred_mean[i_trial])/nn_pred_sd[i_trial]
        mscal_nn += cal ** 2
        if ((i_trial % 100) == 0):
            ave_mscal = mscal_nn/ (1.0 + float(i_trial)) # to remove<<<<<<<<<<<<<<<<<<<<<<<
            ave_mse = mse_nn/ (1.0 + float(i_trial))
            ave_nll = -nll_nn/ (1.0 + float(i_trial))
            #print ('i_trial = %d, mscal = %f, mse = %f, nll = %f' %(i_trial,ave_mscal, ave_mse, ave_nll), flush = 'True')
    nll_nn = -nll_nn
    nll_nn /= float(predict_data_size)
    mse_nn /=  float(predict_data_size)
    mscal_nn /= float(predict_data_size)
    ans = mse_nn, nll_nn, mscal_nn
    return(ans)

def get_xy_preprocess_tranforms(x_train,y_train,xpreprocess,train_data_size,input_dim):
    # derive quantities that can be used to preprocess (x,y) training values (in the case of x this might optionally either be none, prewhitening or axis-rescaling)
    # derive mean and sd of y train values, mean of x train components, if 'prewhitenting then inverse covar of x train cpts, and if axis-rescaling then
    #diag matrix with 1/(sd x-cpt) along the diag
    m_x  = np.average(x_train,axis = 0)
    m_y = np.average(y_train)
    sd_y =  np.std(y_train, dtype=np.float64)
    if (xpreprocess == 'axis_rescale'):
        sd_x =  np.std(x_train, axis =0, dtype=np.float64)
        prep_mat_x = np.diag(1.0/sd_x)
    if (xpreprocess == 'whiten'):
        cov_x = np.dot(np.transpose(x_train - m_x), (x_train - m_x))/float(train_data_size)
        U,S,V = np.linalg.svd(cov_x) #Singular Value Decomposition
        epsilon = 1e-5
        prep_mat_x = np.diag(1.0 / np.sqrt(S + epsilon)).dot(U.T)
    if (xpreprocess == 'none'):
        prep_mat_x = np.identity(input_dim,dtype=np.float64)
    prep_mat_x /= float(input_dim) **0.5  #<<<<<<<<<<<<<<<<<<<<<<<<<  to make compatible with perprocesing applied in SVGP and ditributed runs !!!!!!!!!!!!!!!!!!!!
    ans = m_y, sd_y, m_x, prep_mat_x
    return(ans)
 
def preprocess_x_vals(x_vals, m_x, prep_mat_x):
    #apply preprocessing transforms to x vals
    x_vals = x_vals - m_x
    x_vals = np.transpose( prep_mat_x.dot(np.transpose(x_vals)) )
    return(x_vals)
 
def get_million_paper_results(fname):
    # EDIT THIS probably makes sense only to include exact GP results AND also need to add cache creation time to the training <<<<<<<<<<<<<<<<<!!!!!!
    rmse = -99.
    nll =  - 99.
    time = -99.
    compute = '-null'
    if (fname == '/Users/marfa/documents/reformatted_UCI_array_datasets/bike/data.npy'):
        rmse = 0.220
        nll = 0.119
        time = 41.2
        compute = '1_GPU exact'
    if (fname == '/Users/marfa/documents/reformatted_UCI_array_datasets/ctslice/data.npy'):
        rmse = .218
        nll = -.073
        time = 129.6
        compute = '1-GPU SGPR'
    if (fname == '/Users/marfa/documents/reformatted_UCI_array_datasets/poletele/data.npy'):
        rmse = 0.151
        nll = -.18
        time =  41.5
        compute = '1-GPU exact'
    if (fname ==  '/Users/marfa/documents/reformatted_UCI_array_datasets/protein/data.npy'):
        rmse = .536
        nll =  1.018
        time = 47.9
        compute = '1-GPU exact'
    if (fname == '/Users/marfa/documents/reformatted_UCI_array_datasets/road3d/data.npy'):
        rmse = .101
        nll =   .909
        time = 720.5
        compute = '8-GPU SGPR'
    if (fname == '/Users/marfa/documents/reformatted_UCI_array_datasets/song/data.npy'):
        rmse = .803
        nll =  1.21
        time = 253.4
        compute = '8-GPU exact'
    if (fname == '/Users/marfa/documents/reformatted_UCI_array_datasets/house_electric/data.npy'):
        rmse = .055
        nll =  -.152
        time = 4317.3
        compute = '8-GPU exact'
    ans = rmse, nll, time, compute
    return(ans)
 
def get_short_name(fname):
    short_name = fname
    if (fname == '/Users/marfa/documents/reformatted_UCI_array_datasets/bike/data.npy'):
        short_name = 'bike'
    if (fname == '/Users/marfa/documents/reformatted_UCI_array_datasets/ctslice/data.npy'):
        short_name = 'ctslice'
    if (fname == '/Users/marfa/documents/reformatted_UCI_array_datasets/poletele/data.npy'):
        short_name = 'poletele'
    if (fname ==  '/Users/marfa/documents/reformatted_UCI_array_datasets/protein/data.npy'):
        shortname = 'protein'
    if (fname == '/Users/marfa/documents/reformatted_UCI_array_datasets/road3d/data.npy'):
        shortname = 'road3d'
    if (fname == '/Users/marfa/documents/reformatted_UCI_array_datasets/song/data.npy'):
        shorname = 'song'
    if (fname == '/Users/marfa/documents/reformatted_UCI_array_datasets/house_electric/data.npy'):
        shortname = 'house_electric'
    return(short_name)
 
def preprocess_y_vals(y_vals,m_y,sd_y):
    #apply preprocessing transforms to y vals
    y_vals = y_vals - m_y
    y_vals = y_vals/sd_y
    return(y_vals)

#note all of the GP defs below use zero mean option
 
class ExactGP_Matern(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGP_Matern, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.base_kernel = gpytorch.kernels.MaternKernel(nu=1.5)
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)
           
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class ExactGP_Exp(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGP_Exp, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.base_kernel = gpytorch.kernels.MaternKernel(nu=0.5)
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)
           
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class ExactGP_RBF(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGP_RBF, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.base_kernel = gpytorch.kernels.RBFKernel()
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)
 
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

args = parse_args()
assum_kern = args.assum_kernel_type #default rbf
recal_data_size = args.recal_data_size #default 1000
test_data_size = args.test_data_size #default -999 which is sued to trigger use of test_data_size = int(2/9 x total_data_size)
test_data_size_lim =   args.test_data_size_cap  #default 100000000 - can be used to keep test time manageable on my laptop
xpreprocess = args.x_preprocess # type of pre-proceesing to apply to xvals
seed = args.random_seed #default 42
subset_size = args.subset_size  #default 300
max_num_subsets = args.max_nsubsets #default 10
num_nearest_neighbours = args.number_nn #default 400
xy_data_file = args.xy_input_file # xy input file
res_file = args.results_file # results file
res_file_csv = args.results_file_csv # csv results file
 
print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
print('kernel-type used for param estimation and prediction :')
print(assum_kern)
print('seed = %d' %(seed))
print('for param estimation num-subsets = %d, subset size = %d (fewer subsets used if training data too small)' %(max_num_subsets,subset_size))
print('num nearest neighbours for prediction = %d' %(num_nearest_neighbours))
print('requested test data size = %d' %(test_data_size))
print('recalibration data size = %d' %(recal_data_size))
print('type of preprocessing applied to x values:')
print(xpreprocess)
print('xy data input file')
print(xy_data_file)
print('results file')
print(res_file)
print('csv results file')
print(res_file_csv)
 
#likelihood = gpytorch.likelihoods.GaussianLikelihood()
likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1.000E-06)) #<< (needed to deal with small nv's that arise)
num_adam_iters = 200 # may be overkill, but this is what I used for my synthetic dataset evals so far
 
out_file = open(res_file, 'a')
out_file_csv =  open(res_file_csv, 'a')
 
xy_data_array = np.load(xy_data_file, mmap_mode = 'r')
nrows, ncols = xy_data_array.shape
print('in original npy  data file nrows = %d, ncols =%d' %(nrows, ncols))
xy_data_array = xy_data_array[~np.isnan(xy_data_array).any(axis=1)]
nrows, ncols = xy_data_array.shape
print('after removal of any bad rows,  npy  data file nrows = %d, ncols =%d' %(nrows, ncols))
 
x_data_set = xy_data_array[: :,  : -1 ]
y_data_set = xy_data_array[: : , -1 ]
total_data_size, input_dim = x_data_set.shape
print('total_data_size = %d, input_dim  = %d' %(total_data_size, input_dim))
if (test_data_size == -999):
    test_data_size = int ( (2.0/9.0) * total_data_size )
    print ('actual test-data_size = (2/9 x total_data_size) = %d' %(test_data_size))                             
 
np.random.seed(seed)
perm = np.random.permutation(total_data_size)  
 
train_data_size = total_data_size - (recal_data_size + test_data_size)
 
if (test_data_size > test_data_size_lim):
    test_data_size = test_data_size_lim   # temporary measure to be removed !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    print('!!!!!!!!!!!!!!reduced test_data_size  to %d, but kept same train_data_size = %d, and same recal_data_size = %d' %(test_data_size, train_data_size, recal_data_size))

#replace below with slicing operations  <<<<<<<<<<<<<<<<<<<<<<<
y_train =  np.zeros(train_data_size,dtype=np.float64)
y_recal =  np.zeros(recal_data_size,dtype=np.float64)
y_test  =  np.zeros(test_data_size,dtype=np.float64)
x_train =  np.zeros([train_data_size,input_dim],dtype=np.float64)
x_recal =  np.zeros([recal_data_size,input_dim],dtype=np.float64)
x_test  =  np.zeros([test_data_size,input_dim],dtype=np.float64)
for i in range(train_data_size):
    y_train[i] = y_data_set[perm[i]]
    for j in range(input_dim):
        x_train[i][j] = x_data_set[perm[i]][j]
for i in range(recal_data_size):
    y_recal[i] = y_data_set[perm[i+train_data_size]]
    for j in range(input_dim):
        x_recal[i][j] = x_data_set[perm[i+train_data_size]][j]
for i in range(test_data_size):
    y_test[i] = y_data_set[perm[i+train_data_size+recal_data_size]]
    for j in range(input_dim):
        x_test[i][j] = x_data_set[perm[i+train_data_size+recal_data_size]][j]

num_subsets = math.floor(float(train_data_size)/float(subset_size))
if (num_subsets > max_num_subsets):
    num_subsets = max_num_subsets
print('num_subsets = %d' %(num_subsets))

train_time = 0.0
 
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# preprocess the (x,y) training values  (in the case of x this might optionally either be 'prewhitening' or axis rescaling)
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
print('preprocess training data', flush = True)
tic = time.perf_counter()
  # derive mean and sd of y train values, mean of x train components, inverse covar of x train cpts, and diag matrix with 1/(sd x-cpt) along the diag
m_y, sd_y, m_x, prep_mat_x =  get_xy_preprocess_tranforms(x_train,y_train,xpreprocess,train_data_size,input_dim)
  # Apply transform y <- (y-m_y)/sd_y and x <- inv_M.(x-m_x) to the x and y vals in training data
#xpreprocess == 'axis_rescale'
x_train = preprocess_x_vals(x_train,m_x,prep_mat_x)
y_train = preprocess_y_vals(y_train,m_y,sd_y)
toc = time.perf_counter()
set_up_time = toc - tic
print('set_up time to preprocess training data = %f' %(set_up_time))
train_time += set_up_time
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
 
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
print('generate nn table for nn prediction capability')
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
tic = time.perf_counter()
neigh = NearestNeighbors(n_neighbors=num_nearest_neighbours)
neigh.fit(x_train)
toc = time.perf_counter()
nn_table_time = toc - tic
print('nn_table_time =  %.8f seconds' %(nn_table_time))
train_time +=  nn_table_time
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
print('Phase 1 parameter estimation')
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
tic = time.perf_counter()
x_phase1 = x_train[0:(num_subsets*subset_size), ::]
y_phase1 =  y_train[0:(num_subsets*subset_size)]
est_ls, est_nv, est_ks = get_param_estimates2(x_phase1,y_phase1,input_dim,subset_size,num_subsets,num_adam_iters,assum_kern)
toc = time.perf_counter()
time_phase1 = toc - tic
print('est_ls = %f, est_nv = %f, est_ks = %f, time_phase1 = %f' %(est_ls, est_nv, est_ks, time_phase1))
train_time +=  time_phase1
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
 
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#Phase 2 parameter calibration (only needed for improved uncertainty measures)
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
print('Phase2 estimation: compute mscal from calibation data in order to enable subsequent calibration')
#compute mscal on calibration data in order to be able to perform recalibration
tic = time.perf_counter()
# Apply preprocessing transform to the x,y vals in the calibration data
x_recal = preprocess_x_vals(x_recal,m_x,prep_mat_x)
y_recal = preprocess_y_vals(y_recal,m_y,sd_y)
nn_pred_mean_recal, nn_pred_sd_recal = make_nn_predictions2(x_recal,recal_data_size,x_train,y_train,input_dim,est_ls,est_ks,est_nv,neigh,assum_kern)
mse, nll, mscal = evaluate_nn_predictions2(nn_pred_mean_recal, nn_pred_sd_recal, recal_data_size, y_recal)
toc = time.perf_counter()
time_phase2 = toc - tic
print('obtained mscal = %f (to be used for recalibration purposes) ' %(mscal))
print('[on calibration data: mse = %.5f,  nll = %.5f]'  %(mse, nll))
print('phase2 time (for calibration) = %f' %(time_phase2))
train_time +=  time_phase2
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
 
print('overall train_time = %f' %(train_time))
 
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#obtain predictions from the test data
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
print('use  parameters to obtain predictions from test data')
tic = time.perf_counter()
# Apply preprocessing transform to the x vals in the test data
x_test = preprocess_x_vals(x_test,m_x,prep_mat_x)
# note the recalibration below where est_ks and est_nv are scaled by the phase2 estimated mscal value
nn_pred_mean_test, nn_pred_sd_test = make_nn_predictions2(x_test,test_data_size,x_train,y_train,input_dim,est_ls,(mscal*est_ks),(mscal*est_nv),neigh,assum_kern)
toc = time.perf_counter()
tot_predict_time = toc-tic
per_predict_time = tot_predict_time /float(test_data_size)
print('tot_predict_time = %f, per_predict_time = %f ' %(tot_predict_time,per_predict_time))
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
 
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
print('find performance of those predictions on the test data')
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#first on preprocessed y vals:
prep_y_test = preprocess_y_vals(y_test,m_y,sd_y)
mse, nll, mscal = evaluate_nn_predictions2(nn_pred_mean_test, nn_pred_sd_test, test_data_size, prep_y_test) 
print('for preprocessed test y vals : mse = %.5f,  nll = %.5f, test data mscal = %.5f '  %(mse, nll, mscal))
rmse_million, nll_million, time_million, compute_million = get_million_paper_results(xy_data_file)
print('xy datafile =', xy_data_file , file = out_file, end ='')
print( '     kernel =', assum_kern, '   seed = %d'  %(seed), file = out_file)
print('mse = %.5f,  nll = %.5f, mscal = %.5f, rmse = %.5f,  rmse_million = %f, nll_million = %f, per_predict_time = %f, train_time = %f '  %(mse, nll, mscal, mse**.5, rmse_million, nll_million, per_predict_time, train_time), file = out_file, flush = True)
short_name = get_short_name(xy_data_file)
print('%s,%d,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f'  %(short_name,seed, mse, nll, mscal, mse**.5, per_predict_time, set_up_time, nn_table_time, time_phase1, time_phase2,  train_time), file = out_file_csv, flush = True)