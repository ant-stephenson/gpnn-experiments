# To understand what this is doing see algorithm 1 in paper

import argparse
import torch
import math
import numpy as np
import gpytorch
import time
from itertools import product as cartesian_prod
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance_matrix

def k_se(x1: np.ndarray, x2: np.ndarray, sigma, ls) -> np.ndarray:
    return sigma * np.exp(-distance_matrix(x1, x2)**2/(2*ls**2))


def k_per(x1: np.ndarray, x2: np.ndarray, sigma, ls, p) -> np.ndarray:
    return sigma * np.exp(-2*np.sin(np.pi/p * distance_matrix(x1, x2)/(ls**2))**2)


def k_mat_half(x1: np.ndarray, x2: np.ndarray, sigma, ls) -> np.ndarray:
    return sigma * np.exp(-distance_matrix(x1, x2)/ls)


def k_exp(x1: np.ndarray, x2: np.ndarray, sigma, ls) -> np.ndarray:
    return k_mat_half(x1, x2, sigma, ls)


def k_mat_3half(x1: np.ndarray, x2: np.ndarray, sigma, ls) -> np.ndarray:
    D = distance_matrix(x1, x2)
    return sigma * (1 + np.sqrt(3) * D/ls) * np.exp(-np.sqrt(3)*D/ls)


def parse_args():
    parser = argparse.ArgumentParser()

    # Problem Shape Args
    parser.add_argument('-d', '--dimension',
                        help='Dimensionality of input data',
                        default=10, type=int)
    parser.add_argument('-n_train', '--train-data-size',
                        help='Size of train data set',
                        default=100000, type=int)
    parser.add_argument('-n_test', '--test-data-size',
                        help='Size of test data set',
                        default=1000, type=int)

    # Truth GP Args
    parser.add_argument('-tker', '--true-kernel-type',
                        help='Kernel type to use in data-generation',
                        default='RBF', choices=['RBF', 'Matern', 'Exp'])
    parser.add_argument('-tks', '--true-kernel-scale',
                        help='Truth value for output scale hyperparameter ',
                        default=0.8, type=float)
    parser.add_argument('-tl', '--true-lenscale',
                        help='Truth value for length scale hyperparameter',
                        default=1.0, type=float)
    parser.add_argument('-tnv', '--true-noisevar',
                        help='Truth value for noise_var',
                        default=0.2, type=float)

    # Assumed GP params - in sensitivity analysis two of the hyparams (currently anv and aks) will be held constant while the other is varied
    parser.add_argument('-aks', '--assum-kernel-scale',
                        help='Assumed value for output scale hyperparameter ',
                        default=0.8, type=float)
    parser.add_argument('-al', '--assum-lenscale',
                        help='Assumed value for length scale hyperparameter',
                        default=1.0, type=float)
    parser.add_argument('-anv', '--assum-noisevar',
                        help='Assumed value for noise_var',
                        default=0.2, type=float)

    # Args for param to vary

    parser.add_argument(
        '-varpar', '--param_to_vary',
        help='parameter to be varied in sensitivity analysis',
        default='lenscale', choices=['lenscale', 'noisevar', 'kernelscale'])
    parser.add_argument('-maxval', '--max_parval',
                        help='Largest param value',
                        default=5.0, type=float)
    parser.add_argument('-minval', '--min_parval',
                        help='Smallest param value',
                        default=0.1, type=float)
    parser.add_argument('-numvals', '--num_parvals',
                        help='Number of values between limits',
                        default=40, type=int)

    # prediction params
    parser.add_argument('-numnn', '--number-nn',
                        help='Number of nearest neighbours used for prediction',
                        default=400, type=int)

    # Miscellaneous Args
    parser.add_argument('-seed', '--random-seed',
                        help='Random seed to initialise',
                        default=42, type=int)

    parser.add_argument('-out', '--output_file',
                        help='Name of output file',
                        default="sim_gpnn_limits_results_test", type=str)

    parser.add_argument('-array_idx', '--array_index',
                        help='',
                        default=1, type=int)

    return parser.parse_args()


array_sets = cartesian_prod([int(10**i) for i in range(4, 8)], [
                            "RBF", "Matern", "Exp"])
array_sets_dict = {k+1: {'train_data_size': v[0], 'true_kernel_type': v[1]}
                   for k, v in enumerate(array_sets)}


def generate_spherical_gaus_xvals(train_data_size, test_data_size, input_dim):
    # N(0,(1/d)I_d) x_vals (whose squared-len distro has mode 1)
    tic = time.perf_counter()
    mean_x = np.zeros([input_dim], dtype=np.float64)
    cov_x = np.identity(input_dim)/float(input_dim)
    data_size = train_data_size + test_data_size
    #x = np.random.multivariate_normal(mean_x, cov_x, data_size)
    x = rng.multivariate_normal(mean_x, cov_x, data_size).astype(np.float64)
    x_train = x[:train_data_size, :]
    x_test = x[train_data_size:, :]
    toc = time.perf_counter()
    print('time to generate all required xvals = %f' % (toc-tic))
    return x_train, x_test


def get_subset_yvals(
        all_subset_xvals, combined_subset_size, input_dim, act_kern, act_ls,
        act_nv, act_ks):
    # get y vals based on true kernel family and true param vals
    ymean = np.zeros([combined_subset_size], dtype=np.float64)
    ycovar = gen_ycovar(act_ks, act_nv, act_ls, all_subset_xvals,
                        combined_subset_size, input_dim, act_kern)
    #print('ycovar recovered')
    #y_vals = np.random.multivariate_normal(ymean, ycovar)
    y_vals = rng.multivariate_normal(ymean, ycovar)
    #print('y_vals computed')
    return y_vals


def gen_ycovar(act_ks, act_nv, act_ls, x_double, size, inp_dim, act_kern):
    y_covar = np.zeros([size, size], dtype=np.float64)
    if (act_kern.lower() == 'rbf'):
        y_covar = k_se(x_double, x_double, act_ks, act_ls)
    if (act_kern.lower() == 'matern'):
        y_covar = k_mat_3half(x_double, x_double, act_ks, act_ls)
    if (act_kern.lower() == 'exp'):
        y_covar = k_exp(x_double, x_double, act_ks, act_ls)
    y_covar += act_nv * np.identity(size)
    return (y_covar)

def generate_xy_nn_prediction_subsets(
        x_predict, predict_data_size, x_train, train_data_size,
        num_nearest_neighbours, input_dim, neigh, act_ls, act_ks, act_nv,
        act_kern):
    # Generate a concatination of predict_data_size sets, each of size (1+num_nearest_neighbour), both for x vals and for y vals
    # These are placed in np_xsets and np_ysets respectively
    np_this_xset = np.zeros(
        [(num_nearest_neighbours + 1),
         input_dim],
        dtype=np.float64)
    np_predict_x = np.zeros([1, input_dim], dtype=np.float64)
    np_xsets = np.zeros(
        [(predict_data_size * (num_nearest_neighbours + 1)),
         input_dim],
        dtype=np.float64)
    np_ysets = np.zeros(
        [(predict_data_size * (num_nearest_neighbours + 1))],
        dtype=np.float64)
    mnn_retrieve_time = 0.0
    # first construct x sets
    for i_nnsubset in range(predict_data_size):
        if((i_nnsubset % 10) == 0):
            print('generating nn_xysubsets for predict point %d' % (i_nnsubset))
        for j in range(input_dim):
            np_predict_x[0][j] = x_predict[i_nnsubset][j]
        tic = time.perf_counter()
        neigh_list = neigh.kneighbors(np_predict_x, return_distance=False)
        for i in range(num_nearest_neighbours):
            ind = neigh_list[0][i]
            for j in range(input_dim):
                np_this_xset[i][j] = np_xsets[(
                    i_nnsubset * (num_nearest_neighbours+1)) + i][j] = x_train[ind][j]
        toc = time.perf_counter()
        mnn_retrieve_time += toc - tic
        for j in range(input_dim):
            np_this_xset[num_nearest_neighbours][j] = np_xsets[(
                i_nnsubset * (num_nearest_neighbours+1)) + num_nearest_neighbours][j] = np_predict_x[0][j]
        # now construct y sets based on actual ('correct') kernel family and actual ('correct') param vals
        y_vals = get_subset_yvals(
            np_this_xset, (num_nearest_neighbours + 1),
            input_dim, act_kern, act_ls, act_nv, act_ks)
        for i in range(num_nearest_neighbours+1):
            np_ysets[(i_nnsubset * (num_nearest_neighbours+1)) + i] = y_vals[i]
    mnn_retrieve_time /= float(predict_data_size)
    ans = np_xsets, np_ysets, mnn_retrieve_time
    return (ans)


def evaluate_nn_predictions(
    np_xsets, np_ysets, predict_data_size, num_nearest_neighbours,
        input_dim, assum_ls, assum_ks, assum_nv, assum_kern):
    # nearest neighbour predictions on pre-constructed nn x,y sets + associated x,y target points
    mse_nn = 0.0
    me_nn = 0.0
    nll_nn = 0.0
    mscal_nn = 0.0
    mcal_nn = 0.0
    mnn_predict_time = 0.0
    ymean = np.zeros([num_nearest_neighbours+1], dtype=np.float64)
    np_predict_x = np.zeros([1, input_dim], dtype=np.float64)
    np_nearest_x = np.zeros(
        [num_nearest_neighbours, input_dim],
        dtype=np.float64)
    np_nearest_y = np.zeros([num_nearest_neighbours], dtype=np.float64)

    for i_trial in range(predict_data_size):
        nn_ind = np.s_[(i_trial * (num_nearest_neighbours+1))                       :((i_trial+1) * (num_nearest_neighbours+1)-1)]
        last_ind = (i_trial * (num_nearest_neighbours+1)
                    ) + num_nearest_neighbours
    # if((i_trial % 100) == 0):
    # print(' nn-prediction i_trial = %d' %(i_trial))
    # collect together both nn xvals and target xval for this trial
        np_predict_x[0, :] = np_xsets[last_ind, :]
        np_nearest_x = np_xsets[nn_ind, :]
    # collect together both nn yvals and true target yval for this trial
        true_y = np_ysets[last_ind]
        np_nearest_y = np_ysets[nn_ind]
        # convert to the tensor format required for the gpytorch follow-on processing
        predict_x = torch.from_numpy(np_predict_x)
        nearest_x = torch.from_numpy(np_nearest_x)
        nearest_y = torch.from_numpy(np_nearest_y)
        # set up nn model with the assumed (possibly mispecified) kernel family and parameters and put in eval mode ready to make predictions
        tic = time.perf_counter()
        if (assum_kern.lower() == 'rbf'):
            nn_model = ExactGP_RBF(nearest_x, nearest_y, likelihood)
        if (assum_kern.lower() == 'matern'):
            nn_model = ExactGP_Matern(nearest_x, nearest_y, likelihood)
        if (assum_kern.lower() == 'exp'):
            nn_model = ExactGP_Exp(nearest_x, nearest_y, likelihood)
        nn_model.double()
        nn_model.covar_module.base_kernel.lengthscale = assum_ls
        nn_model.likelihood.noise = assum_nv
        nn_model.covar_module.outputscale = assum_ks
        nn_model.eval()
        output_nn_model = nn_model(predict_x)
    # extract nn_model predictive mean and sd values corresponding to location predict_x
        nn_pred_mean = output_nn_model.mean
        nn_predf_var = output_nn_model.variance  # doesn't account for noise
        nn_pred_var = nn_predf_var + assum_nv
        nn_pred_sd = nn_pred_var**0.5
        toc = time.perf_counter()
        mnn_predict_time += toc - tic
    # add this trial's results to the stats collection
        const = (2.0*math.pi)**0.5
        nll_nn += math.log(1.0/(nn_pred_sd*const))
        nll_nn -= 0.5 * ((true_y - nn_pred_mean)/nn_pred_sd) ** 2
        e_nn = nn_pred_mean - true_y
        me_nn += e_nn
        mse_nn += e_nn ** 2
        cal = (true_y - nn_pred_mean)/nn_pred_sd
        mscal_nn += cal ** 2
        mcal_nn += cal
    # compute average stats over all predictions
    nll_nn = -nll_nn
    nll_nn /= float(predict_data_size)
    mse_nn /= float(predict_data_size)
    me_nn /= float(predict_data_size)
    mscal_nn /= float(predict_data_size)
    mcal_nn /= float(predict_data_size)
    sdcal_nn = mscal_nn - (mcal_nn ** 2)
    wass_nn = ((mcal_nn - 0.0)**2) + ((sdcal_nn - 1.0)**2)
    mnn_predict_time /= float(predict_data_size)
    ans = mse_nn, nll_nn, wass_nn, mscal_nn, me_nn, mnn_predict_time
    return(ans)


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

array_set = array_sets_dict[args.array_index]

if args.array_index != 0:
    args.true_kernel_type = array_set["true_kernel_type"]
    args.train_data_size = array_set["train_data_size"]

act_kern = args.true_kernel_type  # default rbf
act_ks = args.true_kernel_scale  # default 0.8
act_ls = args.true_lenscale  # default 1.0
act_nv = args.true_noisevar  # default 0.2
assum_ks = args.assum_kernel_scale  # default 0.8
assum_ls = args.assum_lenscale  # default 1.0
assum_nv = args.assum_noisevar  # default 0.2
input_dim = args.dimension  # default 10
train_data_size = args.train_data_size  # default 100000
test_data_size = args.test_data_size  # default 1000
seed = args.random_seed  # default 42
num_nearest_neighbours = args.number_nn  # default 400
varypar = args.param_to_vary  # default lenscale
max_varypar = args.max_parval  # default 5.0
min_varypar = args.min_parval  # default 0.1
num_vals = args.num_parvals
out_file_name = args.output_file

print('input dim = %d' % (input_dim))
print('kernel-type and parms used to generate data:')
print(act_kern)
print('act_ls = %f, act_ks = %f, act_nv =%f ' % (act_ls, act_ks, act_nv))
print('assum_ls = %f, assum_ks = %f, assum_nv =%f ' %
      (assum_ls, assum_ks, assum_nv))
print('WILL BE VARYING FOLLOWING ASSUMED PARAMETER VALUE')
print(varypar)
print('over %d values between %f and %f' % (num_vals, min_varypar, max_varypar))
print('seed = %d' % (seed))
print('num nearest neighbours for prediction = %d' % (num_nearest_neighbours))
print('training data size = %d' % (train_data_size))
print('test data size = %d' % (test_data_size))


rng = np.random.default_rng(seed)  # being treated like global param


print('generate x vals', flush=True)
x_train, x_test = generate_spherical_gaus_xvals(
    train_data_size, test_data_size, input_dim)
# being treated like global param
likelihood = gpytorch.likelihoods.GaussianLikelihood()

best_pos_mse = act_nv
perfect_limit_nll = 0.5 * (1.0 + math.log(act_nv) + math.log(2.0*math.pi))
perfect_limit_mscal = 1.0

# set up neigh for prediction purposes (to be applied for all predictions henceforth)
print('generate nn table for nn prediction capability', flush=True)

tic = time.perf_counter()
neigh = NearestNeighbors(n_neighbors=num_nearest_neighbours)
neigh.fit(x_train)
toc = time.perf_counter()
duration = toc - tic
print(' nn table construction complete and took %.8f seconds' %
      (duration), flush=True)


# generate x,y predict points and nearest neighbour sets
tic = time.perf_counter()
np_xsets, np_ysets, mnn_retrieve_time = generate_xy_nn_prediction_subsets(
    x_test, test_data_size, x_train, train_data_size, num_nearest_neighbours,
    input_dim, neigh, act_ls, act_ks, act_nv, act_kern)
toc = time.perf_counter()
duration = toc - tic
print(' xy nn based sets contructed for all %d target s vals complete and took %f seconds' %
      (test_data_size, duration))
print('average time per target x value to retrieve its nearest neighbours = %.8f seconds' %
      (mnn_retrieve_time), flush=True)

# now make predictions based on varying asumptions concerning the kernel family and hyperparameter values


def inc_varypar(i, varypar):
    max_varypar = {'lenscale': 3.0,
                   'kernelscale': 3.0, 'noisevar': 1.0}[varypar]
    min_varypar = {'lenscale': 0.1,
                   'kernelscale': 0.1, 'noisevar': 0.01}[varypar]
    varypar_inc = float(i) * (max_varypar - min_varypar)/float(num_vals)
    return min_varypar + varypar_inc


with open(out_file_name, 'a') as out_file:
    # header = "n,n_test,d,m,seed,k_true,k_model,ks,ls,nv,assum_ks,assum_ls,assum_nv,varypar,mse,nll,mscal"
    # print(header, file=out_file, flush=True)

    for icase in range(3):
        if (icase == 0):
            assum_kern = 'RBF'
        if (icase == 1):
            assum_kern = 'Matern'
        if (icase == 2):
            assum_kern = 'Exp'

        print(
            f'XXXXXXXXXXXX prediction performance using assumed ker = {assum_kern}')
        for varypar in ['lenscale', 'kernelscale', 'noisevar']:
            print('with assum_ls = %f, assum_ks = %f, assum_nv = %f' %
                  (assum_ls, assum_ks, assum_nv))
            assum_vars = {
                'lenscale': assum_ls, 'kernelscale': assum_ks,
                'noisevar': assum_nv}
            for i in range(num_vals+1):
                assum_vars[varypar] = inc_varypar(i, varypar)
                mse, nll, wass, mscal, me, mnn_predict_time = evaluate_nn_predictions(
                    np_xsets, np_ysets, test_data_size, num_nearest_neighbours, input_dim, assum_vars['lenscale'], assum_vars['kernelscale'], assum_vars['noisevar'], assum_kern)

                line = f"{train_data_size},{test_data_size},{input_dim},{num_nearest_neighbours},{seed},{act_kern},{assum_kern},{act_ks},{act_ls},{act_nv},{assum_vars['kernelscale']},{assum_vars['lenscale']},{assum_vars['noisevar']},{varypar},{mse.item()},{nll.item()},{mscal.item()}"
                print(line, file=out_file, flush=True)



