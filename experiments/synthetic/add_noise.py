import math
import numpy as np
import argparse
 
def parse_args():
    parser = argparse.ArgumentParser()
    # Noisevar input param
    parser.add_argument('-noise_var', '--nvar',
                        help='noise variance',
                        default=0.2,type = float)
    # Input and output file args
    parser.add_argument('-xyinfl', '--xy_inp_file',
                        help='xy input file',
                        default='base_oak_xy_data.npy')
    parser.add_argument('-xyoutfl', '--xy_out_file',
                        help='xy vals (noisy)  output file',
                        default='noisy_oak.npy')
    parser.add_argument('-rep_nv_fl', '--nv_report_file',
                        help='output file where noisevar value is appended',
                        default='oak_results_file')
    return parser.parse_args()
 
args = parse_args()
noisevar =  args.nvar  #default 0.2
nv_outfile = open(args.nv_report_file,'a') 
 
xy_data_array = np.load(args.xy_inp_file , mmap_mode = 'r')
nrows, ncols = xy_data_array.shape
print('when running "add-noise"  base npy  data file has nrows = %d, ncols =%d' %(nrows, ncols))
# add iid Laplace noise
xy_data =  np.zeros([nrows,ncols],dtype=np.float64)
for i in range(nrows):
    for j in range(ncols-1):
        xy_data[i][j] = xy_data_array[i][j]
    xy_data[i][ncols-1] = xy_data_array[i][ncols-1] + ((0.5 *noisevar) ** 0.5) * np.random.laplace(loc=0.0, scale=1.0 )
np.save(args.xy_out_file, xy_data)
perfect_limit_nll =  0.5 *(1.0 + math.log(noisevar) + math.log(2.0*math.pi))
print('%f,%f'  %(noisevar,perfect_limit_nll) , file= nv_outfile)