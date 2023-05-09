import numpy as np
import argparse
 
def parse_args():
    parser = argparse.ArgumentParser()
    # Truncation param
    parser.add_argument('-fracfile', '--propfile',
                        help='proportion of file to extract ',
                        default=1.0,type = float)
    # Input and output file args
    parser.add_argument('-xyinfl', '--xy_inp_file',
                        help='xy vals input file',
                        default='noisy_oak.npy')
    parser.add_argument('-xyoutfl', '--xy_out_file',
                        help='xy vals truncated output file',
                        default='trunc_noisy_oak.npy')
    parser.add_argument('-rep_size_fl', '--trunc_report_file',
                        help='output file where truncated len value is appended',
                        default='oak_results_file')
    return parser.parse_args()
 
args = parse_args()
trunc_prop =  args.propfile  #default 1.0
trunclen_outfile = open(args.trunc_report_file, 'a')
xy_data_array = np.load(args.xy_inp_file , mmap_mode = 'r')
 
nrows, ncols = xy_data_array.shape
print('when running "truncate"  base npy  data file has nrows = %d, ncols =%d' %(nrows, ncols))
trunclen = int(trunc_prop*float(nrows))
if (trunc_prop  == 1.0):
    trunclen = nrows
print('trunclen = %d' %(trunclen))
np.save (args.xy_out_file, xy_data_array[0 :trunclen, : ])
print('%d,'  %(trunclen) , file= trunclen_outfile, end = '')