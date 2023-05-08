# gpnn-experiments
Code to reproduce results of the paper. Includes code to run distributed methods and SVGP as well as our own method. Additionally includes code to run simulations as well as scripts to analyse the output data and generate tables and figures.

A "requirements.txt" file is included in the repository with all of the necessary dependencies to reproduce all of these results.
`pip install -r requirements.txt` will install the required packages.

## Main results table
<table border="1" class="dataframe">  <thead>    <tr>      <th></th>      <th></th>      <th></th>      <th colspan="3" halign="left">NLL</th>      <th colspan="3" halign="left">RMSE</th>    </tr>    <tr>      <th></th>      <th></th>      <th></th>      <th>Distributed</th>      <th>OURS</th>      <th>SVGP</th>      <th>Distributed</th>      <th>OURS</th>      <th>SVGP</th>    </tr>    <tr>      <th>Dataset</th>      <th>n</th>      <th>d</th>      <th></th>      <th></th>      <th></th>      <th></th>      <th></th>      <th></th>    </tr>  </thead>  <tbody>    <tr>      <th>Poletele</th>      <th>4.6e+03</th>      <th>19</th>      <td>0.0091 ± 0.015</td>      <td>-0.213 ± 0.021</td>      <td>-0.0667 ± 0.017</td>      <td>0.241 ± 0.0033</td>      <td>0.195 ± 0.0046</td>      <td>0.226 ± 0.0059</td>    </tr>    <tr>      <th>Bike</th>      <th>1.4e+04</th>      <th>13</th>      <td>0.977 ± 0.0057</td>      <td>0.954 ± 0.013</td>      <td>0.93 ± 0.0043</td>      <td>0.634 ± 0.004</td>      <td>0.625 ± 0.0078</td>      <td>0.606 ± 0.0033</td>    </tr>    <tr>      <th>Protein</th>      <th>3.6e+04</th>      <th>9</th>      <td>1.11 ± 0.0051</td>      <td>1.01 ± 0.0019</td>      <td>1.05 ± 0.0059</td>      <td>0.733 ± 0.0038</td>      <td>0.666 ± 0.0017</td>      <td>0.688 ± 0.0043</td>    </tr>    <tr>      <th>Ctslice</th>      <th>4.2e+04</th>      <th>378</th>      <td>-0.159 ± 0.052</td>      <td>-1.26 ± 0.011</td>      <td>0.467 ± 0.016</td>      <td>0.237 ± 0.012</td>      <td>0.132 ± 0.0007</td>      <td>0.384 ± 0.0064</td>    </tr>    <tr>      <th>Road3D</th>      <th>3.4e+05</th>      <th>2</th>      <td>0.685 ± 0.0041</td>      <td>0.371 ± 0.0041</td>      <td>0.608 ± 0.018</td>      <td>0.478 ± 0.0023</td>      <td>0.351 ± 0.0014</td>      <td>0.443 ± 0.008</td>    </tr>    <tr>      <th>Song</th>      <th>4.6e+05</th>      <th>90</th>      <td>1.32 ± 0.0012</td>      <td>1.2 ± 0.0025</td>      <td>1.24 ± 0.0012</td>      <td>0.851 ± 6.7e-05</td>      <td>0.801 ± 0.0025</td>      <td>0.834 ± 0.0011</td>    </tr>    <tr>      <th>Houseelectric</th>      <th>1.6e+06</th>      <th>8</th>      <td>-1.34 ± 0.0013</td>      <td>-1.56 ± 0.042</td>      <td>-1.46 ± 0.0046</td>      <td>0.0626 ± 5.2e-05</td>      <td>0.0506 ± 0.0021</td>      <td>0.0566 ± 0.00011</td>    </tr>  </tbody></table>

## Guide to reproduce UCI results
### Reproducing benchmark methods (Distributed and SVGP)

### Reproducing our method
Our method is run via the "method_of_paper.py" file, using command line, e.g.
`python3 method_of_paper.py -xyinfl bike/data.npy -aker RBF  -seed 11   -n_recal 1000`

### Generating the results table
The results from all of the distributed methods can be combined and then imported into python with the results of SVGP and our method and analysed with the aid of the "transform_results.py" script which imports various functions from the "results_utils.py" file.

## Guide to reproduce simulated results
Run the below snippet with relevant location filled in (and choices of parameters) in a bash script.
```
SCRIPT_DIR=<loc>

VAR_PAR=noisevar
DIM=15

OUTFILE="sim_gpnn_limits_results_d${DIM}"

python3 $SCRIPT_DIR/simulate_GPNN_limits_alg1.py -n_train 100000 -n_test 1000 -d $DIM \
    -tker RBF -tks 0.9 -tl 0.75 -tnv 0.1 -aks 1.0 -al 0.75 -anv 0.2 \
    -varpar $VAR_PAR -numvals 40 -numnn 400 -seed 3 \
    -out "${OUTFILE}" \
    -array_idx 0 \
```

This will output a csv file ("sim_gpnn_limits_results_d15.csv") with all of the results recorded for analysis. Analysis can be conducted with the "analyse_GPNN_limit_sims.py" file.

## Guide to reproducing results on synthetic data
`python3  truncate_xy_file.py  -fracfile 0.0001 -xyinfl noisy_oak_var0.05.npy  -xyoutfl trunc_noisy_oak.npy -rep_size_fl oak_res.csv`
