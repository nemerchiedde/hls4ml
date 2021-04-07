# cppm-nnlar

** README under construction **

checkout packages :
```
git clone ssh://git@gitlab.cern.ch:7999/tcalvet/cppm-nnlar.git
```




## common_io

Sharing plateform for input/ouput data, mainly between `cpp_rnn_emulator` and `hls-cppm` codes. It should be limited to very few small size reference i/o, namely :
* input_data/ : mu=140 - random-E random-gap -  BT_flat - EMBMiddle_eta0.5125_phi0.0125 - /quark3/tcalvet/LAr/phase_2/areus_simulation/2021.02.03_areus_sim/mu140/flat/OF5_rdGap_rdSig/digitization_monitorOF5_eta_0.5125_phi_0.0125_EMMiddle_5GeV_WithNoise_2MBC.root using the 2nd half of the data (correspond to test set a priori, may be few BC overlap, 1M BC, -6 for OFMax alignement, no discontinuities)
   * `data.txt` : digitized ADC output from AREUS simulation
   * `true.txt` : true energy deposit from same simulation
   * `ofmax.txt` : Optimal-Filter (5 BC) + MaxFinder energy prediction for reference
* outputs/ :
   * `reference_prediction.txt` : output from `cpp_rnn_emulator` code using default common training setting (8 timestamps, 4 units) and `data_normalized.txt` (to study : diff with `data.txt`) -- TO BE ADDED
   * `reference_hls.txt` : not necessary, same as previous from running with `hls-cppm` code -- TO BE ADDED
* lstm_weights/ :
   * `weights.h5` : Keras `save_model` output from default common training  (8 timestamps, 4 units)
   * can add more if they are small enough files

To be noted, git not made for sharing data. If this extend too much need an other form of sharing. Also ".txt" files are probably not the best choice ... but current easy way through.





## macros

Set of common macros for various purposes. For now contains :
* `resolution_drawer.C` : draw energy resolutions in different energy bins, takes as inputs ".txt" files one entry per line, as are output by `cpp_rnn_emulator` and `hls-cppm` ... see next section
* `comparison_quartus_cpp.C` : copy of resolution_drawer dedicated to hls/cpp comparison with improved reporting
* `areus_to_txt.py` : turn AREUS outputs (root or transformed to hdf5 with Anne's script) into ".txt" files one entry per line, used to generate the `common_io/input_data/` files
* `areus_data_investigator.C` : read single AREUS output file and dump basic properties to validate data files (counts per E bin, E gaps, OF training coefs, ...)
* `keras_model_generator.py` : script to create a test RNN model, build recurrent data (E*(1/t) for short series, E=param, t=time, E*harmonic-oscillator(t) for longer series), create LSTM network, train and dump some basic information


### comparison_quartus_cpp.C

Macros for plotting all resolution plots for cpp VS hls comparison. Also dump summary text report 

Macros to compile within root and run live with several options

Compilation :
```
g++ comparison_quartus_cpp.C -o resdrawer -lm -g -Wall -pthread -m64 -I/${PATHTOROOTBUILT}/include -L/${PATHTOROOTBUILT}/lib -lGui -lCore -lImt -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lROOTVecOps -lTree -lTreePlayer -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lThread -lMultiProc -lROOTDataFrame -pthread -lm -ldl -rdynamic
```
PATHTOROOTBUILT is to define yourself, most of the flags are default root (part with PATHTOROOTBUILT as well), you can get them from the terminal using :
```
`echo root-config --cflags --glibs` | sed s/" -std=c++11"//g
```
and then the code is run as (no option required for default run) :
```
./resdrawer
```
to see available options
```
./resdrawer -h
```


### resolution_drawer.C

Macros for plotting all resolution plots either for cpp VS hls comparison or analysis and comparison of RNN trainings/perf in various configs. 

#### Running the code

Macros to compile within root and run live with several options (support for RNN analysis to be added, only hlsVScpp really ok)

Compilation :
```
g++ resolution_drawer.C -o resdrawer -lm -g -Wall -pthread -m64 -I/${PATHTOROOTBUILT}/include -L/${PATHTOROOTBUILT}/lib -lGui -lCore -lImt -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lROOTVecOps -lTree -lTreePlayer -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lThread -lMultiProc -lROOTDataFrame -pthread -lm -ldl -rdynamic
```
PATHTOROOTBUILT is to define yourself, most of the flags are default root (part with PATHTOROOTBUILT as well), you can get them from the terminal using :
```
`echo root-config --cflags --glibs` | sed s/" -std=c++11"//g
```
and then the code is run as :
```
./resdrawer
```
run
```
./resdrawer -h
```
to get the list of currently supported options.

The code covers 2 aspects.

**hls vs cpp comparisons : `--hlsvscpp`**

Normally it is configured to work by default on files like Etienne and I used last time. However, one should specify :
* a reference file (true energy) use `--ein` default is https://gitlab.cern.ch/tcalvet/cppm-nnlar/-/blob/master/common_io/input_data/true.txt (that one actually can be kept, unless need to normalize in which case we need to reproduce that file with normalized entries -- flag and support to be added)
* the cpp prediction, to set it use `--cppin`, it defaults to the ofmax prediction now (ref file to be put in git) 
* the hls prediction, to set it use `--hlsin`, it defaults to the ofmax prediction now (ref file to be put in git)
The alignment between the files can be settled with options removing the first/last n elements in any of these files, see help command.


**RNN studies : no flag**

Currently all is hardcodded. Proper support could be designed. For now see examples written by Thomas.


#### Design

The code is designed to produce comparison plots for N methods in bins of true energy, excluding zeros from predictions, and for bins of an additional observable (named after mu cause first to test).

The code then circles around 3 structures : `ResoHolder`, `HistoHolder` and `CompHolder`.
* `ResoHolder` is the object to fill with prediction and true inputs. The "additional observable" is to be specified as this is generally expected to arise from different predictions (like mu) and therefore different BC counters.
* `HistoHolder` and `CompHolder` are built from `ResoHolder` entries via `make_all_histograms`, they are different stucts to organize the plotting.
-- description to detail further

To learn how to use these classes, refer to the function `run_alg_comp` and follow some examples. -- more description to add





## cpp_rnn_emulator

c++ code to calculate the output of a LSTM based RNN. Compile with :
```
make
```
and run with
```
./main_lstm.exe
```

Supported options :
* The RNN weights are set via a `weight.h` file found in `config/X`. By default `X=lstm_weight_reference` but it can be changed to anything in the compiling step. For example, one can target an other RNN available in the repo (config/lstm_weight_valid_l1_d1_t10_u4/weights.h) using :
```
make CONFIG=lstm_weight_valid_l1_d1_t10_u4
```
* The code output is a single `.txt` file containing prediction at each BC, one entry per line. The target output can be modified with the `-o` or `--output` options :
```
./main_lstm.exe -o test.txt
```



## trainAndTest

Package for training and testing NN for shape pulse.

### setup and run on machines with cvmfs mounted

After checkout the package get in the package directory :
```
cd cppm-nnlar/trainAndTest
```
Modify `config.py` for your setup (default config to be made once setup)

Run setup script (each login) :
```
source setup.sh
```
Run the code :
```
python run.py
```

### Side notes

Can run on laptop but requires ROOT and python packages

#### Setup python virtual environment for :

This section is for the explanations, in practice for the `trainAndTest` package just run the `setup.sh` script. It'll take care of installing conda and setuping the environment.

Uses Miniconda https://conda.io/projects/conda/en/latest/index.html

If you don't have miniconda or conda, retrieve miniconda installer and run it (python3) :
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
```
Installer proposes to init conda, say yes. It will also include lines in the bashrc to run conda on start-up. These lines can be commented not to set automatically but I'd advice to allow conda to write them so that you have the setup sequence they suggest in your script.

Next create and activate an environement :
```
conda create --name trainpackage python=3.7.6
conda activate trainpackage
```
And finally install all necessary packages for the trainAndTest framework :
```
conda install -c conda-forge root
conda install -c conda-forge uproot
conda install -c conda-forge h5py
conda install -c conda-forge matplotlib
conda install -c conda-forge tensorflow
conda install -c conda-forge tensorflow-gpu==2.1
conda install -c conda-forge scikit-learn
conda install -c conda-forge pydot
```
Note : package "graphviz" seems already installed by the above

#### Setup ROOT on local machine :

Download root from : [https://root.cern.ch/downloading-root](https://root.cern.ch/downloading-root) (Thomas : I used ver=6.14.04 for my laptop ... testing 6.22/02)

Choose location for your root build `loc=/your/path/` and follow these instructions :
```
cd ${loc}
tar -xvf /path/to/download/root_v${ver}.source.tar.gz
cd root-${ver}
mkdir build
cd build
cmake -Dpython3=ON -DPYTHON_EXECUTABLE=/usr/bin/python3 ..
cmake --build .
```
Note : ROOT build is long (~hour I think)

For each session setup root with this command :
```
source ${loc}/root-${ver}/build/bin/thisroot.sh
```
