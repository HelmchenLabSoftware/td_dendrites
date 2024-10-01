# Unsigned temporal difference errors in cortical L5 apical dendrites during learning

## Description

This program runs the simulations and plots the corresponding panels described in the manuscript 'Unsigned temporal difference errors in cortical L5 dendrites during learning'. This means simulating models of mice with L5 sensory pyramidal neurons learning a go/no-go sensory discrimination task.

### Installation

If conda is not installed yet, do so following these [instructions](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html) (linux recommended). Then make sure conda is up-to-date.
```
conda update conda --all
```

Create a conda environment with your favorite name, here `myenv`
```
conda create -n myenv python=3.10.8
```

Activate the conda environment 
```
conda activate myenv
```

Install the required libraries
```
pip install -r requirements.txt
```

In order to rerun the full pipeline of simulations yourself and replot all the panels, you will also need a functioning version of Matlab (script was written with R2022b). The Matlab script will be necessary to extract the performance traces from the simulations. Without Matlab, you will still be able to plot the presimulated results used for the paper (and all panels not requiring the performance traces from results that you simulated yourself).

### Execution

Before rerunning the simulations, you can test whether the installation was successful. Just enter
```
python main.py plot
```

This will plot all the panels (to the folder panels) from the existing simulation results. This is possible as the 'main' branch contains pre-simulated results in the the 'results' folder. These are ~1.5GB, if you want to avoid downloading these files and simulate them yourself, you can clone the 'clean' branch. To rerun all the simulations (except the extraction of the performance trace), enter
```
python main.py simulate
```

This will overwrite any simulation outcomes present in the 'results' folder. In order to extract the performance traces necessary for some panels yourself, you will have to then run Smith/getperfs.m with Matlab. Once the Matlab script is finished you can run
```
python main.py perf
```

This should integrate the performances extracted by the code from Smith et al. and will allow you to replot all of the panels. If any step is interrupted or skipped, the results files can become corrupted and can only be recovered by rerunning the simulation from the start or downloading the uncorrupted results from the github repository. In case you do not have Matlab installed and nothing was changed in the simulation code, it is also possible to skip the last step. The main github repo already contains the extracted performance traces for the default simulations. The code will therefore be able to reuse those.
