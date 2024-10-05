# Unsigned temporal difference errors in cortical L5 apical dendrites during learning

## Description

This program runs the simulations and plots the corresponding panels described in the manuscript '[Unsigned temporal difference errors in cortical L5 dendrites during learning](https://www.biorxiv.org/content/10.1101/2021.12.28.474360v3)'. This means simulating models of mice with L5 sensory pyramidal neurons learning a go/no-go sensory discrimination task from a sensory representation plagued by distractor signals.

### Installation

These are instructions to setup a functioning conda environment. If conda is not installed yet, do so following these [instructions](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html) (linux recommended). Then make sure conda is up-to-date.
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

In order to rerun the full pipeline of simulations yourself and replot all the panels, you will also need a functioning version of Matlab (script was tested with R2022b). The Matlab script will be necessary to extract the performance traces from the simulations. Without Matlab, you will still be able to plot the pre-simulated results used for the paper (and any panel not requiring the performance traces using the results that you simulated yourself).

### Execution

Before rerunning the simulations, you can test whether the installation was successful. Just enter
```
python main.py plot
```

This will plot all the panels (to the folder `panels`) with the existing simulation results. This is possible as the `main` branch contains pre-simulated results in the `results` folder, which are ~1.5GB. If you want to avoid downloading these files and simulate them yourself, you can clone the `clean` branch instead of the `main` branch. To rerun all the simulations (except the extraction of the performance trace), enter
```
python main.py simulate
```

This will overwrite any simulation outcomes present in the `results` folder. The `results` folder will then contain all the simulation results except for the performance traces, which are pre-computed in the folder `Smith`. In order to extract the performance traces necessary for some panels yourself, you will have to then run `Smith/getperfs.m` with Matlab. Once the Matlab script is finished you can run
```
python main.py perf
```

This should integrate the performances extracted by the code from Smith et al. into the pickle files in the `results` folder and will allow you to replot all of the panels. If any step is interrupted or skipped, the results files can become corrupted and can only be recovered by rerunning the simulation from the start or downloading the uncorrupted results from the github repository. In case you do not have Matlab installed and nothing was changed in the simulation code, it is also possible to skip the last step. Both the `main` and `clean` branch already contain the extracted performance traces for the default simulations. The code will therefore be able to reuse those files.
