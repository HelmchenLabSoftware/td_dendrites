""" Computational model for salience coding in apical dendrites of L5 pyramidal neurons

This is the main script to run the simulations and plot the panels of the manuscript:
"Error representations in apical dendrites of neocortical layer 5 pyramidal neurons during learning" by G. Schoenfeld,
S. Kollmorgen, M.C. Tsai, C. Lewis, S. Han, P. Bethge, A.M. Reuss, A. Aguzzi, W. Senn, V. Mante and F. Helmchen.

Plotting requires the results of the simulations to be present in the 'results' directory. The results can be generated
by running the simulations with the command:
>> python main.py simulate
The simulations might take a while to run depending on the hardware. After the simulations are done, the performance
traces have to be extracted by the Matlab script 'getperfs.m' (from Smith et al., 2004) located in the 'Smith'
directory. After running the Matlab script, the performance trace have to be merged into the main results by running:
>> python main.py perf
Finally, all the panels can be plotted by running:
>> python main.py plot
If no argument is provided, the script will default to plotting all the panels.
"""

import sys
import l5apical

__author__ = "Matthias Chinyen Tsai"
__license__ = "GPL"
__version__ = "2.0.0"

if __name__ == '__main__':

    if len(sys.argv) == 1:
        l5apical.panels.plot_all()
    else:
        if str(sys.argv[1]) == 'plot':
            l5apical.panels.plot_all()

        elif str(sys.argv[1]) == 'simulate':
            l5apical.simulations.run()

        elif str(sys.argv[1]) == 'perf':
            l5apical.simulations.load_smith_perf()

        # Explain correct usage syntax
        else:
            print("Proper usage requires argument 'plot' or 'simulate' or 'perf'. To plot all the panels enter: \n"
                  ">> python main.py plot\n"
                  "\nTo run all the simulations enter:\n"
                  ">> python main.py simulate\n"
                  "\nWarning: if you rerun all the simulations, you will also need to run the Matlab script getperfs.m "
                  "located in the Smith directory after which you will have to enter:\n"
                  ">> python main.py simulate\n"
                  "\nIf you fail to do so, you might corrupt the results and might not be able to plot the panels.")
