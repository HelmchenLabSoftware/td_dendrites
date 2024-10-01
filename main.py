import sys
import l5apical


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
