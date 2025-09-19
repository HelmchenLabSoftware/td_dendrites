"""
Functions to generate the figures of the paper from the simulation results in the 'results' directory
"""

import warnings
import matplotlib
import matplotlib.pyplot as plt
from math import floor
from pickle import load
from scipy import stats
from l5apical.parula import PARULA
from l5apical.helper import *

# Colors
COL_SENSORY = "#2985C4"
COL_ITI = '#cdcdcdff'
COL_PRE = '#004977ff'
COL_TEXTURE = "#ce583aff"
COL_T1 = 'k'
COL_T2 = '#9f9f9fff'
COL_DISTRACTOR = "#4cf3edff"
COL_OUTCOME = "#eda922ff"
COL_RESTORED = "#77c044ff"
COL_HIT = "#25b592ff"
COL_CR = "#794783f8"
GO_NOGO_MAP = matplotlib.colors.LinearSegmentedColormap.from_list(name="", colors=[COL_T2, COL_T1])

# Dimensions for panels
INCH = 2.54
DPI = 450
FIG_W_PAD = 0.1  # inches
FIG_H_PAD = 0.06  # inches
PANEL_LABEL_HEIGHT = 0.11110236
AXIS_WIDTH = 0.5
LINE_WIDTH = 0.7
MARKER_SIZE = 2
FIG_WIDTH = 16 / INCH
PANEL_WIDTH = FIG_WIDTH / 3.
PANEL_HEIGHT = 4.5 / INCH
FONT_SIZE = 6

# Plot directory
PLOT_DIR = ROOT_DIR / "panels"
SVG_DIR = 'svg'

# Processing initializations
N_EXPERTS = 150  # Number of expert trials to plot
N_AV_BIN = 20  # Number of available bins for histograms
BIN_MAX = 6  # Maximal bin ID for histogram
END_T = OUTCOME_T + 2  # Time step up to which to plot


def get_results(simulation: Simulation = Simulation.DEFAULT, expert_aligned: bool = False, theta_0: float = THETA_0
                ) -> list[dict]:
    """
    Align simulation outcomes to the trials at which the simulated agents crossed the expertise thresholds
    :param simulation: Enum identifier of the simulation type
    :param expert_aligned: Whether to align trials to the first expert trial
    :param theta_0: Value of the theta_o parameter (only relevant for mixed selectivity simulations)
    :return: results dictionary with values aligned to expert times
    """
    with open(get_path(simulation=simulation, theta_0=theta_0), 'rb') as handle:
        results = load(handle)
    if expert_aligned:
        expert_t = [int(results[s][K_EXPERT_T]) for s in range(N_SEEDS)]
        before_expert = min(expert_t)
        after_expert = N_TRIALS - max(expert_t)
        sample_ranges = [[expert_t[s] - before_expert, expert_t[s] + after_expert] for s in range(N_SEEDS)]
        for s in range(N_SEEDS):
            for k in [K_LEARNING_T, K_EXPERT_T]:
                results[s][k] = results[s][k] - expert_t[s] + before_expert
            for k in KEYS_1D:
                results[s][k] = results[s][k][sample_ranges[s][0]:sample_ranges[s][1]]
            for k in KEYS_2D:
                results[s][k] = results[s][k][sample_ranges[s][0]:sample_ranges[s][1], :]

    return results


def get_outcome_specific(outcome: Outcome, outcomes: type[np.ndarray | list[np.ndarray]],
                         values: type[np.ndarray | list[np.ndarray]]) -> type[np.ndarray | list[np.ndarray]]:
    """
    From the array of values for each trial, only keep the ones that occurred in trials of a specific outcome and
    replace the other values with nan
    :param outcome: specific outcome to select for
    :param outcomes: array or list of arrays containing the outcomes of each trial
    :param values: array or list of arrays containing the values to pick out depending on the trial outcome
    :return: array or list of arrays with only the value that occurred for the selected outcome
    """
    vs = np.copy(values)
    if isinstance(outcomes, list) and isinstance(values, list):
        if vs[0].ndim == 1:
            for s in range(len(outcomes)):
                vs[s][outcomes[s] != outcome.value] = np.nan
        elif vs[0].ndim == 2:
            for s in range(len(outcomes)):
                vs[s][outcomes[s] != outcome.value, :] = np.nan
        else:
            raise NotImplementedError(vs[0].ndim)

    elif isinstance(outcomes, np.ndarray) and isinstance(values, np.ndarray):
        if vs.ndim == 1:
            vs[outcomes != outcome.value] = np.nan
        elif vs.ndim == 2:
            vs[outcomes != outcome.value, :] = np.nan
        else:
            raise NotImplementedError(vs.ndim)
    else:
        raise NotImplementedError(f"Object types are {type(outcomes)} and {type(values)}")

    return vs


def get_texture_specific(texture: bool, outcomes: type[np.ndarray | list[np.ndarray]],
                         values: type[np.ndarray | list[np.ndarray]]) -> type[np.ndarray | list[np.ndarray]]:
    """
    From the array of values for each trial, only keep the ones that occurred in trials in which a specific texture was
    shown replace the other values with nan
    :param texture: boolean of the selected stimulus. True for go-stimulus (s1 / texture 1), False for nogo-stimulus
    :param outcomes: array or list of arrays containing the outcomes of each trial
    :param values: array or list of arrays containing the values to pick out depending on the trial outcome
    :return: array or list of arrays with only the value that occurred for the selected stimulus
    """
    vs = np.copy(values)
    if texture:
        bad_outcomes = [Outcome.FA.value, Outcome.CR.value]
    else:
        bad_outcomes = [Outcome.HIT.value, Outcome.MISS.value]

    if isinstance(outcomes, list) and isinstance(values, list):
        if vs[0].ndim == 1:
            for s in range(len(outcomes)):
                vs[s][outcomes[s] in bad_outcomes] = np.nan
        elif vs[0].ndim == 2:
            for s in range(len(outcomes)):
                vs[s][outcomes[s] in bad_outcomes, :] = np.nan
        else:
            raise NotImplementedError(vs[0].ndim)

    elif isinstance(outcomes, np.ndarray) and isinstance(values, np.ndarray):
        bad_idxs = np.logical_or(outcomes == bad_outcomes[0], outcomes == bad_outcomes[1])
        if vs.ndim == 1:
            vs[bad_idxs] = np.nan
        elif vs.ndim == 2:
            vs[bad_idxs, :] = np.nan
        else:
            raise NotImplementedError(vs.ndim)
    else:
        raise NotImplementedError(f"Object types are {type(outcomes)} and {type(values)}")

    return vs


def batch_nan_stat(values: type[np.ndarray | list[np.ndarray]], batch_size: int = 10, stat: str = 'mean') -> np.ndarray:
    """
    Extract mean or standard deviation for sequential batches of value arrays.
    :param values: Array (1D or 2D) or list of 1D-arrays of values to process
    :param batch_size: Size of the batches to make
    :param stat: 'mean' or 'std' to specify the statistic to extract
    :return: 1D or 2D array of the batch-dependent statistics (mean or std)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if isinstance(values, list):
            n_batches = len(values[0]) // batch_size
            batch_stat = np.zeros(n_batches)
            for b in range(n_batches):
                if stat == 'mean':
                    batch_stat[b] = np.nanmean(np.array([v[b*batch_size:(b+1)*batch_size] for v in values]))
                elif stat == 'std':
                    batch_stat[b] = np.nanstd(np.array([np.nanmean(v[b*batch_size:(b+1)*batch_size]) for v in values]))
                else:
                    raise NotImplementedError(stat)
        elif isinstance(values, np.ndarray):
            if values.ndim == 2:
                n_batches = values.shape[1] // batch_size
                batch_stat = np.zeros(n_batches)
                for b in range(n_batches):
                    if stat == 'mean':
                        batch_stat[b] = np.nanmean(values[:, b * batch_size:(b + 1) * batch_size])
                    elif stat == 'std':
                        batch_stat[b] = np.nanstd(np.nanmean(values[:, b * batch_size:(b + 1) * batch_size], axis=0))
                    else:
                        raise NotImplementedError(stat)
            elif values.ndim == 1:
                n_batches = values.shape[0] // batch_size
                batch_stat = np.zeros(n_batches)
                for b in range(n_batches):
                    if stat == 'mean':
                        batch_stat[b] = np.nanmean(values[b * batch_size:(b + 1) * batch_size])
                    elif stat == 'std':
                        batch_stat[b] = np.nanstd(values[b * batch_size:(b + 1) * batch_size])
                    else:
                        raise NotImplementedError(stat)
            else:
                raise NotImplementedError(values.ndim)
        else:
            raise NotImplementedError(type(values))

    return batch_stat


def get_significance(p_value: float) -> str:
    """
    Get significance stars for a p-value.
    :param p_value: P-value.
    """
    if p_value > 0.05 or np.isnan(p_value):
        s = 'n.s.'
    elif p_value > 0.01:
        s = '*'
    elif p_value > 0.001:
        s = '**'
    else:
        s = '***'
    return s


def set_style() -> None:
    """
    Set the plotting style, by defining some specific features.
    """
    matplotlib.rc(group='font', family='sans-serif')
    matplotlib.rc(group='font', serif='Arial')
    matplotlib.rc(group='text', usetex='false')
    matplotlib.rcParams['mathtext.default'] = 'regular'
    matplotlib.rcParams['svg.fonttype'] = 'none'
    matplotlib.rcParams['font.size'] = FONT_SIZE
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['xtick.major.size'] = 2.
    matplotlib.rcParams['xtick.major.width'] = 0.5
    matplotlib.rcParams['xtick.minor.size'] = 1.2
    matplotlib.rcParams['xtick.minor.width'] = 0.35
    matplotlib.rcParams['ytick.major.size'] = matplotlib.rcParams['xtick.major.size']
    matplotlib.rcParams['ytick.major.width'] = matplotlib.rcParams['xtick.major.width']
    matplotlib.rcParams['ytick.minor.size'] = matplotlib.rcParams['xtick.minor.size']
    matplotlib.rcParams['ytick.minor.width'] = matplotlib.rcParams['xtick.minor.width']
    matplotlib.rcParams['axes.linewidth'] = AXIS_WIDTH
    matplotlib.rcParams['lines.linewidth'] = LINE_WIDTH
    matplotlib.rcParams['patch.linewidth'] = LINE_WIDTH


def adjust_figure(fig: plt.Figure, w_space: float = None, h_space: float = None) -> None:
    """
    Adjust the layout of the figure.
    :param fig: figure object
    :param w_space: width reserved for space between plots
    :param h_space: height reserved for space between plots
    """
    fig_width, fig_height = fig.get_size_inches()
    bottom = FIG_H_PAD / fig_height
    left = FIG_W_PAD / fig_width
    rect = (left, bottom, 1. - left, 1. - bottom - PANEL_LABEL_HEIGHT / fig_height)
    plt.tight_layout(pad=0., h_pad=0, w_pad=0, rect=rect)
    fig.subplots_adjust(wspace=w_space, hspace=h_space)


def save_or_show(saving: bool, plot_dir: Path, plot_name: str, plot_dpi=600) -> None:
    """
    Save the figure as svg and as pdf or just display the figure with the desired DPI.
    :param saving: Whether to save the figure. If false the figure will just be displayed.
    :param plot_dir: Directory where to save the figure
    :param plot_name: File name under which to save the figure (without suffix)
    :param plot_dpi: DPI to display the figure with
    """

    if saving:
        (plot_dir / SVG_DIR).mkdir(parents=True, exist_ok=True)
        matplotlib.pyplot.savefig(plot_dir / SVG_DIR / f"{plot_name}.svg")
        matplotlib.pyplot.savefig(plot_dir / f"{plot_name}.pdf", dpi=DPI)
        matplotlib.pyplot.close()
    else:
        matplotlib.pyplot.gcf().set_dpi(plot_dpi)
        matplotlib.pyplot.show()


def get_parula_cm() -> matplotlib.colors.ListedColormap:
    """
    Get the parula colormap, which is the default for matlab plots
    :return: The parula colormap
    """
    return matplotlib.colors.ListedColormap(PARULA, name='parula')


def get_performance_cm(reverse: bool = False) -> matplotlib.colors.LinearSegmentedColormap:
    """
    Get the colormap used to depict the evolution of the learning performance
    :param reverse: whether to reverse the colormap
    :return: The performance colormap
    """
    if reverse:
        return matplotlib.colors.LinearSegmentedColormap.from_list(name="", colors=['lime', 'green', 'silver', 'k'])
    else:
        return matplotlib.colors.LinearSegmentedColormap.from_list(name="", colors=['k', 'silver', 'green', 'lime'])


def plot_trace(x, y, color="b", alpha=0.5, label=None, ax=None, z=0, line_style='-') -> None:
    """
    Plot traces either of: (1) mean trace and area showing the standard deviation of the trace or (2) just a trace of
    singular values. Which of both is determined by the type of y. If y is a dict, the
    :param x: array of x-values
    :param y: array of y-values or dictionary with keys 'mean' and 'std' each containing an array
    :param color: color of trace and std area
    :param alpha: transparency of std area
    :param label: label for legend
    :param ax: axis object to plot on
    :param z: zorder of the trace
    :param line_style: style of the line
    """
    ls = {'color': color, 'label': label, 'zorder': z, 'linestyle': line_style}
    if type(y) is dict:
        idxs = np.isfinite(y["mean"])
        y1 = y["mean"] - y["std"]
        y2 = y["mean"] + y["std"]
        if ax is None:
            plt.fill_between(x[idxs], y1[idxs], y2[idxs], color=color, alpha=alpha)
        else:
            ax.fill_between(x[idxs], y1[idxs], y2[idxs], color=color, alpha=alpha)
        if ax is None:
            plt.plot(x[idxs], y["mean"][idxs], **ls)
        else:
            ax.plot(x[idxs], y["mean"][idxs], **ls)
    else:
        idxs = np.isfinite(y)
        if ax is None:
            plt.plot(x[idxs], y[idxs], **ls)
        else:
            ax.plot(x[idxs], y[idxs], **ls)


def plot_t(ax: plt.Axes, y: float, iti: float = 0, end: float = END_T) -> None:
    """
    Plot important time windows of the trial as colored sections of a horizontal bar
    :param ax: axis object to plot on
    :param y: y-axis height at which to plot the bar
    :param iti: Time (in HZ freq) when the inter-trial interval bar to plot begins
    :param end: Time (in HZ freq) when the outcome bar ends
    """
    ls = {'linewidth': 4, 'clip_on': False, 'zorder': 0, 'solid_capstyle': 'butt'}
    ax.plot([iti * HZ, TONE_T], [y, y], COL_ITI, **ls)
    ax.plot([TONE_T, TEXTURE_T], [y, y], COL_PRE, **ls)
    ax.plot([TEXTURE_T, OUTCOME_T - 1], [y, y], COL_TEXTURE, **ls)
    ax.plot([OUTCOME_T - 1, end], [y, y], COL_OUTCOME, **ls)


def plot_perf_eg(saving: bool = True, mean: bool = True, sim: Simulation = Simulation.DEFAULT,
                 save_name: str = 'perf_example'):
    """
    Performance trace example for the default model
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param mean: boolean encoding whether to plot an example or the mean performance
    :param sim: simulation identifier
    :param save_name: name under which to save the plot
    """
    set_style()
    fig, ax = plt.subplots(figsize=(PANEL_WIDTH, PANEL_HEIGHT))
    plt.plot([0, N_TRIALS], [0.5, 0.5], "k:")
    trials = np.arange(N_TRIALS+1)
    results = get_results(simulation=sim)
    if mean:
        perfs = np.zeros((len(results), len(trials)))
        for seed in range(N_SEEDS):
            perfs[seed, :] = results[seed][K_PERFORMANCE]
        mean = np.mean(perfs, axis=0)
        std = np.std(perfs, axis=0)
        ax.fill_between(trials, mean - std, mean + std, clip_on=False, color='k', alpha=0.5, lw=0)
        ax.plot(trials, mean, 'k')
        xp_t = sum([results[seed][K_EXPERT_T] for seed in range(N_SEEDS)]) / N_SEEDS

    else:
        seed = 0
        ax.plot(trials, results[seed][K_PERFORMANCE], 'k')
        xp_t = results[seed][K_EXPERT_T]
    ax.plot([xp_t, xp_t], [0, 1], 'r--', lw=0.5, zorder=0)
    plt.xlim([0, N_TRIALS])
    plt.ylim([0, 1])
    plt.ylabel("Performance")
    plt.xlabel("Trial")

    # Performance evolution colormap
    p_cmap = get_performance_cm()
    cb_width = 0.05
    c_map_ax = ax.inset_axes([0., 0, 1, cb_width], zorder=-10)
    cb = matplotlib.colorbar.ColorbarBase(c_map_ax, cmap=p_cmap, orientation='horizontal')
    cb.outline.set_visible(False)
    cb.set_ticks([])

    # Add the patch to the Axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    adjust_figure(fig=fig)
    save_or_show(saving=saving, plot_dir=PLOT_DIR, plot_name=save_name)


def plot_v_hat(saving=True, outcome_types: tuple[Outcome, ...] = (Outcome.HIT, Outcome.CR),
               save_name: str = 'v_hat_dynamics') -> None:
    """
    Dynamic of the state value estimate (V) during trials of specific outcome types for expert agents and the
    corresponding dynamic changes of the estimate (Delta V) and unsigned changes of the estimate (|Delta V|).
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param outcome_types: tuple of outcome type to plot.
    :param save_name: name under which to save the plot.
    """

    # Get value estimates and trial outcomes
    results = get_results()
    outcomes = [results[s][K_OUTCOME] for s in range(N_SEEDS)]
    v_preds = [results[s][K_V_HAT] for s in range(N_SEEDS)]

    # Figure cosmetics
    n_outcomes = len(outcome_types)
    colors = {Outcome.HIT: COL_HIT, Outcome.CR: COL_CR, Outcome.FA: 'palevioletred', Outcome.MISS: 'olivedrab'}
    y_labels = ('Naïve\n$\widehat{V}$', 'Expert\n$\widehat{V}$', '$\Delta \widehat{V}_t$', '$|\Delta \widehat{V}_t|$')
    labels = {Outcome.HIT: 'Hit', Outcome.CR: 'CR', Outcome.FA: 'FA', Outcome.MISS: 'Miss'}

    # Define time steps to plot before the cue and after the outcome
    x = np.arange(END_T+1, dtype=int)
    simulated_x_ts = np.arange(TONE_T, END_T, dtype=int)

    # Define the trial windows for naive and expert learning stages
    trial_windows = (np.arange(5), np.arange(N_TRIALS - 50, N_TRIALS))

    # Variables to compute the largest y-axis range. This will be used to maintain the same y-scale across all plots
    bottom = 0.
    top = 0.
    y_diff = 0.

    # Init figure
    set_style()
    n_rows = 4 * n_outcomes + 1
    fig_size = (FIG_WIDTH / 4., 4 * n_outcomes / INCH)
    h_ratios = 4 * n_outcomes * [3] + [1]
    fig, axs = plt.subplots(nrows=n_rows, figsize=fig_size, sharex='all', height_ratios=h_ratios)

    # Plot the Hit case and the CR case
    for i in range(n_outcomes):

        # Plot the naive and expert learning stages of the state value estimate
        r0 = i * 4
        outcome_type = outcome_types[i]
        color = colors[outcome_type]
        v_preds_outcome = get_outcome_specific(outcome=outcome_type, outcomes=outcomes, values=v_preds)
        for j, window in enumerate(trial_windows):

            # Process the value estimate in the desired window of trials
            v_preds_outcome_window = np.array([v_preds_outcome[s][window, :] for s in range(N_SEEDS)])
            vp_mean, vp_ste = np.zeros(len(x)), np.zeros(len(x))
            vp_mean[simulated_x_ts] = np.nanmean(v_preds_outcome_window, axis=(0, 1))
            vp_ste[simulated_x_ts] = np.nanstd(v_preds_outcome_window, axis=(0, 1)) / np.sqrt(N_SEEDS)

            # Plot the traces with standard error
            axs[r0 + j].fill_between(x, vp_mean - vp_ste, vp_mean + vp_ste, color=color, alpha=0.3, linewidth=0.)
            axs[r0 + j].plot(x, vp_mean, color, label=labels[outcome_type])

            # Update the lowest and highest values to plot in the y-axis range
            bottom = min(bottom, np.min(vp_mean - vp_ste))
            top = max(top, np.max(vp_mean + vp_ste))
            y_diff = max(top - bottom, y_diff)

        # Compute the signed and unsigned change of state-value estimate during the trial and plot with bars
        v_preds_outcome_window = np.array([v_preds_outcome[s][trial_windows[1], :] for s in range(N_SEEDS)])
        dv_dt = np.diff(v_preds_outcome_window, axis=2)
        for j in (2, 3):

            # Process the data
            if j == 2:
                y = dv_dt
            else:
                y = np.abs(dv_dt)
            dvdt_mean = np.nanmean(y, axis=(0, 1))
            dvdt_ste = np.nanstd(y, axis=(0, 1)) / np.sqrt(N_SEEDS)

            # Bar plot
            axs[r0 + j].bar(
                x=simulated_x_ts[:-1] + 0.5, height=dvdt_mean, width=0.9, color=color, yerr=dvdt_ste, bottom=0.
            )

            # Update the lowest and highest values to plot in the y-axis range
            y_lim = axs[r0 + j].get_ylim()
            y_diff = max(y_lim[1] - y_lim[0], y_diff)

        # Cosmetics like removing axes, setting y-label and adding horizontal dashed line for zero
        for j in range(4):
            axs[r0 + j].plot([x[0], x[-1]], [0., 0.], 'k--', dashes=(5, 2), linewidth=AXIS_WIDTH, clip_on=False)
            axs[r0 + j].xaxis.set_visible(False)
            plt.setp(axs[r0 + j].spines.values(), visible=False)
            axs[r0 + j].tick_params(left=False, labelleft=False)
            axs[r0 + j].set_ylabel(y_labels[j], rotation='horizontal', verticalalignment='bottom', ha='center', y=0)

    # Adjust the y-axis range for the scale to be the same across all plots
    for row in range(n_rows-1):
        ax_bottom = min(axs[row].get_ylim()[0], -0.05)
        axs[row].set_ylim([ax_bottom, ax_bottom + y_diff])

    # Colored illustration of time windows
    plot_t(ax=axs[-1], y=0, end=x[-1])
    axs[-1].set_axis_off()

    # Finalize and save/show the figure
    txt_style = {'ha': 'right', 'va': 'center', 'fontsize': FONT_SIZE}
    for i, outcome_type in enumerate(outcome_types):
        ax = axs[i * 4]
        ax.text(x=0.95, y=0.3, s=labels[outcome_type], color=colors[outcome_type], transform=ax.transAxes, **txt_style)
    plt.suptitle('State-value estimate')
    adjust_figure(fig=fig)
    save_or_show(saving=saving, plot_dir=PLOT_DIR, plot_name=save_name, plot_dpi=150)


def plot_apical_raster(outcome: Outcome, expert_aligned: bool = False) -> None:
    """
    Raster plots of the evolution of responses for each apical dendrite type (sensory and outcome dendrites) for the go
    and the no-go texture selective neuron during a specific trial type of choice.
    :param outcome: Outcome identifier specifying the trial type (Hit, Miss, CR or FA)
    :param expert_aligned: Whether to align trials to the first expert trial
    """
    rows = 2
    cols = 2
    batch_size = 20
    set_style()
    cmap = get_parula_cm()
    cmap.set_bad(color='k')
    cb_width = 0.041
    p_cmap = get_performance_cm(reverse=True)
    if expert_aligned:
        y_ticks = [-400, -200, 0]
        n_trials = 500 + N_EXPERTS
        t_min, t_max = 1 - n_trials + N_EXPERTS, N_EXPERTS
    else:
        y_ticks = [0, 500, 1000, 1500]
        n_trials = N_TRIALS
        t_min, t_max = 1, N_TRIALS
    n_batches = n_trials // batch_size
    keys = [[K_DENDRITE_OUTCOME, K_DENDRITE_OUTCOME], [K_DENDRITE_SENSORY_T1, K_DENDRITE_SENSORY_T2]]
    results = get_results(expert_aligned=expert_aligned)
    outcomes = [results[s][K_OUTCOME] for s in range(N_SEEDS)]
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * PANEL_WIDTH, rows * PANEL_HEIGHT))
    for i in range(rows):
        for j in range(cols):
            key = keys[i][j]
            data = [results[s][key] for s in range(N_SEEDS)]
            outcome_spec_data = np.array(get_outcome_specific(outcome=outcome, outcomes=outcomes, values=data))
            apical = np.zeros((n_batches, N_TIME_STEPS))
            if key == K_DENDRITE_OUTCOME:
                if batch_size > 1:
                    for b in range(n_batches):
                        apical[b, -1] = np.nanmean(outcome_spec_data[:, b * batch_size:(b + 1) * batch_size])
                else:
                    apical[:, -1] = np.nanmean(outcome_spec_data, axis=0)
            else:
                if batch_size > 1:
                    for b in range(n_batches):
                        apical[b, :] = np.nanmean(outcome_spec_data[:, b * batch_size:(b + 1) * batch_size, :],
                                                  axis=(0, 1))
                else:
                    apical = np.nanmean(outcome_spec_data, axis=0)
            img = np.pad(array=apical, pad_width=((0, 0), (HZ, 1)))
            axs[i, j].imshow(img, aspect='auto', extent=(0, END_T, t_max, t_min),
                             vmin=apical_transfer(0.), vmax=1., cmap=cmap)
            plot_t(ax=axs[i, j], y=t_max + (t_max-t_min) * 0.02)
            c_map_ax = axs[i, j].inset_axes([0., cb_width, cb_width, 1 - cb_width])
            cb = matplotlib.colorbar.ColorbarBase(c_map_ax, cmap=p_cmap, orientation='vertical')
            cb.outline.set_visible(False)
            cb.set_ticks([])
            for pos in ['right', 'top', 'bottom', 'left']:
                axs[i, j].spines[pos].set_visible(False)

            axs[i, j].set_ylim([t_max + (t_max-t_min) * 0.04, None])
            axs[i, j].set_xlim([-0.99, None])
            if i == 1:
                axs[i, j].set_xlabel('Trial time')
                axs[i, j].set_xticks([TONE_T + 0.5, TEXTURE_T + 0.5, OUTCOME_T + 0.5],
                                     ['Tone', 'Texture', 'Outcome'], zorder=10)
            else:
                axs[i, j].set_xticks([])
            if j == 0:
                axs[i, j].set_yticks(y_ticks)
                if i == 0:
                    axs[i, j].set_ylabel('Outcome Dendrite\n\nTrial ID')
                else:
                    axs[i, j].set_ylabel('Sensory Dendrite\n\nTrial ID')
            else:
                axs[i, j].set_yticks(y_ticks, [''] * len(y_ticks))
    axs[0, 0].title.set_text('Go neuron')
    axs[0, 1].title.set_text('NoGo neuron')
    adjust_figure(fig=fig, h_space=0.05)


def plot_apical_trace_boxplot(saving=True, verbose: bool = False, outcome: Outcome = None) -> None:
    """
    Evolution of apical activity with learning during different trial windows and for each dendrite type as well as
    box-plots of activity depending on the learning phase and statistical significance between phases. The figure can be
    produced for specific trial outcome types (Hits, Misses, CRs or FAs) or for all trials (outcome=None).
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param verbose: whether to print out p-values
    :param outcome: the outcome type to plot the traces for. If None, all trial types are included.
    """
    # Initializations
    batch_size = 50
    nt = min(N_TRIALS, 500 + N_EXPERTS)
    p_cmap = get_performance_cm()
    cb_width = 0.05
    window_labels = ("Cue", "Touch", "Outcome")
    tcolors = (COL_PRE, COL_TEXTURE, COL_OUTCOME)
    window_ids = list(range(len(window_labels)))
    plot_order = list(reversed(window_ids))
    branch_names = ("Outcome dendrites", "Sensory dendrites")
    n_branches = len(branch_names)
    title_col = (COL_OUTCOME, COL_SENSORY)
    learning_phase_label = ('N', 'L', 'E')
    n_l_phase = len(learning_phase_label)
    n_seeds = N_SEEDS
    if outcome is None:
        plot_name = 'fig_4g_apical_responses'
        outcome_label = 'All trial types'
    elif outcome == Outcome.HIT:
        plot_name = 'fig_s13_apical_responses_hit'
        outcome_label = 'Hit trials'
    elif outcome == Outcome.CR:
        plot_name = 'fig_s13_apical_responses_cr'
        outcome_label = 'CR trials'
    elif outcome == Outcome.FA:
        plot_name = 'fig_s13_apical_responses_fa'
        outcome_label = 'FA trials'
    elif outcome == Outcome.MISS:
        plot_name = 'fig_s13x_apical_responses_miss'
        outcome_label = 'Miss trials'
    else:
        raise ValueError(outcome)

    # Initialize figure that will be split between traces (left) and boxplots (right)
    set_style()
    fig = plt.figure(constrained_layout=True, figsize=(0.5 * FIG_WIDTH, 8 / INCH))
    sub_figs = fig.subfigures(1, 2, width_ratios=[3, 1.5], wspace=0.07)
    axs = sub_figs[0].subplots(nrows=n_branches, ncols=1)

    # Get data for plotting
    trials = np.zeros(0)
    rs = get_results(expert_aligned=True)
    apical_sensory = [rs[s][K_DENDRITE_SENSORY] for s in range(n_seeds)]
    apical_outcome = [rs[s][K_DENDRITE_OUTCOME] for s in range(n_seeds)]

    if outcome is not None:
        outcomes = [rs[s][K_OUTCOME] for s in range(n_seeds)]
        apical_sensory = get_outcome_specific(outcome=outcome, outcomes=outcomes, values=apical_sensory)
        apical_outcome = get_outcome_specific(outcome=outcome, outcomes=outcomes, values=apical_outcome)

    y_out = np.array([apical_transfer(torch.zeros(n_seeds, len(apical_outcome[0]))).numpy(),
                      apical_transfer(torch.zeros(n_seeds, len(apical_outcome[0]))).numpy(),
                      apical_outcome])
    y_sen = np.array([[apical_sensory[s][:, 0] for s in range(n_seeds)],
                      [apical_sensory[s][:, TEXTURE_T - TONE_T] for s in range(n_seeds)],
                      [apical_sensory[s][:, -1] for s in range(n_seeds)]])
    ys = np.stack([y_out, y_sen], axis=0)

    # Plot traces with standard error
    for dendrite_type in range(n_branches):
        for trial_window in window_ids:
            y = {"mean": batch_nan_stat(ys[dendrite_type, trial_window, :, :], batch_size=batch_size),
                 "std": batch_nan_stat(ys[dendrite_type, trial_window, :, :], batch_size=batch_size, stat='std') / np.sqrt(N_SEEDS)}
            trials = np.linspace(start=N_EXPERTS-nt, stop=N_EXPERTS, num=len(y['mean']), endpoint=False)
            plot_trace(
                trials, y, color=tcolors[trial_window], label=window_labels[trial_window], ax=axs[dendrite_type], alpha=0.2
            )

    # Figure cosmetics
    for dendrite_type in range(n_branches):

        # Colorbar of learning progression
        c_map_ax = axs[dendrite_type].inset_axes([0, 0., 1, cb_width])
        cb = matplotlib.colorbar.ColorbarBase(c_map_ax, cmap=p_cmap, orientation='horizontal')
        cb.outline.set_visible(False)
        cb.set_ticks([])

        # Expert line
        axs[dendrite_type].vlines(x=0, ymin=0, ymax=1, color='k', linestyle='--', linewidth=AXIS_WIDTH)

        # Axis and labeling
        axs[dendrite_type].set_xlim([trials[0], trials[-1]])
        axs[dendrite_type].set_ylim([-0.05, 1.01])
        axs[dendrite_type].spines[['top', 'right']].set_visible(False)
        axs[dendrite_type].set_xticks([-500, -400, -300, -200, -100, 0, 100], ['', '-400', '', '-200', '', '0', '100'])
        axs[dendrite_type].set_yticks([0, 0.5, 1.])
        axs[dendrite_type].set_yticklabels([0., 0.5, 1.])
        axs[dendrite_type].spines['left'].set_bounds(0, 1.)
        axs[dendrite_type].set_ylabel('Apical activity')
        axs[dendrite_type].set_title(branch_names[dendrite_type], color=title_col[dendrite_type])
        if dendrite_type == n_branches - 1:
            axs[-1].set_xlabel("Trial ID")
            axs[1].legend(loc='upper left', frameon=False)
        else:
            axs[dendrite_type].set_xticklabels([])
            axs[dendrite_type].text(0 - 0.03 * (trials[-1] - trials[0]), 0.95, outcome_label, ha='right')

    # Get the unaligned data
    rs = get_results(expert_aligned=False)
    apical_sensory = [rs[s][K_DENDRITE_SENSORY] for s in range(n_seeds)]
    apical_outcome = [rs[s][K_DENDRITE_OUTCOME] for s in range(n_seeds)]
    if outcome is not None:
        outcomes = [rs[s][K_OUTCOME] for s in range(n_seeds)]
        apical_sensory = get_outcome_specific(outcome=outcome, outcomes=outcomes, values=apical_sensory)
        apical_outcome = get_outcome_specific(outcome=outcome, outcomes=outcomes, values=apical_outcome)
    y_out = np.array([apical_transfer(torch.zeros(n_seeds, len(apical_outcome[0]))).numpy(),
                      apical_transfer(torch.zeros(n_seeds, len(apical_outcome[0]))).numpy(),
                      apical_outcome])
    y_sen = np.array([[apical_sensory[s][:, 0] for s in range(n_seeds)],
                      [apical_sensory[s][:, TEXTURE_T - TONE_T] for s in range(n_seeds)],
                      [apical_sensory[s][:, -1] for s in range(n_seeds)]])
    ys = np.stack([y_out, y_sen], axis=0)

    # Init the boxplot subfigure
    sub_sub_figs = sub_figs[1].subfigures(2, 1)
    n_trial_windows = 3
    boxplot_style = {'showfliers': False,
                     'widths': 0.4,
                     'medianprops': {'color': 'k', 'linewidth': AXIS_WIDTH, 'solid_capstyle': 'butt'},
                     'boxprops': {'linewidth': 0, 'facecolor': 'k'},
                     'whiskerprops': {'clip_on': False, 'linewidth': AXIS_WIDTH},
                     'capprops': {'clip_on': False, 'linewidth': AXIS_WIDTH}}

    # Init statistics variables
    idx_pair = [[0, 1], [1, 2], [0, 2]]
    str_pair = ['naïve vs. learn', 'learn vs. expert', 'naïve vs. expert']
    stat_style = {'va': 'center', 'ha': 'center', 'color': 'k', 'fontsize': FONT_SIZE-2, 'weight': 'bold'}
    stat_pair_ids = [0, 1, 2]

    # Plot the boxplots and statistical significance
    warnings.simplefilter("ignore")
    for dendrite_type in range(n_branches):
        sub_sub_figs[dendrite_type].text(-0.07, 0.5, 'Apical activity', va='center', rotation='vertical')
        axs = sub_sub_figs[dendrite_type].subplots(nrows=n_trial_windows, ncols=1)
        for ax_i, trial_window in enumerate(plot_order):

            # Get data by learning phase
            data = ys[dendrite_type, trial_window, :, :]
            ys_by_phase = [np.zeros(n_seeds) for _ in range(n_l_phase)]
            for seed in range(n_seeds):
                ys_by_phase[0][seed] = np.nanmean(data[seed, :rs[seed][K_LEARNING_T]])
                ys_by_phase[1][seed] = np.nanmean(data[seed, rs[seed][K_LEARNING_T]:rs[seed][K_EXPERT_T]])
                ys_by_phase[2][seed] = np.nanmean(data[seed, rs[seed][K_EXPERT_T]:])

            # Clean data (sometimes empty learning phase led to NaNs)
            for p in range(n_l_phase):
                ys_by_phase[p] = ys_by_phase[p][~np.isnan(ys_by_phase[p])]

            # Plot the boxplots
            boxplot_style['boxprops']['facecolor'] = tcolors[trial_window]
            axs[ax_i].boxplot(ys_by_phase, positions=stat_pair_ids, patch_artist=True, **boxplot_style)

            # Run the Tuckey HSD statistical test
            if verbose:
                print(f'{branch_names[dendrite_type]} during the {window_labels[trial_window]} window:')
            try:
                p_values = stats.tukey_hsd(*tuple(ys_by_phase)).pvalue
            except RuntimeWarning:
                p_values = np.array([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]])

            # Plot significance stars
            for k in stat_pair_ids:
                s = get_significance(p_value=p_values[idx_pair[k][0], idx_pair[k][1]])
                if s != get_significance(1.):
                    sx = sum(idx_pair[k]) / 2.
                    sy = 1.2 + 0.3 * k
                    axs[ax_i].hlines(y=sy, xmin=idx_pair[k][0], xmax=idx_pair[k][1], color='k', clip_on=False, linewidth=AXIS_WIDTH)
                    axs[ax_i].text(sx, sy + 0.04, s, **stat_style)
                if verbose:
                    print(f'  - {str_pair[k]} {s} (p-value={p_values[idx_pair[k][0], idx_pair[k][1]]:.1e})')

            # Axes cosmetics
            axs[ax_i].set_xlim((-0.4, 2.4))
            axs[ax_i].set_ylim((0, 2))
            axs[ax_i].spines[['top', 'right']].set_visible(False)
            axs[ax_i].tick_params(axis=u'x', which=u'both', length=0)
            axs[ax_i].set_yticks([0, 1., 2.], ['0', '', '2'])
            ax_right = axs[ax_i].twinx()
            ax_right.spines[['top', 'right']].set_visible(False)
            ax_right.set_yticks([])
            ax_right.set_ylabel(window_labels[trial_window], color=tcolors[trial_window], labelpad=7, ha='center', va='center')
            if ax_i == n_trial_windows - 1:
                axs[ax_i].set_xticklabels(['N', 'L', 'E'])
            else:
                axs[ax_i].set_xticklabels([])
    warnings.simplefilter("default")

    # Save or show figure
    save_or_show(saving=saving, plot_dir=PLOT_DIR, plot_dpi=300, plot_name=plot_name)


def f4c_performance_eg(saving: bool = True) -> None:
    """
    Plot an example performance trace for the default simulation
    :param saving: boolean encoding whether to save (or just display) the plot.
    """
    plot_perf_eg(saving=saving, mean=False, sim=Simulation.DEFAULT, save_name='fig_4c_performance_example')


def f4d_expert_v_pred(saving=True) -> None:
    """
    Dynamic of the state value estimate (V) during Hit and CR trials for expert agents and the
    corresponding dynamic changes of the estimate (Delta V) and unsigned changes of the estimate (|Delta V|).
    :param saving: boolean encoding whether to save (or just display) the plot.
    """
    plot_v_hat(saving=saving, outcome_types=(Outcome.HIT, Outcome.CR), save_name='fig_4d_expert_v_pred')


def f4e_apical_dendrites_raster_plots(saving: bool = True) -> None:
    """
    Raster plots of the evolution of responses for each apical dendrite type (sensory and outcome dendrites) for the go
    and the no-go texture selective neuron during either Hit or CR trials.
    :param saving: boolean encoding whether to save (or just display) the plot.
    """
    plot_apical_raster(outcome=Outcome.HIT)
    save_or_show(saving=saving, plot_dir=PLOT_DIR, plot_name='fig_4e_apical_hit', plot_dpi=300)

    plot_apical_raster(outcome=Outcome.CR)
    save_or_show(saving=saving, plot_dir=PLOT_DIR, plot_name='fig_4e_apical_cr', plot_dpi=300)


def f4e_soma_raster_plots(saving: bool = True, expert_aligned: bool = False):
    """
    Raster plots of the evolution of responses for somatic activity of the no-go texture selective neuron during type of
    trial (Hit, CR, FA, Miss).
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param expert_aligned: Whether to align trials to the first expert trial
    """
    rows = 2
    cols = 2
    batch_size = 50
    if expert_aligned:
        y_ticks = [-400, -200, 0]
        n_trials = 500 + N_EXPERTS
        t_min, t_max = 1 - n_trials + N_EXPERTS, N_EXPERTS
    else:
        y_ticks = [0, 500, 1000, 1500]
        n_trials = N_TRIALS
        t_min, t_max = 1, N_TRIALS
    n_batches = n_trials // batch_size
    set_style()
    cmap = get_parula_cm()
    cmap.set_bad(color='k')
    cb_width = 0.041
    p_cmap = get_performance_cm(reverse=True)
    results = get_results(expert_aligned=expert_aligned)
    outcomes = [results[s][K_OUTCOME] for s in range(N_SEEDS)]
    outcome_types = [[Outcome.HIT, Outcome.MISS], [Outcome.FA, Outcome.CR]]
    t_idxs = [T1_IDX, T2_IDX]
    for k in range(2):
        data = [results[s][K_X_SOM_TXT][:, t_idxs[k]] for s in range(N_SEEDS)]
        fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * PANEL_WIDTH, rows * PANEL_HEIGHT))
        for i in range(rows):
            for j in range(cols):
                out_spec_data = np.abs(np.array(get_outcome_specific(outcome=outcome_types[i][j], outcomes=outcomes,
                                                                     values=data)))
                mean_data = np.zeros((n_batches, END_T))
                if batch_size > 1:
                    for b in range(n_batches):
                        mean_data[b, TEXTURE_T] = np.nanmean(out_spec_data[:, b * batch_size:(b + 1) * batch_size])
                else:
                    mean_data = np.nanmean(out_spec_data, axis=0)
                mean_data[np.isnan(mean_data)] = 0
                axs[i, j].imshow(mean_data, aspect='auto', extent=(0, END_T, t_max, t_min), vmin=0.,
                                 vmax=MAX_GAIN, cmap=cmap)
                plot_t(ax=axs[i, j], y=t_max + (t_max-t_min) * 0.02)
                c_map_ax = axs[i, j].inset_axes([0., cb_width, cb_width, 1 - cb_width])
                cb = matplotlib.colorbar.ColorbarBase(c_map_ax, cmap=p_cmap, orientation='vertical')
                cb.outline.set_visible(False)
                cb.set_ticks([])
                for pos in ['right', 'top', 'bottom', 'left']:
                    axs[i, j].spines[pos].set_visible(False)
                axs[i, j].set_ylim([t_max + (t_max-t_min) * 0.04, None])
                axs[i, j].set_xlim([-0.99, None])
                if i == 1:
                    axs[i, j].set_xlabel('Trial time')
                    axs[i, j].set_xticks([TONE_T + 0.5, TEXTURE_T + 0.5, OUTCOME_T + 0.5],
                                         ['Tone', 'Texture', 'Outcome'], zorder=10)
                else:
                    axs[i, j].set_xticks([])
                if j == 0:
                    axs[i, j].set_yticks(y_ticks)
                    if i == 0:
                        axs[i, j].set_ylabel('Go neuron\n\nTrial ID')
                    else:
                        axs[i, j].set_ylabel('NoGo neuron\n\nTrial ID')
                else:
                    axs[i, j].set_yticks(y_ticks, [''] * len(y_ticks))
        axs[0, 0].title.set_text('Lick')
        axs[0, 1].title.set_text('No Lick')

        adjust_figure(fig=fig, h_space=0.05)
        save_or_show(saving=saving, plot_dir=PLOT_DIR, plot_name=f'fig_4e_t{k+1}_soma', plot_dpi=300)


def f4f_sen_dendrite(saving: bool = True, expert_aligned: bool = False) -> None:
    """
    Multi-panel plot showing:
    - Evolution of the pre-synaptic input to sensory dendrites across learning as raster plot
    - Evolution of the sensory dendrite synaptic weight strengths depending on the bottom-up (basal) selectivity of the
      pyramidal neurons (with learning)
    - Evolution of the multiplicative gains of different stimuli with learning
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param expert_aligned: Whether to align trials to the first expert trial
    """

    # Initializations and loading data
    if expert_aligned:
        n_trials = 500 + N_EXPERTS
        t_min, t_max = 1 - n_trials + N_EXPERTS, N_EXPERTS
    else:
        n_trials = N_TRIALS
        t_min, t_max = 1, N_TRIALS
    results = get_results(expert_aligned=expert_aligned)
    batch_size = 50
    n_batches = n_trials // batch_size
    set_style()
    cmap = get_parula_cm()
    cmap.set_bad(color='k')
    p_cmap = get_performance_cm(reverse=True)
    cb_width = 0.043

    # Raster plot of the sensory dendrite pre-synaptic input (which learns to predict |Delta V|)
    data = np.nanmean(np.array([results[s][K_TIMINGS] for s in range(N_SEEDS)]), axis=0)
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(PANEL_WIDTH, 8 / INCH), sharey='all')
    mean_data = np.zeros((n_batches, N_TIME_STEPS))
    if batch_size > 1:
        for b in range(n_batches):
            mean_data[b, :] = np.nanmean(data[b * batch_size:(b + 1) * batch_size, :], axis=0)
    for i in range(2):
        c_map_ax = axs[i].inset_axes([-cb_width, 0, cb_width, 1], zorder=-10)
        cb = matplotlib.colorbar.ColorbarBase(c_map_ax, cmap=p_cmap, orientation='vertical')
        cb.outline.set_visible(False)
        cb.set_ticks([])
        cb.set_label('Trial ID', labelpad=-15)
        cb.ax.tick_params(size=0)

    # Plot traces of the sensory dendrite synapse strengths and of the pyramidal neuron gains
    trials = np.linspace(t_min, t_max, n_batches, endpoint=True)
    plot_cols = [COL_PRE, COL_T1, COL_T2, COL_DISTRACTOR]
    idxs = [0, 1, 2, np.arange(3, N_Z)]
    axes = (0, 0, 0, (0, 2))
    keys = [K_W_AP, K_GAIN]
    x_ticks = [[0, 0.5, 1], [0, 2, 4, 6, 8, 10]]
    x_labs = [r'Apical synaptic weight $w^\mathrm{ap}$', r'Apical gain $g$']
    labels = ['Cue neuron', 'Go neuron', 'NoGo neuron', 'Other neurons']
    for row in range(2):
        data = np.array([results[s][keys[row]] for s in range(N_SEEDS)])
        mean_data = np.zeros((n_batches, 4))
        for i in range(4):
            st_err = np.zeros(n_batches)
            if batch_size > 1:
                for b in range(n_batches):
                    mean_data[b, i] = np.nanmean(data[:, b * batch_size:(b + 1) * batch_size, idxs[i]])
                    st_err[b] = np.nanstd(data[:, b * batch_size:(b + 1) * batch_size, idxs[i]]) / np.sqrt(batch_size)
            else:
                mean_data[:, i] = np.nanmean(data[:, :, idxs[i]], axis=axes[i])
                st_err = np.nanstd(data[:, :, idxs[i]], axis=axes[i])
            axs[row].fill_betweenx(trials, mean_data[:, i] - st_err, mean_data[:, i] + st_err,
                                          color=plot_cols[i], alpha=0.3, linewidth=0.)
        for i in range(4):
            axs[row].plot(mean_data[:, i], trials, plot_cols[i], label=labels[i])
        axs[row].spines[['right', 'top', 'left']].set_visible(False)
        axs[row].set_xlabel(x_labs[row])
        axs[row].set_xticks(x_ticks[row])
        axs[row].set_xlim([x_ticks[row][0], x_ticks[row][-1]])
        axs[row].set_yticks([])
        axs[row].set_ylim([t_min, t_max])
        axs[row].invert_yaxis()

    axs[0].legend(frameon=False, bbox_to_anchor=(1., 1.1), loc='upper right')
    adjust_figure(fig=fig, h_space=0.5)
    save_or_show(saving=saving, plot_dir=PLOT_DIR, plot_name=f'fig_4f_wap_g', plot_dpi=250)


def f4g_apical_window_traces(saving=True, verbose: bool = False) -> None:
    """
    Evolution of apical activity with learning during different trial windows and for each dendrite type.
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param verbose: whether to print out p-values
    """
    plot_apical_trace_boxplot(saving=saving, verbose=verbose)


def f4h_performance(saving=True) -> None:
    """
    Performance traces with and without apical inhibition
    :param saving: boolean encoding whether to save (or just display) the plot.
    """
    n_plot_seeds = N_SEEDS
    set_style()
    fig, ax = plt.subplots(figsize=(PANEL_WIDTH, PANEL_HEIGHT))
    simulations = [Simulation.DEFAULT, Simulation.APICAL_INHIBITION]
    max_t = [N_TRIALS, N_TRIALS_AP_INH]
    plot_col = ["grey", COL_RESTORED]
    plt.plot([0, N_TRIALS_AP_INH], [0.5, 0.5], "k:")
    trials = np.arange(N_TRIALS_AP_INH)
    for sim in reversed(range(len(simulations))):
        results = get_results(simulation=simulations[sim])
        for s in range(n_plot_seeds):
            plt.plot(trials[:max_t[sim]], results[s][K_PERFORMANCE][:max_t[sim]],
                     plot_col[sim], zorder=0)
    plt.xlim([0, max_t[1]])
    y_min = 0.25
    plt.ylim([y_min, 1])
    plt.plot([0, N_TRIALS_AP_INH], [y_min + 0.07, y_min + 0.07], linewidth=3, color="w", zorder=1)
    plt.plot([0, N_TRIALS], [y_min + 0.03, y_min + 0.03], linewidth=3, color=COL_RESTORED, zorder=1)
    plt.plot([N_TRIALS, N_TRIALS_AP_INH], [y_min + 0.03, y_min + 0.03], linewidth=3, color="w", zorder=1)
    plt.ylabel("Performance")
    plt.xlabel("Trial")
    plt.xticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500], labels=[0, "", 1000, "", 2000, "", 3000, ""])
    plt.yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], labels=["", "", "0.6", "", "0.8", "", "1.0"])
    fig.tight_layout()
    fig.subplots_adjust(left=0.15, top=0.85)

    plt.text(-0.02, 0., "Apical\ninhibition", transform=ax.transAxes, ha="right")

    # Add the patch to the Axes
    x0 = 1
    length_x = N_TRIALS_AP_INH - 2 * x0
    ax.add_patch(matplotlib.patches.Rectangle(
        xy=(x0, y_min + 0.06), width=length_x, height=0.02, linewidth=0.7, edgecolor="k", facecolor='none',
        clip_on=False, zorder=2
    ))
    ax.add_patch(matplotlib.patches.Rectangle(
        xy=(x0, y_min + 0.02), width=length_x, height=0.02, linewidth=0.7, edgecolor=COL_RESTORED, facecolor='none',
        clip_on=False, zorder=2
    ))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_label_coords(-0.18, 0.6)
    adjust_figure(fig=fig)
    save_or_show(saving=saving, plot_dir=PLOT_DIR, plot_name='fig_4h_performance')


def f4h_expert_trials(saving=True, verbose: bool = False) -> None:
    """
    Distribution of number of trials necessary to reach expert performance depending on whether the apical dendrites
    were inhibited during the first 1800 trials or not.
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param verbose: boolean encoding whether to print statistical tests between the number of trials to become expert.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(PANEL_WIDTH, 0.7 * PANEL_HEIGHT))
    plot_col = ['k', COL_RESTORED]
    face_col = ['grey', 'white']
    plot_order = [2, 0]
    simulations = [Simulation.DEFAULT, Simulation.APICAL_INHIBITION]
    xp_ts: list[np.ndarray] = [np.zeros(0), np.zeros(0)]
    for sim in (1, 0):
        results = get_results(simulation=simulations[sim])
        expert_t = [results[s][K_EXPERT_T] for s in range(N_SEEDS)]
        xp_ts[sim] = np.array(expert_t)
        plt.barh(plot_order[sim], np.nanmean(expert_t), capsize=3.5, linewidth=1., edgecolor=plot_col[sim],
                 color=(1, 1, 1, 0.), xerr=np.nanstd(expert_t), align='center', fc=face_col[sim])
        plt.plot(expert_t, [plot_order[sim]] * N_SEEDS, 'ko', alpha=1, markersize=1)
        if verbose:
            print(f'{np.nanmean(expert_t):.0f} ± {np.nanstd(expert_t):.0f}')
    plt.barh(1, np.nanmean(xp_ts[1]-1800), capsize=3.5, linewidth=1., edgecolor='k',
             color=(1, 1, 1, 0.), xerr=np.nanstd(xp_ts[1]-1800), align='center')
    plt.plot(xp_ts[1]-1800, [1] * N_SEEDS, 'ko', alpha=1, markersize=1)
    plt.vlines(x=N_TRIALS, ymin=-0.5, ymax=2.5)
    plt.xlabel("Trial")
    plt.xlim([0, N_TRIALS_AP_INH])
    plt.yticks([2, 1, 0], ["Unperturbed", "Corrected", "Perturbed"])

    p_values = [stats.ranksums(xp_ts[0], xp_ts[1])[1], stats.ranksums(xp_ts[0], xp_ts[1] - 1800)[1]]
    if verbose:
        print(f'Perturbed vs. Unperturbed {get_significance(p_values[0])} {p_values[0]:.1e}')
        print(f'Corrected vs. Unperturbed {get_significance(p_values[1])} {p_values[1]:.1e}')
    xs = [3500, 2500]
    for i in range(2):
        plt.plot([xs[i], xs[i]], [i, 2], 'k-')
        s = get_significance(p_values[i])
        if s == get_significance(1.):
            plt.text(xs[i] + 100, i + (2 - i)/2, s, rotation=270, va='center', ha='center')
        else:
            plt.text(xs[i] + 50, i + (2 - i)/2, s, rotation=270, va='center', ha='center', weight='bold')
    fig.tight_layout()

    ax = fig.get_axes()[0]
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    adjust_figure(fig=fig)
    save_or_show(saving=saving, plot_dir=PLOT_DIR, plot_name='fig_4h_expert_trials')


def f4h_w_ap(saving=True) -> None:
    """
    Evolution of the excitatory synaptic weights on the sensory dendrites with learning.
    :param saving: boolean encoding whether to save (or just display) the plot.
    """

    # Plot S1 top-down weight strength evolution
    set_style()
    fig, ax = plt.subplots(figsize=(PANEL_WIDTH, PANEL_HEIGHT))
    sims = [Simulation.DEFAULT, Simulation.APICAL_INHIBITION]
    plotcols = ['grey', COL_RESTORED]
    for i in [1, 0]:
        results = get_results(simulation=sims[i])
        trials = np.arange(len(results[0][K_W_AP]))
        for s in range(N_SEEDS):
            plt.plot(trials, results[s][K_W_AP][:, 1:3], plotcols[i], clip_on=False)

    # Plot style
    y_min = -0.25
    plt.ylim([y_min, 1])
    plt.xlim([0., N_TRIALS_AP_INH])
    plt.ylabel("Synapse strength $w^{\mathrm{ap}}$")
    plt.xlabel("Trial")
    plt.xticks([0, 1000, 2000, 3000, 4000], ['', '1000', '2000', '3000', ''])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    fig.tight_layout()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Apical inhibition bars
    x0 = 3
    y0 = y_min / 7.
    k = (y0 - y_min) / 7.
    x_length = N_TRIALS_AP_INH - 2 * x0
    y_width = 2 * k
    plt.plot([0, N_TRIALS], [y0 - 5 * k, y0 - 5 * k], linewidth=3.6, color=COL_RESTORED, zorder=0)
    ax.add_patch(matplotlib.patches.Rectangle(
        (x0, y0 - 6 * k), x_length, height=y_width, linewidth=0.7, edgecolor=COL_RESTORED, facecolor='none',
        clip_on=False
    ))
    ax.add_patch(matplotlib.patches.Rectangle(
        (x0, y0 - 3 * k), x_length, height=y_width, linewidth=0.7, edgecolor='k', facecolor='none', clip_on=False
    ))
    plt.text(-0.02, 0., "Apical\ninhibition", transform=ax.transAxes, ha="right")
    ax.yaxis.set_label_coords(-0.2, 0.6)

    adjust_figure(fig=fig)
    save_or_show(saving=saving, plot_dir=PLOT_DIR, plot_name='fig_4h_w_ap')


def fs11bc_transfer_f(saving=True) -> None:
    """
    Transfer functions mapping apical excitation to apical activation, apical activation to multiplicative gain and the
    combination of both, which gives the multiplicative gain depending on the apical excitation.
    :param saving: boolean encoding whether to save (or just display) the plot.
    """
    set_style()
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(3 * PANEL_WIDTH, PANEL_HEIGHT))

    x0, x1 = -0.25, 1.25
    x = torch.tensor(np.linspace(x0, x1, 100))
    axs[0].plot(x, np.clip(apical_transfer(x), 0, 1), color="k", label=r'$\varphi (x)$')
    axs[0].set_ylim([-0.05, 1.01])
    axs[0].set_ylabel(r'Apical activity $x^\mathrm{ap}$')
    axs[0].set_xlabel("Apical input")
    axs[0].spines['left'].set_bounds(-0.05, 1)

    axs[1].plot(x, gain(x), color="k", label=r'$g(x)$')
    axs[1].set_ylim([0, None])
    axs[1].set_ylabel(r'Gain modulation $g$')
    axs[1].set_xlabel(r'Apical activity $x^\mathrm{ap}$')
    axs[1].spines['left'].set_bounds(0, MAX_GAIN)

    axs[2].plot(x, gain(apical_transfer(x)), color="k", label=r'$g(\varphi (x))$')
    axs[2].set_ylim([0, MAX_GAIN*1.01])
    axs[2].set_ylabel(r'Gain modulation $g$')
    axs[2].set_xlabel("Apical input")
    axs[2].spines['left'].set_bounds(0, MAX_GAIN)

    for c in range(3):
        axs[c].set_xlim([x0, x1])
        axs[c].spines['top'].set_visible(False)
        axs[c].spines['right'].set_visible(False)
        axs[c].legend(loc='lower right')

    adjust_figure(fig=fig)
    save_or_show(saving=saving, plot_dir=PLOT_DIR, plot_dpi=300, plot_name='fig_s11bc_transfer_functions')


def fs11d_expert_v_pred(saving=True) -> None:
    """
    Dynamic of the state value estimate (V) during HFA trials for expert agents and the
    corresponding dynamic changes of the estimate (Delta V) and unsigned changes of the estimate (|Delta V|).
    :param saving: boolean encoding whether to save (or just display) the plot.
    """
    plot_v_hat(saving=saving, outcome_types=(Outcome.FA,), save_name='fig_s11d_expert_v_pred')


def fs11e_apical_dendrites_raster_plots(saving: bool = True) -> None:
    """
    Raster plots of the evolution of responses for each apical dendrite type (sensory and outcome dendrites) for the go
    and the no-go texture selective neuron during FA trials.
    :param saving: boolean encoding whether to save (or just display) the plot.
    """
    plot_apical_raster(outcome=Outcome.FA)
    save_or_show(saving=saving, plot_dir=PLOT_DIR, plot_name='fig_s11e_apical_fa', plot_dpi=300)


def fs12a_performance_eg(saving: bool = True) -> None:
    """
    Plot an example performance trace for the default simulation
    :param saving: boolean encoding whether to save (or just display) the plot.
    """
    plot_perf_eg(saving=saving, mean=False, sim=Simulation.MIXED_SELECTIVITY, save_name='fig_s12a_performance_example')


def fs12b_wap_traces(saving: bool = False):
    """
    Traces of the sensory dendrite synapse strengths for pyramidal neurons depending on their bottom-up selectivity for
    the go-stimulus (which is encoded by the basal synaptic strength of the texture 1 input).
    :param saving: boolean encoding whether to save (or just display) the plot.
    """
    n_available_bins = N_AV_BIN
    s_idxs = [T1_IDX, T2_IDX]
    colors = [COL_T1, COL_T2]
    c_labels = ['$w^{bas}_{Go}$', '$w^{bas}_{NoGo}$']
    p_cmap = get_performance_cm()
    cb_width = 0.1
    set_style()
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(PANEL_WIDTH, PANEL_HEIGHT))
    trials = torch.linspace(1, N_TRIALS, steps=N_TRIALS)
    results = get_results(simulation=Simulation.MIXED_SELECTIVITY)
    for s in range(len(s_idxs)):
        w_aps = {}
        for result in results:
            w_bas = result[K_W_BAS]
            for i in range(w_bas.shape[1]):
                b = min(floor(w_bas[s_idxs[s], i] * n_available_bins), BIN_MAX)
                if b not in w_aps.keys():
                    w_aps[b] = torch.tensor(result[K_W_AP][:, i])[None, :]
                else:
                    w_aps[b] = torch.cat((w_aps[b], torch.tensor(result[K_W_AP][:, i])[None, :]), dim=0)
        n_bins = len(w_aps)

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [COL_DISTRACTOR, colors[s]])
        for b in range(n_bins):
            axs[s].plot(trials, torch.mean(w_aps[b], dim=0), color=cmap(b/(n_bins-1.)))
        axs[s].set_ylim([-cb_width, 1.1])
        axs[s].spines['left'].set_bounds(0, 1)
        axs[s].set_ylabel('$w^{ap}$')
        axs[s].spines[['right', 'top']].set_visible(False)
        axs[s].set_xlim([trials[0], trials[-1]])
        norm = matplotlib.colors.Normalize(vmin=0.5/n_bins, vmax=1-0.5/n_bins)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        ticks = np.linspace(0., 1, n_bins + 1, endpoint=True)
        cbar = plt.colorbar(mappable=sm, boundaries=ticks, ticks=ticks, ax=axs[s])
        tick_labels: list = [None for _ in range(n_bins + 1)]
        tick_labels[0] = 0
        tick_labels[-1] = n_bins / n_available_bins
        cbar.ax.set_yticklabels(tick_labels)
        cbar.ax.set_ylabel(c_labels[s], rotation=270, labelpad=-1)
        c_map_ax = axs[s].inset_axes([0., 0, 1, cb_width], zorder=-10)
        cb = matplotlib.colorbar.ColorbarBase(c_map_ax, cmap=p_cmap, orientation='horizontal')
        cb.outline.set_visible(False)
        cb.set_ticks([])
    axs[0].set_xticks([])
    axs[0].spines[['bottom']].set_visible(False)
    axs[1].set_xlabel('Trial')

    adjust_figure(fig=fig, h_space=0.3)
    save_or_show(saving=saving, plot_dir=PLOT_DIR, plot_name='fig_s12b_w_ap_traces')


def fs12b_theta0(saving: bool = False) -> None:
    """
    Sensory dendrite synapse strength after 1800 trials for pyramidal neurons depending on their bottom-up selectivity
    for the go-stimulus (which is encoded by the basal synaptic strength of the texture 1 input) and on the chosen
    value for the theta_0 parameter.
    :param saving: boolean encoding whether to save (or just display) the plot.
    """
    n_available_bins = 20
    theta_0 = get_thetas_0()
    s_idxs = [T1_IDX, T2_IDX]
    ns = len(s_idxs)
    n_theta = len(theta_0)
    set_style()
    fig, axs = plt.subplots(nrows=1, ncols=ns, figsize=(PANEL_WIDTH, PANEL_HEIGHT))
    n_bins = BIN_MAX + 1
    w_ap_final_mean = torch.empty((ns, n_theta, n_available_bins))
    for t in range(n_theta):
        results = get_results(simulation=Simulation.MIXED_SELECTIVITY, theta_0=theta_0[t])
        for s in range(ns):
            w_ap_final = {}
            for result in results:
                w_bas = result[K_W_BAS]
                for i in range(w_bas.shape[1]):
                    b = min(floor(w_bas[s_idxs[s], i] * n_available_bins), BIN_MAX)
                    if b not in w_ap_final.keys():
                        w_ap_final[b] = [result[K_W_AP][-1, i]]
                    else:
                        w_ap_final[b] += [result[K_W_AP][-1, i]]

            for b in range(len(w_ap_final)):
                w_ap_final_mean[s, t, b] = np.mean(w_ap_final[b])

    y_ticks = [i / 5 for i in range(6)]
    y_tick_labels: list = ['' for _ in range(6)]
    y_tick_labels[0] = 0
    y_tick_labels[-1] = 1
    c_tick_labels: list = [None for _ in range(n_bins+1)]
    c_tick_labels[0] = 0
    c_tick_labels[-1] = n_bins / n_available_bins
    colors = [COL_T1, COL_T2]

    for k in range(ns):
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [COL_DISTRACTOR, colors[k]])
        for b in range(n_bins):
            axs[k].plot(theta_0, w_ap_final_mean[k, :, b], marker='.', color=cmap(b/(n_bins-1.)), ms=MARKER_SIZE)
        axs[k].spines[['right', 'top']].set_visible(False)
        axs[k].spines['left'].set_bounds(0, 1)
        axs[k].spines['bottom'].set_bounds(theta_0[0], theta_0[-1])
        axs[k].set_xticks(get_thetas_0())
        axs[k].set_yticks(y_ticks, y_tick_labels)
        axs[k].set_xlabel(r'$\theta_0$')

    for k in range(1, ns):
        axs[k].set_yticklabels([])
    axs[0].set_ylabel('$w^{ap}$')

    adjust_figure(fig=fig, w_space=0.15)
    save_or_show(saving=saving, plot_dir=PLOT_DIR, plot_name='fig_s12b_theta_0')


def fs12c_selectivity_traces(saving: bool = False) -> None:
    """
    Evolution of the sensory pyramidal neuron selectivity for the go or no-go stimulus as learning progresses and
    depending on the bottom-up selectivity of the neurons (encoded as the difference in the basal synaptic weights
    between texture 1 and texture 2).
    :param saving: boolean encoding whether to save (or just display) the plot.
    """

    bin_factor = 20
    set_style()
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(PANEL_WIDTH, PANEL_HEIGHT))
    bin_max = 8
    n_bins = 2 * bin_max + 1
    results = get_results(simulation=Simulation.MIXED_SELECTIVITY)
    x_s1 = nan_array((N_SEEDS, N_TRIALS, N_Z))
    x_s2 = nan_array((N_SEEDS, N_TRIALS, N_Z))
    bin_idxs = {}
    for s in range(N_SEEDS):
        x_s1[s, :, :] = get_texture_specific(texture=True, outcomes=results[s][K_OUTCOME], values=results[s][K_X_SOM_TXT])
        x_s2[s, :, :] = get_texture_specific(texture=False, outcomes=results[s][K_OUTCOME], values=results[s][K_X_SOM_TXT])
        w_bas = results[s][K_W_BAS]
        for i in range(N_Z):
            b = max(min(round((w_bas[T1_IDX, i] - w_bas[T2_IDX, i]).item() * bin_factor), bin_max), -bin_max)
            if b not in bin_idxs.keys():
                bin_idxs[b] = [[s, i]]
            else:
                bin_idxs[b] += [[s, i]]
    batch_size = 20
    trials = np.arange(N_TRIALS / batch_size)
    for b in bin_idxs.keys():
        s1_stack = np.vstack([x_s1[idx[0], :, idx[1]] for idx in bin_idxs[b]])
        s2_stack = np.vstack([x_s2[idx[0], :, idx[1]] for idx in bin_idxs[b]])
        if batch_size == 1:
            s1_responses = np.nanmean(s1_stack, axis=0)
            s2_responses = np.nanmean(s2_stack, axis=0)
        else:
            s1_responses = batch_nan_stat(s1_stack, batch_size=batch_size)
            s2_responses = batch_nan_stat(s2_stack, batch_size=batch_size)
        axs.plot(trials, s1_responses-s2_responses, color=GO_NOGO_MAP(b/(2. * bin_max) + 0.5))
    x_ticks = [500, 1000, 1500]
    axs.set_xticks([x/batch_size for x in x_ticks], x_ticks)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=GO_NOGO_MAP, norm=norm)
    boundaries = [b / (n_bins-1) for b in range(n_bins)]
    cbar = plt.colorbar(mappable=sm, ax=axs, boundaries=boundaries, ticks=boundaries)
    cbar.ax.set_yticklabels([f'{b / bin_factor}' if b % 4 == 0 else '' for b in range(-bin_max, bin_max+1)])
    cbar.ax.yaxis.set_tick_params(pad=1)
    cbar.ax.set_ylabel('$w^{bas}_{Go}-w^{bas}_{NoGo}$', rotation=270, labelpad=10)
    p_cmap = get_performance_cm()
    cb_width = 0.04
    c_map_ax = axs.inset_axes([0., 0, 1, cb_width], zorder=-10)
    cb = matplotlib.colorbar.ColorbarBase(c_map_ax, cmap=p_cmap, orientation='horizontal')
    cb.outline.set_visible(False)
    cb.set_ticks([])
    axs.set_ylabel('$x^{som}(Go)-x^{som}(NoGo)}$')
    axs.set_xlabel('Trial')
    axs.set_xlim([0, N_TRIALS / batch_size])
    axs.spines[['right', 'top']].set_visible(False)
    axs.spines['left'].set_bounds(-4, 4)

    adjust_figure(fig=fig, h_space=0.1)
    save_or_show(saving=saving, plot_dir=PLOT_DIR, plot_name='fig_s12c_selectivity_traces')


def fs12cd_selectivity_distribution(saving: bool = False) -> None:
    """
    Distribution (over all neurons that had a minimal response in the texture window) of the go-stimulus vs. the
    no-go stimulus selectivity before and after learning. Non-selective neurons are shown as neurons that had a minimal
    response in the texture window, but were not selective for either stimulus.
    :param saving: boolean encoding whether to save (or just display) the plot.
    """

    set_style()
    results = get_results(simulation=Simulation.MIXED_SELECTIVITY)
    x_s1 = nan_array((N_SEEDS, N_TRIALS, N_Z))
    x_s2 = nan_array((N_SEEDS, N_TRIALS, N_Z))
    txt_responsive = []
    tr_threshold = 0.1
    for s in range(N_SEEDS):
        x_s1[s, :, :] = get_texture_specific(
            texture=True, outcomes=results[s][K_OUTCOME], values=results[s][K_X_SOM_TXT]
        )
        x_s2[s, :, :] = get_texture_specific(
            texture=False, outcomes=results[s][K_OUTCOME], values=results[s][K_X_SOM_TXT]
        )
        for i in range(N_Z):
            if np.mean(results[s][K_X_SOM_TXT][:, i]) > tr_threshold:
                txt_responsive += [[s, i]]
    nn = N_SEEDS * N_Z
    bin_edges = np.arange(-4, 4, 0.2)
    n_bins = len(bin_edges) - 1
    nb2 = int(n_bins / 2)
    unselective_bins = [nb2, nb2+1]
    s2_bins = list(range(nb2))
    s1_bins = list(range(nb2 + 2, n_bins))
    n_ts = 100
    phase = [[0, n_ts], [N_TRIALS - n_ts, N_TRIALS]]
    n_phases = len(phase)
    fig, axs = plt.subplots(nrows=1, ncols=n_phases, figsize=(n_phases*PANEL_WIDTH, PANEL_HEIGHT))
    color_unresponsive = 'white'
    color_unselective = COL_DISTRACTOR
    for i in range(n_phases):
        txt_responsive = []
        for s in range(N_SEEDS):
            for j in range(N_Z):
                if np.mean(results[s][K_X_SOM_TXT][phase[i][0]:phase[i][1], j]) > tr_threshold:
                    txt_responsive += [[s, j]]

        if tr_threshold is not None:
            av_s1 = np.array([np.nanmean(x_s1[tr[0], phase[i][0]:phase[i][1], tr[1]]) for tr in txt_responsive])
            av_s2 = np.array([np.nanmean(x_s2[tr[0], phase[i][0]:phase[i][1], tr[1]]) for tr in txt_responsive])
        else:
            av_s1 = np.nanmean(x_s1[:, phase[i][0]:phase[i][1], :], axis=1)
            av_s2 = np.nanmean(x_s2[:, phase[i][0]:phase[i][1], :], axis=1)
        data = (av_s1 - av_s2).flatten()
        counts, _, patches = axs[i].hist(data, bins=bin_edges)
        for b in s1_bins:
            patches[b].set_facecolor(COL_T1)
        for b in s2_bins:
            patches[b].set_facecolor(COL_T2)
        for b in unselective_bins:
            patches[b].set_facecolor(color_unselective)
        ax_ins = axs[i].inset_axes([0.0, 0.5, 0.5, 0.5])
        numbers = [nn-len(txt_responsive), counts[unselective_bins].sum(), counts[s1_bins].sum(), counts[s2_bins].sum()]
        n_tot = sum(numbers)
        labels = [f'{100 * n / n_tot:.1f}%' for n in numbers]
        ax_ins.pie(numbers, labels=labels, labeldistance=1.2,
                   colors=[color_unresponsive, color_unselective, COL_T1, COL_T2],
                   wedgeprops={'linewidth': AXIS_WIDTH, 'edgecolor': 'k'})
    axs[1].set_yticklabels([])
    ticks_y = matplotlib.ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(100 * x / nn))
    axs[0].set_ylabel('Neurons [%]')
    for i in range(2):
        axs[i].yaxis.set_major_formatter(ticks_y)
        axs[i].set_xlabel('$x^{som}(go)-x^{som}(no-go)}$')
        axs[i].spines[['right', 'top']].set_visible(False)
        axs[i].set_xlim([-4, 4])
        axs[i].set_ylim([0, 0.15 * nn])
        axs[i].set_yticks([0.05 * i * nn for i in range(4)])

    p1 = matplotlib.patches.Patch(color=COL_T1, label='go selective')
    p2 = matplotlib.patches.Patch(color=COL_T2, label='no-go selective')
    p3 = matplotlib.patches.Patch(color=color_unselective, label='non-selective')
    p4 = matplotlib.patches.Patch(color=color_unresponsive, label='unresponsive', lw=AXIS_WIDTH, ec='k')
    axs[1].legend(handles=[p1, p2, p3, p4], handlelength=1, loc='upper right', frameon=False)

    adjust_figure(fig=fig, h_space=0.1)
    save_or_show(saving=saving, plot_dir=PLOT_DIR, plot_name='fig_s12c_selectivity_distributions')


def fs13_apical_dendrites_traces_boxplots(saving: bool = True, verbose: bool = False) -> None:
    """
    Evolution of apical activity with learning during different trial windows and for each dendrite type and each trial
    type (Hit, CR, Miss, FA).
    :param saving: boolean encoding whether to save (or just display) the plot.
    :param verbose: whether to print out p-values
    """
    for outcome in (Outcome.HIT, Outcome.CR, Outcome.FA):
        plot_apical_trace_boxplot(saving=saving, verbose=verbose, outcome=outcome)


def plot_all(saving: bool = True) -> None:
    """
    Plot all the panels.
    :param saving: boolean encoding whether to save (or just display) the plot.
    """
    f4c_performance_eg(saving=saving)  # Performance example for default simulation
    f4d_expert_v_pred(saving=saving)  # Expert V pred trace for Hit and CR trials
    f4e_apical_dendrites_raster_plots(saving=saving)  # S1 apical activity raster plot for Hit and CR
    f4e_soma_raster_plots(saving=saving)  # S1 output raster plot
    f4f_sen_dendrite(saving=saving)  # w^ap and gain
    f4g_apical_window_traces(saving=saving)  # Apical activity evolution with learning for different time windows
    f4h_performance(saving=saving)  # Performances traces with and without apical inhibition
    f4h_expert_trials(saving=saving)  # Expert times
    f4h_w_ap(saving=saving)  # Top-down w^ap to S1 neurons
    fs11bc_transfer_f(saving=saving)  # Transfer functions
    fs11d_expert_v_pred(saving=saving)  # Expert V pred trace for FA trials
    fs11e_apical_dendrites_raster_plots(saving=saving)  # S1 apical activity raster plot for FA
    fs12a_performance_eg(saving=saving)  # Performance example for mixed selectivity simulation
    fs12b_wap_traces(saving=saving)  # Evolution of apical synaptic weights
    fs12b_theta0(saving=saving)  # Dependence on the theta_0 parameter
    fs12c_selectivity_traces(saving=saving)  # Evolution of the selectivity with learning
    fs12cd_selectivity_distribution(saving=saving)  # Distribution of the selectivity before and after learning
    fs13_apical_dendrites_traces_boxplots(saving=saving)  # S1 apical activity traces and boxplots for each trial type
    print('All panels have been plotted successfully!')


if __name__ == '__main__':
    plot_all()
