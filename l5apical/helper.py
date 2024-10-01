import torch
import numpy as np
from typing import Literal
from enum import Enum
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.resolve()  # Root directory
SMITH_DIR = ROOT_DIR / 'Smith'  # Directory containing matlab scripts and matlab derived performance traces

# Main simulation parameters
LR = 0.016  # Common learning rate tuned to replicate experimental number of trials necessary to become expert
THETA = 0.1  # Value of the activity threshold used in the w_ap plasticity rule
N_Z = 203  # Number of distractor neurons
N_TRIALS = 1800  # Default number of trials simulated
N_TRIALS_AP_INH = 4000  # Number of trials in case the apical dendrites are inhibited for the first 1800 trials
MAX_GAIN = 10.  # Maximum possible apical gain
BKG = 0.01  # Background activity of pyramidal neurons
K_V = BKG * MAX_GAIN  # Constant for TD learning plasticity
REWARD = 1.  # Reward value in case of hit
PUNISHMENT = -0.5  # Reward value in case of false alarm
GAMMA = 1.  # TD learning gamma
N_SEEDS = 10  # Number of pseudorandom seeds simulated
HZ = 4  # Number of time steps simulated per second
TONE_T = 0  # Time step when tone happens
TEXTURE_T = HZ + 1  # Time step when texture is perceived
N_TIME_STEPS = TONE_T + 4 * HZ + 2  # Total number of time steps per trial
T1_IDX = 2  # Index of the texture 1 stimulus
T2_IDX = 1  # Index of the texture 2 stimulus
THETA_0 = 4.  # Default value for theta_0 in the mixed selectivity simulations

# Keys identifying various outcomes of the simulation
Key = Literal[
    'outcome', 'correct_trials', 'performance', 'learning_t', 'expert_t', 'v_hat', 'td_delta', 'w_ap', 'x_pre_t',
    'sensory_dendrite', 'sensory_dendrite_t1', 'sensory_dendrite_t2', 'outcome_dendrite', 'z', 'gain', 'w_bas',
    'x_som_texture'
]
K_OUTCOME: Key = 'outcome'
K_CORRECT_TRIALS: Key = 'correct_trials'
K_PERFORMANCE: Key = 'performance'
K_LEARNING_T: Key = 'learning_t'
K_EXPERT_T: Key = 'expert_t'
K_V_HAT: Key = 'v_hat'
K_TD_DELTA: Key = 'td_delta'
K_TIMINGS: Key = 'x_pre_t'
K_W_AP: Key = 'w_ap'
K_W_BAS: Key = 'w_bas'
K_X_SOM: Key = 'x_som_texture'
K_DENDRITE_SENSORY: Key = 'sensory_dendrite'
K_DENDRITE_SENSORY_T1: Key = 'sensory_dendrite_t1'
K_DENDRITE_SENSORY_T2: Key = 'sensory_dendrite_t2'
K_DENDRITE_OUTCOME: Key = 'outcome_dendrite'
K_Z: Key = 'z'
K_GAIN: Key = 'gain'
KEYS_1D = [K_OUTCOME, K_CORRECT_TRIALS, K_PERFORMANCE, K_DENDRITE_OUTCOME]
KEYS_2D = [K_V_HAT, K_TD_DELTA, K_TIMINGS, K_W_AP, K_DENDRITE_SENSORY, K_DENDRITE_SENSORY_T1, K_DENDRITE_SENSORY_T2,
           K_X_SOM, K_Z, K_GAIN]


class Outcome(Enum):
    """
    Enumeration class for the four possible trial outcomes in the go/no-go sensory discrimination task
    """
    HIT = 0
    CR = 1
    FA = 2
    MISS = 3


class Simulation(Enum):
    """
    Enumeration class for the three types of simulations implemented
    """
    DEFAULT = 0
    APICAL_INHIBITION = 1
    MIXED_SELECTIVITY = 2
    SMITH_PERFORMANCE = 3


def apical_transfer(x) -> torch.Tensor:
    """
    Transfer function taking apical input and returning the apical activation
    :param x: apical input tensor
    :return: apical activation tensor
    """
    return torch.sigmoid((MAX_GAIN-1)*x-torch.log(torch.tensor(MAX_GAIN-1)))


def gain(x: torch.Tensor) -> torch.Tensor:
    """
    Transfer function taking apical activation and returning a multiplicative gain
    :param x: apical activation
    :return: multiplicative gain
    """
    return 1 + MAX_GAIN * np.clip(x - 1. / MAX_GAIN, 0, (MAX_GAIN-1) / MAX_GAIN)


def get_path(simulation: Simulation = Simulation.DEFAULT, theta_0: float = THETA_0) -> Path:
    """
    Get the path for the pickle file containing the simulation outcomes.
    :param simulation: Simulation identifier.
    :param theta_0: Value of the theta_o parameter (only relevant for mixed selectivity simulations)
    :return: String of the relative path for the pickle file containing the simulation outcomes
    """
    results_dir = ROOT_DIR / 'results'
    if simulation == Simulation.DEFAULT:
        path = results_dir / 'results_default.pickle'
    elif simulation == Simulation.APICAL_INHIBITION:
        path = results_dir / 'results_perturbed.pickle'
    elif simulation == Simulation.MIXED_SELECTIVITY:
        path = results_dir / f'mixed_selectivity/t_{theta_0}.pickle'
    elif simulation == Simulation.SMITH_PERFORMANCE:
        path = SMITH_DIR / 'performances.mat'

    else:
        raise ValueError(simulation)

    return path


def get_thetas_0() -> list[float]:
    """
    The values of theta_0 values that are tried in the mixed selectivity simulations.
    :return: List of theta_0 values
    """
    return [1., 2., 3., 4., 5., 6., 7., 8.]


def nan_array(shape: (int,)) -> np.ndarray:
    """
    Make a numpy array filled with nan.
    :param shape: Tuple of integers corresponding to the shape of the array
    :return: Numpy array of nan.
    """
    a = np.empty(shape=shape)
    if len(shape) == 1:
        a[:] = np.nan
    elif len(shape) == 2:
        a[:, :] = np.nan
    return a


def makedirs() -> None:
    """
    Creates directories in which simulation outcomes and panels are saved.
    """
    for subdir in ['results/mixed_selectivity', 'panels/svg', 'gogo']:
        p = ROOT_DIR / subdir
        p.mkdir(parents=True, exist_ok=True)
