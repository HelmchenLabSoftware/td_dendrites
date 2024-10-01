from random import seed as rd_seed
from random import shuffle as random_shuffle
from pickle import load, dump, HIGHEST_PROTOCOL
from scipy.io import savemat, loadmat
from l5apical.helper import *


class Agent:
    """
    Agent class simulating how the mouse selects actions and receives rewards according to the go/no-go sensory
    discrimination task.
    """
    def __init__(self, nr_inputs: int = 2, policy_lr: float = LR):
        self.reward = torch.tensor([REWARD])
        self.punishment = torch.tensor([PUNISHMENT])
        self.policy_weights = torch.zeros(nr_inputs)
        self.lick_activity = torch.Tensor([0.])
        self.input_history = torch.zeros(nr_inputs)
        self.lr = policy_lr
        self.a_prob = 0.5

    def integrate_input(self, input_vec: torch.Tensor) -> None:
        """
        Sample an action, observe the resulting outcome and update the action-selection policy network
        :param input_vec: Sensory state representation vector used as input for the policy network
        """
        self.lick_activity += torch.dot(self.policy_weights, input_vec)
        self.input_history += input_vec

    def act(self, go: int) -> (torch.Tensor, bool, int):
        """
        Sample an action, observe the resulting outcome and update the action-selection policy network
        :param go: Integer encoding the go stimulus (if non-zero) or the no-go stimulus (if zero)
        :return: Tuple containing the reward, a boolean encoding whether the lick action was performed and an integer
        encoding whether the best action was performed (1) or not (0).
        """

        # Compute the lick action probability and sample action
        lick_prob = torch.sigmoid(self.lick_activity)
        lick_prob_float = lick_prob.detach().numpy()[0]
        lick = bool(np.random.choice([0, 1], p=[1 - lick_prob_float, lick_prob_float]))

        # Get the outcome depending on the action
        if lick:
            self.a_prob = lick_prob
            if go:
                correct = 1
                reward = self.reward
            else:
                correct = 0
                reward = self.punishment
        else:
            self.a_prob = 1 - lick_prob
            reward = torch.tensor([0.])
            if go:
                correct = 0
            else:
                correct = 1

        return reward, lick, correct

    def update(self, reward: float, x_som: torch.Tensor = None):
        # Update policy network
        self.policy_weights += self.lr * reward * (1 - self.a_prob) * self.input_history * x_som
        self.lick_activity = torch.Tensor([0.])
        self.input_history = torch.zeros(self.policy_weights.size())


def get_params(simulation: Simulation, theta_0: float
               ) -> (int, int, int, list[float], float, torch.Tensor, torch.Tensor):
    """
    Initialize parameters depending on the simulation type.
    :param simulation: Simulation identifier
    :param theta_0: Value of the theta_0 parameter
    :return: Tuple containing the total number of basal inputs, the number of distractor input, the number of pyramidal
    neurons (also corresponding to the size of the state representation vector), the probabilities of each distractor to
    be activated in a trial, the common learning rate parameter, an array of theta values for each pyramidal neuron.
    """

    n_task_stim = 3  # Number of task-relevant stimuli

    # Number of distractor simuli
    if simulation in [Simulation.DEFAULT, Simulation.APICAL_INHIBITION]:
        n_noise_stim = N_Z - n_task_stim
    elif simulation == Simulation.MIXED_SELECTIVITY:
        n_noise_stim = 14
    else:
        raise ValueError(simulation)

    # Basal weights, plasticity thresholds (theta) and the Bernoulli distributions of the distractor stimuli
    n_stim = n_task_stim + n_noise_stim
    weights_apical = 0.05 + torch.rand(N_Z) * 0.1  # Sensory dendrite synaptic strengths
    noise_probs = [float(i / (n_noise_stim + 1)) for i in range(1, n_noise_stim + 1)]
    random_shuffle(noise_probs)
    if simulation == Simulation.MIXED_SELECTIVITY:
        w_bas = torch.nn.functional.normalize(torch.rand((n_stim, N_Z)) ** 6, p=1., dim=0)
        probs = torch.tensor([1, 0.5, 0.5] + noise_probs) / N_TIME_STEPS
        thetas = theta_0 * torch.matmul(probs[:], w_bas) + BKG
    else:
        w_bas = torch.diag(torch.ones(N_Z))
        thetas = THETA * torch.ones(N_Z) + BKG

    return n_stim, n_noise_stim, N_Z, noise_probs, LR, w_bas, thetas.float(), weights_apical


def simulate_seed(simulation: Simulation = Simulation.DEFAULT, seed: int = 0, theta_0: float = THETA_0) -> dict:
    """
    Simulate learning of a go-nogo sensory discrimination task
    :param simulation: Simulation identifier
    :param seed: Pseudorandom seed
    :param theta_0: theta_0 parameter (only used in mixed selectivity simulations)
    :return: Dictionary containing the evolution of all the recorded variables of interest during the simulation
    """

    # Setting random seed
    rd_seed(seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)

    # Initialization of basic params
    apical_active = simulation.value != Simulation.APICAL_INHIBITION.value
    n_trials = N_TRIALS_AP_INH if simulation.value == Simulation.APICAL_INHIBITION.value else N_TRIALS
    n_inputs, noise_inputs, n_z, noise_ps, lr, w_bas, thetas, w_ap = get_params(simulation=simulation, theta_0=theta_0)
    tlr, lr_ap, plr, learning_rate = lr, lr, lr, lr  # Learning rates
    tx_idxs = np.array([T1_IDX, T2_IDX])  # Indices for the texture selective neurons
    n_timings = N_TIME_STEPS  # Number of time steps simulated per trial
    tone_t = TONE_T  # Timing of the tone cue
    texture_t = TEXTURE_T  # Timing of the texture
    outcome_t = n_timings - 1  # Timing of the outcome
    texture_xs = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    base_input_time_idxs = np.zeros(shape=n_inputs)  # Template for time steps at which each stimulus occurs
    base_input_time_idxs[0] = tone_t
    base_input_time_idxs[1] = texture_t
    base_input_time_idxs[2] = texture_t
    x_bas_background = torch.ones(n_z) * BKG

    # Initialize agent and state value estimator
    actor = Agent(nr_inputs=n_z, policy_lr=plr)  # Simulated mouse agent with action selection policy network
    x_pre_t = torch.zeros(outcome_t)  # Pre-synaptic inputs for sensory dendrites (time-step dependent)
    dw_weights = torch.zeros(n_z)

    # Initialization of empty objects in which various simulation variables will be recorded
    outcomes, correct_trials, den_out = tuple(nan_array((n_trials,)) for _ in range(3))
    d_sen_tx, d_sen_t1, d_sen_t2, x_pre_t_h, td_delta_h = tuple(nan_array((n_trials, n_timings)) for _ in range(5))
    w_ap_h, gain_h, x_som_texture_h = tuple(nan_array((n_trials, n_z)) for _ in range(3))
    z_h = np.zeros((n_trials, n_z))
    v_hat_h = np.zeros((n_trials, 1 + n_timings))

    # Simulating trial after trial
    for j in range(n_trials):

        # Lift apical inhibition (in case it was present)
        if j == N_TRIALS and simulation.value == Simulation.APICAL_INHIBITION.value:
            apical_active = True

        # Initialize trial
        texture = np.random.choice([0, 1])  # Randomly sample a texture
        txt_in = texture_xs[texture]  # Get basal activation for sampled texture
        noise_in = torch.tensor([np.random.choice(a=[0, 1], p=[1-pr, pr]) for pr in noise_ps])  # Sample noise
        basal_input = torch.cat(tensors=(txt_in, noise_in), dim=0)  # Full basal activation vector
        input_time_idxs = np.copy(base_input_time_idxs)  # Array for the timing of basal inputs
        input_time_idxs[3:] = np.random.randint(low=0, high=outcome_t, size=noise_inputs)  # Random distractor timings
        input_time_idxs[:][basal_input < 0.5] = n_timings  # Timings of zero-inputs are set after the end of the trial

        # Storage variables
        dv_hat_before = torch.tensor([0.])  # State value change estimate at t-1
        v_hat_h[j, 0] = dv_hat_before.numpy()
        d_sen_tx[j, :outcome_t] = torch.mean(apical_transfer(torch.outer(x_pre_t, w_ap[tx_idxs])), dim=1)
        d_sen_t1[j, :outcome_t] = apical_transfer(x_pre_t * w_ap[T1_IDX])
        d_sen_t2[j, :outcome_t] = apical_transfer(x_pre_t * w_ap[T2_IDX])

        # Simulate each time step of the trial up to right before the action and outcome happen
        for t in range(outcome_t):

            # Simulate pyramidal neurons and compute state value estimate
            current_idxs = np.array(input_time_idxs == t)  # Get the indices of currently non-zero basal inputs
            x_in = torch.zeros((n_inputs,))  # Basal input vector
            x_in[current_idxs] = 1  # Set currently active basal inputs to 1
            x_bas = torch.matmul(x_in, w_bas) + x_bas_background  # Basal activations
            gains = gain(apical_transfer(w_ap * x_pre_t[t]))  # Multiplicative apical gains
            x_som = gains * x_bas  # Somatic activations (also firing rate)
            actor.integrate_input(x_som)
            dv_hat_now = torch.dot(dw_weights, x_som)  # Estimate state value change

            # Update state value estimator, sensory dendrite afferent and sensory dendrite synaptic weight
            dw_weights += learning_rate * dv_hat_now * z_h[j, :] * K_V
            w_ap = torch.clamp(w_ap + lr_ap * x_pre_t[t] * (x_som - thetas) * (1 - w_ap) * w_ap, min=0, max=1)
            x_pre_t[t] += tlr * (abs(dv_hat_now.item()) - x_pre_t[t])

            # Store variables of interest
            z_h[j, :] = GAMMA * z_h[j, :] + x_som.numpy()  # - BKG
            v_hat_h[j, 1 + t] = v_hat_h[j, t] / GAMMA + dv_hat_now.numpy()
            td_delta_h[j, t] = GAMMA * dv_hat_now
            if simulation == Simulation.DEFAULT:
                gain_h[j, current_idxs] = gains[current_idxs].detach().clone().numpy()
            if t == texture_t:
                x_som_texture_h[j, :] = x_som.detach().clone().numpy()

        # Select the action, observe the reward obtained and update the policy network accordingly
        reward, licked, correct_trials[j] = actor.act(texture)

        # Record the outcome and the predicted reward
        if licked:
            r_pred = v_hat_h[j, outcome_t]
            if texture:
                outcomes[j] = Outcome.HIT.value
            else:
                outcomes[j] = Outcome.FA.value
        else:
            r_pred = 0.
            if texture:
                outcomes[j] = Outcome.MISS.value
            else:
                outcomes[j] = Outcome.CR.value

        # Outcome timing apical dendrite activities
        out_salience = torch.abs(reward - r_pred).clone().detach()
        sen_salience = abs(r_pred) * w_ap
        if apical_active:
            x_ap = apical_transfer(sen_salience + out_salience)
            den_out[j] = apical_transfer(out_salience)
            d_sen_tx[j, outcome_t] = torch.mean(apical_transfer(sen_salience[tx_idxs]))
            d_sen_t1[j, outcome_t] = apical_transfer(sen_salience[T1_IDX])
            d_sen_t2[j, outcome_t] = apical_transfer(sen_salience[T2_IDX])
        else:
            x_ap = torch.tensor([0.])
            den_out[j] = 0.
            d_sen_tx[j, outcome_t] = 0.
            d_sen_t1[j, outcome_t] = 0.
            d_sen_t2[j, outcome_t] = 0.

        # Update state-value estimate and policy network according to outcome activity
        gains = gain(x_ap)  # Multiplicative apical gains
        x_som = gains * x_bas_background  # Somatic activations (also firing rate)
        dw_weights += learning_rate * (reward - r_pred) * z_h[j, :] * (x_som - BKG)
        actor.update(reward=reward, x_som=x_som)

        # Record variables of interest for post-simulation analysis
        v_hat_h[j, outcome_t+1] = v_hat_h[j, outcome_t] - r_pred
        d_sen_tx[j, outcome_t] = torch.mean(apical_transfer(sen_salience[tx_idxs]))
        d_sen_t1[j, outcome_t] = apical_transfer(sen_salience[T1_IDX])
        d_sen_t2[j, outcome_t] = apical_transfer(sen_salience[T2_IDX])
        w_ap_h[j, :] = w_ap
        x_pre_t_h[j, :-1] = x_pre_t
        x_pre_t_h[j, outcome_t] = abs(r_pred)
        td_delta_h[j, outcome_t] = reward - r_pred

        # Update weights according to TD error elicited by resetting value estimate to zero at the end of the trial
        dw_weights += - learning_rate * torch.tensor(v_hat_h[j, outcome_t+1] * z_h[j, :]).float() * K_V

    # Stash all recorded simulation outcomes of interest into a single dictionary for saving
    z_h[z_h == 0] = np.nan
    result = {K_OUTCOME: outcomes,
              K_CORRECT_TRIALS: correct_trials,
              K_V_HAT: v_hat_h,
              K_TD_DELTA: td_delta_h,
              K_DENDRITE_SENSORY: d_sen_tx,
              K_DENDRITE_SENSORY_T1: d_sen_t1,
              K_DENDRITE_SENSORY_T2: d_sen_t2,
              K_DENDRITE_OUTCOME: den_out,
              K_Z: z_h,
              K_GAIN: gain_h,
              K_W_AP: w_ap_h,
              K_TIMINGS: x_pre_t_h,
              K_W_BAS: w_bas,
              K_X_SOM: x_som_texture_h}

    return result


def simulate_seeds(simulation: Simulation, theta_0: float = THETA_0):
    """
    Simulate multiple runs of the sensory discrimination task
    :param simulation: Simulation identifier
    :param theta_0: theta_0 parameter (only used in mixed selectivity simulations)
    """
    # String specifying the value of the theta_0 parameter used for the output log
    if simulation == Simulation.MIXED_SELECTIVITY:
        t0_str = f' with Theta_0 = {theta_0}'
    else:
        t0_str = ''

    # Simulate each seed and aggregate all the simulation outcomes in an array of entries for each seed
    results = []
    for s in range(N_SEEDS):
        print(f"\rSimulated {s}/{N_SEEDS} seeds...", end='')
        result = simulate_seed(seed=s, simulation=simulation, theta_0=theta_0)
        results += [result]
    print(f"\rSimulation{t0_str} finished successfully.")

    # Save all the recorded variables of interest
    with open(get_path(simulation=simulation, theta_0=theta_0), 'wb') as handle:
        dump(results, handle, protocol=HIGHEST_PROTOCOL)


def save_trial_outcomes_matlab() -> None:
    """
    Save the arrays containing a bool for each trial describing whether the correct action was taken to matlab files
    """
    for simulation in [Simulation.DEFAULT, Simulation.APICAL_INHIBITION, Simulation.MIXED_SELECTIVITY]:
        results_path = get_path(simulation=simulation)
        if simulation == Simulation.APICAL_INHIBITION:
            number_trials = N_TRIALS_AP_INH
            perf_path = SMITH_DIR / 'outcomes_apical_inhibition.mat'
        elif simulation == Simulation.DEFAULT:
            number_trials = N_TRIALS
            perf_path = SMITH_DIR / 'outcomes_default.mat'
        elif simulation == Simulation.MIXED_SELECTIVITY:
            number_trials = N_TRIALS
            perf_path = SMITH_DIR / 'outcomes_mixed.mat'
        else:
            raise ValueError(simulation)

        with open(results_path, 'rb') as handle:
            load_results = load(handle)
            save_results = {K_CORRECT_TRIALS: np.zeros((N_SEEDS, number_trials))}
            for k in save_results.keys():
                for s in range(N_SEEDS):
                    save_results[k][s, :] = load_results[s][k]
            savemat(perf_path, save_results)
    print('Simulation results saved and ready to extract performance traces.')


def load_smith_perf() -> None:
    """
    Load the performances computed by matlab script and save into the results pickle files
    """

    # Load performance traces
    matlab_data = loadmat(str(get_path(simulation=Simulation.SMITH_PERFORMANCE)))

    # Load other results, insert performances into the results dictionary and save
    simulations = [Simulation.DEFAULT, Simulation.APICAL_INHIBITION, Simulation.MIXED_SELECTIVITY]
    for sim in range(len(simulations)):
        results_path = get_path(simulation=simulations[sim])
        with open(results_path, 'rb') as handle:
            load_results = load(handle)
            for seed in range(N_SEEDS):
                load_results[seed][K_EXPERT_T] = matlab_data[K_EXPERT_T][sim][seed] - 1
                load_results[seed][K_LEARNING_T] = matlab_data[K_LEARNING_T][sim][seed] - 1
                if simulations[sim] == Simulation.APICAL_INHIBITION:
                    load_results[seed][K_PERFORMANCE] = matlab_data["perf_apical_inhibition"][seed]
                elif simulations[sim] == Simulation.DEFAULT:
                    load_results[seed][K_PERFORMANCE] = matlab_data["perf_default"][seed]
                elif simulations[sim] == Simulation.MIXED_SELECTIVITY:
                    load_results[seed][K_PERFORMANCE] = matlab_data["perf_mixed"][seed]
                else:
                    raise ValueError(simulations[sim])

        with open(results_path, "wb") as handle:
            dump(load_results, handle, protocol=HIGHEST_PROTOCOL)
    print('Extracted performance traces have been integrated into simulation results.')


def run():
    """
    Run all simulations and save their outcomes.
    """

    # Make sure the necessary directories exist
    makedirs()

    print('\nSimulating default:')
    simulate_seeds(simulation=Simulation.DEFAULT)

    print('\nSimulating with apical inhibition:')
    simulate_seeds(simulation=Simulation.APICAL_INHIBITION)

    print('\nSimulating with mixed selectivity:')
    for t0 in get_thetas_0():
        simulate_seeds(simulation=Simulation.MIXED_SELECTIVITY, theta_0=t0)

    # Save the learning outcomes in a format that can be processed by a matlab script to extract the performance traces
    save_trial_outcomes_matlab()

    # If performance traces are available, save these. Note: Insufficient in case the performance traces have changed!
    if get_path(simulation=Simulation.SMITH_PERFORMANCE).exists():
        load_smith_perf()
    else:
        raise Warning('You will need to extract the performance traces with the Matlab scripts to be able to plot some '
                      'of the panels')
