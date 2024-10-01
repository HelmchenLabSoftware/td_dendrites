nrseeds = 10;
maxres = 1;
corprob = 0.5;
lfilenames = ["outcomes_default.mat", "outcomes_apical_inhibition.mat", 'outcomes_mixed.mat'];
expert_t = zeros(3, nrseeds);
learning_t = zeros(3, nrseeds);

for f = 1:3
    load(lfilenames(f))
    n_trials = size(correct_trials, 2);
    x = 0:n_trials;
    perf_traces = zeros(nrseeds, n_trials+1);
    for s = 1:nrseeds
        disp({f, s})
        [t_learn, t_expert, pmid, p05]  = runanalysisv3(correct_trials(s,:), maxres, corprob);
        expert_t(f, s) = t_expert;
        learning_t(f, s) = t_learn;
        perf_traces(s,:) = pmid;
    end
    if f == 1
        perf_default = perf_traces;
    elseif f == 2
        perf_apical_inhibition = perf_traces;
    elseif f == 3
        perf_mixed = perf_traces;
    end
    
end

save("performances.mat", "expert_t", "learning_t" ,"perf_apical_inhibition", "perf_default", "perf_mixed")