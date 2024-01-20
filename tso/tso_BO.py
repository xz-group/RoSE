"""
	This file contains code for Bayesian optimization and Cadence simulation
"""
import os
import torch
import random
from botorch.models import SingleTaskGP, ModelListGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.optim import optimize_acqf
import numpy as np
import pandas as pd

# Modify it to your directory
Top_dir = '/home/research/gaoj1/jiangao_design'

# Normalization 
def lookup(spec, goal_spec):
    goal_spec = [float(e) for e in goal_spec]
    norm_spec = []
    for i in range (len(spec)):
        if spec[i] < 0:
            # switch to this if you want the specs from -1 to 1
            # norm_spec.append((spec[i]-goal_spec[i])/(goal_spec[i]-spec[i]))
            norm_spec.append((spec[i]-goal_spec[i])/goal_spec[i])
        else:
            norm_spec.append((spec[i]-goal_spec[i])/(goal_spec[i]+spec[i]))
    return norm_spec

# Reward
def reward_func(spec, goal_spec):
    # Reward: doesn't penalize for overshooting spec, is negative
    rel_specs = lookup(spec, goal_spec)
    pos_val = [] 
    reward = 0.0
    for i,rel_spec in enumerate(rel_specs):
        if(i == 1):
            # Reverse it for current since we try to maximize the specs
            rel_spec = rel_spec*-1.0
        if rel_spec < 0:
            reward += rel_spec
            pos_val.append(0)
        else:
            pos_val.append(1)
    return reward

# Bayesian optimization part
# Target function's input is device parameter and output is the average reward for 16 corners
def target_function(parameter_tensor, specs_ideal):
    reward_result = []
    for parameter in (parameter_tensor):
        num_corner = 16

        # process param vals
        param_val2 = []
        mp1 = str(int(parameter[0])) + "n"
        mp2 = str(int(parameter[1])) + "n"
        mp3 = str(int(parameter[2])) + "n"
        mp4 = str(int(parameter[3])) + "n"
        mn1 = str(int(parameter[4])) + "n"
        mn2 = str(int(parameter[5])) + "n"
        mn3 = str(int(parameter[6])) + "n"
        mn4 = str(int(parameter[7])) + "n"
        cap1 = str(round(float(parameter[8]), 1)) + "p"
        cap2 = str(round(float(parameter[9]), 1)) + "p"
        param_val2.append(mp1)
        param_val2.append(mp2)
        param_val2.append(mp3)
        param_val2.append(mp4)
        param_val2.append(mn1)
        param_val2.append(mn2)
        param_val2.append(mn3)
        param_val2.append(mn4)
        param_val2.append(cap1)
        param_val2.append(cap2)

        # Simulate
        # cur_specs = modify_and_simulate(param_val2, num_corner)
        # cur_specs1 = postprocess(np.array(cur_specs["corner0"]))
        # cur_specs2 = postprocess(np.array(cur_specs["corner1"]))
        # cur_specs3 = postprocess(np.array(cur_specs["corner2"]))
        # cur_specs4 = postprocess(np.array(cur_specs["corner3"]))
        # cur_specs5 = postprocess(np.array(cur_specs["corner4"]))
        # cur_specs6 = postprocess(np.array(cur_specs["corner5"]))
        # cur_specs7 = postprocess(np.array(cur_specs["corner6"]))
        # cur_specs8 = postprocess(np.array(cur_specs["corner7"]))
        # cur_specs9 = postprocess(np.array(cur_specs["corner8"]))
        # cur_specs10 = postprocess(np.array(cur_specs["corner9"]))
        # cur_specs11 = postprocess(np.array(cur_specs["corner10"]))
        # cur_specs12 = postprocess(np.array(cur_specs["corner11"]))
        # cur_specs13 = postprocess(np.array(cur_specs["corner12"]))
        # cur_specs14 = postprocess(np.array(cur_specs["corner13"]))
        # cur_specs15 = postprocess(np.array(cur_specs["corner14"]))
        # cur_specs16 = postprocess(np.array(cur_specs["corner15"]))
        cur_specs1 = np.array(list(cur_specs1))
        cur_specs2 = np.array(list(cur_specs2))
        cur_specs3 = np.array(list(cur_specs3))
        cur_specs4 = np.array(list(cur_specs4))
        cur_specs5 = np.array(list(cur_specs5))
        cur_specs6 = np.array(list(cur_specs6))
        cur_specs7 = np.array(list(cur_specs7))
        cur_specs8 = np.array(list(cur_specs8))
        cur_specs9 = np.array(list(cur_specs9))
        cur_specs10 = np.array(list(cur_specs10))
        cur_specs11 = np.array(list(cur_specs11))
        cur_specs12 = np.array(list(cur_specs12))
        cur_specs13 = np.array(list(cur_specs13))
        cur_specs14 = np.array(list(cur_specs14))
        cur_specs15 = np.array(list(cur_specs15))
        cur_specs16 = np.array(list(cur_specs16))

        # Calculate the reward
        reward1 = reward_func(cur_specs1, specs_ideal)
        reward2 = reward_func(cur_specs2, specs_ideal)
        reward3 = reward_func(cur_specs3, specs_ideal)
        reward4 = reward_func(cur_specs4, specs_ideal)
        reward5 = reward_func(cur_specs5, specs_ideal)
        reward6 = reward_func(cur_specs6, specs_ideal)
        reward7 = reward_func(cur_specs7, specs_ideal)
        reward8 = reward_func(cur_specs8, specs_ideal)
        reward9 = reward_func(cur_specs9, specs_ideal)
        reward10 = reward_func(cur_specs10, specs_ideal)
        reward11 = reward_func(cur_specs11, specs_ideal)
        reward12 = reward_func(cur_specs12, specs_ideal)
        reward13 = reward_func(cur_specs13, specs_ideal)
        reward14 = reward_func(cur_specs14, specs_ideal)
        reward15 = reward_func(cur_specs15, specs_ideal)
        reward16 = reward_func(cur_specs16, specs_ideal)
        reward_arr = np.array([reward1, reward2, reward3, reward4, reward5, reward6, reward7, reward8, reward9, reward10, reward11, reward12, reward13, reward14, reward15, reward16])
        reward = np.average(reward_arr)
        reward_result.append(reward)
    return torch.tensor(reward_result).float()

# Initial random sampling
# check tso.yaml for design space
def generate_initial_data(n, specs_ideal):
    mp1 = torch.randint(10000, 50000, (n,))
    mp2 = torch.randint(10000, 50000, (n,))
    mp3 = torch.randint(10000, 50000, (n,))
    mp4 = torch.randint(50000, 250000, (n,))
    mn1 = torch.randint(2000, 20000, (n,))
    mn2 = torch.randint(2000, 20000, (n,))
    mn3 = torch.randint(2000, 20000, (n,))
    mn4 = torch.randint(10000, 50000, (n,))
    cap1 = torch.FloatTensor(n, ).uniform_(25.0, 50.0)
    cap2 = torch.FloatTensor(n, ).uniform_(1.0, 25.0)
    parameter = torch.stack((mp1,mp2,mp3,mp4,mn1,mn2,mn3,mn4,cap1,cap2), dim = 1)
    parameter = parameter.float()
    exact_obj = target_function(parameter, specs_ideal).unsqueeze(-1)

    # get the best point's spec and parameter
    best_observerd_value = exact_obj.max().item()
    idx = torch.argmax(exact_obj).item()
    best_parameter = parameter[int(idx), :]
    return parameter, exact_obj, best_observerd_value, best_parameter

# Acquisition function to get the next parameter
def get_next_points(parameter, init_reward, best_init_reward, bounds, n_points):
    single_model = SingleTaskGP(parameter, init_reward)
    mll = ExactMarginalLogLikelihood(single_model.likelihood, single_model)
    fit_gpytorch_model(mll)
    EI = qExpectedImprovement(model = single_model, best_f = best_init_reward)

    # adjust the hyperparameter for trade-off between optimization speed and result
    candidates, _ = optimize_acqf(acq_function = EI, bounds = bounds, q = n_points, num_restarts = 200, raw_samples = 1024, options={"batch_limit": 5, "maxiter": 200})
    return candidates

# Sample from sampling space
def get_idealspec():
    # Choose center specs for starting point optimizaiton
    gain = 42.5
    I = 0.055
    PM = 55.0
    UGF = 1.5e6
    idealspec = np.array([gain, I, PM, UGF])

    # change to this for normal BO run
    # gain = random.uniform(40, 45)
    # I = random.uniform(0.01, 0.1)
    # PM = 55.0
    # UGF = random.uniform(1.0e6, 2.0e6)
    # idealspec = np.array([gain, I, PM, UGF])
    return idealspec

# reset()
# Number of initial random sampling points
init_samplepoints = 20
# Number of candidate acquisition function give
num_candidates = 1
specs_ideal = get_idealspec()
print(f"Ideal spec is: {specs_ideal}")
parameter, init_reward, best_init_reward, best_init_parameter = generate_initial_data(init_samplepoints, specs_ideal)
print(f"Initial sample parameters: {parameter.numpy()}")
print(f"Initial sample reward: {init_reward.numpy()}")
print(f"Initial sample best reward: {best_init_reward}")
print(f"Initial sample best parameter: {best_init_parameter.numpy()}")
# Define the design space
bounds = torch.tensor([[10000., 10000., 10000., 50000., 2000., 2000., 2000., 10000., 25., 1.],[50000., 50000., 50000., 250000., 20000., 20000., 20000., 50000., 50.0, 25.0]])
# Number of optimization runs
n_runs = 30
wtdir = Top_dir+"/tso"
log = open(wtdir + '/tso_BO.log', 'a')
log_interval = 1
# Running reward is used to calculate the average reward for each log_interval
running_reward = 0
target_reward = 0
print("Begin!")
for i in range(1, n_runs+1):
    # print(f"Nr. of optimization run: {i}")
    new_candidates = get_next_points(parameter, init_reward, best_init_reward, bounds, num_candidates)
    new_results = target_function(new_candidates, specs_ideal).unsqueeze(-1)
    running_reward += new_results
    # print(f"New candidate is: {new_candidates}  New reward is: {new_results}")
    parameter = torch.cat([parameter, new_candidates])
    init_reward = torch.cat([init_reward, new_results])
    best_init_reward = init_reward.max().item()
    idx = torch.argmax(init_reward).item()
    best_init_parameter = parameter[int(idx), :]
    # print(f"Best performaning point: {best_init_reward}")
    if float(best_init_reward) >= target_reward:
        if i < log_interval:
            running_reward = round(float(running_reward) / i, 3)
        elif i % log_interval == 0:
            running_reward = round(float(running_reward) / log_interval, 3)
        else:
            running_reward = round(float(running_reward) /(i % log_interval), 3)
        print(f"Simulation steps: {i} Mean reward: {running_reward} Best point: {best_init_reward}")
        # Write the simulation steps, mean reward, best reward to log
        log.write(str(i) + "\t" + str(running_reward) + "\t" + str(best_init_reward) + "\t")
        # Write the best reward's parameter to log
        best_init_parameter_arr = best_init_parameter.numpy()
        for i in range (len(best_init_parameter_arr)):
            if (i > 7):
                # capacitance
                log.write(str(round(float(best_init_parameter_arr[i]), 1)) + "\t")
            else:
                log.write(str(int(best_init_parameter_arr[i])) + "\t")
        log.write("\n")
        running_reward = 0
        log.flush()
        print("########## Solved! ##########")
        break
    # logging for each log interval
    if i % log_interval == 0:
        running_reward = round(float(running_reward) / log_interval, 3)
        print(f"Simulation steps: {i} Mean reward: {running_reward} Best point: {best_init_reward}")
        # Write the simulation steps, mean reward, best reward to log
        log.write(str(i) + "\t" + str(running_reward) + "\t" + str(best_init_reward) + "\t")
        # Write the best reward's parameter to log
        best_init_parameter_arr = best_init_parameter.numpy()
        for i in range (len(best_init_parameter_arr)):
            if (i > 7):
                # capacitance
                log.write(str(round(float(best_init_parameter_arr[i]), 1)) + "\t")
            else:
                log.write(str(int(best_init_parameter_arr[i])) + "\t")
        log.write("\n")
        running_reward = 0
        log.flush()
print("Done!")
# end_simulator()
log.close()