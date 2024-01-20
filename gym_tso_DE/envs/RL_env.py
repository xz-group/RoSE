"""
A new ckt environment based on a new structure of MDP
"""
import gym
from gym import spaces
import random
from multiprocessing.dummy import Pool as ThreadPool
from collections import OrderedDict
import yaml
import yaml.constructor
import pickle
import numpy as np
# from tso_BORL_DE.tso_Cadence import *

# Way of ordering the way a yaml file is read
class OrderedDictYAMLLoader(yaml.Loader):
    """
    A YAML loader that loads mappings into ordered dictionaries.
    """

    def __init__(self, *args, **kwargs):
        yaml.Loader.__init__(self, *args, **kwargs)

        self.add_constructor(u'tag:yaml.org,2002:map', type(self).construct_yaml_map)
        self.add_constructor(u'tag:yaml.org,2002:omap', type(self).construct_yaml_map)

    def construct_yaml_map(self, node):
        data = OrderedDict()
        yield data
        value = self.construct_mapping(node)
        data.update(value)

    def construct_mapping(self, node, deep=False):
        if isinstance(node, yaml.MappingNode):
            self.flatten_mapping(node)
        else:
            raise yaml.constructor.ConstructorError(None, None,
                                                    'expected a mapping node, but found %s' % node.id, node.start_mark)

        mapping = OrderedDict()
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            value = self.construct_object(value_node, deep=deep)
            mapping[key] = value
        return mapping

class OPAMP(gym.Env):
    """
    # The following are the Env methods you should know:
    # 1. reset(self): Reset the environment's state. Returns observation.
    # 2. step(self, action): Step the environment by one timestep. Returns observation, reward, done, info.
    # 3. render(self, mode='human'): Render one frame of the environment. The default mode will do something human friendly, such as pop up a window.     
    """     
    
    # Custom Environment that follows gym interface    
    metadata = {'render.modes': ['human']}

    PERF_LOW = -1
    PERF_HIGH = 0

    # Obtains yaml file
    path = os.getcwd()
    CIR_YAML = path+'/eval_engines/tso.yaml'

    def __init__(self):
        self.multi_goal = False
        self.generalize = True
        num_valid = 50
        self.specs_save = False
        self.valid = False
        print("Env is successfully initialized!")
        self.env_steps = 0
        with open(OPAMP.CIR_YAML, 'r') as f:
            # Open tso.yaml file
            yaml_data = yaml.load(f, OrderedDictYAMLLoader)

        # This step is to call for using specifications for training. The specifications are defined in /gen_specs/specs_gen_tso
        # Please check gen_specs.py. In function gen_data, it uses pickle to save the specifications.
        if self.generalize == False:
            specs = yaml_data['target_specs']
        else:
        # When self.generalize == true. This means that, we have already generated target design specifications before runing the training. Then, we can directly load the pre-generated design specifications.
            load_specs_path = OPAMP.path+"/gen_specs/specs_gen_tso"
            # Load the (sampled) specification for policy training.
            with open(load_specs_path, 'rb') as f:
                specs = pickle.load(f)
           
        self.specs = OrderedDict(sorted(specs.items(), key=lambda k: k[0]))
        
        # Default of self.specs_save is false
        if self.specs_save:
            with open("specs_"+str(num_valid)+str(random.randint(1,100000)), 'wb') as f:
                pickle.dump(self.specs, f)
        
        self.specs_ideal = []
        self.specs_id = list(self.specs.keys())
        self.fixed_goal_idx = -1 
        self.num_os = len(list(self.specs.values())[0])
        # Param array
        # Check tso.yaml file.
        params = yaml_data['params']
        self.params = []
        # params.keys() are mp1, mp2, mp3, mp4, mn1, mn2, mn3, mn4, cap1, cap2:  
        self.params_id = list(params.keys())
        # Example: mp1:  !!python/tuple [10000, 50000, 1000]
        for value in params.values():
            param_vec = np.arange(value[0], value[1], value[2])
            self.params.append(param_vec)

        # -1--reduce size; 0--keep the size; 2-- increase the size;
        self.action_meaning = [-1,0,2]
        # This defines the boundary of space.
        # Discrete spaces are used when we have a discrete action/observation space to be defined in the environment. So spaces.Discrete(2) means that we have a discrete variable which can take one of the two possible values.  
        # self.action_space = spaces.Box( np.array([-1,0,0]), np.array([+1,+1,+1]))  # steer, gas, brake
        # Box means that you are dealing with real valued quantities. The first array np.array([-1,0,0] are the lowest accepted values, and the second np.array([+1,+1,+1]) are the highest accepted values. In this case (using the comment) we see that we have 3 available actions:
        # 1. Steering: Real valued in [-1, 1]
        # 2. Gas: Real valued in [0, 1]
        # 3. Brake: Real valued in [0, 1]  
        # Please see the link here: https://stackoverflow.com/questions/44404281/openai-gym-understanding-action-space-notation-spaces-box
        # The link related to space.Tuple(): https://stackoverflow.com/questions/58964267/how-to-create-an-openai-gym-observation-space-with-multiple-features
        self.action_space = spaces.Tuple([spaces.Discrete(len(self.action_meaning))]*len(self.params_id))
        self.observation_space = spaces.Box(
            low=np.array([OPAMP.PERF_LOW]*2*len(self.specs_id)+len(self.params_id)*[1]),
            high=np.array([OPAMP.PERF_HIGH]*2*len(self.specs_id)+len(self.params_id)*[1]))
    
        # Initialize current param/spec observations
        self.cur_specs = np.zeros(len(self.specs_id), dtype=np.float32)
        self.cur_params_idx = np.zeros(len(self.params_id), dtype=np.int32)

        # Get the g* (overall design spec) you want to reach
        self.global_g = []
        for spec in list(self.specs.values()):
                self.global_g.append(float(spec[self.fixed_goal_idx]))
        self.g_star = np.array(self.global_g)
        # Please see the definition in tso.yaml file
        self.global_g = np.array(yaml_data['normalize'])
        # Objective number (used for validation)
        self.obj_idx = 0

    def reset(self):
        # if multi-goal is selected, every time reset occurs, it will select a different design spec as objective
        if self.generalize == True:
            if self.valid == True:
                if self.obj_idx > self.num_os-1:
                    self.obj_idx = 0
                idx = self.obj_idx
                self.obj_idx += 1
            else:
                idx = random.randint(0,self.num_os-1)
            self.specs_ideal = []
            for spec in list(self.specs.values()):
                self.specs_ideal.append(spec[idx])
            self.specs_ideal = np.array(self.specs_ideal)
        else:
            if self.multi_goal == False:
                self.specs_ideal = self.g_star 
            else:
                idx = random.randint(0,self.num_os-1)
                self.specs_ideal = []
                for spec in list(self.specs.values()):
                    self.specs_ideal.append(spec[idx])
                self.specs_ideal = np.array(self.specs_ideal)

        # Applicable only when you have multiple goals, normalizes everything to some global_g
        self.specs_ideal_norm = self.lookup(self.specs_ideal, self.global_g)

        # initialize current parameters
        # Staring point
        self.cur_params_idx = np.array([33, 20, 4, 16, 13, 12, 13, 25, 12, 40])

        # Starting point parameter simulation
        # 16 sets of spec for 16 corners
        self.cur_specs1, self.cur_specs2, self.cur_specs3, self.cur_specs4, self.cur_specs5, self.cur_specs6, self.cur_specs7, self.cur_specs8, self.cur_specs9, self.cur_specs10, self.cur_specs11, self.cur_specs12, self.cur_specs13, self.cur_specs14, self.cur_specs15, self.cur_specs16 = self.update(self.cur_params_idx)
        # Normalization
        cur_spec_norm1 = self.lookup(self.cur_specs1, self.global_g)
        cur_spec_norm2 = self.lookup(self.cur_specs2, self.global_g)
        cur_spec_norm3 = self.lookup(self.cur_specs3, self.global_g)
        cur_spec_norm4 = self.lookup(self.cur_specs4, self.global_g)
        cur_spec_norm5 = self.lookup(self.cur_specs5, self.global_g)
        cur_spec_norm6 = self.lookup(self.cur_specs6, self.global_g)
        cur_spec_norm7 = self.lookup(self.cur_specs7, self.global_g)
        cur_spec_norm8 = self.lookup(self.cur_specs8, self.global_g)
        cur_spec_norm9 = self.lookup(self.cur_specs9, self.global_g)
        cur_spec_norm10 = self.lookup(self.cur_specs10, self.global_g)
        cur_spec_norm11 = self.lookup(self.cur_specs11, self.global_g)
        cur_spec_norm12 = self.lookup(self.cur_specs12, self.global_g)
        cur_spec_norm13 = self.lookup(self.cur_specs13, self.global_g)
        cur_spec_norm14 = self.lookup(self.cur_specs14, self.global_g)
        cur_spec_norm15 = self.lookup(self.cur_specs15, self.global_g)
        cur_spec_norm16 = self.lookup(self.cur_specs16, self.global_g)

        # Observation is a combination of current specs distance from ideal, ideal spec, and current param vals
        self.ob = np.concatenate([cur_spec_norm1, cur_spec_norm2, cur_spec_norm3, cur_spec_norm4, cur_spec_norm5, cur_spec_norm6, cur_spec_norm7, cur_spec_norm8, cur_spec_norm9, cur_spec_norm10, cur_spec_norm11, cur_spec_norm12, cur_spec_norm13, cur_spec_norm14, cur_spec_norm15, cur_spec_norm16, self.specs_ideal_norm, self.cur_params_idx])
        return self.ob

    def step(self, action):
        """
        :param action: is vector with elements between 0 and 1 mapped to the index of the corresponding parameter
        :return: self.ob is an array contains normalized specs and parameter index
                 reward
                 done is a boolean value represents if reach the target spec or not
        """

        # Take action that RL agent returns to change current params
        action = list(np.reshape(np.array(action), (np.array(action).shape[0],)))
        self.cur_params_idx = self.cur_params_idx + np.array([self.action_meaning[a] for a in action])
        self.cur_params_idx = np.clip(self.cur_params_idx, [0]*len(self.params_id), [(len(param_vec)-1) for param_vec in self.params])
        
        # Get current specs and normalize
        self.cur_specs1, self.cur_specs2, self.cur_specs3, self.cur_specs4, self.cur_specs5, self.cur_specs6, self.cur_specs7, self.cur_specs8, self.cur_specs9, self.cur_specs10, self.cur_specs11, self.cur_specs12, self.cur_specs13, self.cur_specs14, self.cur_specs15, self.cur_specs16 = self.update(self.cur_params_idx)
        cur_spec_norm1 = self.lookup(self.cur_specs1, self.global_g)
        cur_spec_norm2 = self.lookup(self.cur_specs2, self.global_g)
        cur_spec_norm3 = self.lookup(self.cur_specs3, self.global_g)
        cur_spec_norm4 = self.lookup(self.cur_specs4, self.global_g)
        cur_spec_norm5 = self.lookup(self.cur_specs5, self.global_g)
        cur_spec_norm6 = self.lookup(self.cur_specs6, self.global_g)
        cur_spec_norm7 = self.lookup(self.cur_specs7, self.global_g)
        cur_spec_norm8 = self.lookup(self.cur_specs8, self.global_g)
        cur_spec_norm9 = self.lookup(self.cur_specs9, self.global_g)
        cur_spec_norm10 = self.lookup(self.cur_specs10, self.global_g)
        cur_spec_norm11 = self.lookup(self.cur_specs11, self.global_g)
        cur_spec_norm12 = self.lookup(self.cur_specs12, self.global_g)
        cur_spec_norm13 = self.lookup(self.cur_specs13, self.global_g)
        cur_spec_norm14 = self.lookup(self.cur_specs14, self.global_g)
        cur_spec_norm15 = self.lookup(self.cur_specs15, self.global_g)
        cur_spec_norm16 = self.lookup(self.cur_specs16, self.global_g)

        # calculate the reward
        reward1 = self.reward(self.cur_specs1, self.specs_ideal)
        reward2 = self.reward(self.cur_specs2, self.specs_ideal)
        reward3 = self.reward(self.cur_specs3, self.specs_ideal)
        reward4 = self.reward(self.cur_specs4, self.specs_ideal)
        reward5 = self.reward(self.cur_specs5, self.specs_ideal)
        reward6 = self.reward(self.cur_specs6, self.specs_ideal)
        reward7 = self.reward(self.cur_specs7, self.specs_ideal)
        reward8 = self.reward(self.cur_specs8, self.specs_ideal)
        reward9 = self.reward(self.cur_specs9, self.specs_ideal)
        reward10 = self.reward(self.cur_specs10, self.specs_ideal)
        reward11 = self.reward(self.cur_specs11, self.specs_ideal)
        reward12 = self.reward(self.cur_specs12, self.specs_ideal)
        reward13 = self.reward(self.cur_specs13, self.specs_ideal)
        reward14 = self.reward(self.cur_specs14, self.specs_ideal)
        reward15 = self.reward(self.cur_specs15, self.specs_ideal)
        reward16 = self.reward(self.cur_specs16, self.specs_ideal)
        reward_arr = np.array([reward1, reward2, reward3, reward4, reward5, reward6, reward7, reward8, reward9, reward10, reward11, reward12, reward13, reward14, reward15, reward16])
        if np.all((reward_arr < 0)):
            reward = np.average(reward_arr)
        elif np.all((reward_arr == 10)):
            reward = np.average(reward_arr)
        else: 
            reward_arr[reward_arr>0] = 0
            reward = np.average(reward_arr)
        done = False

        # Incentivize reward
        if (reward >= 10):
            done = True

        self.ob = np.concatenate([cur_spec_norm1, cur_spec_norm2, cur_spec_norm3, cur_spec_norm4, cur_spec_norm5, cur_spec_norm6, cur_spec_norm7, cur_spec_norm8, cur_spec_norm9, cur_spec_norm10, cur_spec_norm11, cur_spec_norm12, cur_spec_norm13, cur_spec_norm14, cur_spec_norm15, cur_spec_norm16, self.specs_ideal_norm, self.cur_params_idx])
        self.env_steps = self.env_steps + 1

        return self.ob, reward, done, {}

    def lookup(self, spec, goal_spec):
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
    
    def reward(self, spec, goal_spec):
        '''
        Reward: doesn't penalize for overshooting spec, is negative
        '''
        rel_specs = self.lookup(spec, goal_spec)
        pos_val = [] 
        reward = 0.0
        for i,rel_spec in enumerate(rel_specs):
            if(self.specs_id[i] == 'ibias_max'):
                rel_spec = rel_spec*-1.0
            if rel_spec < 0:
                reward += rel_spec
                pos_val.append(0)
            else:
                pos_val.append(1)
        return reward if reward < -0.000005 else 10

    def update(self, params_idx):
        """
        :param action: an int between 0 ... n-1
        :return: specs
        """
        # Get the param vals and set # of corners
        params = [self.params[i][params_idx[i]] for i in range(len(self.params_id))]       
        param_val = OrderedDict(list(zip(self.params_id,params)))
        num_corner = 16
    
        # process param vals
        param_val2 = []
        mp1 = str(param_val['mp1']) + "n"
        mp2 = str(param_val['mp2']) + "n"
        mp3 = str(param_val['mp3']) + "n"
        mp4 = str(param_val['mp4']) + "n"
        mn1 = str(param_val['mn1']) + "n"
        mn2 = str(param_val['mn2']) + "n"
        mn3 = str(param_val['mn3']) + "n"
        mn4 = str(param_val['mn4']) + "n"
        cap1 = str(round(param_val['cap1'], 1)) + "p"
        cap2 = str(round(param_val['cap2'], 1)) + "p"
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
        # check tso_Cadence.py
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
        '''
        gain_arr = np.array([cur_specs1[0], cur_specs2[0], cur_specs3[0], cur_specs4[0], cur_specs5[0], cur_specs6[0], cur_specs7[0], cur_specs8[0], cur_specs9[0], cur_specs10[0], cur_specs11[0], cur_specs12[0], cur_specs13[0], cur_specs14[0], cur_specs15[0], cur_specs16[0]])
        I_arr = np.array([cur_specs1[1], cur_specs2[1], cur_specs3[1], cur_specs4[1], cur_specs5[1], cur_specs6[1], cur_specs7[1], cur_specs8[1], cur_specs9[1], cur_specs10[1], cur_specs11[1], cur_specs12[1], cur_specs13[1], cur_specs14[1], cur_specs15[1], cur_specs16[1]])
        PM_arr = np.array([cur_specs1[2], cur_specs2[2], cur_specs3[2], cur_specs4[2], cur_specs5[2], cur_specs6[2], cur_specs7[2], cur_specs8[2], cur_specs9[2], cur_specs10[2], cur_specs11[2], cur_specs12[2], cur_specs13[2], cur_specs14[2], cur_specs15[2], cur_specs16[2]])
        UGF_arr = np.array([cur_specs1[3], cur_specs2[3], cur_specs3[3], cur_specs4[3], cur_specs5[3], cur_specs6[3], cur_specs7[3], cur_specs8[3], cur_specs9[3], cur_specs10[3], cur_specs11[3], cur_specs12[3], cur_specs13[3], cur_specs14[3], cur_specs15[3], cur_specs16[3]])
        with open('gain.txt', 'a') as f:
            np.savetxt(f, gain_arr, newline=" ")
            f.write('\n')
        with open('I.txt', 'a') as f:
            np.savetxt(f, I_arr, newline=" ")
            f.write('\n')
        with open('PM.txt', 'a') as f:
            np.savetxt(f, PM_arr, newline=" ")
            f.write('\n')
        with open('UGF.txt', 'a') as f:
            np.savetxt(f, UGF_arr, newline=" ")
            f.write('\n')
        '''
        return cur_specs1, cur_specs2, cur_specs3, cur_specs4, cur_specs5, cur_specs6, cur_specs7, cur_specs8, cur_specs9, cur_specs10, cur_specs11, cur_specs12, cur_specs13, cur_specs14, cur_specs15, cur_specs16

def main():
    env = OPAMP()
    env.reset()
    env.step([2,2,2,2,2,2,2,2,2,2])

if __name__ == "__main__":
    main()
