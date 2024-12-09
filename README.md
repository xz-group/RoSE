# RoSE

A robust analog circuit parameter optimization framework with high sampling efficience by synergistically combining Bayesian optimization (BO) and reinforcement learning (RL). 

## About this work

For more details, please refer to our **DAC'23** paper: [_RoSE: Robust Analog Circuit Parameter Optimization with Sampling-Efficient Reinforcement Learning_](https://ieeexplore.ieee.org/document/10247991)

And our **TCAD'24** paper: [_RoSE-Opt: Robust and Efficient Analog Circuit Parameter Optimization with Knowledge-infused Reinforcement Learning_](https://ieeexplore.ieee.org/document/10614385)

## How to Use

Details and code regarding Cadence simulation will be updated soon.

### Environment Setup

This setup requires Anaconda. Run the following command below:

```bash
conda env create -f environment.yml
```

To activate the environment:

```bash
conda activate RoSE
```

### Generate Specs

```
python gen_specs.py --num_specs ##
```

### RoSE

#### BO for optimizing the starting point 

```
# cd to your Cadence folder
cd tso
python tso_BO.py
```

After getting a suboptimal parameter from BO, convert it to a discrete index in the design space

```
# post proccess all suboptimal parameters from BO to index 
# replace this index in line 169 in gym_tso/envs/RL_env.py for training
self.cur_params_idx = np.array([33, 20, 4, 16, 13, 12, 13, 25, 12, 40])
# replace this index in line 169 in gym_tso_DE/envs/RL_env.py for deployment
self.cur_params_idx = np.array([33, 20, 4, 16, 13, 12, 13, 25, 12, 40])
```

#### RL training 

```
# cd to your Cadence folder
python main_train_BORL.py
```

#### RL deployment

```
# cd to your Cadence folder
python DE_tso_BORL.py
```

## Citation

If you use this framework for your research, please cite our [DAC'23 paper](https://ieeexplore.ieee.org/document/10247991):

```
@inproceedings{gao2023rose,
  title={RoSE: Robust Analog Circuit Parameter Optimization with Sampling-Efficient Reinforcement Learning},
  author={Gao, Jian and Cao, Weidong and Zhang, Xuan},
  booktitle={2023 60th ACM/IEEE Design Automation Conference (DAC)},
  pages={1--6},
  year={2023},
  organization={IEEE}
}
```

our [TCAD'24 paper](https://ieeexplore.ieee.org/document/10614385):

```
@article{cao2024rose,
  title={Rose-opt: Robust and efficient analog circuit parameter optimization with knowledge-infused reinforcement learning},
  author={Cao, Weidong and Gao, Jian and Ma, Tianrui and Ma, Rui and Benosman, Mouhacine and Zhang, Xuan},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems},
  year={2024},
  publisher={IEEE}
}
```

## Contact Information

If you have any questions regarding using this framework, please feel free to contact us at [gao.jian3@northeastern.edu](mailto:gao.jian3@northeastern.edu).

## Version History

* 0.1
  * Initial Release

## License

This framework is licensed under the `GNU3` License - see the [LICENSE.md](LICENSE) file for details
