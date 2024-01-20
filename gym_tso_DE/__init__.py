from gym.envs.registration import register

register(
    id='tso-v2',
    entry_point='gym_tso_DE.envs:OPAMP',
)