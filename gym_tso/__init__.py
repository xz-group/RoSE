from gym.envs.registration import register

register(
    id='tso-v0',
    entry_point='gym_tso.envs:OPAMP',
)