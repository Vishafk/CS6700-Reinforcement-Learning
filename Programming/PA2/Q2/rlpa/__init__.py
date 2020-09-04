from gym.envs.registration import register

register(
		id='chakra-v0',
		entry_point='rlpa.envs:chakra',
		max_episode_steps=40
		)

register(
		id='visham-v0',
		entry_point='rlpa.envs:visham',
		max_episode_steps=40
		)
