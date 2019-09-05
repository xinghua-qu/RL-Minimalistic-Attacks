# RL-Minimalistic-Attacks

# Dependencies:
	python 3.6
	tensorflow
	gym[atari]
	stable_baselines
	cv2
	pybrain
	multiprocessing
	matplotlib
# Run the code with command:
python DQN_Minimalistic_Attack.py -g 'Pong' -a 'dqn' -n 5 -t 0.9 -r 'run_name'
python A2C_Minimalistic_Attack.py -g 'Pong' -a 'a2c' -n 5 -t 0.9 -r 'run_name'
python PPO2_Minimalistic_Attack.py -g 'Pong' -a 'ppo2' -n 5 -t 0.9 -r 'run_name'
python ACKTR_Minimalistic_Attack.py -g 'Pong' -a 'acktr' -n 5 -t 0.9 -r 'run_name'
