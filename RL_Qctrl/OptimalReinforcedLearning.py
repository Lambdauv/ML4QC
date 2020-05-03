import time
import tensorflow as tf
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import wrappers
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.networks.q_network import QNetwork
from tensorflow import keras
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.trajectories.trajectory import to_transition
from tf_agents.eval.metric_utils import log_metrics
import logging
logging.getLogger().setLevel(logging.INFO)
from tf_agents.utils.common import function
from tf_agents.metrics import tf_metrics
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies.random_tf_policy import RandomTFPolicy

from tf_agents.agents.ddpg import actor_network
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.ddpg import ddpg_agent

from tf_agents.trajectories.time_step import StepType
from tf_agents.trajectories.time_step import TimeStep

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

from ML4QC.tf_tensormath import qc_helper
from ML4QC.tf_tensormath import basicOps
from ML4QC.RL_Qctrl.environment import PulseResponseEnv

sigmax = tf.constant([[0,1],[1,0]],dtype=tf.complex128)
sigmaz = tf.constant([[1,0],[0,-1]],dtype=tf.complex128)
sigmay = sigmaz@sigmax/1j
sigxsigx=basicOps.tensorDirectProd(sigmax,sigmax)
sigysigy=basicOps.tensorDirectProd(sigmay,sigmay)
sigzs = [basicOps.tensorDirectProd(sigmaz,tf.eye(2,dtype=tf.complex128)),basicOps.tensorDirectProd(tf.eye(2,dtype=tf.complex128),sigmaz)]
sigxs = [basicOps.tensorDirectProd(sigmax,tf.eye(2,dtype=tf.complex128)),basicOps.tensorDirectProd(tf.eye(2,dtype=tf.complex128),sigmax)]
sigys = [basicOps.tensorDirectProd(sigmay,tf.eye(2,dtype=tf.complex128)),basicOps.tensorDirectProd(tf.eye(2,dtype=tf.complex128),sigmay)]

summary_writer = tf.summary.create_file_writer(logdir = './TFruns',flush_millis=1 * 1000)
summary_writer.set_as_default()
strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())#
train_step = tf.compat.v1.train.get_or_create_global_step()
fc_layer_params = [128]
update_period = 50
Niterations = 1000000
TotalTrainingIterations = Niterations#250000//update_period
nRandTotalSteps = 100000
log_interval = 10000
num_eval_episodes = 1
train_steps_per_iteration = 100

with tf.device('/device:CPU:0'):
	Utarget = np.eye(4,dtype=np.complex128)
	Utarget[3,3]=-1
	Utarget = tf.constant(Utarget) 
	rho0s = qc_helper.genAllPaulyCombine(2)
	num_parallel_environments = 1
	env = PulseResponseEnv.PulseResponseEnv(Utarget,rho0s)
	if num_parallel_environments == 1:
		py_env = env
	else:
		py_env = parallel_py_environment.ParallelPyEnvironment([lambda: env] * num_parallel_environments)
	tf_env = tf_py_environment.TFPyEnvironment(py_env)

with strategy.scope():
	target_update_tau=0.05
	target_update_period=5
	ou_stddev=0.2
	ou_damping=0.15
	actor_net = actor_network.ActorNetwork(
		tf_env.time_step_spec().observation,
		tf_env.action_spec(),
		fc_layer_params=[128],
	)
	critic_net_input_specs = (tf_env.time_step_spec().observation,
							  tf_env.action_spec())

	critic_net = critic_network.CriticNetwork(
		critic_net_input_specs,
		observation_fc_layer_params=[128],
		action_fc_layer_params=[128],
		joint_fc_layer_params=[128],
	)
	agent = ddpg_agent.DdpgAgent(
		tf_env.time_step_spec(),
		tf_env.action_spec(),
		actor_network=actor_net,
		critic_network=critic_net,
		actor_optimizer=tf.compat.v1.train.AdamOptimizer(
			learning_rate=0.001),
		critic_optimizer=tf.compat.v1.train.AdamOptimizer(
			learning_rate=0.001),
		ou_stddev=ou_stddev,
		ou_damping=ou_damping,
		target_update_tau=target_update_tau,
		target_update_period=target_update_period,
		dqda_clipping=None,
		td_errors_loss_fn=keras.losses.Huber(reduction="none"),
		gamma=0.995,
		reward_scale_factor=1,
		gradient_clipping=None,
		debug_summaries=False,
		summarize_grads_and_vars=False,
		train_step_counter=train_step)
	agent.initialize()

with tf.device('/cpu:0'):
	replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
		data_spec = agent.collect_data_spec,
		batch_size = tf_env.batch_size,
		max_length = 100000
	)
	replay_buffer_observer = replay_buffer.add_batch
	env_steps = tf_metrics.EnvironmentSteps(prefix='Train')
	average_return = tf_metrics.AverageReturnMetric(
				prefix='Train',
				buffer_size=num_eval_episodes,
				batch_size=tf_env.batch_size)
	train_metrics = [
		tf_metrics.NumberOfEpisodes(),
		env_steps,
		average_return,
		tf_metrics.AverageEpisodeLengthMetric(
			prefix='Train',
			buffer_size=num_eval_episodes,
			batch_size=tf_env.batch_size),
	]
	initial_collect_policy = RandomTFPolicy(tf_env.time_step_spec(),tf_env.action_spec())
	init_driver = DynamicStepDriver(
		tf_env,
		initial_collect_policy,
		observers=[replay_buffer.add_batch],
		num_steps=nRandTotalSteps
	)
	collect_driver = DynamicStepDriver(
		tf_env,
		agent.collect_policy,
		observers = [replay_buffer_observer]+train_metrics,
		num_steps = update_period
	)
	collect_driver.run = function(collect_driver.run)
	init_driver.run = function(init_driver.run)
	agent.train = function(agent.train)
	final_time_step, final_policy_state = init_driver.run()
	time_acc = 0
	env_steps_before = env_steps.result().numpy()
	dataset = replay_buffer.as_dataset(
		sample_batch_size = 64,
		num_steps = 2,
		num_parallel_calls = 3
	).prefetch(3)
	iterator = iter(strategy.experimental_distribute_dataset(dataset))
	def train_step_fun(dist_experiences):
		per_example_losses = strategy.experimental_run_v2(agent.train, args=(dist_experiences,))
		mean_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_example_losses.loss, axis=None)
		#mean_loss = per_example_losses.loss
		return mean_loss
	train_step_fun = function(train_step_fun)
	time_step = None
	policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
	for iteration in range(Niterations):
		#print('Start Iteration %dth' %(iteration))
		start_time = time.time()
		time_step,policy_state = collect_driver.run(time_step,policy_state)
		with strategy.scope():
			for subStep in range(train_steps_per_iteration):
				trajectories, buffer_info = next(iterator)
				train_loss = train_step_fun(trajectories)
				#print("\r[{}, {}] loss:{:.5f}".format(iteration,subStep,train_loss.numpy()),end="")
		time_acc += time.time() - start_time
		#print('\nIteration %dth done' %(iteration))
		if train_step.numpy() % log_interval == 0:
			logging.info('env steps = %d, average return = %f', env_steps.result(),
						 average_return.result())
			env_steps_per_sec = (env_steps.result().numpy() -
								 env_steps_before) / time_acc
			logging.info('%.3f env steps/sec', env_steps_per_sec)
			tf.summary.scalar(
				name='env_steps_per_sec',
				data=env_steps_per_sec,
				step=env_steps.result())
			time_acc = 0
			env_steps_before = env_steps.result().numpy()
			
	for train_metric in train_metrics:
		train_metric.tf_summaries(train_step=env_steps.result())

