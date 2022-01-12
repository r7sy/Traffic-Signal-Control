from model import *
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import numpy as np
import tensorboard
import os
from datetime import datetime
import random
import cityflow
import gym
from tlc_baselines.world import World
from tlc_baselines.environment import TSCEnv
from tlc_baselines.generator import LaneVehicleGenerator,IntersectionVehicleGenerator
from tlc_baselines.metric import TravelTimeMetric
import json

# Taken from https://github.com/rlworkgroup/garage
def compute_advantages(discount,
                       gae_lambda,
                       max_len,
                       baselines,
                       rewards,
                       name='compute_advantages'):
    with tf.name_scope(name):
        # Prepare convolutional IIR filter to calculate advantages
        gamma_lambda = tf.constant(float(discount) * float(gae_lambda),
                                   dtype=tf.float32,
                                   shape=[max_len, 1, 1])
        advantage_filter = tf.compat.v1.cumprod(gamma_lambda,
                                                axis=0,
                                                exclusive=True)

        # Calculate deltas
        pad = tf.zeros_like(baselines[:, :1])
        baseline_shift = tf.concat([baselines[:, 1:], pad], 1)
        deltas = rewards + discount * baseline_shift - baselines

        # Convolve deltas with the discount filter to get advantages
        deltas_pad = tf.expand_dims(tf.concat(
            [deltas, tf.zeros_like(deltas[:, :-1])], 1),
                                    axis=2)
        adv = tf.nn.conv1d(deltas_pad,
                           advantage_filter,
                           stride=1,
                           padding='VALID')
        advantages = tf.reshape(adv, [-1])

    return advantages


# Taken from https://github.com/rlworkgroup/garage
def discounted_returns(discount, max_len, rewards, name='discounted_returns'):
    with tf.name_scope(name):
        gamma = tf.constant(float(discount),
                            dtype=tf.float32,
                            shape=[max_len, 1, 1])
        return_filter = tf.math.cumprod(gamma, axis=0, exclusive=True)
        rewards_pad = tf.expand_dims(tf.concat(
            [rewards, tf.zeros_like(rewards[:, :-1])], 1),
                                     axis=2)
        returns = tf.nn.conv1d(rewards_pad,
                               return_filter,
                               stride=1,
                               padding='VALID')

    return returns


def runEpisode(env, agents, baseline, greedy, collect=True, gamma=0.99, gae_lam=0.95, steps=5,modified_gae=True):
    observations = env.reset()
    counter = 0
    values = []
    values2 = []
    states = []
    afterstates = []
    actions = []
    log_probs = []
    rewards = []

    while counter < steps:
        obs2 = []
        n_agents = len(agents)
        observations = np.vstack(observations)
        observations = observations.reshape([1, n_agents, -1])
        _actions = []
        _log_probs = []
        for i, agent in enumerate(agents):
            means = agents[i].policy(observations[:, i, :])
            action_distribution = tfp.distributions.Categorical(logits=means)
            action_taken = agents[i].get_action(observations[:, i, :], greedy)

            log_prob = action_distribution.log_prob(action_taken)
            _log_probs.append(log_prob)
            _actions.append(action_taken)
            obs2.append(agent.get_ob())
        obs2 = np.vstack(obs2).reshape([1, n_agents, -1])
        log_probs.append(np.vstack(_log_probs).reshape([1, n_agents, -1]))
        actions.append(np.vstack(_actions).reshape([1, n_agents, -1]))
        states.append(observations)
        afterstates.append(obs2)
        baselines = baseline(obs2)
        baselines2 = baseline(observations)
        values.append(baselines)
        values2.append(baselines2)
        rewards_ = []
        for _ in range(20):
            observations, new_rewards, done, _ = env.step(_actions)
            rewards_.append(new_rewards)
        new_rewards = np.mean(rewards_, axis=0)
        rewards.append(new_rewards)
        counter += 1
        if all(done):
            break
    if collect:
        values = np.vstack(values)
        values = values.reshape((-1, n_agents, 1))
        states = np.vstack(states)
        states = states.astype('float32')
        afterstates = np.vstack(afterstates)
        afterstates = afterstates.astype('float32')
        rewards = np.vstack(rewards)
        rewards = rewards.reshape((-1, n_agents, 1))
        rewards = rewards.astype('float32')
        if not modified_gae:
            gae = [tf.reshape(compute_advantages(gamma, gae_lam, rewards.shape[0], values[:, i, 0].reshape([1, -1]),
                                                 rewards[:, i, 0].reshape([1, -1])), [1, -1]) for i in range(n_agents)]
        else:
            gae = [tf.reshape(compute_advantages(gamma, gae_lam, rewards.shape[0],
                                                 util(states, afterstates, agent.policy.action_space_size, i, baseline)[:,
                                                 i, 0].reshape([1, -1]), rewards[:, i, 0].reshape([1, -1])), [1, -1])
                   for i in range(n_agents)]

        gae = np.vstack(gae)
        gae = np.transpose(gae)
        actions = np.vstack(actions)
        actions = actions.reshape([-1, n_agents, 1])
        log_probs = np.vstack(log_probs)
        log_probs = log_probs.reshape([-1, n_agents, 1])
        returns = np.transpose(np.vstack(
            [discounted_returns(gamma, rewards.shape[0], rewards[:, i, 0].reshape([1, -1])).numpy() for i in
             range(n_agents)]), axes=[1, 0, 2])
        returns = np.vstack(returns).reshape([-1, n_agents, 1])
        return states, actions, rewards, values, gae, log_probs, returns, env.eng.get_average_travel_time(), afterstates
    else:
        return env.eng.get_average_travel_time()


def util(states,afterstates,action_space_size,agent_indice,baseline):
  new_states= np.copy(afterstates)
  new_states[:,agent_indice,-action_space_size:]= states[:,agent_indice,-action_space_size:]
  return baseline(new_states).numpy()

os.environ['PYTHONHASHSEED']=str(42)
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

roadNetFile="roadnet_4_4.json"
flowFile="anon_4_4_750_0.6_synthetic.json"
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/4by4/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
config = {
    "interval": 1.0,
    "seed": 0,
    "dir": "./",
    "roadnetFile": roadNetFile,
    "flowFile": flowFile,
    "rlTrafficLight": True,
    "saveReplay": True,
    "roadnetLogFile": "roadnet_4by4.json",
    "replayLogFile": "log.txt"
}
with open('config.json', 'w') as fp:
    json.dump(config, fp)



world = World('./config.json', thread_num=1)
agents=[]
policy= Policy(32,len(world.intersections[0].phases))
n_agents=len(world.intersections)
baseline= Baseline(32,16,n_agents)
for ind,i in enumerate(world.intersections):
    action_space = gym.spaces.Discrete(len(i.phases))
    agents.append(MyAgent(
        action_space,
        LaneVehicleGenerator(world, i, ["lane_count"], in_only=False, average=None),
        LaneVehicleGenerator(world, i, ["lane_waiting_count"], in_only=False, average="all", negative=True),
        policy
    ))
metric = TravelTimeMetric(world)
env = TSCEnv(world, agents, metric)


optimizer1 = tf.keras.optimizers.Adam(learning_rate=0.0001)
optimizer2= tf.keras.optimizers.Adam(learning_rate=0.0001)

c=0
with train_summary_writer.as_default():
  while c<200:
    states,actions,rewards,values,gae,log_probs,returns,travel_time,afterstates= runEpisode(env,agents,baseline,False,collect=True,gamma=0.75,gae_lam=0.95,steps=3*180+10)
    baseline_loss= baseline.train(returns[:-10],afterstates[:-10],optimizer1,20,4,True)
    tf.summary.scalar('baseline_loss', baseline_loss, step=c)
    policy_losses= policy.train(gae[:-10].reshape([-1,]),states[:-10].reshape([-1,states.shape[2]]),actions[:-10].reshape([-1]),log_probs [:-10,:,0].reshape([-1]),optimizer2,128,2,True,c2=0.001)
    tf.summary.scalar('agent_policy_loss', policy_losses, step=c)
    tf.summary.scalar('travel_time',travel_time,step=c)
    tf.summary.scalar('rewards',np.sum(rewards),step=c)
    print("Episode {}: collected {} rewards, average travel time was {}".format(c,np.sum(rewards),travel_time))
    env.eng.set_save_replay(False)
    if c%50==0:
      print("Running greedy episode and saving next replay")
      states,actions,rewards,values,gae,log_probs,returns,travel_time,_= runEpisode(env,agents,baseline,True,collect=True,gamma=0.95,gae_lam=0.95,steps=3*180)
      print("Greedy travel time: {}".format(travel_time))
      env.eng.set_save_replay(True)
      env.eng.set_replay_file('/replay_{}.json'.format(c))
    c+=1

states,actions,rewards,values,gae,log_probs,returns,travel_time,_= runEpisode(env,agents,baseline,True,collect=True,gamma=0.95,gae_lam=0.95,steps=3*180)
print("Training finished: agent achieves {} travel time when acting greedily.".format(travel_time))
