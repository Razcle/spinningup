import tensorflow as tf
import numpy as np
import gym
import random
import matplotlib.pyplot as plt
from collections import deque

plt.ion()


class Experience:
    def __init__(self, observation, action, reward, terminal):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.terminal = terminal

        @property
        def is_terminal(self):
            return self.next_observation is None


class ReplayBuffer:
    def __init__(self, size=100):
        self.size = size
        self._experiences = []
        self._next_to_overwrite = 0

    def _is_full(self):
        return len(self._experiences) == self.size

    def add(self, experience: Experience):
        if self._is_full():
            self._experiences[self._next_to_overwrite] = experience
            self._next_to_overwrite = (self._next_to_overwrite + 1) % self.size
        else:
            self._experiences.append(experience)

    def sample(self, batch_size):
        return random.choices(self._experiences, k=batch_size)

    def __len__(self):
        return len(self._experiences)


class MyEnvironment:

    def __init__(self, env, state_size=4):
        self.env = env
        self.state_size = state_size
        self.obs_buffer = deque(maxlen=state_size)


    def step(self, action):
        obs, rew, done, _ = self.env.step(action)
        self.obs_buffer.append(obs)
        state = list(self.obs_buffer)

        return state, rew, done

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.state_size):
            self.obs_buffer.append(obs)
        return list(self.obs_buffer)

# class ExperienceBuffer:
#
#     def __init__(self,  obs_dim, max_size=10000, state_size=1):
#         self.max_size = max_size
#         self.observations = np.empty((max_size,) + obs_dim)
#         self.rewards = np.empty((max_size,))
#         self.actions = np.empty((max_size,))
#         self.terminals = np.empty((max_size,), dtype=bool)
#         self.state_size = state_size
#         self.current = 0
#         self.count = 0
#
#
#     def add_to_buffer(self, obs, action, reward, is_terminal):
#             self.observations[self.current] = obs
#             self.rewards[self.current] = reward
#             self.actions[self.current] = action
#             self.terminals[self.current] = is_terminal
#             self.current = (self.current + 1) % self.max_size
#             self.count +=1
#
#     def _get_index(self):
#         suitable = False
#         while not suitable:
#             init_ind = np.random.randint(0, min(self.count, self.max_size) - self.state_size)
#             if not np.any(self.terminals[init_ind: init_ind + self.state_size]):
#                 suitable = True
#         return init_ind
#
#     def get_last_state(self):
#         if self.current > self.state_size:
#             return np.stack(self.observations[self.current - self.state_size: self.current], axis=-1)[np.newaxis, :]
#         else:
#             return np.stack(self.observations[self.max_size - self.state_size: self.max_size], axis=-1)[np.newaxis, :]
#
#     def get_batch(self,  batch_size=32):
#
#         state_batch = []
#         next_state_batch = []
#         rewards_batch = []
#         terminal_batch = []
#         action_batch = []
#
#         while len(state_batch) < batch_size:
#             init_ind = self._get_index()
#             state_batch.append(np.stack(self.observations[init_ind: init_ind + self.state_size], axis=-1))
#             next_state_batch.append(np.stack(self.observations[init_ind + 1: init_ind + self.state_size + 1], axis=-1))
#             terminal_batch.append(self.terminals[init_ind + self.state_size])
#             action_batch.append(self.actions[init_ind + self.state_size -1])
#             rewards_batch.append(self.rewards[init_ind + self.state_size - 1])
#
#         return np.array(state_batch), np.array(next_state_batch), np.array(action_batch), np.array(rewards_batch), np.array(terminal_batch)


def q_network(x, num_actions, activation=tf.nn.relu, output_activation=None):
    """ Takes in a state of shape (batch, 105, 80, 4) and returns an estimate
    of the action value for each of the possible actions"""
    # x = tf.layers.conv2d(x, filters=32, kernel_size=8,
    #                      strides=4, activation=activation)
    # x = tf.layers.conv2d(x, filters=64, kernel_size=4,
    #                      strides=2, activation=activation)
    # x = tf.layers.conv2d(x, filters=64, kernel_size=3,
    #                      strides=1, activation=activation)
    x = tf.layers.flatten(x)
    x = tf.layers.dense(x, 32, activation=activation)
    return tf.layers.dense(x, num_actions, activation=output_activation)


def copy_network_parameters(sess):
    """
    Copies the model parameters of one estimator to another.
    Args:
      sess: Tensorflow session instance
    """
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith('q_network')]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith('target')]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)


def preprocess(img):

    # def to_grayscale(img):
    #     return np.mean(img, axis=2, dtype=np.uint8)
    #
    # def downsample(img):
    #     return img[::2, ::2]
    #
    return img

def train(env_name='CartPole-v0',  batch_size=32, discount=0.99, lr=1e-3, render=True, buffer_size=50000,
          target_update_freq=1000, max_actions=10000000, state_size=4):

    # get the environment
    env = MyEnvironment(gym.make(env_name), state_size=state_size)
    num_actions = env.env.action_space.n
    obs_dim = env.env.observation_space.shape
    fig, ax = plt.subplots()

    # get the collected experience
    state_ph = tf.placeholder(dtype=tf.float32, shape=(None,) + obs_dim + (state_size,))
    actions_taken = tf.placeholder(dtype=tf.int32, shape=(None,))
    targets_ph = tf.placeholder(dtype=tf.float32, shape=(None,))

    # create all the parameters and get the action values
    with tf.variable_scope('q_network', reuse=tf.AUTO_REUSE):
        action_values = q_network(state_ph, num_actions)
        actions = tf.argmax(action_values, axis=1)
        taken_action_values = tf.reduce_sum(action_values * tf.one_hot(actions_taken, num_actions), axis=1)
    with tf.variable_scope('target_network', reuse=tf.AUTO_REUSE):
        target_action_values = tf.reduce_max((q_network(state_ph, num_actions)), axis=1)

    loss = tf.reduce_mean((targets_ph - taken_action_values)**2, axis=0)

    train_op = tf.train.AdamOptimizer(lr).minimize(loss)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    copy_network_parameters(sess)

    # Create the data-history buffer
    total_actions = 0
    returns = []
    buffer = ReplayBuffer(size=buffer_size)

    # initialise episode-specific variables
    observation = preprocess(env.reset())  # first obs comes from starting distribution
    done = False  # signal from environment that episode is over

    # render first episode of each epoch
    finished_rendering_this_epoch = False

    def act_epsilon_greedy(t, exp_buffer, init_epsilon=1.0, final_epsilon=0.001, decay_steps=10000):
        epsilon = ((final_epsilon - init_epsilon)/decay_steps) * t + init_epsilon
        if np.random.rand() < epsilon:
            return np.random.randint(0, num_actions)
        else:
            state = exp_buffer.get_last_state()
            return sess.run(actions, feed_dict={state_ph: state})[0]

    ret = 0
    while total_actions < max_actions:

        # rendering
        if (not finished_rendering_this_epoch) and render:
            env.render()

        action = act_epsilon_greedy(total_actions, buffer)
        next_observation, reward, done, info = (None, 0., True, None) if done else env.step(action)

        experience = Experience(observation, action, reward, next_observation)
        buffer.add(experience)
        observation = next_observation
        total_actions += 1
        ret += discount * reward

        if done:
            # reset episode-specific variables
            obs, done = env.reset(), False
            returns.append(ret)
            ret = 0.0

            # won't render again this epoch
            finished_rendering_this_epoch = True

        if total_actions % 50 == 0:
            experiences = buffer.sample(batch_size)
            targets_batch = reward_batch + (1 - terminal_batch) * discount * sess.run(target_action_values, feed_dict={state_ph: next_state_batch})
            _loss, _ , _av = sess.run([loss, train_op, action_values], feed_dict={state_ph: state_batch,
                                      targets_ph: targets_batch,
                                      actions_taken: action_batch})

        if total_actions % target_update_freq == 0:
            copy_network_parameters(sess)
            finished_rendering_this_epoch = False

        if total_actions % 1000 == 0.0:
            ax.cla()
            ax.plot(returns)
            plt.draw()
            plt.pause(0.1)


if __name__ == '__main__':
    train()






