import tensorflow as tf
import numpy as np
import gym
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from collections import deque


class Experience:
    def __init__(self, observation, action, reward, next_observation):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.next_observation = next_observation

    @property
    def is_terminal(self):
        return self.next_observation is None


class ReplayBuffer:
    def __init__(self, size=50000):
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

    def __init__(self, env_name, state_size=4):
        self.env = gym.make(env_name)
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
    x = tf.layers.dense(x, 30, activation=activation)
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
          target_update_freq=1000, max_actions=10000000, state_size=1):

    # get the environment
    env = MyEnvironment(env_name, state_size=state_size)
    num_actions = env.env.action_space.n
    obs_dim = env.env.observation_space.shape
    fig, (ax, ax1, ax2, ax3) = plt.subplots(1, 4)


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
    _loss = 0.0
    returns = []
    greedy_returns = []
    losses = []
    epsilons = [1.0]
    buffer = ReplayBuffer(size=buffer_size)

    # initialise episode-specific variables
    epsilon = 1.0
    observation = preprocess(env.reset())  # first obs comes from starting distribution
    done = False  # signal from environment that episode is over

    # render first episode of each epoch
    finished_rendering_this_epoch = False

    def evaluate():
        rets = []
        state = env.reset()
        ret = 0.0
        done = False
        for i in range(10):
            while not done:
                action = act_epsilon_greedy(0.0, state)
                next_state, rew, done = env.step(action)
                state = next_state
                ret += rew
            state = env.reset()
            rets.append(ret)
        return np.mean(rets)

    def act_epsilon_greedy(epsilon, observation):
        if np.random.rand() < epsilon:
            return np.random.randint(0, num_actions)
        else:
            state = np.stack(observation, axis=-1)[np.newaxis, :]
            return sess.run(actions, feed_dict={state_ph: state})[0]

    def update_epsilon(t, epsilon, init_epsilon=1.0, final_epsilon=0.1, decay_steps=10000):
        if epsilon > final_epsilon:
            epsilon = ((final_epsilon - init_epsilon)/decay_steps) * t + init_epsilon
        return epsilon


    def get_targets(experiences):
        state_batch = []
        target_batch = []
        actions_batch = []
        for exp in experiences:
            state_batch.append(np.stack(exp.observation, axis=-1))
            actions_batch.append(action)
            if exp.is_terminal:
                target_batch.append(exp.reward)
            else:
                next_av = sess.run(target_action_values,
                                   feed_dict= {state_ph: np.stack(exp.next_observation, axis=-1)[np.newaxis, :]})
                target_batch.append(exp.reward + discount * next_av[0])
        return np.array(state_batch), np.array(actions_batch), np.array(target_batch)


    ret = 0
    while total_actions < max_actions:

        # rendering
        if (not finished_rendering_this_epoch) and render:
            env.env.render()

        action = act_epsilon_greedy(epsilon, observation)
        epsilon = update_epsilon(total_actions, epsilon)

        next_observation, reward, done = (None, 0.0, True) if done else env.step(action)

        experience = Experience(observation, action, reward, next_observation)
        buffer.add(experience)
        observation = next_observation
        total_actions += 1
        ret += reward

        if done:
            # reset episode-specific variables
            obs, done = env.reset(), False
            returns.append(ret)
            epsilons.append(epsilon)
            losses.append(_loss)
            ret = 0.0
            av_greedy_return = evaluate()
            print('greedy', av_greedy_return)
            greedy_returns.append(av_greedy_return)
            ax.cla()
            ax1.cla()
            ax2.cla()
            ax3.cla()
            ax.set_ylim(0, 200)
            ax.plot(returns)
            ax1.set_ylim(0, 200)
            ax1.plot(greedy_returns)
            ax2.plot(epsilons)
            ax3.plot([np.mean(losses[i-10:i]) for i in range(10, len(losses))])
            plt.pause(0.1)


            # won't render again this epoch
            finished_rendering_this_epoch = True

        if total_actions > batch_size and (total_actions % 20) == 0:
            experiences = buffer.sample(batch_size)
            state_batch, actions_batch, target_batch = get_targets(experiences)
            _loss, _ , _av = sess.run([loss, train_op, action_values], feed_dict={state_ph: state_batch,
                                      targets_ph: target_batch,
                                      actions_taken: actions_batch})


        if total_actions % target_update_freq == 0:
            copy_network_parameters(sess)
            finished_rendering_this_epoch = False


if __name__ == '__main__':
    train()






