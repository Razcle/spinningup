import tensorflow as tf
import numpy as np
import gym


def q_network(x, num_actions, activation=tf.nn.relu, output_activation=None):
    """ Takes in a state of shape (batch, 105, 80, 4) and returns an estimate
    of the action value for each of the possible actions"""
    x = tf.layers.conv2d(x, filters=32, kernel_size=8,
                         strides=4, activation=activation)
    x = tf.layers.conv2d(x, filters=64, kernel_size=4,
                         strides=2, activation=activation)
    x = tf.layers.conv2d(x, filters=64, kernel_size=3,
                         strides=1, activation=activation)
    x = tf.layers.flatten(x)
    x = tf.layers.dense(x, 512, activation=activation)
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


def to_grayscale(img):
    return np.mean(img, axis=2)/255


def downsample(img):
    return img[::2, ::2]


def preprocess(img):
    return to_grayscale(downsample(img))


def get_batch(state_bf, action_bf, reward_bf, batch_size=32):
    inds = np.random.randint(low=0, high=len(state_bf) -1, size=batch_size)
    state_batch = np.stack(state_bf, axis=0)[inds, :]
    next_state_batch = np.stack(state_bf, axis=0)[inds + 1, :]
    rewards_batch = np.array(reward_bf)[inds]
    action_batch = np.array(action_bf)[inds]
    return state_batch, next_state_batch, action_batch, rewards_batch


def train(env_name='Pong-v0', epochs=10, batch_size=32, discount=0.9, lr=0.01, render=True, buffer_size=1000):

    # get the environment
    env = gym.make(env_name)
    num_actions = env.action_space.n

    # get the collected experience
    state_ph = tf.placeholder(dtype=tf.float32, shape=(None, 105, 80, 4))
    actions_taken = tf.placeholder(dtype=tf.int32, shape=(None,))
    next_state_ph = tf.placeholder(dtype=tf.float32, shape=(None, 105, 80, 4))
    rewards_ph = tf.placeholder(dtype=tf.float32, shape=(None,))

    # create all the parameters and get the action values
    with tf.variable_scope('q_network', reuse=tf.AUTO_REUSE):
        action_values = q_network(state_ph, num_actions)
        actions = tf.argmax(action_values, axis=1)
        taken_action_values = tf.reduce_sum(action_values * tf.one_hot(actions_taken, num_actions), axis=1)
    with tf.variable_scope('target_network', reuse=tf.AUTO_REUSE):
        future_action_values = tf.reduce_max(tf.stop_gradient(q_network(next_state_ph, num_actions)), axis=1)

    target = rewards_ph + discount * future_action_values
    loss = tf.reduce_mean((taken_action_values - target)**2, axis=0)

    train_op = tf.train.RMSPropOptimizer(lr).minimize(loss)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    copy_network_parameters(sess)

    epsilon = 1.0
    def train_epoch():


        # Create the data-history buffer
        total_actions = 0
        state_bf = []
        next_state_bf = []
        reward_bf = []
        action_bf = []



        # reset episode-specific variables
        obs = preprocess(env.reset())  # first obs comes from starting distribution
        state_bf.append(np.stack([obs, obs, obs, obs], axis=-1))
        done = False  # signal from environment that episode is over
        ep_rews = []  # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        def act_epsilon_greedy(epsilon, state):
            if np.random.rand() < epsilon:
                return np.random.randint(0, 5)
            else:
                return sess.run(actions, feed_dict={state_ph: state})

        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            state = []
            rew = 0
            act = act_epsilon_greedy(epsilon, state_bf[-1][np.newaxis, :])
            while len(state) < 4:
                obs, rew, done, _ = env.step(act)
                state.append(preprocess(obs))
                rew += rew
            state_bf.append(np.stack(state, axis=-1))
            reward_bf.append(rew)
            action_bf.append(act)
            total_actions += 1

            if done:
                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []

                # won't render again this epoch
                #finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if total_actions > buffer_size:
                    break

            if total_actions % 100 == 0:
                state_batch, next_state_batch, action_batch, reward_batch = get_batch(state_bf, action_bf, reward_bf, batch_size)
                _loss, _ = sess.run([loss, train_op], feed_dict={state_ph: state_batch,
                                          next_state_ph: next_state_batch,
                                          actions_taken: action_batch,
                                          rewards_ph: reward_batch})
                print('loss ', _loss)

            if total_actions % 500 == 0:
                copy_network_parameters(sess)

        return np.mean(reward_bf[-1000:])

    for i in range(epochs):
        if epsilon > 0.1:
            epsilon *= 0.9
        av_reward = train_epoch()
        print('{0}: average_reward: {1:.2f}'.format(i, av_reward))

if __name__ == '__main__':
    train()






