import os
import random
import numpy as np
from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import RMSprop
from keras.initializers import Orthogonal
import tensorflow as tf
import datetime
import gym


def neuralNetwork(shape, space):
    x_input = Input(shape)
    x = Dense(512, input_shape=(shape,), activation="relu", kernel_initializer="he_uniform")(x_input)
    x = Dropout(0.2)(x)
    x = Dense(256, activation="relu", kernel_initializer="he_uniform")(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation="relu", kernel_initializer="he_uniform")(x)

    x = Dense(space, activation="linear", kernel_initializer="he_uniform")(
        x)  # vystupna vrstva s 2 uzlami (vpravo, vlavo)
    model = Model(inputs=x_input, outputs=x, name="Inverted_Pendulum_model")
    model.compile(loss="mse", optimizer=RMSprop(learning_rate=0.0003, rho=0.35, epsilon=0.01), metrics=["accuracy"])
    model.summary()
    return model


class Agent:
    def __init__(self):
        self.env = gym.make('CartPole-v1', render_mode="rgb_array")
        self.memory = deque(maxlen=2500)
        self.EPOCHS = 1000
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        self.gamma = 0.95  # miera odmeny za uz pouzity genom
        self.epsilon = 1.00  # miera nahodneho vystupu narozdiel od toho co si mysli ze je dobre
        self.e_min = 0.001  # najnizsia hodnota miery
        self.e_dec = 0.5  # znizenie miery pri tom, ked je naucena
        self.gen_size = 150  # velkost jednej generacie skusit zvysit
        self.train_s = 1000
        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = 'logs/dqn/' + self.current_time
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)

        self.model = neuralNetwork((self.state_size,), self.action_size)

    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) > self.train_s:
            if self.epsilon > self.e_min:
                self.epsilon *= self.e_dec
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state))

    def replay(self):
        if len(self.memory) < self.train_s:
            return
        minibatch = random.sample(self.memory, min(len(self.memory), self.gen_size))

        state = np.zeros((self.gen_size, self.state_size))
        next_state = np.zeros((self.gen_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(self.gen_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        target = self.model.predict(state)
        target_next = self.model.predict(next_state)

        for i in range(self.gen_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))
        self.model.fit(state, target, batch_size=self.gen_size, verbose=0)

    def load(self, name):
        self.model = load_model(name)

    def save(self, name):
        self.model.save(name)

    def run(self):
        max_i = 0
        counter = 0

        for epoch in range(self.EPOCHS):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            flag = True
            i = 0
            while not done:
                # self.env.render()
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                if not done or i == self.env._max_episode_steps - 1:
                    reward = reward
                else:
                    reward = -100
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1

                if done:
                    self.steps.append(i)
                    if i > max_i: max_i = i;
                    print("epoch: {}/{}, iterations in epoch: {}, max_i: {}, exploration rate: {:.2}".format(epoch, self.EPOCHS, i, max_i, self.epsilon))

                    with self.summary_writer.as_default():
                        tf.summary.scalar('Steps in one Epoch', i, step=epoch)  # graf
                        # tf.summary.scalar('losses', self.losses[epoch], step=epoch)

                    if i >= 450 and flag == True:
                        counter += 1
                        flag = False
                    if counter == 3:
                        print("Saving model")
                        self.save_model("trained_model.h5")
                        return
                i += 1
                self.replay()


if __name__ == "__main__":
    agent = Agent()
    agent.run()
