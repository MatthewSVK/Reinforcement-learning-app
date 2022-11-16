import os
import random
import numpy as np
from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import RMSprop
import gym


def neural_model(shape, space):
    x_input = Input(shape)
    x = Dense(350, input_shape=(shape,), activation="relu", kernel_initializer="he_uniform")(x_input)
    x = Dense(175, activation="relu", kernel_initializer="he_uniform")(x)
    x = Dense(80, activation="relu", kernel_initializer="he_uniform")(x)
    x = Dense(space, activation="linear", kernel_initializer="he_uniform")(
        x)  # vystupna vrstva s 2 uzlami (vpravo, vlavo)

    model = Model(inputs=x_input, outputs=x, name="Inverted_Pendulum_model")
    model.compile(loss="mse", optimizer=RMSprop(learning_rate=0.0003, rho=0.35, epsilon=0.01), metrics=["accuracy"])
    model.summary()
    return model


class Agent:
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.memory = deque(maxlen=2000)
        self.EPOCHS = 500
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        self.gamma = 0.95  # miera odmeny za uz pouzity genom
        self.epsilon = 1.00  # miera nahodneho vystupu narozdiel od toho co si mysli ze je dobre
        self.e_min = 0.001  # najnizsia hodnota miery
        self.e_dec = 0.999  # znizenie miery pri tom, ked je naucena
        self.gen_size = 64  # velkost jednej generacie
        self.train_s = 1000

        self.model = neural_model(self.state_size, self.action_size)

    # funkcia na zapamatanie stavu, ktory potom sa pouzije znova na trenovanie
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.train_s:
            if self.epsilon > self.e_min:
                self.epsilon *= self.e_dec

    # trenovanie s uz pouzitymi stavmi zo zoznamu
    def replay(self):
        if len(self.memory) < self.train_s:
            return
        # nahodne zvolena mini generacia zo zoznamu
        mini_gen = random.sample(self.memory, min(len(self.memory), self.gen_size))
        state = np.zeros((self.gen_size, self.state_size))
        next_state = np.zeros((self.gen_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(self.gen_size):
            state[i] = mini_gen[i][0]
            action.append(mini_gen[i][1])
            reward.append(mini_gen[i][2])
            next_state[i] = mini_gen[i][3]
            done.append(mini_gen[i][4])

        # predikcia dalsieho stavu

        target = self.model.predict(state)
        t_next = self.model.predict(next_state)

        for i in range(self.gen_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # udelenie odmeny podla maximalizacneho vzorca
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(t_next[i]))

        self.model.fit(state, target, batch_size=self.gen_size, verbose=0)

    def save_model(self, name):
        self.model.save(name)

    def model_load(self, name):
        self.model = load_model(name)

    def run(self):
        for epoch in range(self.EPOCHS):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                self.env.render()
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                if not done or i == self.env._max_episode_steps - 1:
                    reward = reward
                else:
                    reward = -200

                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1

                if done:
                    print("episode: {}/{}, score: {}, epsilon: {:.2}".format(epoch, self.EPOCHS, i, self.epsilon))

                self.replay()

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state))


if __name__ == "__main__":
    agent = Agent()
    agent.run()
