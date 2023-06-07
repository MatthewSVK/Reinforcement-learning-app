import os
import random
import gym
import numpy as np
from collections import deque
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import Adam, RMSprop
import datetime
from matplotlib import animation
import matplotlib.pyplot as plt


def NeuralNetwork(shape, space):
    x_input = Input(shape)
    x = Dense(1024, input_shape=(shape,), activation="relu", kernel_initializer="he_uniform")(x_input)
    x = Dropout(0.2)(x)
    x = Dense(512, activation="relu", kernel_initializer="he_uniform")(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation="relu", kernel_initializer="he_uniform")(x)

    x = Dense(space, activation="linear", kernel_initializer="he_uniform")(
        x)  # vystupna vrstva s 2 uzlami (vpravo, vlavo)
    model = Model(inputs=x_input, outputs=x, name="Inverted_Pendulum_model")
    model.compile(loss="mse", optimizer=RMSprop(learning_rate=0.0003, rho=0.35, epsilon=0.01), metrics=["accuracy"])
    model.summary()
    return model


def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

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
        self.total_steps = []
        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = '/content/drive/MyDrive/Cartpole/logs/dqn/' + self.current_time
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)

        self.model = NeuralNetwork(self.state_size, self.action_size)

        # self.model = NeuralNetwork((self.state_size,), self.action_size)

        # load_model("temporary_cartpole20230508-103709.h5")

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
        self.model.save("/content/drive/MyDrive/Cartpole/" + name)

    def run(self):
        count = 0
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
                    print(
                        "epoch: {}/{}, score: {}, exploration rate: {:.2}".format(epoch, self.EPOCHS, i, self.epsilon))
                    with self.summary_writer.as_default():
                        tf.summary.scalar('Steps in one Epoch', i, step=epoch)  # graf

                    self.total_steps.append(i)
                    mean = np.mean(self.total_steps)
                    if i > mean:
                        print("Saving partialy trained model")
                        self.save("temporary_cartpole" + self.current_time + ".h5")

                    if i < 400:
                        count = 0
                    if i >= 400:
                        count += 1
                        flag = False
                        if count == 3:
                            print("Saving trained model as cartpole-dqn.h5")
                            self.save("cartpole-dqn.h5")
                            return
                self.replay()

    def test(self):
        self.load("trained_model.h5")
        frames = []
        for e in range(50):
            state = self.env.reset()
            done = False
            i = 0
            while not done:
                frames.append(self.env.render())
                action = self.env.action_space.sample()
                result = self.env.step(action)
                state, _, done, _ = result[:4]  # Unpack the first four values
                i += 1
                if done:
                    print("epoch: {}/{}, score: {}".format(e, self.EPOCHS, i))
                    break
        self.env.close()
        save_frames_as_gif(frames)


if __name__ == "__main__":
    agent = Agent()
    agent.test()
