# -*- coding: utf-8 -*-
# @FileName : DQN_keras.py
# @Project  : MAHJONG AI
# @Author   : WANG Jianxing
# @Time     : 2021/4/21 16:06
import random
from keras.callbacks import TensorBoard
from keras import backend
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
import numpy as np
from collections import deque
from tqdm import tqdm
from mahjong.ReinforcementLearning.experience import ExperienceBuffer
import math


# # Own Tensorboard class
# class ModifiedTensorBoard(TensorBoard):
#
#     # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.step = 1
#         self.writer = tf.compat.v1.summary.FileWriter(self.log_dir)
#
#     # Overriding this method to stop creating default log writer
#     def set_model(self, model):
#         pass
#
#     # Overrided, saves logs with our step number
#     # (otherwise every .fit() will start writing from 0th step)
#     def on_epoch_end(self, epoch, logs=None):
#         self.update_stats(**logs)
#
#     # Overrided
#     # We train for one batch only, no need to save anything at epoch end
#     def on_batch_end(self, batch, logs=None):
#         pass
#
#     # Overrided, so won't close writer
#     def on_train_end(self, _):
#         pass
#
#     # Custom method for saving own metrics
#     # Creates writer, writes custom metrics and closes writer
#     def update_stats(self, **stats):
#         self._write_logs(stats, self.step)


class DQN:
    def __init__(self):
        # trained model
        self.model = self.create_model()

        # target model
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_buffer = deque(maxlen=BUFFER_SIZE)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def preprocess(self, states, rewards, actions, cnt):  # (n,195,34,1),(n,),(n,1,34)
        for i in range(len(states)-1):
            round_cnt = cnt[i]
            # For a same round, set done = False
            if cnt[i+1] == round_cnt:
                self.replay_buffer.append((states[i], actions[i], rewards[i], states[i+1], False))
            # For a different round, set done = True
            else:
                self.replay_buffer.append((states[i], actions[i], rewards[i], states[i+1], True))
        # return math.ceil(len(self.replay_buffer)/BATCH_SIZE)
        return self.replay_buffer

    def create_model(self):
        state_shape = (195,34,1)
        action_shape = (1,34)

        model = Sequential()

        model.add(Conv2D(256, (3,3), input_shape=state_shape))
        model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3), input_shape=state_shape))
        model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        
        model.add(Flatten())
        model.add(Dense(64))

        model.add(Dense(action_shape, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=LR), metrics=['accuracy'])
        return model

    def get_estimated_Q(self, state):
        return self.model.predict(np.array(state))

    def train(self, terminal_state):
        mini_batch = random.sample(self.replay_buffer, BATCH_SIZE)

        all_current_state = np.array([transition[0] for transition in mini_batch])
        all_current_Q = self.model.predict(all_current_state)  # TODO: whether can predict directly

        all_next_state = np.array([transition[3] for transition in mini_batch])
        all_next_Q = self.target_model.predict(all_next_state)

        X,y = [],[]
        for i, (current_state, action, reward, new_state, done) in enumerate(mini_batch):
            if not done:
                max_next_Q = np.max(all_next_Q[i])
                new_Q = reward + GAMMA*max_next_Q
            else:
                new_Q = reward

            # Update Q value for given state
            current_qs = all_current_Q[i]  # current_qs: (1,34)
            for idx,val in enumerate(action):
                if val == 1:
                    current_qs[idx] = new_Q
                break

            # append to training data
            X.append(current_state)
            y.append(current_qs)

        # Train as one batch
        self.model.fit(np.array(X), np.array(y), shuffle=False, batch_size=BATCH_SIZE)

        # Update target network counter every ju
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter == UPDATE_FREQUENCY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
            return self.target_model


LR = 0.00025
GAMMA = 0.99
BATCH_SIZE = 32
UPDATE_FREQUENCY = 3  # 每三局更新一次
BUFFER_SIZE = 10000
EPSILON = 1.0
# 每次从buffer中读一次数据就训练一次，一共100次。训练一次从buffer中随机取32batch训练
EPISODE = 100

DQN_agent = DQN()
FILE = 'experiment_2021_04_21_12_13_25.h5'


for episode in tqdm(range(1, EPISODE+1)):
    new_buffer = DQN_agent.preprocess(ExperienceBuffer(10).read_experience(FILE))

    new_model = False
    step = 0

    # When new_model changes from False to True, meaning finished 3 局
    while not new_model:
        done = new_buffer[step][4]
        new_model = DQN_agent.train(done)
        step += 1

    # Update target model

# TODO: Change model structure
# TODO: Figure out how to train a supervised learning Q function, whether can just replace y(discard) with y'(reward)
# TODO: Use new Q function as the discard model to play in env
# TODO: Replace discard model with new target model in env every 3 rounds (How?)
# TODO: New epsilon-greedy should be added into env

# if np.random.random() > EPSILON:
#     # Get action from Q table
#     action = np.argmax(DQN_agent.get_estimated_Q(current_state))
# else:
#     # Get random action
#     action = np.random.randint(0, env.ACTION_SPACE_SIZE)
