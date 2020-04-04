# -*- coding: utf8 -*0
import numpy as np
from collections import deque
import tensorflow as tf
import time
import random

def genModel(stateSpace, actionSpace):  # build neural net to be used for DDQN
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(32, input_shape=(stateSpace,), activation=tf.keras.activations.relu,
                                    kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in',
                                                                                             distribution='normal',
                                                                                             seed=None)))
    model.add(tf.keras.layers.Dense(32, activation=tf.keras.activations.relu,
                                    kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in',
                                                                                             distribution='normal',
                                                                                             seed=None)))
    model.add(tf.keras.layers.Dense(16, activation=tf.keras.activations.relu,
                                    kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in',
                                                                                             distribution='normal',
                                                                                             seed=None)))
    model.add(tf.keras.layers.Dense(actionSpace, activation=tf.keras.activations.linear))
    model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss=tf.keras.losses.Huber())
    return model


class Agent():
    # ...
    def __init__(self):
        self.memory = deque(maxlen=int(1e6))
        self.gamma = 0.995
        self.stateSpace = 23
        self.actionSpace = 5
        self.batchSize = 64
        self.model = genModel(self.stateSpace, self.actionSpace)
        self.targetModel = genModel(self.stateSpace, self.actionSpace)
        self.targetModel = tf.keras.models.clone_model(self.model)
        self.targetModel.set_weights(self.model.get_weights())

    '''
    def get_batch(self):
        mini_batch = random.sample(self.memory, self.batchSize)
        X_batch = np.empty((0, self.stateSpace), dtype=np.float64)
        Y_batch = np.empty((0, self.actionSpace), dtype=np.float64)

        for mem in mini_batch:  # get original predictions, get q value of next state, and update original
            # predictions, orig state = x, updated preds = y
            q = self.model.predict(mem[0])  # get prediction from state

            if mem[4]:
                qn = mem[3]
            else:
                qNextMain = self.model.predict(mem[2])
                qNextTarget = self.targetModel.predict(mem[2])  # 用target网络预测
                maxAction = np.argmax(qNextMain[0])
                maxQNext = qNextTarget[0][maxAction]
                qn = mem[3] + self.gamma * maxQNext
            q[0][mem[1]] = qn  # replace predicted q values with calculated value for action taken

            X_batch = np.append(X_batch, mem[0])  # append state to X values
            Y_batch = np.append(Y_batch, q)  # append updated predictions to Y values

        return X_batch, Y_batch
    '''
    def get_batch_fast(self):
        trans_batch = random.sample(self.memory, self.batchSize)
        trans_batch = np.asarray(trans_batch)
        # (state, action, next_state, reward, done)
        state_batch = trans_batch[:, 0:self.stateSpace]
        action_batch = np.array(trans_batch[:, self.stateSpace], dtype=int)
        nState_batch = trans_batch[:, self.stateSpace + 1:2 * self.stateSpace + 1]
        reward_batch = trans_batch[:, 2 * self.stateSpace + 1]
        done_batch = np.array(trans_batch[:, 2*self.stateSpace+2], dtype=bool)
        q_batch = self.model.predict(state_batch)
        q_next_main_batch = self.model.predict(nState_batch)
        q_next_target_batch = self.targetModel.predict(nState_batch)
        act_max_batch = np.argmax(q_next_main_batch, axis=1)
        # a_onehot = int(np.eye(self.batchSize, self.actionSpace)[act_max_batch])
        q_max_next_batch = q_next_target_batch[np.arange(self.batchSize), act_max_batch]

        for i in range(self.batchSize):
            if done_batch[i]:
                qn = reward_batch[i]
            else:
                qn = reward_batch[i] + q_max_next_batch[i] * self.gamma
            q_batch[i][action_batch[i]] = qn
        return action_batch, q_batch


if __name__ == '__main__':
# main
    A = Agent()
    A.memory = deque(maxlen=int(1e6))
    for i in range(int(1e5)):
        state = np.random.rand(1, 23)
        state_ = np.random.rand(1, 23)
        done = bool(np.random.randint(2))
        done_ = bool(np.random.randint(2))
        reward = np.random.randint(20)
        action = np.random.randint(5)
        reward = np.random.sample()

        transition = np.append(state,action)
        transition = np.append(transition, state_)
        transition = np.append(transition, reward)
        transition = np.append(transition,done)
        A.memory.append(transition)
    #(state, action, next_state, reward, done)
    '''
    trans_batch = random.sample(A.memory, A.batchSize)
    trans_batch = np.asarray(trans_batch)
    state_batch = trans_batch[:, 0:A.stateSpace]
    action_batch = trans_batch[A.stateSpace]
    nState_batch = trans_batch[:, A.stateSpace+1:2*A.stateSpace+1]
    reward_batch = trans_batch[2*A.stateSpace+1]
    '''
    t1 = time.time()

    for i in range(10000):
        X, Y = A.get_batch_fast()
        #print(X.shape)
        #print(Y.shape)
    t2 = time.time()
    print("Time of 100 Batches = {0}".format(t2-t1))
# self.memory.append([lastState, action, state, reward, done])
