
#Mistakes/Learning
#reward mechanism goves constant positive reward for whole episode
#Relu and Prelu activation was used and have negative effect over q function approximation
#as reward becomes negative when the episode ends
#Decreased the value of both positive and negative reward so that neural net can fit the q function without
#altering the value of q function at any other state


#Thinking
#TD in backward veiw faster policy evaluation


import gym
from random import random
from random import randint
import numpy as np
import sys
import math
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.functional as F
from torch.nn import Parameter

def constrain(value , min , max):
    if (value < min):
        return min
    elif (value > max):
        return max
    else :
        return value

class cartpole_agent():

    def __init__(self , env):
        self.epsilon = 1.0
        self.epsilon_decay = 0.94
        self.gamma = 0.99
        self.epi_max = 3000
        self.reward = 0
        self.sum_loss = None
        self.replay_buf = []
        self.replay_buf_size = 100
        self.loss_buffer = []
        self.model_qf = nn.Sequential(nn.Linear(4,20),nn.Linear(20,25),nn.Linear(25,2))
        self.model_pol = nn.Sequential(nn.Linear(4,20) ,nn.Linear(20,25) ,nn.Linear(25,2))
        self.update_pol()
        self.loss = torch.nn.MSELoss()
        self.optimizer = optim.Adagrad(self.model_qf.parameters(),lr=0.001 , weight_decay=0.88)

    def update_pol(self):
        for i in range (len(self.model_qf)):
            # sh = self.model_pol[i].weight.shape()
            # for j in range(len(sh[1])):
            #     for k in range(len(self.model_pol[i].weight[]))
            try:
                self.model_pol[i].weight = Parameter(torch.tensor(self.model_qf[i].weight))
                self.model_pol[i].bias = Parameter(torch.tensor(self.model_qf[i].bias))

            except Exception as e:
                continue

    def train(self, current_state , action , new_state , reward):
        qf = self.model_qf.forward(torch.tensor(current_state).float())
        current_q_value = qf[action]
        new_qf = self.model_qf.forward(torch.tensor(new_state).float())
        new_action = self.policy(new_state)
        next_q_value = qf[new_action]
        qf[action] = reward + self.gamma * (next_q_value)
        self.train_nn(current_state , qf , reward)
        return new_action


    def train_nn(self , input , target , reward):
        input = torch.from_numpy(input).float()\
        input = Variable(input , requires_grad = False)
        target = Variable(target , requires_grad = False)

        output = self.model_qf.forward(input)
        losses = self.loss.forward(output, target)
        if(self.sum_loss is None):
            self.sum_loss = losses
        else:
            self.sum_loss = self.sum_loss + losses

        if(reward == -1.0/10.0):
            self.loss_buffer.append(self.sum_loss)
            self.sum_loss.backward()
            self.optimizer.step()
            # print(self.sum_loss)
            self.sum_loss = None
        # print(losses)



    def policy(self , state):
        c = random()
        act = 0
        if ((c > 0) & (c < self.epsilon)):
            act = randint(0,1)
        else:
            # qf = self.model_pol(torch.tensor(state).float())
            #
            # for i in range(len(qf)):
            #     if(i == 0):
            #         j = qf[i]
            #     else:
            #         j = j + qf[i]
            # if((qf[0]/j) > c):
            #     act = 0
            # else:
            #     act = 1
            act = self.greedy(state);
            # print("greedy")
        # print(act)
        return act;

    def greedy(self , state):
        last = -1000.0
        temp = np.float(0.0)
        act = 0

        q_value = self.model_pol.forward(torch.tensor(state).float())

        for i in range(len(q_value)):
            if(q_value[i] > last):
                last = q_value[i]
                act = i

        return act

env = gym.make('CartPole-v0')
env.force_mag = 8.0
current_state = env.reset()
action = 1
final = 0
env.reset()
agent = cartpole_agent(env)
last_update = 0
for i in range(agent.epi_max):
    env.reset()
    sum_t = 0
    action = agent.policy(current_state)
    env.reset()
    for t in range(100):
        # env.render()
        print(action)
        # action = action_convertion(action)
        new_state , reward , done, info = env.step(action)
        # print(reward)
        if(done == True):
            reward = -1.0
        reward = reward / 10.0
        action = agent.train(current_state , action ,new_state , reward);
        current_state = new_state
        if done:
            print("Episode finished after {} timesteps".format(t+1) , i)            break
    if((i-last_update) == 15):
        # break
        agent.epsilon = max(agent.epsilon * agent.epsilon_decay,0.1)
            #weight of nn_pol = qf
        # print(agent.model_qf[0].weight[0,0],agent.model_pol[0].weight[0,0])
        last_update = i
        agent.update_pol()
        # print(agent.model_qf[0].weight[0,0],agent.model_pol[0].weight[0,0])
        if(final == 1):
            break
torch.save(agent.model_qf.state_dict(), "./Users/vedantneekhra/Desktop/cartpole3")
plt.plot(agent.loss_buffer , color = 'r' , marker='p' , linewidth=0)
plt.show()
env.close()

