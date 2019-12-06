# Deep Reinforcement Learning

![Image of Cartpole](Image/cartPole.gif)

## Introduction
I have developed a value-based reinforcement agent for cartpole module of openAI gym. I have Neural Network a non-linear function approximator for approximating q-function. 

## Value Iterator 
I have implemented forward Temporal Difference zero for value iterations. 
![Image of Value Iterator](Image/TD(0).png)
I have used temporal difference zero because it works good for such small state-action space.
 
## Function Approximator
I have used Neural Network as a function approximator. Neural Network also is used in two forms one in which we provide state and action value both to get the value of q function and another being in which we provide state gets q value for all action possible. I have used the second form as your action space is small and static.
Major draw of function is that they cannot be complete become equal to required q function. They always have some nosie. More detailed problems with funciton approimator is discussed inn this [paper](https://www.ri.cmu.edu/pub_files/pub1/thrun_sebastian_1993_1/thrun_sebastian_1993_1.pdf)

## Control And Policy
For control, I have used SARSA. In SARSA firstly when are in a state(S_n)  and then selecting action according to the policy. Then Reward is awarded and the agent arrives at a new state then the agent again selects the action according to the policy.  
The policy which I have used is Epsilon greedy. 

