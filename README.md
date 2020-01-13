# Deep Reinforcement Learning

![Image of Cartpole](Image/cartPole.gif)

## Introduction
I have developed a value-based reinforcement agent for cartpole module of openAI gym. I have Neural Network a non-linear function approximator for approximating q-function. 

## Value Iterator 
I have implemented forward Temporal Difference zero in forward veiw for value iterations. 
![Image of Value Iterator](Image/TD(0).png)
I have used temporal difference zero because unlike Monte Carlo method, it updates the value funtion during the run.
 
## Function Approximator
I have used Neural Network as a function approximator. Neural Network also is used in two forms one in which we provide state and action value both to get the value of q function and another being in which we provide state gets q value for all action possible. I have used the second form as your action space is small and static.
Major draw of function is that they cannot be complete become equal to required q function. They always have some nosie. More detailed problems with funciton approimator is discussed inn this [paper](https://www.ri.cmu.edu/pub_files/pub1/thrun_sebastian_1993_1/thrun_sebastian_1993_1.pdf)

## Control And Policy
For control, I have used SARSA. In SARSA firstly when are in a state(S_n)  and then selecting action according to the policy. Then Reward is awarded and the agent arrives at a new state then the agent again selects the action according to the policy.  
The policy which I have used is Epsilon greedy. 

## Learning 
- Relu and PRelu activation was used and have negative effect over q function approximation as reward becomes negative when the episode ends.
- Decreased the value of both positive and negative reward so that neural net can fit the q function without altering the value of q function at any other state.
- Unstable nature of neural net - using neural net directly in place of q table in highly unstable. Techique used for stablization of neural net :
    - Used two neural net, one for q-function for value iterator and other q-function for policy.
    - Update q-function for policy after some episodes of value iterator so that value becomes optimal for that policy.
- Other than using replay buffer as it is memory efficient way, I feed forward the network after every state transition and backpropogated it at the end of the episode. This also stabilized the neural net.

## TODO
- Modify value iterator to TD(lamda) in Backward Veiw (feel it will converge value function earlier because agent receives negative reward at the end of the episode and using backward veiw will propogate it backward earlier).

