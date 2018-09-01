# Reinforcement Learning Framework

## Markov Decision Process (MDP)

- a (finite) set of state $S$
- a (finite) set of actions $A$
- a set of rewards $R$
- one-step dynamics of the environment
- a discount rate $\gamma$

## Policy

- $pi: S -> A$
- A deterministic policy maps a state to an action

- $pi: S \cdot A -> [0, 1]$

- A stochastic policy maps a state to the probability that the agent takes an action

## State-Value Function

- For each state the state-value function yields the expected return if the agent started in that state and then followed the policy for all time steps

- A state-value function always corresponds to a particular policy

- The state-value function is denoted by $v _\pi$

## Optimality

- A policy $\pi'$ is better than or equal to a policy $\pi$ only if its state value function is better than or equal to the state value function for $\pi'$ for all states

- An optimal policy $\pi_*$ satisfies $\pi _* >= \pi$ for all policies $\pi$

- An optimal policy is guaranteed to exist but may not be unique
  
- All optimal policies have the same state-value function called the optimal state-value function denoted by $v_*$

## Action-Value Function

- For each state $s$ and action $a$ the action-value function yields the expected return if the agent started in state $s$ then chooses action $a$ and then uses the policy to choose its actions for all time steps

- A action-value function always corresponds to a particular policy

- The action-value function is denoted by $q _\pi$
  
- All optimal policies have the same action-value function called the optimal state-value function denoted by $q_*$

## Iterative Policy Evaluation

- Updates the Bellman Expectation Equation to work as an iterative function which is then used to estimate the state-value function $v_\pi$

- Takes an MDP environment and a policy $\pi$ outputs a state-value function $v_\pi$

## Policy Improvement

1. Start with random policy $\pi$ from action state
2. Use iterative policy evaluation to get the value function $v _\pi$
3. Construct the action-value function $q _\pi$ from the value function $v _\pi$
4. Use the action-value function $q _\pi$ to get a policy $\pi'$ that is at least as good as policy $\pi$
5. Repeat

## Policy Iteration

TBA

## Episodes

- AN episode is a finite sequence $S_0,A_0,R_1,S_1,A_1,R_2,...,S_T$
- For any episode the agents goal is to find policy $\pi$ to maximise expected cumulative reward

## The Prediction Problem

- Given a policy $\pi$, determine the value function $v_\pi$ by interacting with the environment
- Basis for MonteCarlo Prediction algorithm

