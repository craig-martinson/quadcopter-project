import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from agents.policy_search import PolicySearch_Agent
from agents.basic_agent import Basic_Agent
from task import Task

def plot_quadcopter(x, y, yaw, length=0.1):
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=0.25, head_width=0.25)

show_animation = True
num_episodes = 1000
target_pos = np.array([10., 10., 10.])
task = Task(target_pos=target_pos)
agent = PolicySearch_Agent(task) 
#agent = Basic_Agent(task) 

# simulation bounds
lower_bounds = task.sim.lower_bounds / 10
upper_bounds = task.sim.upper_bounds / 10

for i_episode in range(1, num_episodes+1):

    if show_animation:
        plt.xlim((lower_bounds[0], upper_bounds[0]))
        plt.ylim((lower_bounds[1], upper_bounds[1]))
        plt.autoscale(False)
        plt.grid(True)
        target_pos
        plt.plot(target_pos[0], target_pos[1], "xr")

    # start a new episode
    state = agent.reset_episode() 

    while True:
        action = agent.act(state) 
        next_state, reward, done = task.step(action)
        agent.step(reward, done)
        state = next_state

        if show_animation:
            pos = list(task.sim.pose)          
            plot_quadcopter(pos[0], pos[1], pos[4])    
            plt.title("Altitude = {:7.3f}".format(pos[2]))    
            plt.pause(0.0001)
        
        if done:
            print("\rEpisode = {:4d}, score = {:7.3f} (best = {:7.3f})".format(
                i_episode, agent.score, agent.best_score), end="") 
            plt.cla()
            break
    sys.stdout.flush()

print("\n")