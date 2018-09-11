import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 0.]) 
        self.initial_dist_to_target = self.distance(self.target_pos, init_pose[:3])

    def distance_squared(self, ptA, ptB):
        """ Returns the square of the distance between ptA and ptB. """
        return (
            ((ptA[0] - ptB[0]) ** 2) +
            ((ptA[1] - ptB[1]) ** 2) +
            ((ptA[2] - ptB[2]) ** 2)
        )

    def distance(self, ptA, ptB):
        """ Returns the distance between ptA and ptB. """
        dist = np.linalg.norm(ptA - ptB)
        return dist

    def distance2d(self, ptA, ptB):
        """ Returns 2d the distance between ptA and ptB. """
        ptA[2] = 0
        ptB[2] = 0
        dist = np.linalg.norm(ptA - ptB)
        return dist

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        
        # distance reward
        #d = self.distance(self.sim.pose[:3], self.target_pos)
        #reward = 1 - d / self.initial_dist_to_target
   
        reward = np.tanh(1 - 0.003*(abs(self.sim.pose[:3] - self.target_pos))).sum()

        if self.sim.pose[2] > self.target_pos[2]:
            reward = -1.0 * abs(reward)

        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state