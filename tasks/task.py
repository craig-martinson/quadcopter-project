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

        self.state_size = self.action_repeat * 9
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 0.]) 
 
    def get_reward(self):
        """Uses current pose of sim to return reward."""
        
        # Calculate distance from target
        dist_from_target = np.sqrt(np.square(self.sim.pose[:3] - self.target_pos).sum())
        #dist_from_target_squared = np.square(self.sim.pose[:3] - self.target_pos).sum()

        # Calculate distance from vertical axis
        #dist_from_axis = np.sqrt(np.square(self.sim.pose[:2] - self.target_pos[:2]).sum())

        # Calculate angular deviation
        angular_deviation = np.sqrt(np.square(self.sim.pose[3:]).sum())
        
        # Calculate velocity in xy plane
        #non_z_v = np.square(self.sim.v[:2]).sum()

        # Calculate overall velocity
        #vel = np.square(self.sim.v[:3]).sum()

        penalty = 0.004 * dist_from_target * dist_from_target
        #penalty = 0.015 * dist_from_target_squared

        # Penalty based on angular deviation to encourage the quadcopter to remain upright
        penalty += 0.008 * angular_deviation

        # Penalty based on movement in the xy plane to encourage the quadcopter to fly vertically
        #if dist_from_axis > 4:
        #   penalty += 0.1

        # Penalty for high velocity to encourage quadcopter to fly slower
        #if vel > 10.0:
        #   penalty += 0.1

        #if self.sim.pose[2] > self.target_pos[2] + 5:
        #    penalty += 0.01

        bonus = 1.0
        #if dist_from_target < 0.5:
        #    bonus += 0.01
            
        # Calculate reward
        reward = bonus - penalty
        # Reward for time to encourage quadcopter to remain in the air
        #reward += 0.001 * self.sim.time

        # clamp reward to [-1, 1]
        #return min(1.0, max(-1.0, reward))
        return np.tanh(reward)


    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
            pose_all.append(self.sim.v)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        pose_all = np.append(self.sim.pose, self.sim.v)
        state = np.concatenate([pose_all] * self.action_repeat) 
        return state