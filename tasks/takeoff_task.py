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
        
        # Calculate distance from target
        dist_from_target = np.sqrt(np.square(self.sim.pose[:3] - self.target_pos).sum())

        # Calculate distance from vertical axis
        dist_from_axis = np.sqrt(np.square(self.sim.pose[:2] - self.target_pos[:2]).sum())

        # Calculate angular deviation
        angular_deviation = np.sqrt(np.square(self.sim.pose[3:]).sum())
        
        # Calculate velocity in xy plane
        non_z_v = np.square(self.sim.v[:2]).sum()

        # Calculate overall velocity
        vel = np.square(self.sim.v[:3]).sum()

        reward = 0

        if dist_from_target > 1.5:
            # Penalties to encourage the quadcopter to fly toward target
            reward = 1.0 - 0.01 * dist_from_target

            # Penalty based on angular deviation to encourage the quadcopter to remain upright
            reward -= 0.01 * angular_deviation

            # Penalty based on movement in the xy plane to encourage the quadcopter to fly vertically
            if dist_from_axis > 4:
                reward -= 0.1

            # Penalty for high velocity to encourage quadcopter to fly slower
            if vel > 10.0:
                reward -= 0.1

        else:
            # Rewards to encourage loitering
            reward += 0.2

            if vel > 1.0:
                reward -= 0.1

            # Reward for time to encourage quadcopter to remain in the air
            reward += 0.001 * self.sim.time

        # clamp reward to [-1, 1]
        #reward = min(1.0, max(-1.0, reward))
        return reward


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