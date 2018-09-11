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

        self.on_ground = True
        self.turnpoint = False


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
        """ Returns the distance between ptA and ptB. """
        ptA[2] = 0
        ptB[2] = 0
        dist = np.linalg.norm(ptA - ptB)
        return dist

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        
        done = False
        d = self.distance(self.sim.pose[:3], self.target_pos)
        h_dist = self.distance2d(self.sim.pose[:3], self.target_pos)
        v_dist = abs(self.target_pos[2] - self.sim.pose[2])

        # distance penalty
        #reward = -0.01 * d**2

        #reward = 1.0 - d / self.initial_dist_to_target
        #reward = 0.8 * (self.initial_dist_to_target - d)
        
        #reward = self.target_pos[2] - v_dist
        #reward += -0.5 * h_dist
        #reward += -0.5 * v_dist
        reward = np.tanh(1 - (0.003 * d))
        #reward = -0.1 * d2d + 0.12 * self.sim.pose[2]
        #reward = -1.0 * d
        # time penalty
        #reward += -0.01 * self.sim.time
    
        #reward = 0
        bonus = 0.1
        
        if self.on_ground:
            if self.sim.pose[2] > 0.1:
                # bonus for taking off
                reward += bonus
                self.on_ground = False
                print("Takeoff!")
            else:
                #reward based on height above ground [0->1]
                #reward += max(1.0, self.sim.pose[2])
                #reward = np.tanh(1 - (0.002 * d))
                #reward += 10 * self.sim.pose[2]
                print("Trying to takeoff - alt={}m".format(self.sim.pose[2]))
 
        elif not self.turnpoint:
            # bonus for reaching turnpoint
            if d < 2:
                reward += bonus
                self.turnpoint = True
                print("Reached turnpoint")
            else:
                #reward based on distance from turnpoint [0->1]
                reward += 1 - max(1.0, d / self.initial_dist_to_target)
                print("Flying to turnpoint...")

        elif self.sim.pose[2] < 0.1:
            # bonus for landing
            reward += bonus
            done = True
            print("Landed")
        else:
             #reward based on height above ground [0->1]
            reward += 1.0 - max(1.0, 0.01 * self.sim.pose[2])
            print("Trying to land...")

        #if d2d > 5:
        #    reward -= 1
            #done = True

        #reward = -abs(self.target_pos[2] - self.sim.pose[2])
        #reward = self.target_pos[2] - (self.target_pos[2] - self.sim.pose[2])
        # bonus for being close to target
        #if d < 1:
        #    reward += 1
        #    goal_achieved = True

        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        
        #reward = np.tanh(1 - 0.003*d)

        return done, reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        total_reward = 0
        pose_all = []

        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            
            reward_done, reward = self.get_reward()
            total_reward += reward 

           # if reward_done:
           #     done = True
  
            pose_all.append(self.sim.pose)
            
        next_state = np.concatenate(pose_all)
        return next_state, total_reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        self.on_ground = True
        self.turnpoint = False
        return state