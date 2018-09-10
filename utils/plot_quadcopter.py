from math import cos, sin
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Patch to 3d axis to remove margins around x, y and z limits.
# Taken from here: https://stackoverflow.com/questions/16488182/removing-axes-margins-in-3d-plot
###patch start###
from mpl_toolkits.mplot3d.axis3d import Axis
if not hasattr(Axis, "_get_coord_info_old"):
    def _get_coord_info_new(self, renderer):
        mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
        mins += deltas / 4
        maxs -= deltas / 4
        return mins, maxs, centers, deltas, tc, highs
    Axis._get_coord_info_old = Axis._get_coord_info
    Axis._get_coord_info = _get_coord_info_new
###patch end###

class Quadrotor():
    """
    Class for plotting a quadrotor
    Original author: Daniel Ingram (daniel-s-ingram)
    https://github.com/AtsushiSakai/PythonRobotics
    """
    def __init__(self, x=0, y=0, z=0, roll=0, pitch=0, yaw=0, reward=0, title=None, filepath=None, size=2.0):
        self.p1 = np.array([size / 2, 0, 0, 1]).T
        self.p2 = np.array([-size / 2, 0, 0, 1]).T
        self.p3 = np.array([0, size / 2, 0, 1]).T
        self.p4 = np.array([0, -size / 2, 0, 1]).T

        self.x_data = []
        self.y_data = []
        self.z_data = []
        self.reward_data = []

        # start
        self.sx = x
        self.sy = y
        self.sz = z

        # target
        self.tx = 0
        self.ty = 0
        self.tz = 0

        #fig = plt.figure(figsize=(16,12), dpi=72)
        fig = plt.figure(dpi=150)
        self.ax = plt.subplot2grid((24, 24), (0, 0), colspan=24, rowspan=18, projection='3d')
        self.ax5 = plt.subplot2grid((24, 24), (20, 0), colspan=24, rowspan=4)
        self.update_pose(x, y, z, roll, pitch, yaw, reward, title, filepath)

    def set_target(self, x, y, z):
        self.tx = x
        self.ty = y
        self.tz = z

    def update_pose(self, x, y, z, roll, pitch, yaw, reward, title, filepath):
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.reward_data.append(reward)
        self.x_data.append(x)
        self.y_data.append(y)
        self.z_data.append(z)

        self.plot(title, filepath)

    def transformation_matrix(self):
        x = self.x
        y = self.y
        z = self.z
        roll = self.roll
        pitch = self.pitch
        yaw = self.yaw
        return np.array(
            [[cos(yaw) * cos(pitch), -sin(yaw) * cos(roll) + cos(yaw) * sin(pitch) * sin(roll), sin(yaw) * sin(roll) + cos(yaw) * sin(pitch) * cos(roll), x],
             [sin(yaw) * cos(pitch), cos(yaw) * cos(roll) + sin(yaw) * sin(pitch) *
              sin(roll), -cos(yaw) * sin(roll) + sin(yaw) * sin(pitch) * cos(roll), y],
             [-sin(pitch), cos(pitch) * sin(roll), cos(pitch) * cos(yaw), z]
             ])

    def clear(self):
        self.x_data = []
        self.y_data = []
        self.z_data = []
        self.reeard_data = []

    def plot(self, title, filepath):
        T = self.transformation_matrix()

        p1_t = np.matmul(T, self.p1)
        p2_t = np.matmul(T, self.p2)
        p3_t = np.matmul(T, self.p3)
        p4_t = np.matmul(T, self.p4)

        #plt.cla()
        self.ax.cla()
        
        plt.suptitle(title, fontsize=14)

        # plot start
        self.ax.scatter(self.sx, self.sy, self.sz, zdir='z', c='g')

        # plot target
        self.ax.scatter(self.tx, self.ty, self.tz, zdir='z', c='b')

        # plot rotors
        self.ax.plot([p1_t[0], p2_t[0], p3_t[0], p4_t[0]],
                     [p1_t[1], p2_t[1], p3_t[1], p4_t[1]],
                     [p1_t[2], p2_t[2], p3_t[2], p4_t[2]], 'k.', zdir='z')

        # plot frame
        self.ax.plot([p1_t[0], p2_t[0]], [p1_t[1], p2_t[1]],
                     [p1_t[2], p2_t[2]], 'r-', zdir='z')
        
        self.ax.plot([p3_t[0], p4_t[0]], [p3_t[1], p4_t[1]],
                     [p3_t[2], p4_t[2]], 'r-', zdir='z')

        # plot track
        self.ax.plot(self.x_data, self.y_data, self.z_data, 'b:', zdir='z')

        x_bounds = max(5, self.tx + 5)
        y_bounds = max(5, self.ty + 5)
        z_bounds = max(5, self.tz + 5)
        self.ax.set_xlim(-x_bounds, x_bounds)
        self.ax.set_ylim(-y_bounds, y_bounds)
        self.ax.set_zlim(0, z_bounds)

        # Plot reward
        self.ax5.plot(self.reward_data, label='Reward', c=[0,0,0,0.7], linewidth=1.0)
        #self.ax5.set_xlim(0, max(30, len(self.reward_data)))
        #self.ax5.set_ylim(-1, 1)
        self.ax5.set_xlabel('t [s]')
        self.ax5.set_ylabel('Reward')

        if filepath:
            plt.savefig(filepath)
        else:
            plt.pause(0.001)