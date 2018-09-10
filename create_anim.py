import os
import numpy as np
import pickle
import imageio
import argparse
from utils.plot_quadcopter import Quadrotor

def get_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('num_episodes', type=int, default=5,
                        help='Number of episodes to render')

    return parser.parse_args()

def main():
    
    in_args = get_input_args()

    # Load results file
    with open('data/results0.bin', 'rb') as pickleFile:
        results = pickle.load(pickleFile)

    exportPath = './videos/frames/'
    if not os.path.exists(exportPath):
        os.makedirs(exportPath)

    # Sort wrt episode reward
    episodeRewards = [np.sum(r['reward']) for r in results]
    resultIndices = np.argsort(episodeRewards)

    # Get top results
    resultIndices = list(reversed(resultIndices))[0:in_args.num_episodes]

    # Render top episodes
    images = []

    for iE, e in enumerate(resultIndices):
        res = results[e]

        # Generate episode title
        reward = np.sum(res['reward'])
        title = "Rank:{:02} - Episode:{:04} - Reward:{:.3f}".format(iE, e, reward)
        
        # Render all frames in this episode
        for i in range(0, len(res['x'])):
            filepath = "{}frame{:04}_{:04}.png".format(exportPath, iE, i)
            print("Processing: {}".format(filepath))
        
            if i == 0:
                q = Quadrotor(x=res['x'][0], 
                y=res['y'][0], 
                z=res['z'][0], 
                roll=res['phi'][0],
                pitch=res['theta'][0], 
                yaw=res['psi'][0],
                reward=res['reward'][0], 
                title=title,
                filepath=filepath)

                q.set_target(0.0, 0.0, 50.0)
            else:
                q.update_pose(x=res['x'][i], 
                y=res['y'][i], 
                z=res['z'][i], 
                roll=res['phi'][i],
                pitch=res['theta'][i], 
                yaw=res['psi'][i],
                reward=res['reward'][i],
                title=title,
                filepath=filepath)

            images.append(imageio.imread(filepath))
            
    # Save all frames to animated gif
    imageio.mimsave("{}movie.gif".format(exportPath), images)

# Call to main function to run the program
if __name__ == "__main__":
    main()