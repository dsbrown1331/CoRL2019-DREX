import numpy as np
import cv2
import csv
import os
import torch
from os import path, listdir
import gaze_heatmap as gh
import time
# TODO: add masking part for extra games
from baselines.common.trex_utils import preprocess, mask_score

cv2.ocl.setUseOpenCL(False)


def normalize_state(obs):
    return obs / 255.0


def normalize(obs, max_val):
    # TODO: discard frames with no gaze
    if(max_val != 0):
        norm_map = obs/float(max_val)
    else:
        norm_map = obs
    return norm_map


# need to grayscale and warp to 84x84
def GrayScaleWarpImage(image):
    """Warp frames to 84x84 as done in the Nature paper and later work."""
    width = 84
    height = 84
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    #frame = np.expand_dims(frame, -1)
    return frame


def MaxSkipAndWarpFrames(trajectory_dir, img_dirs, frames):
    """take a trajectory file of frames and max over every 3rd and 4th observation"""
    num_frames = len(frames)
    # print('total images:', num_frames)
    skip = 4

    sample_pic = np.random.choice(
        listdir(path.join(trajectory_dir, img_dirs[0])))
    image_path = path.join(trajectory_dir, img_dirs[0], sample_pic)
    pic = cv2.imread(image_path)
    obs_buffer = np.zeros((2,)+pic.shape, dtype=np.uint8)
    max_frames = []
    for i in range(num_frames):
        # TODO: check that i should max before warping.
        img_name = frames[i] + ".png"
        img_dir = img_dirs[i]

        if i % skip == skip - 2:
            obs = cv2.imread(path.join(trajectory_dir, img_dir, img_name))

            obs_buffer[0] = obs
        if i % skip == skip - 1:
            obs = cv2.imread(path.join(trajectory_dir, img_dir, img_name))
            obs_buffer[1] = obs

            # warp max to 80x80 grayscale
            image = obs_buffer.max(axis=0)
            warped = GrayScaleWarpImage(image)
            max_frames.append(warped)
    return max_frames

def GrabSingleFrames(trajectory_dir, img_dirs, frames):
    """take a trajectory file of frames and max over every 3rd and 4th observation"""
    num_frames = len(frames)
    
    sample_pic = np.random.choice(
        listdir(path.join(trajectory_dir, img_dirs[0])))
    image_path = path.join(trajectory_dir, img_dirs[0], sample_pic)
    frames = []
    for i in range(num_frames):
        # TODO: check that i should max before warping.
        img_name = frames[i] + ".png"
        img_dir = img_dirs[i]

        obs = cv2.imread(path.join(trajectory_dir, img_dir, img_name))

        warped = GrayScaleWarpImage(obs)
        frames.append(warped)
    return frames


def StackFrames(frames):
    import copy
    """stack every four frames to make an observation (84,84,4)"""
    stacked = []
    stacked_obs = np.zeros((84, 84, 4))
    for i in range(len(frames)):
        if i >= 3:
            stacked_obs[:, :, 0] = frames[i-3]
            stacked_obs[:, :, 1] = frames[i-2]
            stacked_obs[:, :, 2] = frames[i-1]
            stacked_obs[:, :, 3] = frames[i]
            stacked.append(np.expand_dims(copy.deepcopy(stacked_obs), 0))
    return stacked


def MaxSkipGaze(gaze, heatmap_size):
    """take a list of gaze coordinates and max over every 3rd and 4th observation"""
    num_frames = len(gaze)
    skip = 4

    obs_buffer = np.zeros((2,)+(heatmap_size, heatmap_size), dtype=np.float32)
    max_frames = []
    for i in range(num_frames):
        g = gaze[i]
        g = np.squeeze(g)
        if i % skip == skip - 2:
            obs_buffer[0] = g
        if i % skip == skip - 1:
            obs_buffer[1] = g
            image = obs_buffer.max(axis=0)
            max_frames.append(image)
    if np.isnan(max_frames).any():
        print('nan max gaze map created')
        exit(1)

    return max_frames


def CollapseGaze(gaze_frames, heatmap_size):
    import copy
    """combine every four frames to make an observation (84,84)"""
    stacked = []
    stacked_obs = np.zeros((heatmap_size, heatmap_size))
    for i in range(len(gaze_frames)):
        if i >= 3:
            # Sum over the gaze frequency counts across four frames
            stacked_obs = gaze_frames[i-3]
            stacked_obs = stacked_obs + gaze_frames[i-2]
            stacked_obs = stacked_obs + gaze_frames[i-1]
            stacked_obs = stacked_obs + gaze_frames[i]

            # Normalize the gaze mask
            max_gaze_freq = np.amax(stacked_obs)
            stacked_obs = normalize(stacked_obs, max_gaze_freq)

            stacked.append(np.expand_dims(
                copy.deepcopy(stacked_obs), 0))  # shape: (1,7,7)

    return stacked


def MaxSkipReward(rewards):
    """take a list of rewards and max over every 3rd and 4th observation"""
    num_frames = len(rewards)
    skip = 4
    max_frames = []
    obs_buffer = np.zeros((2,))
    for i in range(num_frames):
        r = rewards[i]
        if i % skip == skip - 2:

            obs_buffer[0] = r
        if i % skip == skip - 1:

            obs_buffer[1] = r
            rew = obs_buffer.max(axis=0)
            max_frames.append(rew)
    return max_frames


def StackReward(rewards):
    import copy
    """combine every four frames to make an observation"""
    stacked = []
    stacked_obs = np.zeros((1,))
    for i in range(len(rewards)):
        if i >= 3:
            # Sum over the rewards across four frames
            stacked_obs = rewards[i-3]
            stacked_obs = stacked_obs + rewards[i-2]
            stacked_obs = stacked_obs + rewards[i-1]
            stacked_obs = stacked_obs + rewards[i]

            stacked.append(np.expand_dims(copy.deepcopy(stacked_obs), 0))
    return stacked


def get_sorted_traj_indices(env_name, dataset):
    # need to pick out a subset of demonstrations based on desired performance
    # first let's sort the demos by performance, we can use the trajectory number to index into the demos so just
    # need to sort indices based on 'score'
    game = env_name
    # Note, I'm also going to try only keeping the full demonstrations that end in terminal
    traj_indices = []
    traj_scores = []
    traj_dirs = []
    traj_rewards = []
    traj_gaze = []
    traj_frames = []
    traj_actions = []
    print('traj length: ', len(dataset.trajectories[game]))
    for t in dataset.trajectories[game]:
        traj_indices.append(t)
        traj_scores.append(dataset.trajectories[game][t][-1]['score'])
        # a separate img_dir defined for every frame of the trajectory as two different trials could comprise an episode
        traj_dirs.append([dataset.trajectories[game][t][i]['img_dir']
                          for i in range(len(dataset.trajectories[game][t]))])
        traj_rewards.append([dataset.trajectories[game][t][i]['reward']
                             for i in range(len(dataset.trajectories[game][t]))])
        traj_gaze.append([dataset.trajectories[game][t][i]['gaze_positions']
                          for i in range(len(dataset.trajectories[game][t]))])
        traj_frames.append([dataset.trajectories[game][t][i]['frame']
                            for i in range(len(dataset.trajectories[game][t]))])
        traj_actions.append([dataset.trajectories[game][t][i]['action']
                            for i in range(len(dataset.trajectories[game][t]))])

    sorted_traj_indices = [x for _, x in sorted(
        zip(traj_scores, traj_indices), key=lambda pair: pair[0])]
    sorted_traj_scores = sorted(traj_scores)
    sorted_traj_dirs = [x for _, x in sorted(
        zip(traj_scores, traj_dirs), key=lambda pair: pair[0])]
    sorted_traj_rewards = [x for _, x in sorted(
        zip(traj_scores, traj_rewards), key=lambda pair: pair[0])]
    sorted_traj_gaze = [x for _, x in sorted(
        zip(traj_scores, traj_gaze), key=lambda pair: pair[0])]
    sorted_traj_frames = [x for _, x in sorted(
        zip(traj_scores, traj_frames), key=lambda pair: pair[0])]
    sorted_traj_actions = [x for _, x in sorted(
        zip(traj_scores, traj_actions), key=lambda pair: pair[0])]

    print("Max human score", max(sorted_traj_scores))
    print("Min human score", min(sorted_traj_scores))

    # so how do we want to get demos? how many do we have if we remove duplicates?
    seen_scores = set()
    non_duplicates = []
    for i, s, d, r, g, f, a in zip(sorted_traj_indices, sorted_traj_scores, 
                                    sorted_traj_dirs, sorted_traj_rewards, sorted_traj_gaze, sorted_traj_frames, sorted_traj_actions):
        if s not in seen_scores:
            seen_scores.add(s)
            non_duplicates.append((i, s, d, r, g, f, a))
    print("num non duplicate scores", len(seen_scores))
    if env_name == "spaceinvaders":
        start = 0
        skip = 3
    elif env_name == "revenge":
        start = 0
        skip = 1
    elif env_name == "qbert":
        start = 0
        skip = 3
    elif env_name == "mspacman":
        start = 0
        skip = 1
    else:   # TODO: confirm best logic for all games
        start = 0
        skip = 3
    num_demos = 12
    # demos = non_duplicates[start:num_demos*skip + start:skip]
    demos = non_duplicates  # don't skip any demos
    return demos


def get_preprocessed_trajectories_onestate(env_name, dataset, data_dir, use_gaze, gaze_conv_layer):
    """returns an array of trajectories corresponding to what you would get running checkpoints from PPO
       demonstrations are grayscaled, maxpooled, stacks of 4 with normalized values between 0 and 1 and
       top section of screen is masked

       This version will return two datasets, one with gaze for TREX and one without gaze for BC on single states and actions
    """

    demos = get_sorted_traj_indices(env_name, dataset)
    human_scores = []
    human_demos = []
    human_rewards = []
    human_gaze = []
    bc_data = []

    print('len demos: ', len(demos))
    for indx, score, img_dir, rew, gaze, frame, action in demos:
        print(score)
        print('before frames', len(frame), 'before actions', len(action))
        
        human_scores.append(score)

        # traj_dir = path.join(data_dir, 'screens', env_name, str(indx))
        traj_dir = path.join(data_dir, env_name)
        maxed_traj = MaxSkipAndWarpFrames(traj_dir, img_dir, frame)
        stacked_traj = StackFrames(maxed_traj)
        print('after frames', len(stacked_traj))
   

        demo_norm_mask = []
        # normalize values to be between 0 and 1 and have top part masked
        for ob in stacked_traj:
            demo_norm_mask.append(mask_score(ob, env_name)[0])  # masking
        
        demo_onestate_mask = GrabSingleFrames(traj_dir, img_dir, frame)
        for i,ob in demo_onestate_mask:
            demo_onestate_mask[i] = mask_score(ob, env_name)[0]

        #postprocess actions to keep the action that is after the framestacks
        #print("pre", action[:416])
        #all_actions = action[15::4]  #assume that we need to create a buffer of four stacked frames and that we make decision after that
        #for now let's just repeat the last action if we need one more (ie if there isn't an action for the last observation)
        #while len(stacked_actions) < len(demo_norm_mask):
        #    print("buffering action list with repeated last action!!")
        #    stacked_actions.append(action[-1])
        #print("post", stacked_actions[:100])

        print("len human demos", len(demo_norm_mask))
        assert(len(demo_onestate_mask) == len(action))
        sa_list = list(zip(demo_onestate_mask, action))
        
      



        
        human_demos.append(demo_norm_mask)
        bc_data.append(sa_list)

        

        # skip and stack reward
        maxed_reward = MaxSkipReward(rew)
        stacked_reward = StackReward(maxed_reward)
        human_rewards.append(stacked_reward)

        if use_gaze:
            # generate gaze heatmaps as per Ruohan's algorithm
            h = gh.DatasetWithHeatmap()
            if gaze_conv_layer == 1:
                conv_size = 26
            elif gaze_conv_layer == 2:
                conv_size = 11
            elif gaze_conv_layer == 3:
                conv_size = 9
            elif gaze_conv_layer == 4:
                conv_size = 7
            else:
                print('Invalid Gaze conv layer. Must be between 1-4.')
                exit(1)
            g = h.createGazeHeatmap(gaze, conv_size)

            maxed_gaze = MaxSkipGaze(g, conv_size)
            stacked_gaze = CollapseGaze(maxed_gaze, conv_size)
            human_gaze.append(stacked_gaze)
            print('stacked gaze: ', stacked_gaze[0].shape)

    # if(use_gaze):
    #     print(len(human_demos[0]), len(human_rewards[0]), len(human_gaze[0]))
    #     print(len(human_demos), len(human_rewards), len(human_gaze))

    return human_demos, human_scores, human_rewards, human_gaze, bc_data


def get_preprocessed_trajectories(env_name, dataset, data_dir, use_gaze, gaze_conv_layer):
    """returns an array of trajectories corresponding to what you would get running checkpoints from PPO
       demonstrations are grayscaled, maxpooled, stacks of 4 with normalized values between 0 and 1 and
       top section of screen is masked
    """

    demos = get_sorted_traj_indices(env_name, dataset)
    human_scores = []
    human_demos = []
    human_rewards = []
    human_gaze = []

    print('len demos: ', len(demos))
    for indx, score, img_dir, rew, gaze, frame, action in demos:
        print(score)
        print('before frames', len(frame), 'before actions', len(action))
        
        human_scores.append(score)

        # traj_dir = path.join(data_dir, 'screens', env_name, str(indx))
        traj_dir = path.join(data_dir, env_name)
        maxed_traj = MaxSkipAndWarpFrames(traj_dir, img_dir, frame)
        stacked_traj = StackFrames(maxed_traj)
        print('after frames', len(stacked_traj))
   

        demo_norm_mask = []
        # normalize values to be between 0 and 1 and have top part masked
        for ob in stacked_traj:
            demo_norm_mask.append(mask_score(ob, env_name)[0])  # masking
        
        #postprocess actions to keep the action that is after the framestacks
        #print("pre", action[:416])
        stacked_actions = action[15::4]  #assume that we need to create a buffer of four stacked frames and that we make decision after that
        #for now let's just repeat the last action if we need one more (ie if there isn't an action for the last observation)
        while len(stacked_actions) < len(demo_norm_mask):
            print("buffering action list with repeated last action!!")
            stacked_actions.append(action[-1])
        #print("post", stacked_actions[:100])

        print("len human demos", len(demo_norm_mask))
        print("len stacked actions", len(stacked_actions))
        assert(len(demo_norm_mask) == len(stacked_actions))
        sa_list = list(zip(demo_norm_mask, stacked_actions))
        
        #TODO: might need to be fancier....
        print('after actions', len(stacked_actions))
        



        
        human_demos.append(sa_list)

        

        # skip and stack reward
        maxed_reward = MaxSkipReward(rew)
        stacked_reward = StackReward(maxed_reward)
        human_rewards.append(stacked_reward)

        if use_gaze:
            # generate gaze heatmaps as per Ruohan's algorithm
            h = gh.DatasetWithHeatmap()
            if gaze_conv_layer == 1:
                conv_size = 26
            elif gaze_conv_layer == 2:
                conv_size = 11
            elif gaze_conv_layer == 3:
                conv_size = 9
            elif gaze_conv_layer == 4:
                conv_size = 7
            else:
                print('Invalid Gaze conv layer. Must be between 1-4.')
                exit(1)
            g = h.createGazeHeatmap(gaze, conv_size)

            maxed_gaze = MaxSkipGaze(g, conv_size)
            stacked_gaze = CollapseGaze(maxed_gaze, conv_size)
            human_gaze.append(stacked_gaze)
            print('stacked gaze: ', stacked_gaze[0].shape)

    # if(use_gaze):
    #     print(len(human_demos[0]), len(human_rewards[0]), len(human_gaze[0]))
    #     print(len(human_demos), len(human_rewards), len(human_gaze))

    return human_demos, human_scores, human_rewards, human_gaze


ACTION_MEANING = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",
}


def translate_action(a, action_meanings):
	#take action from akanksha's stuff and map it to the 0:num_actions that i'm using for ale
    if ACTION_MEANING[a] in action_meanings:
	    return action_meanings.index(ACTION_MEANING[a])
    else: #just no-op since action is meaningless
        return 0



def debug_action_translation():
	#debugging code
	import gym

	env = gym.make('SpaceInvadersNoFrameskip-v4')
	action_meanings = env.unwrapped.get_action_meanings()
	print(action_meanings)
	for a in [0, 1, 2, 3, 4, 11, 12]:
		print(a, translate_action(a, action_meanings))

if __name__=="__main__":
    debug_action_translation()