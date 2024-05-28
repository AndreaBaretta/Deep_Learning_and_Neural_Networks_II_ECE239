import cv2
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from utils import preprocess #this is a helper function that may be useful to grayscale and crop the image


class EnvWrapper(gym.Wrapper):
    def __init__(
        self,
        env:gym.Env,
        skip_frames:int=4,
        stack_frames:int=4,
        initial_no_op:int=50,
        do_nothing_action:int=0,
        **kwargs
    ):
        """the environment wrapper for CarRacing-v2

        Args:
            env (gym.Env): the original environment
            skip_frames (int, optional): the number of frames to skip, in other words we will
            repeat the same action for `skip_frames` steps. Defaults to 4.
            stack_frames (int, optional): the number of frames to stack, we stack 
            `stack_frames` frames to form the state and allow agent understand the motion of the car. Defaults to 4.
            initial_no_op (int, optional): the initial number of no-op steps to do nothing at the beginning of the episode. Defaults to 50.
            do_nothing_action (int, optional): the action index for doing nothing. Defaults to 0, which should be correct unless you have modified the 
            discretization of the action space.
        """
        super(EnvWrapper, self).__init__(env, **kwargs)
        self.initial_no_op = initial_no_op
        self.skip_frames = skip_frames
        self.stack_frames = stack_frames
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(stack_frames, 84, 84),
            dtype=np.float32
        )
        self.do_nothing_action = do_nothing_action
        
    def reset(self, **kwargs):
        
        state, _ = self.env.reset(**kwargs)

        for _ in range(self.initial_no_op):
            state, _, _, _,info = self.env.step(self.do_nothing_action)
            pass

        self.return_state=  np.stack([preprocess(state)]*self.stack_frames, axis=0)
        return self.return_state, info
    
    def step(self, action):
        total_reward = 0

        for i in range(self.skip_frames):
            state, reward, terminated,truncated,info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
            pass

        self.return_state[:-1] = self.return_state[1:]
        self.return_state[-1] = preprocess(state) 
        return self.return_state, total_reward, terminated,truncated,info

    
   