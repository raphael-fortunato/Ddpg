import numpy as np
import tensorflow as tf
from running_mean_std import RunningMeanStd
# from mpi_running_mean_std import RunningMeanStd
import pdb
import traceback

class Normalizer:
    """
    Normalizes state and vectors through running means and running stds. Based on open ai's stable baselines
    """
    def __init__(self, env_params, gamma ,clip_obs=5, clip_rew=5, eps=1e-8):
        with tf.variable_scope('obs_rms'):
            self.obs_rms = RunningMeanStd(shape=(env_params['observation'],))
        with tf.variable_scope('ret_rms'):
            self.ret_rms = RunningMeanStd(shape=(1,))
        self.clip_obs = clip_obs
        self.clip_rew =clip_rew
        self.epsilon  = eps
        self.disc_reward =np.array([0])
        self.gamma =.99


    def normalize_state(self, obs, training=True):

        observation = obs
        if training:
            self.obs_rms.update(np.array(observation))
        observation = np.clip((observation - self.obs_rms.mean) / np.sqrt(self.obs_rms.var +self.epsilon), -self.clip_obs, self.clip_obs)
        return observation


    def normalize_reward(self, reward, training=True):

        if training:
            self.disc_reward = self.disc_reward * self.gamma +reward
            self.ret_rms.update(self.disc_reward.flatten())
            r = np.clip(reward / np.sqrt(self.ret_rms.var + self.epsilon), -self.clip_rew, self.clip_rew)
        return r


    def load(load_path, venv):
        """
        Loads a saved VecNormalize object.

        :param load_path: the path to load from.
        :param venv: the VecEnv to wrap.
        :return: (VecNormalize)
        """
        with open(load_path, "rb") as file_handler:
            norm = pickle.load(file_handler)

        return norm

    def save(self, save_path):
        with open(save_path, "wb") as file_handler:
            pickle.dump(self, file_handler)
