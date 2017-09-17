# coding: utf-8

import os

import brica1.gym
import numpy as np
<<<<<<< HEAD
=======
import six.moves.cPickle as pickle

from ml.network import make_network
from ml.agent import Agent
from lightsaber.tensorflow.util import initialize, get_session
>>>>>>> bc706aacb9ef2daef6bd1329bf6a16aa50434a26

from config.model import CNN_FEATURE_EXTRACTOR, CAFFE_MODEL, MODEL_TYPE

from config.log import APP_KEY
import logging
app_logger = logging.getLogger(APP_KEY)

use_gpu = int(os.getenv('GPU', '-1'))


class VVCComponent(brica1.Component):
    image_feature_count = 1
    cnn_feature_extractor = CNN_FEATURE_EXTRACTOR
    model = CAFFE_MODEL
    model_type = MODEL_TYPE
    image_feature_dim = 256 * 6 * 6

    def __init__(self, n_output=10240, n_input=1):
        # image_feature_count = 1
        super(VVCComponent, self).__init__()

        self.use_gpu = use_gpu
        self.n_output = n_output
        self.n_input = n_input

    def set_model(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def fire(self):
        observation = self.get_in_port('Isocortex#V1-Isocortex#VVC-Input').buffer
        # call feature extractor
        obs_array = self.feature_extractor.feature(observation, self.image_feature_count)

        self.results['Isocortex#VVC-BG-Output'] = obs_array
        self.results['Isocortex#VVC-UB-Output'] = obs_array


class BGComponent(brica1.Component):
    def __init__(self, n_input=10240, n_output=1, agent=None):
        super(BGComponent, self).__init__()
        self.use_gpu = use_gpu
        actions = [0, 1, 2]
        self.input_dim = n_input

    def start(self):
        return 0

    def end(self, reward):  # Episode Terminated
        features = self.get_in_port('Isocortex#VVC-BG-Input').buffer
        self.agent.stop_episode_and_train(features, reward)

    def fire(self):
        reward = self.get_in_port('RB-BG-Input').buffer[0]
        rotation = self.get_in_port('RB-BG-Input').buffer[1]
        movement = self.get_in_port('RB-BG-Input').buffer[2]
        observation = self.get_in_port('RB-BG-Input').buffer[3]
        features = self.get_in_port('Isocortex#VVC-BG-Input').buffer

        action = self.agent.act_and_train(features, reward, rotation, movement, observation)
        app_logger.info('action {}, reward {}'.format(action, reward))

        self.results['BG-Isocortex#FL-Output'] = np.array([action])


class UBComponent(brica1.Component):
    def __init__(self):
        super(UBComponent, self).__init__()
        self.use_gpu = use_gpu
        data_size = 10**5
        replay_size = 32
        hist_size = 1
        initial_exploration = 10**3
        dim = 10240
        vvc_input = np.zeros((hist_size, dim), dtype=np.uint8)
        self.last_state = vvc_input
        self.state = vvc_input
        self.time = 0

    def end(self, action, reward):
        self.time += 1

    def fire(self):
        self.state = self.get_in_port('Isocortex#VVC-UB-Input').buffer
        action, reward = self.get_in_port('Isocortex#FL-UB-Input').buffer

        self.last_state = self.state.copy()
        self.time += 1


class FLComponent(brica1.Component):
    def __init__(self):
        super(FLComponent, self).__init__()
        self.last_action = np.array([0])

    def fire(self):
        action = self.get_in_port('BG-Isocortex#FL-Input').buffer
        reward = self.get_in_port('RB-Isocortex#FL-Input').buffer
        self.results['Isocortex#FL-MO-Output'] = action
        self.results['Isocortex#FL-UB-Output'] = [self.last_action, reward]

        self.last_action = action
