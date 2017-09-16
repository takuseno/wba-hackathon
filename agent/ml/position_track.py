from __future__ import print_function

import csv
import glob
import sys
import time

import numpy as np

class PositionTrack():
    def __init__(self):
        self.accumulated_position = np.array([0, 0, 0])
        self.accumulated_last_position = np.array([0, 0, 0])
        self.forward = np.array([0, 0, 1])
        self.accumulated_rotation = 0
        self.depth_th = 10.0
        self.debug = False

    def get_position(self):
        return self.accumulated_position
        
    def get_rotation(self):
        return self.accumulated_rotation

    def get_velocity(self):
        return self.accumulated_position - self.accumulated_last_position

    def reset(self):
        self.accumulated_position = np.array([0, 0, 0])
        self.accumulated_rotation = 0
    
    def step(self, observation, rotation, movement):
        self.accumulated_rotation += rotation
        self.accumulated_rotation %= 360
        if self.debug:
            print(np.mean(observation['depth']), rotation, movement, self.accumulated_rotation, self.accumulated_position)
        self.accumulated_last_position = self.accumulated_position
        rad = np.radians(self.accumulated_rotation)
        if np.mean(observation['depth']) >= self.depth_th:
            self.accumulated_position = self.accumulated_position + np.array([np.sin(rad), 0, np.cos(rad)]) * movement
