import build_graph
import numpy as np
import tensorflow as tf
from position_track import PositionTrack

class Agent:
    def __init__(self, model, dnds, num_actions, name='global', lr=2.5e-4, gamma=0.99):
        self.num_actions = num_actions
        self.gamma = gamma
        self.t = 0
        self.name = name
        self.dnds = dnds

        act, train, update_local, action_dist, state_value = build_graph.build_train(
            model=model,
            dnds=dnds,
            num_actions=num_actions,
            optimizer=tf.train.RMSPropOptimizer(learning_rate=7e-4, decay=.99, epsilon=0.1),
            scope=name
        )

        self._act = act
        self._train = train
        self._update_local = update_local
        self._action_dist = action_dist
        self._state_value = state_value

        self.initial_state = np.zeros((1, 258), np.float32)
        self.rnn_state0 = self.initial_state
        self.rnn_state1 = self.initial_state
        self.last_obs = None
        self.last_reward = None
        self.last_action = None
        self.last_value = None

        self.states = []
        self.rewards = []
        self.actions = []
        self.values = []
        self.encodes = []
        self.rotations = []
        self.movements = []
        self.positions = []
        self.directions = []
        self.position_changes = []
        self.pos_track = PositionTrack()

    def set_summary_writer(self, summary_writer):
        self.summary_writer = summary_writer

    def append_experience(self, action, encode, advantage):
        self.dnds[action].write(encode, advantage)

    def train(self, bootstrap_value):
        actions = np.array(self.actions, dtype=np.uint8)
        returns = []
        R = bootstrap_value
        for r in reversed(self.rewards):
            R = r + 0.99 * R
            returns.append(R)
        returns = np.array(list(reversed(returns)), dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)

        advantages = returns - values

        for i in range(len(advantages)):
            self.append_experience(actions[i], self.encodes[i], returns[i])

        summary, loss = self._train(self.states, self.initial_state, self.initial_state, self.rotations,
                self.movements, actions, returns, advantages, self.positions, self.directions, self.position_changes)
        self.summary_writer.add_summary(summary, self.t)
        self._update_local()
        return loss

    def act(self, obs):
        normalized_obs = np.zeros((1, 84, 84, 4), dtype=np.float32)
        normalized_obs[0] = np.array(obs, dtype=np.float32) / 255.0
        prob, rnn_state = self._act(normalized_obs, self.rnn_state0, self.rnn_state1)
        action = np.random.choice(range(self.num_actions), p=prob[0])
        self.rnn_state0, self.rnn_state1 = rnn_state
        return action

    def act_and_train(self, obs, reward, rotation, movement, observation):
        prob, rnn_state, encode, place_cell = self._act([obs], self.rnn_state0, self.rnn_state1, [rotation], [movement])
        action = np.random.choice(range(self.num_actions), p=prob[0])
        value = self._state_value([obs], self.rnn_state0, self.rnn_state1, [rotation], [movement])[0][0]
        self.pos_track.step(observation, rotation, movement)

        if len(self.states) == 50:
            self.train(self.last_value)
            self.states = []
            self.rewards = []
            self.actions = []
            self.values = []
            self.encodes = []
            self.rotations = []
            self.movements = []
            self.positions = []
            self.directions = []
            self.position_changes = []

        if self.last_obs is not None:
            self.states.append(self.last_obs)
            self.rewards.append(reward - self.last_reward)
            self.actions.append(self.last_action)
            self.values.append(self.last_value)
            self.encodes.append(self.last_encode)
            self.rotations.append(self.last_rotation)
            self.movements.append(self.last_movement)
            self.positions.append(self.last_position)
            self.directions.append(self.last_direction)
            self.position_changes.append(self.last_position_change)

        self.t += 1
        self.rnn_state0, self.rnn_state1 = rnn_state
        self.last_obs = obs
        self.last_reward = reward
        self.last_action = action
        self.last_value = value
        self.last_encode = encode[0]
        self.last_rotation = rotation
        self.last_movement = movement
        self.last_position = self.pos_track.get_position()
        self.last_direction = self.pos_track.get_rotation()
        self.last_position_change = self.pos_track.get_velocity()
        return action, place_cell[0], self.last_position

    def stop_episode_and_train(self, obs, reward, done=False):
        self.pos_track.reset()
        if len(self.states) > 0:
            self.states.append(self.last_obs)
            self.rewards.append(reward - self.last_reward)
            self.actions.append(self.last_action)
            self.encodes.append(self.last_encode)
            self.values.append(self.last_value)
            self.rotations.append(self.last_rotation)
            self.movements.append(self.last_movement)
            self.positions.append(self.last_position)
            self.directions.append(self.last_direction)
            self.position_changes.append(self.last_position_change)
            self.train(0)
            self.stop_episode()

    def stop_episode(self):
        self.rnn_state0 = self.initial_state
        self.rnn_state1 = self.initial_state
        self.last_obs = None
        self.last_reward = None
        self.last_action = None
        self.last_value = None
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.encodes = []
        self.rotations = []
        self.movements = []
        self.positions = []
        self.directions = []
        self.position_changes = []
