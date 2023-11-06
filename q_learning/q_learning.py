import random


class QLearning:
    _FLAP, _NO_FLAP = 0, 1
    _MIN_V, _MAX_V = -8, 10
    _MAX_X, _MAX_Y = 288, 512
    _ACTIONS = [_FLAP, _NO_FLAP]

    def __init__(self, epsilon=0.1, alpha=0.1, gamma=1):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self.Q_table = {
            ((y_pos, pipe_top_y, x_dist, velocity), action): 0
            for y_pos in range(15)
            for pipe_top_y in range(15)
            for x_dist in range(15)
            for velocity in range(-8, 11)
            for action in range(2)
        }

    def observe(self, s1, a, r, s2, end):
        """this function is called during training on each step of the game where
        the state transition is going from state s1 with action a to state s2 and
        yields the reward r. If s2 is a terminal state, end==True, otherwise end==False.

        Unless a terminal state was reached, two subsequent calls to observe will be for
        subsequent steps in the same episode. That is, s1 in the second call will be s2
        from the first call.
        """
        s1 = self.state_encoder(s1)
        s2 = self.state_encoder(s2)

        future_reward = (
            max(self.Q_table[(s2, self._NO_FLAP)], self.Q_table[(s2, self._FLAP)])
            if not end
            else 0
        )

        self.Q_table[(s1, a)] = self.Q_table[(s1, a)] + self.alpha * (
            r + self.gamma * future_reward - self.Q_table[(s1, a)]
        )

    def training_policy(self, state):
        """Returns the index of the action that should be done in state while training the agent.
        Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

        training_policy is called once per frame in the game while training
        """
        state = self.state_encoder(state)
        threshold = random.uniform(0, 1)
        if threshold < self.epsilon:
            return random.choice(self._ACTIONS)
        elif self.Q_table[(state, self._FLAP)] == self.Q_table[(state, self._NO_FLAP)]:
            return random.choice(self._ACTIONS)
        elif self.Q_table[(state, self._FLAP)] > self.Q_table[(state, self._NO_FLAP)]:
            return self._FLAP
        else:
            return self._NO_FLAP

    def policy(self, state):
        """Returns the index of the action that should be done in state when training is completed.
        Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

        policy is called once per frame in the game (30 times per second in real-time)
        and needs to be sufficiently fast to not slow down the game.
        """
        state = self.state_encoder(state)
        if self.Q_table[(state, self._FLAP)] == self.Q_table[(state, self._NO_FLAP)]:
            return random.choice(self._ACTIONS)
        elif self.Q_table[(state, self._FLAP)] > self.Q_table[(state, self._NO_FLAP)]:
            return self._FLAP
        else:
            return self._NO_FLAP

    def state_encoder(self, state):
        y_pos = max(0, min(state["player_y"], 512))
        pipe_top_y = max(25, min(state["next_pipe_top_y"], 192))
        x_dist = max(0, min(state["next_pipe_dist_to_player"], 288))
        velocity = max(-8, min(state["player_vel"], 10))

        enc_y_pos = self._get_interval(512, 15, y_pos)
        enc_pipe_top_y = self._get_interval(512, 15, pipe_top_y)
        enc_x_dist = self._get_interval(288, 15, x_dist)

        return (enc_y_pos, enc_pipe_top_y, enc_x_dist, velocity)

    def _get_interval(self, total_size, interval, value):
        interval_width = total_size / interval
        partition = value // interval_width

        if partition < 0:
            return 0
        if partition > interval - 1:
            return interval - 1

        return partition
