from time import time
import numpy as np
import matplotlib.pyplot as plt


class Position:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash(str(self))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "(%s, %s)" % (self.x, self.y)

    def take_action(self, action, allowed_actions):

        if action not in allowed_actions or action == 'WAIT':
            return self

        elif action == 'UP':
            return Position(self.x, self.y - 1)

        elif action == 'DOWN':
            return Position(self.x, self.y + 1)

        elif action == 'LEFT':
            return Position(self.x - 1, self.y)

        elif action == 'RIGHT':
            return Position(self.x + 1, self.y)


class State:

    def __init__(self, robber=Position(1, 1), police=Position(4, 4)):
        self.robber = robber
        self.police = police

    def __eq__(self, other):
        return self.robber == other.robber and self.police == other.police

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        return "Robber: %s, Police: %s" % (self.robber, self.police)


class Environment:

    """ World is a 4x4 grid """

    def __init__(self):
        self.state = State()
        self.bank = Position(2, 2)
        self.valid_actions = ['WAIT', 'UP', 'DOWN', 'LEFT', 'RIGHT']

    def __repr__(self):
        return self.state.__repr__()

    def step(self, action):
        new_state = self._get_new_state(action)
        reward = self._reward()
        return new_state, reward

    def render(self):
        self._plot_grid()
        self._plot_state()
        plt.show()

    @staticmethod
    def _plot_grid():
        plt.axis([0, 4, 0, 4])
        plt.plot(1.5, 2.5, 'g*', markersize=25)
        plt.grid()
        ax = plt.gca()
        ax.set_xticks(np.arange(0, 4))
        ax.set_yticks(np.arange(0, 4))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # ax.set_aspect('equal')

    def _plot_state(self):
        xr, yr = self._remap_position(self.state.robber)
        xp, yp = self._remap_position(self.state.police)
        plt.plot(xp, yp, 'bo', markersize=20)
        plt.plot(xr, yr, 'ro', markersize=20)

    @staticmethod
    def _remap_position(position):
        x = position.x - 0.5
        y = 4.5 - position.y  # should work for bank robbing env
        return x, y

    def sample_action(self):
        return self.valid_actions[np.random.randint(5)]

    def reset(self):
        self.state = State(Position(1, 1), Position(4, 4))

    def _get_new_state(self, action):
        allowed_robber_actions = self._allowed_actions()
        allowed_police_actions = self._allowed_actions(police=True)

        new_robber_position = self.state.robber.take_action(action, allowed_robber_actions)

        police_action = allowed_police_actions[np.random.randint(len(allowed_police_actions))]
        new_police_position = self.state.police.take_action(police_action, allowed_police_actions)

        new_state = State(new_robber_position, new_police_position)
        self.state = new_state

        return new_state

    def _reward(self):

        bank_reward = 1
        penalty = -10

        robber = self.state.robber
        police = self.state.police

        if robber == self.bank and robber != police:
            return bank_reward

        elif robber == police:
            return penalty

        else:
            return 0

    def _allowed_actions(self, police=False):

        not_allowed = set()
        if police:
            position = self.state.police
            not_allowed.add('WAIT')
        else:
            position = self.state.robber

        # Bounds of map
        if position.x == 1:
            not_allowed.add('LEFT')

        if position.y == 1:
            not_allowed.add('UP')

        if position.x == 4:
            not_allowed.add('RIGHT')

        if position.y == 4:
            not_allowed.add('DOWN')

        allowed = [a for a in self.valid_actions if a not in not_allowed]
        return allowed


if __name__ == '__main__':

    env = Environment()
    env.render()

    for i in range(10):
        a = env.sample_action()
        s, r = env.step(a)
        print(s)
        print(r)
        env.render()

    steps = 1e6

    before = time()
    for i in range(1, int(steps + 1)):
        # a = env.sample_action()
        env.step('RIGHT')

        if i % (steps / 10) == 0:
            percent = (i / steps) * 100
            n = int(percent/10)
            print("%3d%% " % percent + "|" + "=" * n + (10 - n) * "-" + "|")

    print("Total time: %s seconds" % (time() - before))
