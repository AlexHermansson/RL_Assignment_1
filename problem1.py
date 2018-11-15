import numpy as np
import matplotlib.pyplot as plt


class Position:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __ne__(self, other):
        return not self.__eq__(other)

    def manhattan(self, other):
        return int(np.abs(self.x - other.x) + np.abs(self.y - other.y))

    def take_action(self, action, allowed_actions):

        if action not in allowed_actions:
            return

        if action == 'UP':
            self.y -= 1

        elif action == 'DOWN':
            self.y += 1

        elif action == 'LEFT':
            self.x -= 1

        elif action == 'RIGHT':
            self.x += 1


class State:

    def __init__(self, p=None, m=None, done=False):
        self.done = done
        if p is None:
            self.player = Position(1, 1)
        else:
            self.player = p

        if m is None:
            self.minotaur = Position(5, 5)

        else:
            self.minotaur = m


class Environment:

    def __init__(self, t=15):
        self.T = t
        self.p = Position(1, 1)
        self.m = Position(5, 5)
        self.G = Position(5, 5)
        self.done = False
        self.valid_actions = {'UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT'}
        self.transition_probabilities = {}
        self._fill_probabilities()

    def reward(self, action=None):

        win_reward = 1
        loose_reward = -1

        # Terminal rewards
        if action is None:
            if self.p == self.G and self.m != self.p:
                return win_reward

            elif self.p == self.m:
                return loose_reward

            else:
                return 0

        # Non-terminal rewards
        else:
            if self.p == self.G and self.m != self.p:
                return win_reward

            elif self.p == self.m:
                return loose_reward

            else:
                return 0

    def _allowed_actions(self, position, minotaur=False):

        not_allowed = set()

        # Bounds of map
        if position.x == 1:
            not_allowed.add('LEFT')

        if position.y == 1:
            not_allowed.add('UP')

        if position.x == 6:
            not_allowed.add('RIGHT')

        if position.y == 5:
            not_allowed.add('DOWN')

        if minotaur:
            not_allowed.add('WAIT')
            return self.valid_actions - not_allowed

        # Walls
        if position.x == 2 and position.y in (1, 2, 3):
            not_allowed.add('RIGHT')

        if position.x == 3 and position.y in (1, 2, 3):
            not_allowed.add('LEFT')

        if position.y == 4 and position.x in (2, 3, 4, 5):
            not_allowed.add('DOWN')

        if position.y == 5:
            if position.x in (2, 3, 4, 5):
                not_allowed.add('UP')

                if position.x == 4:
                    not_allowed.add('RIGHT')

                if position.x == 5:
                    not_allowed.add('LEFT')

        if position.x == 4 and position.y in (2, 3):
            not_allowed.add('RIGHT')

        if position.x == 5 and position.y in (2, 3):
            not_allowed.add('LEFT')

        if position.y == 2 and position.x in (5, 6):
            not_allowed.add('DOWN')

        if position.y == 3 and position.x in (5, 6):
            not_allowed.add('UP')

        return self.valid_actions - not_allowed

    def _fill_probabilities(self):

        states = self._get_all_states()

        for state in states:
            for next_state in states:
                for action in self.valid_actions:
                    prob = self._transition_probability(next_state, state, action)
                    self.transition_probabilities[(next_state, state, action)] = prob

    def _get_all_states(self):
        xs = [i + 1 for i in range(6)]
        ys = [i + 1 for i in range(5)]

        states = [State(done=True)]
        for px in xs:
            for py in ys:
                for mx in xs:
                    for my in ys:
                        p = Position(px, py)
                        m = Position(mx, my)
                        states.append(State(p, m))
        return states

    def _transition_probability(self, next_state, state, action):

        allowed_actions_player = self._allowed_actions(state.player)
        allowed_actions_minotaur = self._allowed_actions(state.minotaur, minotaur=True)
        num_allowed_minotaur = len(allowed_actions_minotaur)

        if next_state.done:
            if state.player == self.G or state.player == state.minotaur or state.done:
                return 1

        elif state.player.take_action(action, allowed_actions_player) == next_state.player and \
                state.minotaur.manhattan(next_state.minotaur) == 1:
            return 1 / num_allowed_minotaur
        # todo: Changes for when the minotaur is allowed to stay

        return 0

    def visualize_maze(self):
        pass


if __name__ == '__main__':

    env = Environment()
    print(len(env.transition_probabilities))
