import pickle
import numpy as np
import matplotlib.pyplot as plt

## Gian is nice

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

    def manhattan(self, other):
        return int(np.abs(self.x - other.x) + np.abs(self.y - other.y))

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

    def __init__(self, p=None, m=None, done=False):
        self.done = done
        if p is None and m is None:
            self.player = Position(-1, -1)
            self.minotaur = Position(-5, -5)
        else:
            self.player = p
            self.minotaur = m

    def __eq__(self, other):
        return self.player == other.player and self.minotaur == other.minotaur and self.done == other.done

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        if self.done:
            return "Done"

        else:
            return "Player: %s, Minotaur: %s)" % (self.player, self.minotaur)


class Environment:

    def __init__(self, t=15, transition_prob=None):
        self.T = t
        self.p = Position(1, 1)
        self.m = Position(5, 5)
        self.G = Position(5, 5)
        self.done = False
        self.valid_actions = {'UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT'}
        self.transition_probabilities = {}
        self.count=0
        if transition_prob is None:
            self._fill_probabilities()
        else:
            self.transition_probabilities = transition_prob

    def reward(self, state, action=None):

        win_reward = 1
        loose_reward = -1

        # Terminal rewards
        if action is None:
            if state.player == self.G and state.minotaur != state.player:
                return win_reward

            elif state.player == state.minotaur:
                return loose_reward

            else:
                return 0

        # Non-terminal rewards
        else:
            if state.player == self.G and state.minotaur != state.player:
                return win_reward

            elif state.player == state.minotaur:
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

        states = self.get_all_states()

        for state in states:
            for next_state in states:
                for action in self.valid_actions:
                    prob = self._transition_probability(next_state, state, action)
                    if prob > 0:
                        self.transition_probabilities[(next_state, state, action)] = prob

    @staticmethod
    def get_all_states():
        xs = range(1, 7)
        ys = range(1, 6)

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

        elif not state.done:

            ###### THIS IS THE NEW CONDITION 14510
            if state.player == state.minotaur and not next_state.done:

                return 0

            new_position = state.player.take_action(action, allowed_actions_player)
            if new_position == next_state.player and state.minotaur.manhattan(next_state.minotaur) == 1:
                return 1 / num_allowed_minotaur

        # todo: Change here when the minotaur is allowed to stay

        return 0

    def visualize_maze(self):
        pass


def DP(environment, time_horizon=15):

    all_states = environment.get_all_states()
    all_actions = environment.valid_actions

    # Init V
    V = {}
    for state in all_states:
        V[state] = [environment.reward(state)]


    for t in range(1, time_horizon-1):
        for state in all_states:
            action_values = []
            for action in all_actions:
                r = environment.reward(state, action)
                sum = r
                for next_state in all_states:

                    try:
                        p = environment.transition_probabilities[(next_state, state, action)]
                    except KeyError:
                        p = 0

                    v = V[next_state][t-1]
                    sum += p * v

                action_values.append(sum)

            V[state].append(max(action_values))

    return V



def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':

    trans_prob = load_obj('T')
    T = 15

    env = Environment(transition_prob=trans_prob)
    V = DP(env, T)

    init_s = State(Position(1, 1), Position(5, 5))
    print(V[init_s])
