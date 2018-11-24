import pickle
from itertools import product
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

    def manhattan(self, other):
        return int(np.abs(self.x - other.x) + np.abs(self.y - other.y))


class State:

    def __init__(self, robber=Position(1, 1), police=Position(3, 2)):
        self.robber = robber
        self.police = police
        self.actions = {'UP': [], 'DOWN': [], 'LEFT': [], 'RIGHT': [], 'WAIT': []}

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
        self.initial_state = State()
        self.banks = [Position(1, 1),Position(1,3),Position(6,3),Position(6,1)]
        self.valid_actions = ['WAIT', 'UP', 'DOWN', 'LEFT', 'RIGHT']
        self.states = []
        self._fill_states()
        self.state_to_int = {s: i for i, s in enumerate(self.states)}
        self._fill_probabilities()

    def __repr__(self):
        return self.state.__repr__()

    def render(self):#todo do i need this?
        self._plot_grid()
        self._plot_state()
        plt.show()

    @staticmethod
    def _plot_grid():
        plt.axis([0, 6, 0, 3])
        plt.plot(1.5, 2.5, 'g*', markersize=25)
        plt.grid()
        ax = plt.gca()
        ax.set_xticks(np.arange(0, 6))
        ax.set_yticks(np.arange(0, 3))
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

    def reward(self, state):

        bank_reward = 0.1
        penalty = -0.5

        robber = state.robber
        police = state.police

        if robber in self.banks and robber != police:
            return bank_reward

        elif robber == police:
            return penalty

        else:
            return 0

    def _allowed_actions(self,state, is_police=False):

        not_allowed = set()
        police=state.police
        robber=state.robber
        if is_police:
            not_allowed.add('WAIT')
            if police.x>robber.x:
                not_allowed.add('RIGHT')
            elif police.x<robber.x:
                not_allowed.add('LEFT')
            if police.y > robber.y:
                not_allowed.add('DOWN')
            elif police.y < robber.y:
                not_allowed.add('UP')

            if police.x == 1:
                not_allowed.add('LEFT')

            if police.y == 1:
                not_allowed.add('UP')

            if police.x == 6:
                not_allowed.add('RIGHT')

            if police.y == 3:
                not_allowed.add('DOWN')

        else:

            # Bounds of map
            if robber.x == 1:
                not_allowed.add('LEFT')

            if robber.y == 1:
                not_allowed.add('UP')

            if robber.x == 6:
                not_allowed.add('RIGHT')

            if robber.y == 3:
                not_allowed.add('DOWN')

        allowed = [a for a in self.valid_actions if a not in not_allowed]
        return allowed

    def _fill_probabilities(self):

        for state in self.states:
            for s_1, next_state in enumerate(self.states):
                for action in self.valid_actions:
                    prob = self._transition_probability(next_state, state, action)
                    if prob > 0:
                        state.actions[action].append((s_1, prob))

    def _fill_states(self):
        xs = range(1, 7)
        ys = range(1, 4)
        for r in product(xs, ys):
            for p in product(xs, ys):
                self.states.append(State(Position(*r), Position(*p)))

    def _transition_probability(self, next_state, state, action):

        allowed_actions_robber = self._allowed_actions(state)
        allowed_actions_police = self._allowed_actions(state,is_police=True)
        num_allowed_police = len(allowed_actions_police)

        if state.robber == state.police:
            if next_state==self.initial_state:
                return 1
            else:
                return 0

        else:
            new_position = state.robber.take_action(action, allowed_actions_robber)
            new_positions_police=[state.police.take_action(a,allowed_actions_police) for a in allowed_actions_police]
            if new_position == next_state.robber and state.police.manhattan(next_state.police) == 1 and next_state.police in new_positions_police:
                return 1 / num_allowed_police

        return 0

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def visualize_policy(opt_policy, police, env):

    plot_grid()
    plot_police(police)
    for x in range(1, 7):
        for y in range(1, 4):
            player = Position(x, y)
            if player != police:
                state = State(player, police)
                state_index = env.state_to_int[state]
                action_index = opt_policy[state_index]
                action = env.valid_actions[int(action_index)]
                plot_action(action, player)

    plt.show()

def plot_action(action, position):

    x, y = remap_position(position)

    l = 0.1
    arrow_width = 0.10

    if action == 'LEFT':
        start_x = x + l/2
        start_y = y
        plt.arrow(start_x, start_y, -l, 0, head_width=arrow_width)

    elif action == 'RIGHT':
        start_x = x - l/2
        start_y = y
        plt.arrow(start_x, start_y, l, 0, head_width=arrow_width)

    elif action == 'UP':
        start_x = x
        start_y = y - l/2
        plt.arrow(start_x, start_y, 0, l, head_width=arrow_width)

    elif action == 'DOWN':
        start_x = x
        start_y = y + l / 2
        plt.arrow(start_x, start_y, 0, -l, head_width=arrow_width)

    elif action == 'WAIT':
        plt.plot(x, y, 'bo', markersize=7)

    else:
        raise ValueError('Not a valid action!')

def plot_grid():

    plt.axis([0, 6, 0, 3])
    '''l1 = [(1, 5), (1, 1)]
    l2 = [(4, 4), (0, 1)]
    l3 = [(2, 2), (5, 2)]
    l4 = [(4, 4), (4, 2)]
    l5 = [(4, 6), (3, 3)]
    plt.plot(l1[0], l1[1], 'k')
    plt.plot(l2[0], l2[1], 'k')
    plt.plot(l3[0], l3[1], 'k')
    plt.plot(l4[0], l4[1], 'k')
    plt.plot(l5[0], l5[1], 'k')'''
    plt.grid()
    ax = plt.gca()
    ax.set_xticks(np.arange(0, 6))
    ax.set_yticks(np.arange(0, 3))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')

def plot_police(position):
    x, y = remap_position(position)
    plt.plot(x, y, 'ro', markersize=20)

def remap_position(position):
    x = position.x - 0.5
    y = 3.5 - position.y
    return x, y

def VI(environment, lambd=0.9, epsilon=0.01, delta=10):
    '''Value Iteration algorithm'''
    limit=epsilon*(1-lambd)/lambd
    all_states = np.asarray(environment.states)
    all_actions = environment.valid_actions
    iter=0
    # Init V
    V_n = np.zeros(len(all_states))
    V_n1 = np.zeros(len(all_states))
    for s, state in enumerate(all_states):
        V_n[s] = environment.reward(state)#todo: should I initialize like this or all zeros?

    # Init optimal policy
    optimal_policy = np.zeros(len(all_states),dtype=int)

    while delta>limit:
        iter+=1
        for s, state in enumerate(all_states):
            action_values = np.array([])
            for action in all_actions:
                r = environment.reward(state)
                sum = 0
                for s_1, p in state.actions[action]:
                    sum += p * V_n[s_1]
                action_values = np.append(action_values, r+lambd*sum)
            V_n1[s] = np.max(action_values)

        delta=np.linalg.norm(V_n1-V_n)
        V_n=np.copy(V_n1)

    print('VI converged after', iter, 'iterations.')
    for s, state in enumerate(all_states):
        action_values = np.array([])
        for action in all_actions:
            r = environment.reward(state)
            sum = 0
            for s_1, p in state.actions[action]:
                sum += p * V_n[s_1]
            action_values = np.append(action_values, r + lambd * sum)
        optimal_policy[s]=np.argmax(action_values)

    return V_n, optimal_policy


if __name__ == '__main__':

    #env = Environment()
    #save_obj(env, 'envp2')
    env = load_obj('envp2')
    V, policy=VI(env)
    police=Position(2,3)
    visualize_policy(policy,police,env)