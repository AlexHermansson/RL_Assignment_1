import time
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
        self.states = []
        self._fill_states()
        self.state_to_int = {s: i for i, s in enumerate(self.states)}

    def __repr__(self):
        return self.state.__repr__()

    def get_state(self):
        return self.state_to_int[self.state]

    def step(self, action):
        action = self.valid_actions[action]
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

        return self.state_to_int[new_state]

    def _reward(self):
        # Rescaled so that |r| < 1 for all r.
        bank_reward = 0.1
        penalty = -1

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

    def _fill_states(self):
        xs = range(1, 5)
        ys = range(1, 5)
        for r in product(xs, ys):
            for p in product(xs, ys):
                self.states.append(State(Position(*r), Position(*p)))


class QAgent:
    """ Convert states & actions to integers before using this class! """
    def __init__(self, num_states=256, num_actions=5):

        self.Q = np.zeros((num_states, num_actions))
        self.updates = np.ones((num_states, num_actions))  # update counts for each Q-value
        self.discount = 0.8
        self.num_states, self.num_actions = self.Q.shape

    def Q_update(self, state, action, reward, new_state):
        lr = 1 / self.updates[state, action] ** (2 / 3)  # separate learning rate for each Q-value
        old_Q = self.Q[state, action]
        max_Q = np.max(self.Q[new_state])
        self.Q[state, action] = (1 - lr) * old_Q + lr * (reward + self.discount * max_Q)

        self.updates[state, action] += 1

    def random_action(self):
        return np.random.randint(self.num_actions)

    def choose_action(self, state, epsilon):

        if np.random.uniform() < epsilon:
            return np.random.randint(self.num_actions)

        return np.argmax(self.Q[state])


class SARSAAgent:

    def __init__(self, num_states=256, num_actions=5, eps=0.1):

        self.Q = np.zeros((num_states, num_actions))
        self.updates = np.ones((num_states, num_actions))  # update counts for each Q-value
        self.discount = 0.8
        self.num_states, self.num_actions = self.Q.shape
        self.epsilon = eps

    def Q_update(self, state, action, reward, new_state, new_action):
        lr = 1 / self.updates[state, action] ** (2 / 3)  # separate learning rate for each Q-value
        Q = self.Q[state, action]
        Q_next = self.Q[new_state, new_action]
        self.Q[state, action] = (1 - lr) * Q + lr * (reward + self.discount * Q_next)

        self.updates[state, action] += 1

    def choose_action(self, state):

        if np.random.uniform() < self.epsilon:
            return np.random.randint(self.num_actions)

        return np.argmax(self.Q[state])


def q_training():
    env = Environment()
    agent = QAgent()

    prev_state = env.get_state()  # outputs as an int, instead of a State object
    initial_state = prev_state
    random_state = np.random.randint(256)
    print('Random state: %d' % random_state)

    steps = 1e7
    before = time.time()

    V_initial = []
    V_random = []

    print("Learning to heist the bank! \n")
    for step in range(1, int(steps + 1)):

        action = agent.random_action()
        state, reward = env.step(action)
        agent.Q_update(prev_state, action, reward, state)

        V_initial.append(np.max(agent.Q[initial_state]))
        V_random.append(np.max(agent.Q[random_state]))
        prev_state = state

        if step % (steps / 10) == 0:
            percent = (step / steps) * 100
            n = int(percent / 10)
            print("%3d%% " % percent + "|" + "=" * n + (10 - n) * "-" + "|")

    print("Total time: %.0f seconds" % (time.time() - before))

    plt.semilogx(V_initial, c='r', label='Initial state')
    plt.semilogx(V_random, c='b', label='Random state')
    plt.legend()
    plt.title('Learning with Q-learning')
    plt.xlabel('Steps')
    plt.ylabel('$V^{\pi}(s_0)$', rotation=0)
    plt.show()

    rewards = 0
    steps = 1e5
    env.reset()
    for step in range(int(steps)):
        action = agent.choose_action(env.get_state(), epsilon=0)
        _, reward = env.step(action)
        rewards += reward
        # env.render()
        # time.sleep(0.5)

    print("Avarage reward: %0.2f" % (rewards / steps))


def sarsa_training():
    env = Environment()
    steps = 1e7

    epsilons = [0.1, 0.2, 0.3, 0.4]
    for epsilon in epsilons:

        agent = SARSAAgent(eps=epsilon)
        state = env.get_state()  # outputs as an int, instead of a State object
        action = agent.choose_action(state)
        initial_state = state

        V_initial = []

        before = time.time()
        print("Learning to heist the bank! \n")
        for step in range(1, int(steps + 1)):

            next_state, reward = env.step(action)
            next_action = agent.choose_action(next_state)
            agent.Q_update(state, action, reward, next_state, next_action)

            V_initial.append(np.max(agent.Q[initial_state]))

            state = next_state
            action = next_action

            if step % (steps / 10) == 0:
                percent = (step / steps) * 100
                n = int(percent / 10)
                print("%3d%% " % percent + "|" + "=" * n + (10 - n) * "-" + "|")

        print("Total time: %.0f seconds" % (time.time() - before))

        # plt.plot(V_initial, label='$\epsilon = %.1f$' % epsilon)
        plt.semilogx(V_initial, label='$\epsilon = %.1f$' % epsilon)

    plt.title('Learning with SARSA')
    plt.xlabel('Steps')
    plt.ylabel('$V^{\pi}(s_0)$', rotation=0)
    plt.legend()
    plt.show()


if __name__ == '__main__':

    q_training()
