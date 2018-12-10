import pickle

import numpy as np

class TeacherChoice(object):
    def __init__(self, num_teachers, reward_method, alpha_schedule):
        """
        reward_method: choices = ['avg_all_time', 'rolling_avg']
        """
        assert reward_method in ['avg_all_time', 'rolling_avg']

        self.num_teachers = num_teachers
        self.reward_method = reward_method
        self.alpha_schedule = alpha_schedule
        # number of times each teacher was chosen
        self.num_times_chosen = np.zeros(num_teachers)
        # list of rewards received after learning from each teacher
        self.rewards = [[] for _ in range(num_teachers)]
        # Q values associated with each teacher
        self.q = np.zeros(num_teachers)
        # the teacher that was previously chosen. Init to random
        self.prev_chosen_teacher = np.random.choice(num_teachers)
        # history of teachers that were chosen
        self.chosen_teachers = []

    def max_reward_teacher(self):
        """Return the teacher with the largest associated average reward over 
        all time."""
        return np.argmax([np.mean(r) for r in self.rewards])

    def max_q_value_teacher(self):
        """Return the teacher with the largest associated Q value."""
        return np.argmax(self.q)

    def choose_teacher(self, t):
        raise NotImplementedError

    def update_chosen_teacher(self, teacher):
        self.num_times_chosen[teacher] += 1
        self.prev_chosen_teacher = teacher
        self.chosen_teachers.append(teacher)

    def update_schedule(self, t):
        """Update the relevant schedules."""
        pass

    def store_reward(self, rewards, contexts):
        if hasattr(rewards, '__iter__'):
            assert len(rewards) == len(contexts)
            for r, c in zip(rewards, contexts):
                self._store_reward(r, c)
        else:
            self._store_reward(rewards, contexts)

    def _store_reward(self, reward, context=None):
        self.rewards[self.prev_chosen_teacher].append(reward)
        if self.self.alpha_schedule is not None:
            self.q[self.prev_chosen_teacher] += self.alpha_schedule.epsilon * (reward - self.q[self.prev_chosen_teacher])

    def save(self, filename):
        data = {
            'num_teachers': self.num_teachers,
            'num_times_chosen': self.num_times_chosen,
            'rewards': self.rewards,
            'chosen_teachers': self.chosen_teachers
        }
        with open(filename, 'wb') as outfile:
            pickle.dump(data, outfile)
        print('Saved teacher choice data to', filename)


class RandomBandit(TeacherChoice):
    def choose_teacher(self):
        """Choose a teacher uniformly at random."""
        teacher = np.random.choice(self.num_teachers)
        self.update_chosen_teacher(teacher)
        return teacher
  

class EpsilonGreedyBandit(TeacherChoice):
    def __init__(self, num_teachers, reward_method, alpha_schedule, eps_schedule):
        TeacherChoice.__init__(self, num_teachers, reward_method, alpha_schedule)
        self.eps_schedule = eps_schedule

    def get_epsilon(self):
        return self.eps_schedule.epsilon

    def update_schedule(self, t):
        self.eps_schedule.update(t)
        if self.alpha_schedule is not None:
            self.alpha_schedule.update(t)

    def choose_teacher(self):
        """Choose the teacher using epsilon-greedy strategy."""
        if np.random.random() <= self.get_epsilon():
            # choose a random teacher
            teacher = np.random.choice(self.num_teachers)
        else:
            if self.reward_method == 'avg_all_time':
                teacher = self.max_reward_teacher()
            elif self.reward_method == 'rolling_avg':
                teacher = self.max_q_value_teacher()
        
        self.update_chosen_teacher(teacher)
        return teacher


class VDBEBandit(TeacherChoice):
    def __init__(self, num_teachers, reward_method, alpha_schedule, 
            inv_sensitivity, delta='auto'):
        super(TeacherChoice, self).__init__(num_teachers, reward_method, alpha_schedule)
        self.inv_sensitivity = inv_sensitivity
        if delta == 'auto':
            self.delta = 1. / num_teachers
        self.epsilon = 1.

    def get_epsilon(self):
        return self.epsilon

    def update_schedule(self, t):
        # Update alpha schedule
        if self.alpha_schedule is not None:
            self.alpha_schedule.update(t)

    def _store_reward(self, reward, context=None):
        """Store the reward and update the value of epsilon using VDBE."""
        # previous reward estimate
        if self.reward_method == 'avg_all_time':
            prev_q = np.mean(self.rewards[self.prev_chosen_teacher])
        elif self.reward_method == 'rolling_avg':
            prev_q = self.q[self.prev_chosen_teacher]

        # store reward
        self.rewards[self.prev_chosen_teacher].append(reward)
        if self.self.alpha_schedule is not None:
            self.q[self.prev_chosen_teacher] += self.alpha_schedule.epsilon * (reward - self.q[self.prev_chosen_teacher])

        # current reward estimate
        if self.reward_method == 'avg_all_time':
            curr_q = np.mean(self.rewards[self.prev_chosen_teacher])
        elif self.reward_method == 'rolling_avg':
            curr_q = self.q[self.prev_chosen_teacher]

        # update epsilon
        abs_td_error = abs(curr_q - prev_q)
        f = (1 - np.e**(-abs_td_error / self.inv_sensitivity)) / (1 + np.e**(-abs_td_error / self.inv_sensitivity))
        self.epsilon = self.delta * f + (1. - self.delta) * self.epsilon


class UCB1Bandit(TeacherChoice):
    def __init__(self, num_teachers, reward_method, alpha_schedule):
        super(TeacherChoice, self).__init__(num_teachers, reward_method, alpha_schedule)
        self.t = 0

    def update_schedule(self, t):
        # Update alpha schedule and set value of t.
        if self.alpha_schedule is not None:
            self.alpha_schedule.update(t)
        self.t = t

    def choose_teacher(self):
        u = np.sqrt((2. * np.log(self.t) * np.ones(self.num_teachers)) / self.num_times_chosen)
        if self.reward_method == 'avg_all_time':
            q = np.array([np.mean(r) for r in self.rewards])
        elif self.reward_method == 'rolling_avg':
            q = self.q
        return np.argmax(q + u)


