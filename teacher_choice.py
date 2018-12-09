import numpy as np

class TeacherChoice(object):
    def choose_teacher(self, t):
        raise NotImplementedError

    def update_chosen_teacher(self, teacher):
        self.num_times_chosen[teacher] += 1
        self.prev_chosen_teacher = teacher

    def update_schedule(self, t):
        """Update the epsilon schedule."""
        pass

class NonContextualBandit(TeacherChoice):
    def __init__(self, num_teachers):
        self.num_teachers = num_teachers
        # number of times each teacher was chosen
        self.num_times_chosen = np.zeros(num_teachers)
        # list of rewards received after learning from each teacher
        self.rewards = [[] for _ in range(num_teachers)]
        # the teacher that was previously chosen. Init to random
        self.prev_chosen_teacher = np.random.choice(num_teachers)

class RandomBandit(NonContextualBandit):
    def choose_teacher(self):
        """Choose a teacher uniformly at random."""
        teacher = np.random.choice(self.num_teachers)
        self.update_chosen_teacher(teacher)
        return teacher
        
class EpsilonGreedyBandit(NonContextualBandit):
    def __init__(self, num_teachers, eps_schedule):
        super(NonContextualBandit, self).__init__(num_teachers)
        self.eps_schedule = eps_schedule

    def update_schedule(self, t):
        self.eps_schedule.update(t)

    def max_reward_teacher(self):
        """Return the teacher with the largest associated average reward."""
        return np.argmax(np.mean(r) for r in self.rewards)

    def choose_teacher(self):
        """Choose the teacher using epsilon-greedy strategy."""
        if np.random.random() <= self.eps_schedule.epsilon:
            # choose a random teacher
            teacher = np.random.choice(self.num_teachers)
        else:
            teacher = self.max_reward_teacher()
        
        self.update_chosen_teacher(teacher)
        return teacher

    # def store_teacher_rewards(self, reward):
    #     if hasattr(reward, '__iter__'):

    #     else: