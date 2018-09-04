import numpy as np
np.random.seed(21)

def generate_bandits(n):
    means = np.random.randn(n)
    return means

class EpsilonGreedyStrategy:
    
    def __init__(self, eps, n=10):
        self.eps = eps
        self.n = n
        self.sigma = 0.1
        self.true_means = generate_bandits(n)
        self.rewards = []
        self.estimates = np.zeros(n)
        self.counts = np.zeros(n)
        
    def pull_slot_many_times(self, T):
        for _ in range(T):
            self.pull_slot()
    
    def pull_slot(self):
        r = np.random.rand()
        if r < self.eps:
            self.explore()
        else:
            self.exploit()
    
    def exploit(self):
        choice = np.argmax(self.estimates)
        self.play(choice)
    
    def explore(self):
        choice = np.random.randint(self.n)
        self.play(choice)
        
    def play(self, choice):
        noise = self.sigma * np.random.randn()
        reward = self.true_means[choice] + noise
        self.rewards += [reward]
        prev_estimate = self.estimates[choice] * self.counts[choice]
        self.estimates[choice] = (prev_estimate + reward) / (self.counts[choice] + 1)
        self.counts[choice] += 1
        
    def compute_avg_reward(self):
        numerator = np.cumsum(self.rewards)
        denominator = np.cumsum(np.ones_like(self.rewards))
        return numerator / denominator
