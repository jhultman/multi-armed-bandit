import numpy as np
import matplotlib.pyplot as plt
from bandit import EpsilonGreedyStrategy

def run_bandit_experiment(eps_grid, n_trials=1000, T=500, n_actions=10):
    results = np.zeros((4, T))
    for i, eps in enumerate(eps_grid):
        for _ in range(n_trials):
            strat = EpsilonGreedyStrategy(eps=eps, n=n_actions)
            strat.pull_slot_many_times(T)
            avg = strat.compute_avg_reward()
            results[i, :] += avg / n_trials
    return results


def plot_bandit_experiment(results, eps_grid):
    fig, ax = plt.subplots()
    
    for i, eps in enumerate(eps_grid):
        ax.plot(results[i, :], label='$ \epsilon $: {0:.2f}'.format(eps))
        ax.legend()

    ax.set_title('Comparison of $ \epsilon $-greedy strategies')
    ax.set_xlabel('t')
    ax.set_ylabel('Mean reward')
    plt.savefig('../images/experiment.png')


def main():
    eps_grid = [0.01, 0.05, 0.1, 0.2]
    results = run_bandit_experiment(eps_grid, n_trials=500)
    plot_bandit_experiment(results, eps_grid)


if __name__ == '__main__':
    main()
