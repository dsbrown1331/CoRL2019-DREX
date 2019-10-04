import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv) < 2:
    print("usage: python plot_degredation.py [env_name]")
    sys.exit()
env_name = sys.argv[1]
reader = open(env_name + "_batch_rewards.csv")
noise_levels = []
returns = []
for line in reader:
    parsed = line.split(", ")
    if parsed[0] == "demos":
        demo_returns = np.array([float(r) for r in parsed[1:]])
        print(demo_returns)
    else:
        noise_levels.append(float(parsed[0]))
        returns.append([float(r) for r in parsed[1:]])
returns = np.array(returns)
print(noise_levels)
print(returns)
#plot the average of the demos in line
demo_ave = np.mean(demo_returns)
demo_std = np.std(demo_returns)

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'xx-large',
          'figure.figsize': (5, 4),
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)

#plt.plot([0,1.0],[demo_ave, demo_ave])
plt.fill_between([0.01, 1.0], [demo_ave - demo_std, demo_ave - demo_std], [demo_ave + demo_std, demo_ave + demo_std], alpha = 0.3)
plt.plot([0.01,1.0],[demo_ave, demo_ave], label='demos')
plt.fill_between(noise_levels, np.mean(returns, axis=1)-np.std(returns, axis=1), np.mean(returns, axis=1) + np.std(returns, axis=1), alpha = 0.3)
plt.plot(noise_levels, np.mean(returns, axis = 1),'-.', label="bc")
#plot the average of pure noise in dashed line for baseline
plt.fill_between([0.01, 1.0], [np.mean(returns[-1]) - np.std(returns[-1]), np.mean(returns[-1]) - np.std(returns[-1])],
                        [np.mean(returns[-1]) + np.std(returns[-1]), np.mean(returns[-1]) + np.std(returns[-1])], alpha = 0.3)
plt.plot([0.01,1.0], [np.mean(returns[-1]), np.mean(returns[-1])],'--', label="random")
plt.legend(loc="best")
plt.xlabel("Epsilon")
plt.xticks([0.0, 0.25, 0.5, 0.75, 1.0])
plt.ylabel("Return")
plt.tight_layout()
plt.savefig(env_name + "_degredation_plot.png")
plt.show()
