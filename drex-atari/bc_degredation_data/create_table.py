import numpy as np
import matplotlib.pyplot as plt
import sys

env_names = {'beamrider':'Beam Rider', 'breakout':'Breakout', 'enduro':'Enduro',
            'pong':'Pong', 'qbert':'Q*bert', 'seaquest':'Seaquest', 'spaceinvaders':'Space Invaders'}


writer = open("performance_table.txt",'w')
for env_name in env_names:
    reader = open(env_name + "_batch_rewards.csv")
    bc_returns = []
    demo_returns = []
    for line in reader:
        parsed = line.split(", ")
        if parsed[0] == "demos":
            demo_returns = np.array([float(r) for r in parsed[1:]])

        elif float(parsed[0]) == 0.01:
            bc_returns = [float(r) for r in parsed[1:]]

    writer.write("{} & {} & {} & & () & {} & ({:.1f}) \\\\ \n".format(env_names[env_name], np.mean(demo_returns), np.max(demo_returns), np.mean(bc_returns), np.std(bc_returns)))
