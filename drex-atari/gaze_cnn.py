import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Net(nn.Module):
    def __init__(self, gaze_loss_type):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 16, 7, stride=3)  # 26x26
        self.conv2 = nn.Conv2d(16, 16, 5, stride=2)  # 11x11
        self.conv3 = nn.Conv2d(16, 16, 3, stride=1)  # 9x9
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1)  # 7x7
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 1)

        self.gaze_loss_type = gaze_loss_type

    def cum_return(self, traj, gaze_conv_layer=0):
        #print(gaze_conv_layer, "cum_return")
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        sum_abs_rewards = 0
        #print(traj.size())
        x = traj.permute(0, 3, 1, 2)  # get into NCHW format
        # compute forward pass of reward network (we parallelize across frames so batch size is length of partial trajectory)
        x1 = F.leaky_relu(self.conv1(x))
        x2 = F.leaky_relu(self.conv2(x1))
        x3 = F.leaky_relu(self.conv3(x2))
        x4 = F.leaky_relu(self.conv4(x3))
        x = x4.view(-1, 784)
        x = F.leaky_relu(self.fc1(x))
        r = self.fc2(x)
        sum_rewards += torch.sum(r)
        sum_abs_rewards += torch.sum(torch.abs(r))

        # prepare conv map to be returned for gaze loss
        conv_map_traj = []
        conv_map_stacked = torch.tensor([[]])
        if gaze_conv_layer == 1:
            gaze_conv = x1
            gaze_size = 26
        elif gaze_conv_layer == 2:
            gaze_conv = x2
            gaze_size = 11
        elif gaze_conv_layer == 3:
            gaze_conv = x3
            gaze_size = 9
        elif gaze_conv_layer == 4:
            gaze_conv = x4
            gaze_size = 7
        else:
            pass
            #print('invalid gaze_conv_layer. Must be between 1-4.')
            # exit(0)

        if self.gaze_loss_type is not None and gaze_conv_layer != 0:
            # sum over all dimensions of the conv map
            conv_map = gaze_conv.sum(dim=1)

            # normalize the conv map
            traj_length = traj.shape[0]
            min_x = torch.min(torch.min(conv_map, dim=1)[0], dim=1)[0]
            max_x = torch.max(torch.max(conv_map, dim=1)[0], dim=1)[0]

            min_x = min_x.reshape(traj_length, 1).repeat(
                1, gaze_size).unsqueeze(-1).expand(traj_length, gaze_size, gaze_size)
            max_x = max_x.reshape(traj_length, 1).repeat(
                1, gaze_size).unsqueeze(-1).expand(traj_length, gaze_size, gaze_size)

            x_norm = (conv_map - min_x)/(max_x - min_x)
            conv_map_traj.append(x_norm)

            conv_map_stacked = torch.stack(conv_map_traj)

        return sum_rewards, sum_abs_rewards, conv_map_stacked

    def forward(self, traj_i, traj_j, gaze_conv_layer=0):
        #print(gaze_conv_layer, "forward")
        '''compute cumulative return for each trajectory and return logits'''
        cum_r_i, abs_r_i, conv_map_i = self.cum_return(traj_i, gaze_conv_layer)
        cum_r_j, abs_r_j, conv_map_j = self.cum_return(traj_j, gaze_conv_layer)
        return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)), 0), abs_r_i + abs_r_j, conv_map_i, conv_map_j
