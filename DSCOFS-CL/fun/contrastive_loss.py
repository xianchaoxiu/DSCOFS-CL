import torch
import torch.nn as nn


class InstanceLoss(nn.Module):
    def __init__(self, train_num, temperature, device):
        super(InstanceLoss, self).__init__()
        self.train_num = train_num
        self.temperature = temperature
        self.device = device

        self.mask = self.mask(train_num)
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def mask(self, train_num):
        N = 2 * train_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(train_num):
            mask[i, train_num + i] = 0
            mask[train_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.train_num
        z = torch.cat((z_i, z_j), dim=0)
        sim = torch.matmul(z, z.T) / self.temperature

        sim_i_j = torch.diag(sim, self.train_num)
        sim_j_i = torch.diag(sim, -self.train_num)

        positive_labels = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_labels = sim[self.mask].reshape(N, -1)

        label1 = torch.zeros(N).to(positive_labels.device).long()
        label2 = torch.cat((positive_labels, negative_labels), dim=1)
        loss = self.criterion(label2, label1)

        return loss