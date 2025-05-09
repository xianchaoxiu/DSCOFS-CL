import torch.nn as nn
import torch


class Networks(nn.Module):
    def __init__(self, class_num, train_num, dim):
        super(Networks, self).__init__()
        self.X = nn.Parameter(torch.Tensor(dim, class_num))
        self.X = nn.init.xavier_uniform_(self.X)
        self.M = nn.Parameter(torch.Tensor(train_num, train_num))
        self.M = nn.init.xavier_uniform_(self.M)
    def forward(self, input):
        output1 = (self.M - torch.diag(torch.diag(self.M))).T.mm(input)
        output2 = output1.mm(self.X)
        return output1 , output2 ,  self.X, self.M - torch.diag(torch.diag(self.M))