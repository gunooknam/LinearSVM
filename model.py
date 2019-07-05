import torch
import torch.nn as nn

class LinearSVM(nn.Module):

    def __init__(self):
        super(LinearSVM, self).__init__()
        self.fc = nn.Linear(2, 1) # ( 500, 2 ) (2, 1) => (500, 1) 로 계산

    def forward(self, x):
        h = self.fc(x)
        return h