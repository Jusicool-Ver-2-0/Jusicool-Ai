import torch
import torch.nn as nn
from dataloader import *
from model import *
from trainer import *

def main():
    x_train, y_train, x_test, y_test=get_dataset()
    input_size = 4
    hidden_size = 2
    num_layers = 1
    num_classes = 1
    model=GRU(num_classes, input_size, hidden_size, num_layers, x_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    train(model, x_train, y_train, criterion, optimizer, 10000)
    test(model, x_test, y_test, criterion)

if __name__=='__main__':
    main()