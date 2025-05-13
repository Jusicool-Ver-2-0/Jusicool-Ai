from dataloader import *
from model import *
from trainer import *

def main():
    train_loader, val_loader, x_test, y_test = get_dataset()
    input_size = x_test.shape[2]
    hidden_size = 32
    num_layers = 1
    num_classes = 1
    model=GRU(num_classes, input_size, hidden_size, num_layers, seq_length=3)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    train(model, train_loader, val_loader, criterion,optimizer, 1000)
    test(model, x_test, y_test, criterion)

if __name__=='__main__':
    main()