import torch
import torch.nn.functional as F
def train(model, x_train, y_train, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        outputs = model(x_train)
        optimizer.zero_grad()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        print("Epoch: ", epoch + 1, "Loss:", loss.item())

def test(model, x_test, y_test, criterion):
    model.eval()
    with torch.no_grad():
        outputs = model(x_test)
        loss = criterion(outputs, y_test)

    print("Test Loss: ", loss.item())

