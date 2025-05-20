import torch
import torch.nn.functional as F

def train(model, train_loader, val_loader, criterion, optimizer, epochs, writer):
    best_loss = float('inf')
    best_weights = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for x_batch, y_batch in train_loader:
            outputs = model(x_batch).squeeze(1)
            y_batch = y_batch.squeeze(1)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        writer.add_scalar('Training loss', avg_train_loss, epoch)
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                outputs = model(x_val).squeeze(1)
                y_val = y_val.squeeze(1)
                loss = criterion(outputs, y_val)
                val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            writer.add_scalar('validation Loss', avg_val_loss, epoch)

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_weights = model.state_dict()

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    if best_weights:
        model.load_state_dict(best_weights)
        torch.save(model.state_dict(), 'best_model.pth')


def test(model, x_test, y_test, criterion):
    model.eval()
    with torch.no_grad():
        outputs = model(x_test)
        predicted = (outputs >= 0.5).float()
        y_test=y_test.squeeze(1)
        predicted=predicted.squeeze(1)
        correct = (predicted == y_test).sum().item()
        accuracy = correct / y_test.size(0)
        print(f"Accuracy: {accuracy*100:.2f}%")