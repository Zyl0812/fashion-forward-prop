import torch


def train_epoch(model, device, train_loader, loss_fn, optimizer):
    model.train()
    train_loss = 0.0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader)


def val_step(model, device, val_loader, loss_fn):
    model.eval()
    val_loss = 0.0
    val_dataset_num = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            val_loss += loss.item() * inputs.shape[0]
            val_dataset_num += inputs.shape[0]
    return val_loss / val_dataset_num


def test_epoch(model, device, test_loader, loss_fn):
    model.eval()
    test_correct_num = 0
    test_dataset_num = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            pred = output.argmax(dim=1)
            test_correct_num += pred.eq(targets).sum().item()
            test_dataset_num += inputs.shape[0]
    return test_correct_num / test_dataset_num
