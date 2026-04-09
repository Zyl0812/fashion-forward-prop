import torch


def train_epoch(encoder, decoder, device, train_loader, loss_fn, optimizer):
    encoder.train()
    decoder.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        vectors = encoder(inputs)
        outputs = decoder(vectors)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()
    return train_loss / len(train_loader)


def val_step(encoder, decoder, device, val_loader, loss_fn):
    encoder.eval()
    decoder.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            vectors = encoder(inputs)
            outputs = decoder(vectors)
            loss = loss_fn(outputs, targets)
            val_loss += loss.item()
    return val_loss / len(val_loader)


def test_step(encoder, decoder, device, test_loader, loss_fn):
    encoder.eval()
    decoder.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            vectors = encoder(inputs)
            outputs = decoder(vectors)
            loss = loss_fn(outputs, targets)
            test_loss += loss.item()
    return test_loss / len(test_loader)
