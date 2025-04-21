import torch.nn.functional as F
from dscnn.transform import sp_transform


def train(model, epoch, train_loader, optimizer, log_interval):
    model.train()

    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data = sp_transform(data)  # Transform on CPU and move back to GPU

        output = model(data)
        loss = F.nll_loss(output.squeeze(), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )

        losses.append(loss.item())

    return losses


def test(model, epoch, test_loader, optimizer):
    model.eval()

    correct = 0
    for data, target in test_loader:
        data = sp_transform(data)
        output = model(data)

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
