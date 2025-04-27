import torch
import torch.optim as optim
from model import SimpleCNN_INQ
from dataset import get_cifar10
from train import train, test, update_fixed_mask
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='INQ Training on CIFAR-10')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use (default: 0)')
    parser.add_argument('--INQ', action="store_true")
    args = parser.parse_args()
    return args

# -------- Main Script --------
def main():
    args = parse_args()

    if torch.cuda.is_available():
        device = f'cuda:{args.gpu}'
    else:
        device = 'cpu'
    print(f"Using device: {device}")

    trainloader, testloader = get_cifar10()

    total_epochs = 150
    prune_schedule = [0.3, 0.3, 0.4]  # 30%, 30%, 40%
    prune_interval = total_epochs // len(prune_schedule)

    model = SimpleCNN_INQ().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    train_losses = []
    test_losses = []

    for epoch in range(1, total_epochs+1):
        avg_train_loss = train(model, device, trainloader, optimizer, epoch)
        avg_test_loss, _ = test(model, device, testloader)
        scheduler.step()

        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)

        if args.INQ: 
            # Update fixed masks periodically
            if epoch % prune_interval == 0 and prune_schedule:
                prune_percent = prune_schedule.pop(0)
                print(f"--> Fixing {prune_percent*100:.1f}% of weights")
                model.update_masks(prune_percent)

    # ---- Save Losses after Training ----
    torch.save({
        'train_losses': train_losses,
        'test_losses': test_losses,
    }, 'loss_record_INQ.pt')
    print("✅ Saved training and test losses to 'loss_record.pt'!")



if __name__ == '__main__':
    main()

