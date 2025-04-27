import torch
import matplotlib.pyplot as plt

# Load saved losses
loss_record = torch.load('train_losses.pt')
train_losses = loss_record
#test_losses = loss_record['test_accuracies']

# Plot
plt.plot(train_losses, label='Train Loss')
#plt.plot(test_losses, label='Test accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('plot_inq.png') 
