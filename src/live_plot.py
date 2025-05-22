import matplotlib.pyplot as plt

def live_plot():
    """Initialize interactive plotting"""
    plt.ion()
    fig, ax = plt.subplots()
    return fig, ax, [], [], []

def update_live_plot(fig, ax, current_epochs, train_losses, val_losses, epoch, train_loss, val_loss):
    """Update the live training/validation loss plot"""
    epochs.append(epoch)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    ax.clear()
    ax.plot(epochs, train_losses, label='Train Loss', marker='o')
    ax.plot(epochs, val_losses, label='Validation Loss', marker='x')
    ax.set_title('Training and Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)

    fig.canvas.draw()
    fig.canvas.flush_events()
