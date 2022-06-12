import matplotlib.pyplot as plt

def plot_process(_history, _title):
    # accuracy
    fig1 = plt.figure(1)
    plt.plot(_history.history['accuracy'], label='Training Acc')
    plt.plot(_history.history['val_accuracy'], label='Test Acc')
    plt.title(_title)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    # loss
    fig2 = plt.figure(2)
    plt.plot(_history.history['loss'], label='Training Loss')
    plt.plot(_history.history['val_loss'], label='Test Loss')
    plt.title(_title)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plt.show()

    # accuracy, loss

    fig3 = plt.figure(3)
    plt.plot(_history.history['accuracy'], label='Training Acc')
    plt.plot(_history.history['val_accuracy'], label='Test Acc')
    plt.plot(_history.history['loss'], label='Training Loss')
    plt.plot(_history.history['val_loss'], label='Test Loss')
    plt.title(_title)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Acc, Loss')
    plt.show()
    # learning rate
    try:
        fig4, axs = plt.subplots(2, 1)
        fig4.suptitle(_title)
        axs[0].plot(_history.history['loss'], label='Training Loss')
        axs[0].plot(_history.history['val_loss'], label='Test Loss')
        axs[1].plot(_history.history['lr'], label='Learning Rate')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
    except KeyError as e:
        print("ERROR: There is no learning rate in the training history")
    plt.show()