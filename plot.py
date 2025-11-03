import matplotlib.pyplot as plt

def plot_training_results(results,save=False):
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))

    axes[0, 0].plot(results['average_train_batch_loss'], label='Train Loss', color='blue', marker='o')
    axes[0, 0].set_title('Average Loss per Epoch')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(results['average_train_batch_acc'], label='Train Accuracy', color='green', marker='o')
    axes[0, 1].set_title('Average Accuracy per Epoch')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    plt.tight_layout()
    plt.show()
    if save:
        fig.savefig('results_plots.png')
        print('figure save successfully')