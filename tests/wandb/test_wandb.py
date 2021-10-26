import wandb


if __name__ == '__main__':
    wandb.init(project="my-test-project")
    train_acc = 0.3
    train_loss = 0.005
    wandb.log({'accuracy': train_acc, 'loss': train_loss})
    wandb.config.dropout = 0.2
