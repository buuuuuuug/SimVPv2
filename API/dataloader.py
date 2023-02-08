from .dataloader_moving_mnist import load_data as load_mnist


def load_data(dataname, batch_size, val_batch_size, num_workers, data_root, **kwargs):
    if dataname == 'mmnist':
        return load_mnist(batch_size, val_batch_size, num_workers, data_root)