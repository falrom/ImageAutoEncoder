from models import Model_CAE
import os

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

print('\n** GPU selection:', os.environ["CUDA_VISIBLE_DEVICES"])


def lr_decay_strategy(lr0, step, epoch, decay, minimun):
    if epoch < 300:
        return 0.002
    if epoch < 700:
        return 0.001
    if epoch < 900:
        return 0.0001
    return 0.00001


if __name__ == '__main__':
    cae = Model_CAE()
    cae.train(
        train_data_path='data/TFRdata/BSDS500_64.tfrecords',
        test_data_dir='data/test',
        epoch_volume=63000,
        epoch_to_train=1000,
        loss_func='l1',
        # time_str='20190510170936',
        decay_strategy=lr_decay_strategy
    )
