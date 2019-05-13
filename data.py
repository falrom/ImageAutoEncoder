import matplotlib.pyplot as plt
import tensorflow as tf
import os

from utils import progress_bar
from utils import cv_imread


def generate_TFR(source_dir, patch_size=64, step=32):
    target_dir = './data/TFRdata'
    target_path = os.path.join(target_dir, os.path.split(source_dir)[-1] + '_' + str(patch_size) + '.tfrecords')
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    print('Input  dir  : %s' % source_dir)
    print('Output path : %s' % target_path)

    exts = {'.jpg', 'jpeg', '.png'}
    filenames = [name for name in os.listdir(source_dir) if os.path.splitext(name)[-1].lower() in exts]
    writer = tf.python_io.TFRecordWriter(target_path)
    patch_num = 0
    for img_num, filename in enumerate(filenames, start=1):
        filepath = os.path.join(source_dir, filename)
        img = cv_imread(filepath, 'RGB')

        hgt, wdt, c = img.shape
        count_h = int((hgt - patch_size) / step + 1)
        count_w = int((wdt - patch_size) / step + 1)
        start_h = 0
        for h in range(count_h):
            start_w = 0
            for w in range(count_w):
                patch_R = img[start_h:start_h + patch_size, start_w:start_w + patch_size, 0]
                patch_G = img[start_h:start_h + patch_size, start_w:start_w + patch_size, 1]
                patch_B = img[start_h:start_h + patch_size, start_w:start_w + patch_size, 2]
                example = tf.train.Example(features=tf.train.Features(feature={
                    'R': tf.train.Feature(bytes_list=tf.train.BytesList(value=[patch_R.tostring()])),
                    'G': tf.train.Feature(bytes_list=tf.train.BytesList(value=[patch_G.tostring()])),
                    'B': tf.train.Feature(bytes_list=tf.train.BytesList(value=[patch_B.tostring()]))
                }))
                writer.write(example.SerializeToString())
                start_w += step
            start_h += step
        progress_bar(img_num, len(filenames))
        patch_num += count_h * count_w
    writer.close()
    print('\nPatch number:', patch_num)


def dataset_TFR(TFR_path, batch_size, shuffle=None, repeat=True):
    """
    Dataset from TFR file.
    """
    patch_size = int(os.path.split(TFR_path)[-1].split('.')[0].split('_')[-1])

    def example_process(exa):
        ims = tf.parse_single_example(exa, features={
            'R': tf.FixedLenFeature([], tf.string),
            'G': tf.FixedLenFeature([], tf.string),
            'B': tf.FixedLenFeature([], tf.string)
        })
        im_R = ims['R']
        im_G = ims['G']
        im_B = ims['B']
        im_R = tf.decode_raw(im_R, tf.uint8)
        im_G = tf.decode_raw(im_G, tf.uint8)
        im_B = tf.decode_raw(im_B, tf.uint8)
        im_R = tf.reshape(im_R, [patch_size, patch_size, 1]) / 255
        im_G = tf.reshape(im_G, [patch_size, patch_size, 1]) / 255
        im_B = tf.reshape(im_B, [patch_size, patch_size, 1]) / 255

        # return {'R': im_R, 'G': im_G, 'B': im_B}
        return tf.concat([im_R, im_G, im_B], 2)

    dataset = tf.data.TFRecordDataset(TFR_path)
    if shuffle is not None:
        dataset = dataset.shuffle(buffer_size=shuffle)
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(map_func=example_process, batch_size=batch_size, num_parallel_batches=4))
    # dataset = dataset.map(example_process)
    # dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=16)
    iterator = dataset.make_one_shot_iterator()
    batch = iterator.get_next()
    return batch


def _test_tfr(path):
    get_batch = dataset_TFR(path, 3, 63000)
    sess = tf.InteractiveSession()
    for i in range(10):
        imgs = sess.run(get_batch)
        plt.figure()
        plt.imshow(imgs[1])
    sess.close()
    plt.show()
    os.system('PAUSE')


def dataset_IMG(source_dir):
    exts = {'.jpg', 'jpeg', '.png'}
    filenames = [name for name in os.listdir(source_dir) if os.path.splitext(name)[-1].lower() in exts]
    filenames.sort()
    test_batches = {'imgs': [], 'names': []}
    for img_name in filenames:
        img_path = os.path.join(source_dir, img_name)
        img = cv_imread(img_path, 'RGB')
        test_batches['imgs'].append(img / 255)
        test_batches['names'].append(img_name)
    return test_batches


if __name__ == '__main__':
    generate_TFR('./data/BSDS500')  # 63000
    # _test_tfr('./data/TFRdata/BSDS500_64.tfrecords')
