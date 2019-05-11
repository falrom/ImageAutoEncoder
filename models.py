import tensorflow as tf
import numpy as np
import time
import os

from layers import conv_autoencoder
from layers import MSE_float
from layers import PSNR_float
from layers import PSNR_uint8
from layers import loss_L1
from layers import Outputs

from data import dataset_TFR
from data import dataset_IMG

from utils import print_info
from utils import record_info
from utils import get_time_str
from utils import cv_imread
from utils import cv_imwrite


# class Model_base:
#
#     def __init__(self, model_name, name):
#         self.model_name = model_name
#         self.name = name
#
#         self.info_top = {}
#         self.info_top['name'] = name
#         self.info_top['model_name'] = model_name
#
#     def _build_train(self):
#         # needs: inputs, states, inference, tails, loss, optimizer, logs
#         raise NotImplementedError('No implementation for function: _build_train')
#
#     def _build_test(self):
#         raise NotImplementedError('No implementation for function: _build_test')
#
#     def _build_evaluate(self):
#         raise NotImplementedError('No implementation for function: _build_evaluate')
#
#     def _build_predict(self):
#         raise NotImplementedError('No implementation for function: _build_predict')
#
#     def _get_train_batch(self):
#         raise NotImplementedError('No implementation for function: _get_train_batch')
#
#     def _get_test_batch(self):
#         raise NotImplementedError('No implementation for function: _get_test_batch')
#
#     def train(self,
#               train_data_path,
#               test_data_path,
#               time_str=None,
#               train_batch_size=64,
#               test_batch_size=512,
#               test_interval=100,
#               save_interval=500,
#               loss_func='l2',
#               optimizer='adam',
#               learning_rate=0.001,
#               decay=None):
#         # params:
#         time_str = time_str or get_time_str()
#         self.loss_func = loss_func
#         self.optimizer = optimizer
#         self.train_data_path = train_data_path
#         self.test_data_path = test_data_path
#         self.train_batch_size = train_batch_size
#         self.test_batch_size = test_batch_size
#
#         # paths:
#         log_dir = os.path.join('./logs', self.model_name, self.name, time_str)
#         ckpt_dir = os.path.join('./checkpoints', self.model_name, self.name, time_str)
#         if not os.path.exists(log_dir):
#             os.makedirs(log_dir)
#         if not os.path.exists(ckpt_dir):
#             os.makedirs(ckpt_dir)
#         latest_ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
#
#         # info:
#         self.info_train = {}
#         self.info_train['time_str'] = str(time_str)
#         self.info_train['train_batch_size'] = str(train_batch_size)
#         self.info_train['test_batch_size'] = str(test_batch_size)
#         self.info_train['loss_func'] = str(loss_func)
#         self.info_train['optimizer'] = str(optimizer)
#         self.info_train['learning_rate'] = str(learning_rate)
#         self.info_train['decay'] = str(decay)
#         self.info_train['train_data_path'] = str(train_data_path)
#         self.info_train['test_data_path'] = str(test_data_path)
#         print('\n\n********** Train **********')
#         print_info([self.info_train])
#         print('********** ***** **********')
#         record_info([self.info_top, self.info_train], os.path.join(log_dir, 'info.txt'))
#
#         # define graph:
#         print('\n** Define graph...')
#         self.train_graph = tf.Graph()
#         with self.train_graph.as_default():
#             self._build_train()
#             # logs:
#             log_writer = tf.summary.FileWriter(log_dir)
#             log_writer.add_graph(self.train_graph)
#             log_writer.flush()
#             # saver:
#             saver_all = tf.train.Saver(max_to_keep=0, name='saver_all')
#         print('Done.')
#
#         # datasets:
#         self.train_sess = tf.Session(graph=self.train_graph)
#         sess = self.train_sess
#         print('\n** Generate datasets...')
#         print('train data path:', train_data_path)
#         print('test  data path:', test_data_path)
#         with self.train_graph.as_default():
#             train_batches = self._get_train_batch()
#             test_batches = self._get_test_batch()
#         print('Done.')
#
#         # init:
#         if latest_ckpt_path:
#             saver_all.restore(sess, latest_ckpt_path)
#         else:
#             sess.run(self.variable_init)
#         step = tf.train.global_step(sess, self.global_step)
#
#         self.train_sess.close()
#         print('\nALL DONE.')
#
#     def test(self):
#         raise NotImplementedError('No implementation for function: test')
#
#     def evaluate(self):
#         # self.info_evaluate = {}
#         pass
#
#     def predict(self):
#         raise NotImplementedError('No implementation for function: predict')


class Model_CAE():

    def __init__(self, name=None, ratio=3, channels=32, use_pooling=False, use_subpixel=True):
        self.model_name = 'CAE'
        self.name = name or self.model_name + ('_r%dc%d' % (ratio, channels))
        self.ratio = ratio
        self.channels = channels
        self.use_pooling = use_pooling
        self.use_subpixel = use_subpixel

        self.MODE_TRAIN = 'TRAIN'
        self.MODE_EVAL = 'EVAL'
        self.MODE_PREDICT = 'PREDICT'

        self.info_top = {}
        self.info_top['name'] = self.name
        self.info_top['model_name'] = self.model_name
        self.info_top['ratio'] = str(self.ratio)
        self.info_top['channels'] = str(self.channels)
        self.info_top['use_pooling'] = str(self.use_pooling)
        self.info_top['use_subpixel'] = str(self.use_subpixel)

    def _inference(self, inputs):
        return conv_autoencoder(inputs=inputs,
                                ratio=self.ratio,
                                channels=self.channels,
                                name=self.name,
                                use_pooling=self.use_pooling,
                                use_subpixel=self.use_subpixel)

    def _build(self, mode):
        # inputs:
        with tf.name_scope(name='inputs'):
            self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3], name='inputs')
        if mode in {self.MODE_TRAIN, self.MODE_EVAL}:
            with tf.name_scope(name='labels'):
                self.labels = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3], name='labels')

        # states:
        if mode in {self.MODE_TRAIN}:
            with tf.variable_scope('states'):
                self.global_step = tf.Variable(0, trainable=False, name=tf.GraphKeys.GLOBAL_STEP)
                self.learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

        # inference:
        self.encoded, self.decoded = self._inference(self.inputs)

        # tails:
        if mode in {self.MODE_TRAIN}:
            self.mse = MSE_float(predictions=self.decoded, labels=self.labels)
            self.psnr_float = PSNR_float(mse=self.mse)
            self.outputs = Outputs(predictions=self.decoded)
            # loss:
            if self.loss_func == 'l1':
                self.loss = loss_L1(predictions=self.decoded, labels=self.labels)
            elif self.loss_func == 'l2':
                self.loss = self.mse
            else:
                raise Exception('Unknown loss function: ' + str(self.loss_func))
            # optimizer:
            if self.optimizer == 'adam':
                self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
                    self.loss, global_step=self.global_step)
            elif self.optimizer == 'sgd':
                self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(
                    self.loss, global_step=self.global_step)
            else:
                raise Exception('Unknown optimizer: ' + str(self.optimizer))
            self.variable_init = tf.global_variables_initializer()
        if mode in {self.MODE_EVAL}:
            self.mse = MSE_float(predictions=self.decoded, labels=self.labels)
            self.psnr_float = PSNR_float(mse=self.mse)
            self.outputs = Outputs(predictions=self.decoded)
            self.psnr_uint8 = PSNR_uint8(outputs=self.outputs, labels=self.labels)

    def _get_epoch(self, step, batch_size, epoch_volume):
        return (step * batch_size) // epoch_volume + 1

    def _lr_update(self, lr0, step, epoch, decay, strategy, minimun=1e-10):
        # custom strategy：
        if hasattr(strategy, '__call__'):
            return strategy(lr0, step, epoch, decay, minimun)
        # no decay:
        if decay is None:
            return lr0
        # built-in decay methods:
        if strategy == 'exponent':
            return max(lr0 * (decay ** step), minimun)
        elif strategy == 'linear':
            return max(lr0 - decay * step, minimun)
        elif strategy == 'Inverse':
            return max(lr0 / (step * decay + 1), minimun)
        else:
            raise Exception('Unknown strategy: ' + str(strategy))

    def train(self,
              train_data_path,
              test_data_dir,
              epoch_volume,
              epoch_to_train=None,
              time_str=None,
              train_batch_size=64,
              steps=None, max_steps=None,
              log_print_interval=50,
              test_interval=500,
              save_interval=10000,
              loss_func='l2',
              optimizer='adam',
              learning_rate=0.001,
              decay=None,
              decay_epoch=1,
              decay_strategy='exponent'):
        # params:
        time_str = time_str or get_time_str()
        self.loss_func = loss_func
        self.optimizer = optimizer

        # paths:
        log_dir = os.path.join('./logs', self.model_name, self.name, time_str)
        ckpt_dir = os.path.join('./checkpoints', self.model_name, self.name, time_str)
        test_imgs_dir = os.path.join(log_dir, 'test_imgs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        if not os.path.exists(test_imgs_dir):
            os.makedirs(test_imgs_dir)
        latest_ckpt_path = tf.train.latest_checkpoint(ckpt_dir)

        # info:
        self.info_train = {}
        self.info_train['time_str'] = str(time_str)
        self.info_train['train_batch_size'] = str(train_batch_size)
        self.info_train['loss_func'] = str(loss_func)
        self.info_train['optimizer'] = str(optimizer)
        if isinstance(decay_strategy, str):
            self.info_train['learning_rate'] = str(learning_rate)
            self.info_train['decay'] = str(decay)
            self.info_train['decay_strategy'] = str(decay_strategy)
        else:
            self.info_train['learning_rate'] = 'CustomStrategy'
        self.info_train['train_data_path'] = str(train_data_path)
        self.info_train['test_data_dir'] = str(test_data_dir)
        print('\n\n********** Train **********')
        print_info([self.info_train])
        print('********** ***** **********')
        record_info([self.info_top, self.info_train], os.path.join(log_dir, 'info.txt'))

        # define graph:
        print('\n** Define graph...')
        self.train_graph = tf.Graph()
        with self.train_graph.as_default():
            self._build(self.MODE_TRAIN)
            # logs:
            log_train_MSE = tf.summary.scalar('MSE_train', self.mse)
            log_test_MSE = tf.summary.scalar('MSE_test', self.mse)
            log_train_PSNR = tf.summary.scalar('PSNR_train', self.psnr_float)
            log_test_PSNR = tf.summary.scalar('PSNR_test', self.psnr_float)
            log_lr = tf.summary.scalar('learning_rate', self.learning_rate)
            test_PSNR_mean = tf.placeholder(tf.float32, name='PSNR_mean')
            log_test_PSNR_mean = tf.summary.scalar('PSNR_mean_test', test_PSNR_mean)
            log_writer = tf.summary.FileWriter(log_dir)
            log_writer.add_graph(self.train_graph)
            log_writer.flush()
            # saver:
            saver_all = tf.train.Saver(max_to_keep=0, name='saver_all')
        print('Done.')

        # datasets:
        print('\n** Generate datasets...')
        print('train data path:', train_data_path)
        print('test  data  dir:', test_data_dir)
        with self.train_graph.as_default():
            get_train_batch = dataset_TFR(train_data_path, train_batch_size, epoch_volume)
            test_batches = dataset_IMG(test_data_dir)
        print('Done.')

        print('\n** Initialize and prepare...')

        # init:
        sess = tf.Session(graph=self.train_graph)
        if latest_ckpt_path:
            saver_all.restore(sess, latest_ckpt_path)
        else:
            sess.run(self.variable_init)

        step = tf.train.global_step(sess, self.global_step)
        epoch = self._get_epoch(step, train_batch_size, epoch_volume)
        steps_to_run = None
        if steps or max_steps:
            steps_to_run = steps or max(max_steps - step, 0)

        # define process functions:
        def train_once(step, epoch=None, pring_log=True):
            train_batch = sess.run(get_train_batch)
            # lr = learning_rate
            # if decay:
            #     lr = learning_rate * (decay ** step)
            feed_dic = {
                self.inputs: train_batch,
                self.labels: train_batch,
                self.learning_rate: lr
            }
            mse, mse_log, psnr, psnr_log, lr_log, _ = sess.run(
                [self.mse, log_train_MSE, self.psnr_float, log_train_PSNR, log_lr, self.train_op], feed_dic)
            log_writer.add_summary(mse_log, step)
            log_writer.add_summary(psnr_log, step)
            log_writer.add_summary(lr_log, step)

            if pring_log:
                log = 'step: %d  lr: %.8f  train-loss: %.10f  train-PSNR: %.6f' % (step, lr, mse, psnr)
                if epoch is not None:
                    log = ('epoch: %d ' % epoch) + log
                print(log)

        def test_all(step, epoch=None, pring_log=True, save_dir=None):
            if pring_log:
                print('--------------------------------------------------------------')
                print('Test all:')
            img_num = len(test_batches['imgs'])
            psnr_sum = 0
            for tb in range(img_num):
                img = test_batches['imgs'][tb][np.newaxis, :]
                name = test_batches['names'][tb]
                feed_dic = {
                    self.inputs: img,
                    self.labels: img
                }
                run_list = [self.mse, log_test_MSE, self.psnr_float, log_test_PSNR]
                if save_dir is not None:
                    run_list.append(self.outputs)
                run_results = sess.run(run_list, feed_dic)
                if save_dir is None:
                    mse, mse_log, psnr, psnr_log = run_results
                else:
                    mse, mse_log, psnr, psnr_log, outputs = run_results
                    name_no_ext = os.path.splitext(name)[0]
                    if epoch is not None:
                        cv_imwrite(
                            os.path.join(save_dir,
                                         'epoch_%d_step_%d_%s_psnr_%.4f.png' % (epoch, step, name_no_ext, psnr)),
                            outputs[0], 'RGB')
                    else:
                        cv_imwrite(os.path.join(save_dir, 'step_%d_%s_psnr_%.4f.png' % (step, name_no_ext, psnr)),
                                   outputs[0], 'RGB')
                log_writer.add_summary(mse_log, step)
                log_writer.add_summary(psnr_log, step)
                log_writer.flush()
                psnr_sum += psnr
                if pring_log:
                    log = 'step: %d  test-loss: %.10f  test-PSNR: %.6f' % (step, mse, psnr)
                    if epoch is not None:
                        log = ('epoch: %d ' % epoch) + log
                    log = ('img: %s ' % name) + log
                    print(log)
            psnr_mean = psnr_sum / img_num
            log_writer.add_summary(sess.run(log_test_PSNR_mean, {test_PSNR_mean: psnr_mean}), step)
            if pring_log:
                print('PSNR-mean: %.6f (img_num: %d)' % (psnr_mean, img_num))
                print('--------------------------------------------------------------')
            return psnr_mean

        def save_once(step, pring_log=True):
            save_path = os.path.join(ckpt_dir, get_time_str())
            saver_all.save(
                sess=sess,
                save_path=save_path,
                global_step=step,
                write_meta_graph=False)
            if pring_log:
                print('save:', save_path)
            return save_path

        print('Done.')

        # run:
        save_path = None
        print('\n** Begin training:')
        if latest_ckpt_path is None:
            test_all(0, 0, True)
            save_path = save_once(0)
        else:
            test_all(step, epoch, True)

        save_flag_final = False
        save_flag_max = False
        psnr_max = 0
        lr = self._lr_update(learning_rate, step, epoch, decay, decay_strategy)
        t = time.time()
        while (steps_to_run is None) or (steps_to_run > 0):
            step = tf.train.global_step(sess, self.global_step) + 1
            epoch_old = epoch
            epoch = self._get_epoch(step, train_batch_size, epoch_volume)
            if epoch_to_train and (epoch > epoch_to_train):
                break
            if epoch_old != epoch:
                # test_all(step - 1, epoch_old, True, test_imgs_dir)
                if isinstance(decay_strategy, str):
                    if epoch_old % decay_epoch == 0:
                        lr = self._lr_update(learning_rate, step, epoch, decay, decay_strategy)
                else:
                    lr = self._lr_update(learning_rate, step, epoch, decay, decay_strategy)
            save_flag_final = True
            save_flag_max = False
            if (step % log_print_interval) == 0:
                train_once(step, epoch, pring_log=True)
            else:
                train_once(step, epoch, pring_log=False)
            if (step % test_interval) == 0:
                print('time: train_%d %.6fs' % (test_interval, time.time() - t))
                t = time.time()
                psnr_tmp = test_all(step, epoch, True)
                print('time: test_once %.6fs' % (time.time() - t))
                if psnr_tmp > psnr_max:
                    test_all(step, epoch, False, test_imgs_dir)
                    psnr_max = psnr_tmp
                    print('psnr_max: %.6f epoch: %d step: %d' % (psnr_max, epoch, step))
                    save_flag_max = True
                t = time.time()
            if (step % save_interval) == 0 or save_flag_max:
                t = time.time()
                save_path = save_once(step)
                save_flag_final = False
                save_flag_max = False
                print('time: save_once %.6fs' % (time.time() - t))
                t = time.time()
            if steps_to_run is not None:
                steps_to_run -= 1

        if save_flag_final:
            save_path = save_once(step)
        sess.close()
        print('\nALL DONE.')
        return save_path

    def encode(self,
               ckpt_path,
               source_dir,
               encoded_dir):
        pass

    def decode(self,
               ckpt_path,
               encoded_dir,
               decoded_dir):
        pass

    def evaluate(self,
                 ckpt_path,
                 source_dir,
                 # encoded_dir,
                 decoded_dir):
        # info:
        self.info_evaluate = {}
        self.info_evaluate['ckpt_path'] = str(ckpt_path)
        self.info_evaluate['source_dir'] = str(source_dir)
        self.info_evaluate['decoded_dir'] = str(decoded_dir)
        print('\n\n********** Evaluate **********')
        print_info([self.info_evaluate])
        print('********** ******** **********')
        if not os.path.exists(decoded_dir):
            os.makedirs(decoded_dir)

        # define graph:
        print('\n** Define graph...')
        self.predict_graph = tf.Graph()
        with self.predict_graph.as_default():
            self._build(self.MODE_PREDICT)
            # saver:
            saver_all = tf.train.Saver(max_to_keep=0, name='saver_all')
        print('Done.')

        # init:
        print('\n** Initialize and prepare...')
        sess = tf.Session(graph=self.predict_graph)
        saver_all.restore(sess, ckpt_path)

        # run:
        print('\n** Begin processing:\n')
        exts = {'.jpg', 'jpeg', '.png'}
        filenames = [name for name in os.listdir(source_dir) if os.path.splitext(name)[-1].lower() in exts]
        psnr_uint8_sum = 0
        for img_in_name in filenames:
            img_in_path = os.path.join(source_dir, img_in_name)
            img_out_name = os.path.splitext(img_in_name)[0] + '_decoded.png'
            img_out_path = os.path.join(decoded_dir, img_out_name)
            img_in = cv_imread(img_in_path, 'RGB') / 255
            feed_dic = {
                self.inputs: img_in,
                self.labels: img_in
            }
            psnr_uint8, img_out = sess.run([self.psnr_uint8, self.outputs], feed_dic)
            cv_imwrite(img_out_path, img_out[0], 'RGB')
            psnr_uint8_sum += psnr_uint8
            print('img: %s PSNR-uint8: %.6f' % (img_in_name, psnr_uint8))
        psnr_uint8_mean = psnr_uint8_sum / len(filenames)
        print('\nPSNR-uint8 mean: %.6f' % psnr_uint8_mean)

        sess.close()
        print('\nALL DONE.')


if __name__ == '__main__':
    pass
