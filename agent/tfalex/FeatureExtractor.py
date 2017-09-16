import tensorflow as tf
from mynet import AlexNet as MyNet
import numpy as np

DEFAULT_MEAN_IMAGE = './tfalex/ilsvrc_2012_mean.npy'


class FeatureExtractor():
    def __init__(self, sess_name, sess_config, in_size=227, out_dim=9216):

        self.batchsize = 1
        self.out_dim = out_dim
        self.in_size = in_size
        self.outcome = 'pool5'

        print('Building AlexNet')
        self.sess = tf.Session(config=sess_config)

        # TODO: take care of shapes (Batch-Size, W, H, Channel)
        self.x = tf.placeholder(tf.float32, [1, self.in_size, self.in_size, 3])
        # self.y = tf.placeholder(tf.float32, [None, self.])
        self.net = self._build_network()
        self.out = self.net.layers['pool5']

        # Initialize AlexNet
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print('Loading Weights...')
        self.net.load('./tfalex/mynet.npy', self.sess)
        print('Done!')

        # load mean image, mean.shape: (256, 256, 3)
        mean_image = np.load(DEFAULT_MEAN_IMAGE).transpose(1, 2, 0)

        # reshape mean (227, 227, 3)
        cropwidth = 256 - self.in_size
        start = cropwidth // 2
        stop = start + self.in_size
        self.mean_image = mean_image[start:stop, start:stop, :].copy()

    def _build_network(self):
        with tf.name_scope('AlexNet'):
            # remove [15:23] from the network
            mynet = MyNet({'data': self.x})
        return mynet

    def predict(self, data_x):
        # Forwarding
        return self.sess.run(self.out, feed_dict={self.x: data_x})

    def __image_feature(self, camera_image):
        # Feature Extractor per camera_image
        # subtract from mean image
        # TODO: take care of transpose (self.batch, in_size, in_size, 3)
        x_batch = np.ndarray((self.batchsize, self.in_size, self.in_size, 3), dtype=np.float32)
        image = np.asarray(camera_image).astype(np.float32)
        # image = np.asarray(camera_image).transpose(2, 0, 1)[::-1].astype(np.float32)

        image -= self.mean_image

        x_batch[0] = image
        x_data = np.asarray(x_batch)

        # make prediction
        feature = self.predict(x_data).reshape(self.out_dim)
        return feature * 255.0

    def feature(self, observation, image_feature_count=1):
        # called by module.py VVC component

        image_features = []
        depth = []

        for i in range(image_feature_count):
            image_features.append(self.__image_feature(observation["image"][i]))
            depth.append(observation["depth"][i])

        if image_feature_count == 1:
            return np.r_[image_features[0], depth[0]]
        elif image_feature_count == 4:
            return np.r_[image_features[0], image_features[1], image_features[2], image_features[3],
                         depth[0], depth[1], depth[2], depth[3]]
        else:
            print('not supported: number of camera')
            # app_logger.error("not supported: number of camera")
