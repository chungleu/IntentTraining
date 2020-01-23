import tensorflow as tf
import tensorflow_hub as hub

import logging

USE_1_URL = 'https://tfhub.dev/google/universal-sentence-encoder/1'


#Suppress tensorflow logging messages
tf_logging = tf.compat.v1.logging
tf_logging.set_verbosity(tf_logging.WARN)


class USEEncoder(object):
    """ Wrapper for Universal Sentence Encoder 1"""
    logger = logging.getLogger(__name__)

    def __init__(self, encoder_url=USE_1_URL):
        embed = hub.Module(encoder_url)

        self.sentences = tf.placeholder(dtype=tf.string, shape=[None])
        self.embedding_fun = tf.cast(embed(self.sentences), tf.float32)
        self.sess = tf.Session()
        self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        self.dim = self.sess.run(self.embedding_fun, feed_dict={self.sentences: ["this is a test"]}).shape[1]
        self.logger.info('USE Encoder: {} Ready! dim:{}'.format(encoder_url, self.dim))

    def encode(self, sentences):
        """
        Compute sentence embeddings for a list of sentences

        :param sentences: List of sentences
        :return: numpy.ndarray of shape (num_sentences, encoder.dim)
        """
        return self.sess.run(self.embedding_fun, feed_dict={self.sentences: sentences})