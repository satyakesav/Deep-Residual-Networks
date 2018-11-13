import tensorflow as tf
from Model import Cifar
from DataReader import load_data, train_valid_split
import os


def configure():
    flags = tf.app.flags

    ### YOUR CODE HERE
    flags.DEFINE_integer('resnet_version', 2, 'the version of ResNet')
    flags.DEFINE_integer('resnet_size', 3, 'n: the size of ResNet-(6n+2) v1 or ResNet-(9n+2) v2')
    flags.DEFINE_integer('batch_size', 20, 'training batch size')
    flags.DEFINE_integer('num_classes', 10, 'number of classes')
    flags.DEFINE_integer('save_interval', 5, 'save the checkpoint when epoch MOD save_interval == 0')
    flags.DEFINE_integer('first_num_filters', 16, 'number of classes')
    flags.DEFINE_float('weight_decay', 2e-4, 'weight decay rate')
    flags.DEFINE_string('modeldir', 'model_v2', 'model directory')
    ### END CODE HERE

    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS


def main(_):
    sess = tf.Session()
    print('---Prepare data...')

    ### YOUR CODE HERE
	# Download cifar-10 dataset from https://www.cs.toronto.edu/~kriz/cifar.html
    data_dir = "cifar-10-batches-py"
    ### END CODE HERE

    x_train, y_train, x_test, y_test = load_data(data_dir)
    x_train_new, y_train_new, x_valid, y_valid = train_valid_split(x_train, y_train)

    model = Cifar(sess, configure())

    ### YOUR CODE HERE
    model.train(x_train, y_train, 40)
    model.test_or_validate(x_test, y_test, [5, 10, 15, 20, 25, 30, 35, 40])
    ### END CODE HERE

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '9'
    tf.app.run()
