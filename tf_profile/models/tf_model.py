from .RNN.lstm import LstmFn, LstmFn_horovod
from .RNN.seq2seq import Seq2seqFn, Seq2seqFn_horovod
from .RNN.nasnet import NasNetFn, NasNetFn_horovod
from .RNN.deepspeech import DeepSpeech2Fn, DeepSpeech2Fn_horovod
from .CNN.CNN_utils import (Dataset,
                            build_convNet,
                            build_convNet_horovod)
import horovod.tensorflow as hvd
import tensorflow as tf


class RNN_Nasnet:
    def make_model(batchsize=None):
        op, loss = NasNetFn(batchsize=batchsize)
        return op, loss


class RNN_Seq2seq:
    def make_model(batchsize=None):
        op, loss = Seq2seqFn(batchsize=batchsize)
        return op, loss


class RNN_Deepspeech:
    def make_model(batchsize=None):
        op, loss = DeepSpeech2Fn(batchsize=batchsize)
        return op, loss


class RNN_Lstm:
    def make_model(batchsize=None):
        op, loss = LstmFn(batchsize=batchsize)
        return op, loss


class CNN_Resnet:
    def make_model(batchsize=None):
        dataset = Dataset("imagenet")
        return build_convNet("resnet50", dataset, batchsize)


class CNN_Vgg16:
    def make_model(batchsize=None):
        dataset = Dataset("imagenet")
        return build_convNet("vgg16", dataset, batchsize)


class CNN_Inception3:
    def make_model(batchsize=None):
        dataset = Dataset("imagenet")
        return build_convNet("inception3", dataset, batchsize)


class CNN_Alexnet:
    def make_model(batchsize=None):
        dataset = Dataset("cifar10")
        return build_convNet("alexnet", dataset, batchsize)


class RNN_horovod_Nasnet:
    def make_model(batchsize=None):
        op, loss = NasNetFn_horovod(batchsize=batchsize)
        return op, loss


class RNN_horovod_Seq2seq:
    def make_model(batchsize=None):
        op, loss = Seq2seqFn_horovod(batchsize=batchsize)
        return op, loss


class RNN_horovod_Deepspeech:
    def make_model(batchsize=None):
        op, loss = DeepSpeech2Fn_horovod(batchsize=batchsize)
        return op, loss


class RNN_horovod_Lstm:
    def make_model(batchsize=None):
        op, loss = LstmFn_horovod(batchsize=batchsize)
        return op, loss


class CNN_horovod_Resnet:
    def make_model(batchsize=None):
        dataset = Dataset("imagenet")
        return build_convNet_horovod("resnet50", dataset, batchsize)


class CNN_horovod_Vgg16:
    def make_model(batchsize=None):
        dataset = Dataset("imagenet")
        return build_convNet_horovod("vgg16", dataset, batchsize)


class CNN_horovod_Inception3:
    def make_model(batchsize=None):
        dataset = Dataset("imagenet")
        return build_convNet_horovod("inception3", dataset, batchsize)


class CNN_horovod_Alexnet:
    def make_model(batchsize=None):
        dataset = Dataset("cifar10")
        return build_convNet_horovod("alexnet", dataset, batchsize)

class Horovod_Allreduce:
    def make_model(batchsize=None):

        hvd.init()
        hvd_op_list = []
        dtype = tf.float32

        data_a = tf.random.uniform(
            [1024, 1024], 0.0, 1.0, dtype=dtype)
        data_b = tf.random.uniform(
            [1024, 1024], 0.0, 1.0, dtype=dtype)
        tensor = data_a + data_b
        for i in range(9):
            tensor = data_a + tensor 
            summed = hvd.allreduce(tensor, average=False)
            hvd_op_list.append(summed)
        final_op = tf.shape_n(hvd_op_list)        

        return final_op, None 

# set the name of available models
_model_name = {
        'nasnet': RNN_Nasnet,
        'seq2seq': RNN_Seq2seq,
        'deepspeech': RNN_Deepspeech,
        'lstm': RNN_Lstm,
        'resnet50': CNN_Resnet,
        'vgg16': CNN_Vgg16,
        'alexnet': CNN_Alexnet,
        'inception3': CNN_Inception3,
        'allreduce': Horovod_Allreduce

    }

_model_name_horovod = {
        # 'nasnet': RNN_horovod_Nasnet,
        'seq2seq': RNN_horovod_Seq2seq,
        'deepspeech': RNN_horovod_Deepspeech,
        'lstm': RNN_horovod_Lstm,
        'resnet50': CNN_horovod_Resnet,
        'vgg16': CNN_horovod_Vgg16,
        'alexnet': CNN_horovod_Alexnet,
        'inception3': CNN_horovod_Inception3,
        'allreduce': Horovod_Allreduce
}


# return if the model available
def exist_model(model, horovod=False):
    if horovod is True:
        if model in _model_name_horovod:
            return True
        else:
            return False
    else:
        if model in _model_name:
            return True
        else:
            return False


# return the op and loss of model
def get_model(model, batchsize=None, horovod=False):
    if horovod is True:
        if model in _model_name_horovod:
            return _model_name_horovod[model].make_model(batchsize=batchsize)
        else:
            raise ValueError('model: %s doesn\'t exist' % str(model))
    else:
        if model in _model_name:
            return _model_name[model].make_model(batchsize=batchsize)
        else:
            raise ValueError('model: %s doesn\'t exist' % str(model))


def get_model_list(horovod=False):
    if horovod is True:
        return _model_name_horovod.keys()
    else:
        return _model_name.keys()
