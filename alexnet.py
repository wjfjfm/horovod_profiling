import horovod.tensorflow as hvd

print('before init')
hvd.init()
print('after init')