import os

# models = ['alexnet', 'lstm', 'resnet50', 'vgg16', 'deepspeech', 'seq2seq', 'inception3', 'nasnet']
# gpus = [2,4]
models = ['nasnet']
gpus = [4]
data_size = 1024 * 1024 * 1024 / 4

for gpu in gpus:
    for model in models:
        with open('allreduce_shape/%s_allreduce_shape.txt' % model, 'r') as f:
            content = f.readlines()
        size = 0
        for line in content:
            size += int(line)
        ite = 1
        while ite * size < data_size:
            ite *= 2
        

        commands = []
        commands.append('ALLREDUCE_MODEL=%s ALLREDUCE_ITE=%d python profile.py -m allreduce --session 1 --step 80 --horovod -n %d -o allreduce_profile4/%s_gpu_%d_ite_%d_size_%.2fMB_total_%.2fMB -t' % (model, ite, gpu, model, gpu, ite, size*4/1024/1024, size*ite*4/1024/1024  ))

        for cmd in commands:
            try:
                print(cmd)
                os.system(cmd)
            except:
                print("Error while executing:\n  " + cmd)