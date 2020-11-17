import os

models = ['alexnet', 'lstm', 'resnet50', 'vgg16', 'deepspeech', 'seq2seq', 'inception3', 'nasnet']
gpus = [2,4,6,8]

tested = [('alexnet', 2), ('lstm', 2)]
for gpu in gpus:
    for model in models:
        if (model, gpu) in tested:
            continue

        commands = []
        commands.append('ALLREDUCE_MODEL=%s python profile.py -m allreduce --session 1 --step 80 --horovod -n %d -o allreduce_profile/%s_gpu_%d -t' % (model, gpu, model, gpu))

        for cmd in commands:
            try:
                print(cmd)
                os.system(cmd)
            except:
                print("Error while executing:\n  " + cmd)