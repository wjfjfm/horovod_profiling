import os

models = ['alexnet', 'lstm', 'resnet50', 'vgg16', 'deepspeech', 'seq2seq', 'inception3', 'nasnet']
# gpus = [2,4]
# models = ['nasnet']
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
        commands.append('mpirun -np 8 -H 10.150.144.216:4,10.150.144.218:4 \
       -bind-to none -map-by slot \
    -x LD_LIBRARY_PATH -x PATH \
    -x NCCL_SOCKET_IFNAME=^lo,docker0 \
    -mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_exclude lo,docker0 \
	--allow-run-as-root \
-x ALLREDUCE_MODEL=%s -x ALLREDUCE_ITE=%d python profile.py --horovod -m allreduce --session 1 --step 80 --horovod -o allreduce_profile4/%s_gpu_2*4_ite_%d_size_%.2fMB_total_%.2fMB -t' % (model, ite, model, ite, size*4/1024/1024, size*ite*4/1024/1024  ))

        for cmd in commands:
            try:
                print(cmd)
                os.system(cmd)
            except:
                print("Error while executing:\n  " + cmd)