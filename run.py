import os
import itertools

nKBs = [pow(2,i) for i in range(11)]
gpus = [2, 4, 6, 8]
Max_iteration = 8192
Max_KB_size = 1024*1024*8

for nKB, gpu in itertools.product(nKBs, gpus):
    ite_times = min(Max_iteration, Max_KB_size/nKB)
    commands = []
    commands.append("N_KB_PER_TENSOR=%d N_ITERATION_TIMES=%d python profile.py -m allreduce --horovod -n %d -o nKB_%d_ite_%d_gpu_%d --session 1 --step 80 -t" % (nKB, ite_times, gpu, nKB, ite_times, gpu) )
    commands.append("N_KB_PER_TENSOR=%d N_ITERATION_TIMES=%d python profile.py -m allreduce --horovod -n %d -o nKB_%d_ite_%d_gpu_%d --graph" % (nKB, ite_times, gpu, nKB, ite_times, gpu))
    commands.append("N_KB_PER_TENSOR=%d N_ITERATION_TIMES=%d python profile.py -m allreduce --horovod -n %d -o nKB_%d_ite_%d_gpu_%d -session 1 --step 10 --timeline1" % (nKB, ite_times, gpu, nKB, ite_times, gpu))

    for cmd in commands:
        try:
            print(cmd)
            os.system(cmd)
        except:
            print("Error while executing:\n  " + cmd)

