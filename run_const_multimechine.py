import os
import itertools

nKBs = [pow(2,i) for i in range(21)]
Max_iteration = 128
Max_KB_size = 1024*1024

for nKB in nKBs:
    ite_times = min(Max_iteration, Max_KB_size/nKB)
    commands = []
    commands.append("mpirun -np 8 -H 10.150.144.216:4,10.150.144.218:4 \
       -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -x NCCL_SOCKET_IFNAME=^lo,docker0 \
    -mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_exclude lo,docker0 \
	--allow-run-as-root \
    -x N_KB_PER_TENSOR=%d -x N_ITERATION_TIMES=%d python profile.py -m allreduce_const --horovod -o nKB_%d_ite_%d_gpu_2*4 --session 1 --step 80 -t" % (nKB, ite_times, nKB, ite_times) )

    for cmd in commands:
        try:
            print(cmd)
            os.system(cmd)
        except:
            print("Error while executing:\n  " + cmd)

