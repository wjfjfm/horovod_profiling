mpirun -np 8 -H 172.23.232.139:4,172.23.232.166:4 \
       -bind-to none -map-by slot \
    -x LD_LIBRARY_PATH -x PATH \
    -x NCCL_SOCKET_IFNAME=^lo,docker0 \
    -mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_exclude lo,docker0 \
    -x HOROVOD_MPI_THREADS_DISABLE=1 \
-x NCCL_IB_DISABLE=0 -x NCCL_IB_HCA=mlx5_1 \
-x NCCL_IB_GID_INDEX=0 -x NCCL_IB_CUDA_SUPPORT=1 \
--allow-run-as-root \
python profile.py -o real_model_result_ib -m alexnet --session 1 --step 200 -t