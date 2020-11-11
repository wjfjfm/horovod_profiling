# profiling tool

Profiling tool provide a Reproducible, Easy-to-use, Reusable tool to do profiling to some commonly used CNN and RNN models.

It can ouput the time-use and loss by step, metadata of tfprof and tensorflow timeline. you can customize the session times and step times, and you can dicide if the horovod and multi-GPUs turn on or off.

## Code Structure

### File Structure

- `profiling` : project to do real device profiling
  - `profile.py`: the command line tool to use directly
  - `README.md`: the README document of this project
  - `Dockerfile`: the Docker to run this project
  - `tests`: the unitest programs
  - `tf_profile`: the source code of tensorflow profiling
    - `tf_profile.py` : the python API used by profile.py to do profiling
    - `models` : the module to call different models
      - `tf_model.py` : the python API used by tf_profile.py to call models
      - `RNN` : the source code of RNN models
      - `CNN` : the source code of CNN models

### Program Structure

![program_structure](README.assets/profiling_program_structure.png
)

## Installation

### Install by Dockerfile

Using Docker to get a quickstart.

```bash
sudo docker build -t profile .
sudo docker run profile
```

### Install on Native Machine

Install OpenMPI which is the base of horovod

```bash
sudo apt-get install -y openmpi-bin openmpi-doc libopenmpi-dev openssh-client
```

Install tensorflow and horovod.

```bash
python -m pip install tensorflow==1.15
python -m pip install horovod
```

## Quickstart

To get all models’ Single GPU time of steps *(-t)*, set output folder *(-o)* to ./output/  (default 10 sessions 600 steps)

```bash
mkdir output
python profile.py –o output -t
```

To get the list of models available

```bash
python profile.py --list
```

To get resnet50’s *(-m)* tfprof *(-p)* and timeline *(-timeline)* metadata, use 1 session and 20 step

```bash
python profile.py –m resnet50 –p –-timeline --session 1 --step 20
```

To get all models' computation graphs *(-g)* and set output folder *(-o)* to ./graphs/, don't need to run model *(--session 0)*

```bash
mkdir graphs
python profile.py -o graphs --graph --session 0
```

Two way to use horovod to run the model:

To get all models’ 4 GPUs horovod performance

- let the python script to setup horovodrun:
  
  ```bash
  python profile.py --horovod -n 4
  ```

- setup horovodrun manually

  ```bash
  horovodrun -np 4 -H localhost:4 python profile.py –o
  ```

To to the allreduce test:

```bash
N_KB_PER_TENSOR=64 python profile.py -m allreduce --horovod -n 8 -o nKB_64_gpu_8 --session 1 --step 55 -t
N_KB_PER_TENSOR=64 python profile.py -m allreduce --horovod -n 8 -o nKB_64_gpu_8 --graph
N_KB_PER_TENSOR=64 python profile.py -m allreduce --horovod -n 8 -o nKB_64_gpu_8 --session 1 --step 10 --timeline1
```

## Usage

use a cmd-line tool `python profile.py` to do the profiling

- `-h` or `--help`: get the hlep information

- `--list`: list out all the models available

- `-m <model name1 name2>`
  - select which model to be run
  - default all models

- `--horovod`
  - open horovod mode by force
  - horovod will auto open if use horovodrun with GPUs > 1

- `-n`
  - set the gpu_num by force

- `-o <folder_path>`
  - set the output folder of the *.csv* and other output
  - (default to *-o ./*)

- `-g` or `--graph`
  - let the program to output the computing graphs of model
  - will be saved in `-o`'s folder
  - **notice:** even works when *--session 0*, **no** influence to performance

- `-t` or `--time`
  - let the program to output time of steps
  - *.csv* default to *model_GPUs_horovod.csv*
  - will be saved in `-o`'s folder

- `-l` or `--loss`
  - let the program to output loss of steps
  - will be saved in `-o`'s folder
  - **notice:** this process may have a little influence to performance

- `-p` or `--tfprof`
  - let the program to output the *tfprof* metadata
  - will be saved in `-o`'s folder
  - **notice:** this process may have influence to performance

- `--timeline`
  - let the program to output the timeline metadata
  - will be saved in `-o`'s folder
  - **notice:** this process may have influence to performance

- `--session <session_num>`
  - set the *session_num per model* of the profiling
  - default to 10

- `--step <step_num>`
  - set the *step_num per session*
  - default to 600

## Models Information

- `VGG16`
  - **default batchsize**: 64
  - **dataset**: imagenet (all one) [224,224,3]
- `resnet50`
  - **default batchsize**: 32
  - **dataset**: imagenet (all one) [224,224,3]
- `inception3`
  - **default batchsize**: 32
  - **dataset**: imagenet (all one) [299,299,3]
- `nasnet`
  - **default batchsize**: 5
  - **dataset**: imagenet (random) [224,224,3]
- `alexnet`
  - **default batchsize**: 128
  - **dataset**: cifar10 (all one) [32,32,3]
- `lstm`
  - **default batchsize**: 16
- `seq2seq`
  - **default batchsize**: 16
- `deepspeech`
  - **default batchsize**: 16

