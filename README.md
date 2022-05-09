# SpMM
Graph Algorithms using SpMM on GPU

Author: Bin Ma, Peng Zhou

[GitHub Link](https://link-url-here.org)

## Prerequisites
The following dependencies or above versions are requiredfor this project
```
gcc 8.3
cmake 3.9
cuda 10.1
```

## Getting Started
If running on HPC, the following modules are required to load, use the following commands
```
module load cmake
module load cuda
```

To compile the project, use the following commands
```
mkdir bin
cd bin
cmake ..
make
```

To run project with slurm, create `job.sl` at `bin/main/` directory with the following content
```
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=5:00:00
#SBATCH --partition=gpu 
#SBATCH --output=cpujob.out
#SBATCH --gres=gpu:v100:1

module purge
module load gcc/8.3.0
module load nvidia-hpc-sdk

./main
```

To run the project, go to `bin/main/`

Running on local machine:
```
./main
```

Running on slurm:
```
sbatch job.sl
```
