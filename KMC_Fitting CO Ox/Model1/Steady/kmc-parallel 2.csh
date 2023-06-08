#!/bin/csh

#SBATCH -p batch
#SBATCH -o myMPI.o%j
#SBATCH -N 1 -n 48
#SBATCH -t 72:00:00   #max walltime is 24hr
#SBATCH --mail-type=END
#SBATCH --mail-user=yliu129@uh.edu        #your email id

/home/yliu228/zacros_3.01_parallel/build/zacros.x 
