#!/bin/csh

#SBATCH -p batch
#SBATCH -o myMPI.o%j
#SBATCH -N 2 -n 48
#SBATCH -t 100:00:00   #max walltime is 24hr
#SBATCH --mail-type=END
#SBATCH --mail-user=mpdunn@cougarnet.uh.edu       #your email id

zacros.x 
