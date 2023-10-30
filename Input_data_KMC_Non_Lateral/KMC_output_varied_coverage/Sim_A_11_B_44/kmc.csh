#!/bin/csh

#SBATCH -p batch
#SBATCH -o myMPI.o%j
#SBATCH -N 1 -n 48
#SBATCH -t 72:00:00   #max walltime is 24hr
#SBATCH --mail-type=END
#SBATCH --mail-user=klkusima@cougarnet.uh.edu       #your email id

zacros.x 
