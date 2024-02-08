#!/bin/csh

#SBATCH -p batch
#SBATCH -o myMPI.o%j
#SBATCH -N 1 -n 20
#SBATCH -t 72:00:00   
#SBATCH --mail-type=END
#SBATCH --mail-user=klkusima@cougarnet.uh.edu 

zacros.x 
