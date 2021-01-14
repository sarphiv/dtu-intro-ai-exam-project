#!/bin/sh
### Job queue
#BSUB -q hpc 

### Job name
#BSUB -J self-driving-car-reinforce

### Cores to request
#BSUB -n 28

### Force cores to be on same host
#BSUB -R "span[hosts=1]" 

### Amount of memory per core
#BSUB -R "rusage[mem=800MB]" 

### Maximum amount of memory before killing task
#BSUB -M 6GB 

### Wall time (HH:MM), how long before killing task
#BSUB -W 24:00

### Output and error file. %J is the job-id -- 
### -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -oo Output_%J.out 
#BSUB -eo Error_%J.err

module unload python
module load python3
python3 -m pip install --user numpy pygame shapely pandas matplotlib
python3 -m pip install --user torch==1.7.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
python3 'train.py' > output.txt
