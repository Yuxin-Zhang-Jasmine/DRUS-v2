#!/bin/bash

#SBATCH --job-name=theCurrentJob
#SBATCH --qos=all
#SBATCH --nodelist=budbud022
#SBATCH --cpus-per-task=60
#SBATCH --output=theCurrentJob.out
#SBATCH --error=theCurrentJob.err
#SBATCH --time=540

export config=        # pathToDRUS-v2/configs/imagenet_256_3c.yml
export modelPath=     # vitro | vivo | CAROTIDcross  
export MATLAB_PATH=   # pathToObservation_SVDresults/ 

# activate micromamba
source ~/.bashrc
micromamba activate ddrm

# debug an virtual environment library error
echo $LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/micromamba/yzhang2018@ec-nantes.fr/envs/ddrm/lib/
echo $LD_LIBRARY_PATH

# launch the python script
python -u pathToDRUS-v2/main.py 
--ni 
--config $config  
--doc $modelPath  
--matlab_path $MATLAB_PATH
--ckpt         # the name of the selected ckpt in the modelPath  
--timesteps    # 50 performs well 
--deg          # DRUSdeno | DRUS | HtH
--image_folder # restoredImagesFolderName

