#!/bin/bash

#SBATCH --job-name=longDRUS
#SBATCH --qos=all
#SBATCH --nodelist=budbud015
#SBATCH --cpus-per-task=60
#SBATCH --output=longDRUS.out
#SBATCH --error=longDRUS.err
#SBATCH --time=500

export config=/home/yzhang2018@ec-nantes.fr/TMI2023/DRUS-v2/configs/imagenet_256_3c.yml
export modelPath=vivo  
export MATLAB_PATH=/home/yzhang2018@ec-nantes.fr/TMI2023/Observation_SVDresults/

# activate micromamba
source ~/.bashrc
micromamba activate ddrm


# launch python script
# python -c "from PIL import Image; print('ok')" #no problem
echo $LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/micromamba/yzhang2018@ec-nantes.fr/envs/ddrm/lib/
echo $LD_LIBRARY_PATH

python -u /home/yzhang2018@ec-nantes.fr/TMI2023/DRUS-v2/main.py --ni --config $config  --doc $modelPath  --ckpt model004000.pt --matlab_path $MATLAB_PATH --timesteps 50 --deg DRUS --image_folder longDRUS

