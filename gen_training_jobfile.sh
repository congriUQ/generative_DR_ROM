# Script that generates and submits a job file to train semi-supervised Stokes flow model
N_SUPERVISED=128
N_UNSUPERVISED=1024
DIM_Z=10
#Set up file paths
DATESTR=`date +%m-%d-%H-%M-%N`	#datestring for jobfolder name
PROJECTDIR="/home/constantin/python/projects/generative_DR_ROM"
# Set JOBNAME by hand for every job!
JOBNAME="train_supervised=${N_SUPERVISED}_unsupervied=${N_UNSUPERVISED}_dim_z=${DIM_Z}_${DATESTR}"
JOBDIR="/home/constantin/python/jobs/$JOBNAME"
JOBSCRIPT="${JOBDIR}/generativeSurrogate.py"

#Create job directory and copy source code
rm -rf $JOBDIR
mkdir $JOBDIR
SOLUTIONSCRIPT="${PROJECTDIR}/generativeSurrogate.py"
cp  $SOLUTIONSCRIPT $JOBSCRIPT
cp -r "$PROJECTDIR/poisson_fem" $JOBDIR
cp "$PROJECTDIR/ROM.py" $JOBDIR
cp "$PROJECTDIR/Data.py" $JOBDIR
cp "$PROJECTDIR/GenerativeSurrogate.py" $JOBDIR
cp "$PROJECTDIR/myutil.py" $JOBDIR


#Change directory to job directory; completely independent from project directory
cd $JOBDIR

#construct job file
echo "#!/bin/bash" >> ./job_file.sh
echo "#SBATCH --job-name=${JOBNAME}" >> ./job_file.sh
echo "#SBATCH --partition batch_SNB,batch_SKL" >> ./job_file.sh
echo "#SBATCH --output=/home/constantin/OEfiles/${JOBNAME}.%j.out" >> ./job_file.sh
echo "#SBATCH --mincpus=24" >> ./job_file.sh
echo "#SBATCH --mail-type=ALL" >> ./job_file.sh
echo "#SBATCH --mail-user=mailscluster@gmail.com " >> ./job_file.sh
echo "#SBATCH --time=1000:00:00" >> ./job_file.sh

#Activate fenics environment and run python
echo "source /home/constantin/.bashrc" >> ./job_file.sh
echo "conda activate genDRROM" >> ./job_file.sh
#echo "while true; do" >> ./job_file.sh
echo "python -Ou ./generativeSurrogate.py ${N_SUPERVISED} ${N_UNSUPERVISED} ${DIM_Z}" >> ./job_file.sh
#echo "done" >> ./job_file.sh
echo "" >> job_file.sh


chmod +x job_file.sh
#directly submit job file
sbatch job_file.sh
#execute job_file.sh in shell
#./job_file.sh
