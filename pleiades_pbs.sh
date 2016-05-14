#!/bin/bash
pbs_output_folder='/home2/linsleyd/slab_projects/pbs_files'
maindir=/home2/linsleyd/slab_projects/draw_classify

for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
do
	output_file=$(printf "%s/svrt_%i.pbs" "$pbs_output_folder" "$i")


	thisdata=$(printf "res_results_problem_%i" "$i")
	modelfile=$(printf "%s_model" "$thisdata")
	modeldir=$(printf "%s/%s" "$maindir" "$thisdata")

	echo "#!/bin/tcsh" > $output_file
	echo "#PBS -X -l walltime=2:00:00,nodes=gpgpu+1:ppn=1" >> $output_file
	echo "#PBS -N drew_linsley_fmri_connectivity" >> $output_file
	echo $(printf "#PBS -o /home2/linsleyd/slab_projects/pbs_output/output_%i" "$i") >> $output_file
	echo $(printf "#PBS -e /home2/linsleyd/slab_projects/pbs_output/error_%i" "$i") >> $output_file
	echo "#PBS -m abe -M drewlinsley@gmail.com" >> $output_file

	echo "module load base" >> $output_file
	echo "module load imagemagick/6.9.1.7" >> $output_file
	echo "module load hdf5/1.8.15" >> $output_file
	echo "module load blas/1.0" >> $output_file
	echo "module load anaconda/linsleyd" >> $output_file
	echo "module load cuda" >> $output_file

	echo $(printf "cd %s" "$maindir") >> $output_file
	echo $(printf "python ec2-train-draw-classify.py  --dataset=%s" "$thisdata") >> $output_file
	echo $(printf "python ec2_classifier.py  --dataset=%s --model_dir=%s --model_file=%s" "$thisdata" "$modeldir" "$modelfile") >> $output_file

    qsub $output_file
done
