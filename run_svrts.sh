#!/bin/bash
#for i in {1..20}
#for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
maindir=/home/ubuntu/draw_classify
for i in 1
do
	thisdata=$(printf "res_results_problem_%i" "$i")
	modelfile=$(printf "%s_model" "$thisdata")
	THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python ec2-train-draw-classify.py  --dataset=$thisdata
	python ec2_classifier.py  --dataset=$thisdata --model_dir=$maindir --model_file=$thisdata
done
