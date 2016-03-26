#!/bin/bash
#for i in {1..20}
#for i in 1 2 3 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
for i in 1
do
	thisdata=$(printf "res_results_problem_%i" "$i")
	THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python ec2-train-draw-classify.py  --dataset=$thisdata
done
