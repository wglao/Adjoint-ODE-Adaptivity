#!/bin/bash
# Information about the parameters
# $1: number of nodes
# $2: number of GPUs per nodes: Lonestar6: 3, Frontera/maverick2: 4

rm *.e*
rm *.o*
rm arguement_files*

for param in $(seq 1 1 $1)
do
	python Generating_argurment_files.py --files $param --GPU_per_node $2
	sbatch ./main_serial.sh $param $PWD
done


