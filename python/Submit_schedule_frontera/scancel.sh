#!/bin/bash
for param in $(seq $1 1 $2)
do
	scancel $param
done


