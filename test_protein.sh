#!/bin/bash
i=5
while(($i<=4000))
do
	python eval.py --threshould $i
	i=`expr $i + 5`
done
