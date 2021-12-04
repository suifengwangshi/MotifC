#!/usr/bin/bash

Results=${1}
if [ -f ./results/${Results}.txt ]; then
    rm ./results/${Results}.txt
fi

for experiment in $(ls ./${Results}/)
do
    echo "working on ${experiment}."
    
    echo ${experiment} $(cat ./${Results}/${experiment}/record.txt | grep "mean"| awk -F '\t' '{print $2,$3}') >> ./results/${Results}.txt
done
