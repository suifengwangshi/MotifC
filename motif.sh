#!/usr/bin/bash

Data=${1}

for experiment in $(ls ./${Data}/)
do
    echo "working on ${experiment}."
    if [ ! -d ./motifs/${experiment} ]; then
        mkdir ./motifs/${experiment}
    else
        continue
    fi
    
    python motif_finder.py -d `pwd`/${Data}/${experiment}/data \
                           -n ${experiment} \
                           -g 0 \
                           -c `pwd`/models/${experiment} \
                           -o `pwd`/motifs/${experiment}
done
