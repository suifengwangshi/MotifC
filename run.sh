#!/usr/bin/bash
Datadir=${1}
ModelPath=${2}
PythonFile=${3}
for experiment in $(ls ./$Datadir/)
do
    echo "working on $experiment."
    if [ ! -d ./models/$ModelPath/$experiment ]; then
        mkdir ./models/$ModelPath/$experiment
    else
        continue
    fi

    python $PythonFile -d `pwd`/$Datadir/$experiment/data \
                         -n $experiment \
                         -g 0 \
                         -b 100 \
                         -lr 0.001 \
                         -e 20 \
                         -w 0.0005 \
                         -c `pwd`/models/$ModelPath/$experiment
done
