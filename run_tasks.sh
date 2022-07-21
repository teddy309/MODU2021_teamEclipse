#!/bin/bash

#. conda/bin/activate
#conda activate [가상환경 이름:team#] #임시(lss)

cd scripts
echo run task1:COLA !!
python pipeline_COLA.py #task1:OK

echo run task2:WiC !!
python pipeline_WiC.py #task2

echo run task3:COPA !!
python pipeline_COPA.py #task3:OK

echo run task4:BoolQ !!
python pipeline_BoolQ.py #task4:OK