#!/bin/bash

#. conda/bin/activate
#conda activate lss #임시

#echo run myMain.py!!
#$run python main.py

cd scripts
echo run task1:COLA !!
python pipeline_COLA.py #task1:OK
#python trainer.py

#echo run task2:WiC !!
#python pipeline_WiC.py #task2

echo run task3:COPA !!
python pipeline_COPA.py #task3:OK

echo run task4:BoolQ !!
python pipeline_BoolQ.py #task4:OK