#! /bin/bash
sleep 5h
python train.py -e 40 -b 32 -l 1e-2 -p 20 -o SGD --augment
python train.py -e 40 -b 32 -l 1e-2 -p 20 -o SGD 
python train.py -e 20 -b 32 -l 1e-2 -p 20 -o SGD -wd 0 -m 0
python train.py -e 20 -b 32 -l 1e-2 -p 20 -o SGD -wd 0
python train.py -e 20 -b 32 -l 1e-2 -p 20 -o SGD -m 0
