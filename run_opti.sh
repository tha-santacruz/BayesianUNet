#! /bin/bash
python train.py -e 30 -b 32 -l 1e-2 -p 20 -o SGD 
python train.py -e 30 -b 32 -l 1e-2 -p 30 -o SGD 
python train.py -e 30 -b 32 -l 1e-2 -p 40 -o SGD 