#! /bin/bash
python train.py -e 40 -b 32 -l 1e-2 -m 0 -wd 0 -p 20 -o SGD 
python train.py -e 40 -b 32 -l 1e-2 -wd 0 -p 20 -o SGD 
python train.py -e 40 -b 32 -l 1e-2 -m 0 -p 20  -o SGD 