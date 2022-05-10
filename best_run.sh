#! /bin/bash
sleep 13h
python train.py -e 60 -b 32 -l 0.01 -p 20 -o SGD -m 0.9 -wd 1e-8