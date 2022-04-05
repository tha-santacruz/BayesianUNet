#! /bin/bash

python train.py -e 15 -b 32 -l 1e-2 -p 20 -o SGD --amp True
python train.py -e 15 -b 32 -l 1e-2 -p 20 -o SGD --bilinear True
python train.py -e 15 -b 32 -l 1e-2 -p 20 -o SGD --amp True --bilinear True
