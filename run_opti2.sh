#! /bin/bash
python train.py -e 30 -b 32 -l 1e0 -p 2 -o SGD  
python train.py -e 30 -b 32 -l 1e0 -p 20 -o SGD 
python train.py -e 30 -b 32 -l 1e0 -p 50 -o SGD 
python train.py -e 30 -b 32 -l 1e0 -p 100 -o SGD

python train.py -e 20 -b 32 -l 1e-1 -p 2 -o SGD 
python train.py -e 20 -b 32 -l 1e-1 -p 20 -o SGD 
python train.py -e 20 -b 32 -l 1e-1 -p 50 -o SGD 
python train.py -e 20 -b 32 -l 1e-1 -p 100 -o SGD 

python train.py -e 15 -b 32 -l 1e-2 -p 2 -o SGD 
python train.py -e 15 -b 32 -l 1e-2 -p 20 -o SGD 
python train.py -e 15 -b 32 -l 1e-2 -p 50 -o SGD 
python train.py -e 15 -b 32 -l 1e-2 -p 100 -o SGD 

