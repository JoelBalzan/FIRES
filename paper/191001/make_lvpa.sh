#!/bin/bash

fires -f 191001 -p lvpa --ncpu 12 --config-dir ~/Documents/GitHub/FIRES/paper/191001/ -v  --seed 3 --sefd 1.2 --scint --override-param "tau=1.78" -o /home/joel/Documents/GitHub/FIRES/paper/191001/
fires -f 191001_nn -p lvpa --ncpu 12 --config-dir ~/Documents/GitHub/FIRES/paper/191001/ -v  --seed 3 --scint --override-param "tau=1.78" -o /home/joel/Documents/GitHub/FIRES/paper/191001/
fires -f 191001_nn_ns -p lvpa --ncpu 12 --config-dir ~/Documents/GitHub/FIRES/paper/191001/ -v  --seed 3 --override-param "tau=0" -o /home/joel/Documents/GitHub/FIRES/paper/191001/

fires -f 191001 -p pa --ncpu 12 --config-dir ~/Documents/GitHub/FIRES/paper/191001/ -v  --seed 3 --sefd 1.2 --scint --override-param "tau=1.78" -o /home/joel/Documents/GitHub/FIRES/paper/191001/
fires -f 191001_nn -p pa --ncpu 12 --config-dir ~/Documents/GitHub/FIRES/paper/191001/ -v  --seed 3 --scint --override-param "tau=1.78" -o /home/joel/Documents/GitHub/FIRES/paper/191001/
fires -f 191001_nn_ns -p pa --ncpu 12 --config-dir ~/Documents/GitHub/FIRES/paper/191001/ -v  --seed 3 --override-param "tau=0" -o /home/joel/Documents/GitHub/FIRES/paper/191001/