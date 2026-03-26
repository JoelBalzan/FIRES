#!/bin/bash

fires -f 240318A_nn_ns -p lvpa --config-dir ~/Documents/GitHub/FIRES/examples/20240318A/ -v --override-param "tau=0" "propagation.scintillation.enable=false" "observation.sefd=0" "observation.target_snr=0"
fires -f 240318A_nn -p lvpa --config-dir ~/Documents/GitHub/FIRES/examples/20240318A/ -v --override-param "observation.sefd=0" "observation.target_snr=0"
fires -f 240318A -p lvpa --config-dir ~/Documents/GitHub/FIRES/examples/20240318A/ -v --override-param "observation.sefd=0" "observation.target_snr=110"
