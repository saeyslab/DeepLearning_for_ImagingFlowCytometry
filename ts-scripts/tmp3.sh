#!/bin/bash
cd /home/maximl/workdir/basic_nn

source venv/bin/activate
python main.py cv configs/cv-eos-3-deepflow.json ~/runs/simple_nn/eos/deepflow_narrow/cv_s23_lowepslr_longrun ~/IFC/data/
