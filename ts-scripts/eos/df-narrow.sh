#!/bin/bash
cd /home/maximl/workdir/basic_nn

source venv/bin/activate
python main.py cv configs/eos/cv-3-deepflow_narrow.json ~/runs/simple_nn/eos/deepflow_narrow/s23 ~/IFC/data/
