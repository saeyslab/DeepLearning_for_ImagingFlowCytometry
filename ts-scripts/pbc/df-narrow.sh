#!/bin/bash
cd /home/maximl/workdir/basic_nn

source venv/bin/activate
python main.py cv configs/pbc/cv-8-deepflow_narrow.json ~/runs/simple_nn/pbc/deepflow_narrow/d23_early ~/IFC/data/
