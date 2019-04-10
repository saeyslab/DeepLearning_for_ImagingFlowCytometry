#!/bin/bash
cd /home/maximl/workdir/basic_nn

source venv/bin/activate
python main.py cv configs/cv-eos-3-resnet18-loweps.json ~/runs/simple_nn/eos/resnet18/cv_s23_loweps ~/IFC/data/
