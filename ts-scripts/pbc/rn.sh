#!/bin/bash
cd /home/maximl/workdir/basic_nn

source venv/bin/activate
python main.py cv configs/pbc/cv-8-resnet18.json ~/runs/simple_nn/pbc/resnet18/d23_early ~/IFC/data/