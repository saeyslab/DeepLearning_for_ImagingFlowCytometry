#!/bin/bash
cd /home/maximl/workdir/basic_nn

source venv/bin/activate
python main.py cv configs/eos/cv-3-deepflow.json ~/runs/simple_nn/eos/deepflow/s12_3chan ~/IFC/data/
