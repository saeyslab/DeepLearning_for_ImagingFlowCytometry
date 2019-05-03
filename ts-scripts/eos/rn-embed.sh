#!/bin/bash
cd /home/maximl/workdir/basic_nn

source venv/bin/activate
python main.py embed configs/eos/embed-resnet18.json . ~/IFC/data/
