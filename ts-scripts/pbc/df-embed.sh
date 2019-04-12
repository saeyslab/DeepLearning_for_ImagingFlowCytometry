#!/bin/bash
cd /home/maximl/workdir/basic_nn

source venv/bin/activate
python main.py embed configs/pbc/embed-deepflow.json . ~/IFC/data/
