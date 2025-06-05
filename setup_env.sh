#!/usr/bin/env bash
# Simple script to create and setup a Python virtual environment for this project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

