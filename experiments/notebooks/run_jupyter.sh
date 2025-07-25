#!bin/bash

jupyter notebook --port=8888 --ip=0.0.0.0 --allow-root --NotebookApp.token='' --notebook-dir 'experiments/notebooks'
