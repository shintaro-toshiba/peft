#!/bin/bash

for file in lsf_scripts/llama3_math/lora/tmp/*.sh; do
    if [ -f "$file" ]; then
        echo "---- ---- ----"
        echo $file
        bsub < $file
        echo
    else
        echo "No matching files found."
    fi
done
