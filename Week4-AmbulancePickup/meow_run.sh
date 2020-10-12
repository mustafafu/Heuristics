#!/bin/bash

input=$1

output_tmp=`basename $input`

echo > meow_out_$output_tmp

python3 meow_amb.py --input $input --output meow_out_$output_tmp
