#!/bin/bash

echo "downloading tf-alex..."
dir=./agent/tfalex
curl -o ${dir}/mynet.npy https://www.dropbox.com/s/caz3uunv267ogks/tf_alexnet.npy
