#!/bin/sh
base_dir='./environment'
if [ $# -gt 0 ]; then
  n_env=$1
else
  n_env=4
fi
for i in `seq $n_env`; do
  dst_dir="${base_dir}_$i"
  echo "rm -r $dst_dir"
  rm -r $dst_dir
  echo "cp -r $base_dir $dst_dir"
  cp -r $base_dir $dst_dir
done
