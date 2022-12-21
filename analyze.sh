#!/bin/sh

make clean
find . -name abc | xargs -n2 rm
make
./proc_stat_report.py
