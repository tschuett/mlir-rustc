#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import csv
import sys
import os
from tempfile import NamedTemporaryFile
from tabulate import tabulate


def takeSecond(elem):
    """Take second element for sort."""
    return elem[1]


def takeThird(elem):
    """Take third element for sort."""
    return int(elem[2])


def takeFifth(elem):
    """Take fifth element for sort."""
    return int(elem[4])


def normalize_path(p: str) -> str:
    return os.path.basename(p).removesuffix(".o\"")


def main() -> int:
    """The main function."""

    abc_files = []
    for root, dirs, files in os.walk('.'):
        for name in files:
            if os.path.join(root, name).endswith("/abc"):
                abc_files.append(os.path.join(root, name))

    tf = NamedTemporaryFile()
    with open(tf.name, 'w') as outfile:
        for fname in abc_files:
            with open(fname) as infile:
                text = infile.read()
                outfile.write(text)

    with open(tf.name, newline='') as csvfile:
        proc_stat = csv.reader(csvfile, delimiter=',', quotechar='|')
        proc_stat = list(proc_stat)
        proc_stat.sort(key=takeThird, reverse=True)
        print()
        print("total time:")
        print()
        print(tabulate(proc_stat[:5], headers=[
              "exe", "file", "total time", "user time", "rss"]))

    return 0


if __name__ == '__main__':
    sys.exit(main())
