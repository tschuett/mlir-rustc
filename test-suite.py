#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import sys
from os import listdir
from os.path import isfile, join


def test_syntax_only():
    mypath = "testsuite/syntaxonly"
    onlyfiles = [join(mypath, f)
                 for f in listdir(mypath) if isfile(join(mypath, f))]
    print(onlyfiles)
    for f in onlyfiles:
        try:
            return subprocess.run(["tools/rustc/rustc", "--fsyntax-only",
                                   f"--path={f}"], check=True, encoding="utf-8",
                                  stdout=subprocess.PIPE).stdout.strip()
        except subprocess.CalledProcessError as error:
            print("rustc --fsyntax-only failed")
            print("Is this really a rust file?")
            print(error)
            sys.exit(-1)


def test_with_sema():
    mypath = "testsuite/withsema"
    onlyfiles = [join(mypath, f)
                 for f in listdir(mypath) if isfile(join(mypath, f))]
    print(onlyfiles)
    for f in onlyfiles:
        try:
            return subprocess.run(["tools/rustc/rustc", "--fwith-sema",
                                   f"--path={f}"], check=True, encoding="utf-8",
                                  stdout=subprocess.PIPE).stdout.strip()
        except subprocess.CalledProcessError as error:
            print("rustc --fwith-sema failed")
            print("Is this really a rust file?")
            print(error)
            sys.exit(-1)


def main() -> int:
    """The main function."""

    test_syntax_only()
    test_with_sema()

    return 0


if __name__ == '__main__':
    sys.exit(main())
