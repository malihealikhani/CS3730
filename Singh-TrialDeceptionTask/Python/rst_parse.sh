#!/bin/bash
pyexec=../venv/bin/python
echo "Changing to appropriate directory"
cd gCRF_dist/src/
$pyexec parse.py -D ../texts/fileList.txt
