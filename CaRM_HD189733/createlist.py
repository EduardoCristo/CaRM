#!/bin/python
import os
import sys

paths = {}
for line in sys.stdin:
   path = line.strip()
   paths[path] = path
print (','.join( sorted( paths.keys() )))
