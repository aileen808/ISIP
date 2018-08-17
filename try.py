# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 15:52:38 2018

@author: AZhang6
"""

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                     help='an integer for the accumulator',default = 5)
# parser.add_argument('--sum', dest='accumulate', action='store_const',
#                     const=sum, default=max,
#                     help='sum the integers (default: find the max)')

args = parser.parse_args()
print(args.accumulate(args.integers))