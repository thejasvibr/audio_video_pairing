# -*- coding: utf-8 -*-
"""
Script which looks at the modification time of a set of files
This can then be used to compare and check for matching
audio-video files

Created on Sat Sep 09 09:30:54 2017

@author: tbeleyur
"""
import datetime as dt
from glob import glob
import os

file_path =  'C:\\Users\\tbeleyur\\Documents\\SUMMER_2017_local\\actrack\\2017-09-08\\wav\\'
all_files = glob(file_path+'Mic00*')
mtimes= map(os.path.getmtime,all_files)

conv_to_ymdhms = lambda x : dt.datetime.fromtimestamp(x).strftime('%Y-%m-%d-%H-%M-%S')

ymdhms_times = map(conv_to_ymdhms,mtimes)