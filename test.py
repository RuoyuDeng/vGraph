from distutils.file_util import write_file
from subprocess import check_output
from subprocess import *
import os

tmp_file = open('result.log',"w")
tmp_file.write("123\n345")
tmp_file.close()


tmp_file = open('result.log',"w")
tmp_file.write("abc")
tmp_file.close()