import os
from tensorflow.python.platform import gfile
import math

def get_max_length():
  #data_dir = "/media/yy/9a19ad59-dbd6-40b3-8b68-4589aea51b4a1/yy/workspace/kws/aishell/data_aishell/wav/train"
  data_dir = "/media/yy/9a19ad59-dbd6-40b3-8b68-4589aea51b4a1/yy/workspace/kws/hello_lenovo_traindata/file/Audio_files"
  max_time_duration = 0
  less_than_6000 = 0
  #search_path = os.path.join(data_dir, '*','*.wav')
  search_path = os.path.join(data_dir, '*','*','*','*.wav')
  for wav_path in gfile.Glob(search_path):
    length_path = "%s%s"%(wav_path,"_time.txt")
    print("length_path %s wav_path %s"%(length_path,wav_path))
    fd = open(length_path)
    time_duration = 0
    while 1:
      lines = fd.read().splitlines()
      if not lines:
          break
      for line in lines:
        if "Length (seconds):" in line:
          last_position=-1
          while True:
              position=line.find(" ",last_position+1)
              if position==-1:
                  break
              last_position=position
          b=float(line[last_position:])
          time_duration = math.ceil(b) * 1000
          if time_duration > max_time_duration:
              max_time_duration = time_duration
          if time_duration <= 6000:
              less_than_6000 += 1
          print("Length: length_path %s %s %s %s max_time_duration %s less_than_6000 %s"%(length_path,line[last_position:],b,time_duration,max_time_duration,less_than_6000))
  print("max_time_duration %s"%(max_time_duration))



get_max_length()
