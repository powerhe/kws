#!/bin/bash

echo "====="$1

#DOWNLOAD_PATH="/media/yy/9a19ad59-dbd6-40b3-8b68-4589aea51b4a1/yy/workspace/kws/hello_lenovo_traindata/file/Audio_files"

DOWNLOAD_PATH2="/media/yy/9a19ad59-dbd6-40b3-8b68-4589aea51b4a1/yy/workspace/kws/THCHS-30/corpus/data/"
if [ $1 ]; then
    DOWNLOAD_PATH=$1
fi

echo $DOWNLOAD_PATH
"""
wav_paths=$(find $DOWNLOAD_PATH -iname '*.wav')
for wav_path in $wav_paths ; do
    echo ""$wav_path
    dir_path=$(dirname $wav_path)
    file_name=$(basename $wav_path)
    length_path=$dir_path"/"$base"/"$file_name"_time.txt"
    echo $length_path
    sox $wav_path -n stat > $length_path 2>&1
done
"""

wav_paths=$(find $DOWNLOAD_PATH2 -iname '*.wav')
for wav_path in $wav_paths ; do
    echo ""$wav_path
    dir_path=$(dirname $wav_path)
    file_name=$(basename $wav_path)
    length_path=$dir_path"/"$base"/"$file_name"_time.txt"
    echo $length_path
    sox $wav_path -n stat > $length_path 2>&1
done
