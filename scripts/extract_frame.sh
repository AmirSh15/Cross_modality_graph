#!/bin/bash
output_address="/media/amir_shirian/Amir/Datasets/Sound_Recognition/AudioSet"
input_dir="Train_balanced"
input_address="/media/amir_shirian/Amir/Datasets/Sound_Recognition/AudioSet/"$input_dir
if [ ! -d $output_address"/"$input_dir"_video_imgs"  ];then
	mkdir $output_address"/"$input_dir"_video_imgs"
fi

for dir in "$input_address"/*/""
do
for file in "$dir"*.mp4""
do
	name=$(basename "$file" .mp4)
	echo "$name"
	PTHH=$output_address"/"$input_dir"_video_imgs/"$name
	DIR_ARRAY=(${dir//// })

	if [ ${DIR_ARRAY[-2]} != $input_dir ];then
	  folder_name=""
    i=-1
    while true
	  do
	    echo $i "${DIR_ARRAY[${i}]}"
	    if [ "${DIR_ARRAY[${i}]}" != $input_dir ];then
        folder_name="${DIR_ARRAY[${i}]} "$folder_name
        i=$((i-1))
      else
        break
      fi
	  done
	  folder_name="${folder_name:0:-1}"
  else
    folder_name=${DIR_ARRAY[-1]}
  fi

  PTHH=$output_address"/"$input_dir"_video_imgs/"$folder_name"/"$name
  echo $PTHH

	if [ ! -d "$PTHH"  ];then
		mkdir -p "$PTHH"
	fi

	ffmpeg -i "$file" -f image2 -vf fps=30 -qscale:v 2 "$PTHH/img_%05d.jpg"
done
done