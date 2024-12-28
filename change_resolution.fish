#!/usr/local/bin/fish

set fps 25
set resolution 640x360

if test -z $argv
    echo "Usage: ffmpeg_scripts.fish <path>"
    exit 1
end

if not test -d $argv
    echo "The path provided is not a directory"
    exit 1
end

cd $argv

for file in *.mp4 
    mkdir -p output
    set output_path output/$file
    ffmpeg -i $file -r $fps -s $resolution $output_path
end

cd -