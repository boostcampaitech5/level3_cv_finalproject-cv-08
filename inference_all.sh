#! /bin/bash

echo "Video2Roll inference.."
echo "$1"
# python Video2Roll_inference.py --video_name "$1" --img_root "./data/MIDItest/images/testing" --label_root "./data/MIDItest/labels/testing/"
python Video2Roll_inference.py --video_name "$1" --img_root "./data/ytdataset/images/training" --label_root "./data/ytdataset/labels_audeo/training/"
echo "Roll2Midi inference.."
python Roll2Midi_inference.py --video_name "$1"
echo "midi synthesizing.."
python Midi_synth.py --video_name "$1"
