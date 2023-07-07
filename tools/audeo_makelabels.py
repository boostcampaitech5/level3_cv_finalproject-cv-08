import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle


import mido
from mido import MidiFile
from copy import deepcopy
import argparse


def main(args):
    midi_path = args.original_label_path
    image_path = args.image_path
    save_path = args.save_label_path

    img_dir_list = sorted(os.listdir(image_path))
    img_dir_list = [os.path.join(image_path, filename) for filename in img_dir_list]
    
    fps = 25
    A1 = 21

    os.makedirs(save_path, exist_ok=True)

    for filename in tqdm(img_dir_list):
        img_len = len(os.listdir(filename))
        filename_withoutdir = filename.split("/")[-1]
        
        # pianoyt
        img_id = filename_withoutdir.split("_")[0]
        mid = MidiFile(os.path.join(midi_path, f'audio_{img_id}.0.midi'))
        
        #miditest
        # img_id = filename_withoutdir.split(".")[0]
        # mid = MidiFile(os.path.join(midi_path, f'{img_id}.mid'))
        
        bpm = mido.tempo2bpm(mid.tracks[0][0].tempo)
        bps = bpm / 60.0
        sec_per_tick = mid.ticks_per_beat * bps

        midi_tracks = mid.tracks[1]
        
        np_roll = np.zeros((img_len, 88))
        
        time_sum = 0
        
        for msg in midi_tracks:
            if hasattr(msg, "note"):
                time_sum += msg.time
                msg.time = time_sum
        
        for msg in midi_tracks:
            if hasattr(msg, "note"):
                msg.note -= A1
                msg.time = int((msg.time / sec_per_tick) * fps)
                
        processed_midi = []
        idx_start, idx_end = 0, 0
        for msg in midi_tracks:
            if hasattr(msg, "note"):
                if msg.velocity != 0:
                    processed_midi.append({"note": msg.note, "velocity": msg.velocity, "start": msg.time, "end": -1})
                else:
                    is_note_end = False
                    for midi_line in processed_midi:
                        if midi_line['note'] == msg.note and midi_line['end'] == -1:
                            midi_line['end'] = msg.time - 1 # 두 note가 붙어버리는 것을 방지
                            is_note_end = True
                            break
                    if not is_note_end:
                        raise Exception
                        
        midi_dict = {}
        
        for i in range(1, img_len+1):
            midi_dict[i] = np.zeros(88)
            
        if args.onset_mode:
            for midi_line in processed_midi:
                np_midi_line = np.zeros(88)
                np_midi_line[midi_line['note']] = midi_line['velocity']
                midi_dict[midi_line['start']] += np_midi_line
        else:
            for midi_line in processed_midi:
                for frame in range(midi_line['start'], midi_line['end']):
                    np_midi_line = np.zeros(88)
                    np_midi_line[midi_line['note']] = midi_line['velocity']
                    midi_dict[frame] += np_midi_line
        
        with open(os.path.join(save_path, filename_withoutdir + ".pkl"), "wb") as f:
            pickle.dump(midi_dict, f)
            
        label_stack = []
        for k, v in midi_dict.items():
            label_stack.append(v)
            
        label_stack = np.stack(label_stack)
        
        for i in range(50, label_stack.shape[0], 50):
            os.makedirs(os.path.join(args.save_midi_path, filename_withoutdir), exist_ok=True)
            np.savez(os.path.join(args.save_midi_path, filename_withoutdir, f"{i-50+1}-{i+1}.npz"), midi=label_stack[i-50:i])
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--original_label_path", type=str, default='../data/PianoYT/pianoyt_MIDI/', help='default: "../data/PianoYT/pianoyt_MIDI/"')
    parser.add_argument("--image_path", type=str, default='../data/ytdataset/images/training/', help="default: '../data/ytdataset/images/training/")
    parser.add_argument("--save_label_path", type=str, default="../data/ytdataset/labels_audeo/training/", help='default : "../data/ytdataset/labels_audeo/training/"')
    parser.add_argument("--save_midi_path", type=str, default="../data/ytdataset/midi/training/", help='default: "../data/ytdataset/midi/training/"')
    parser.add_argument("--onset_mode", action="store_true")
    
    args = parser.parse_args()
    
    main(args)