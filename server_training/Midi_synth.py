import argparse
import glob
import os

import librosa
import numpy as np
import pretty_midi
import soundfile as sf


# Synthesizing Audio using Fluid Synth
class MIDISynth:
    def __init__(self, out_folder, video_name, instrument, out_path, midi=True):
        self.video_name = video_name
        # synthesize midi or roll
        self.midi = midi
        # synthsized output dir, change to your own path
        self.syn_dir = out_path
        self.min_key = 3
        self.max_key = 83
        self.frame = 50
        self.piano_keys = 88
        if self.midi:
            self.midi_out_folder = os.path.join(out_folder, video_name)
            self.syn_dir = os.path.join(self.syn_dir, "w_Roll2Midi/")
            self.process_midi()
        else:
            self.est_roll_folder = os.path.join(out_folder, video_name)
            self.syn_dir = os.path.join(self.syn_dir, "wo_Roll2Midi/")
            self.process_roll()
        self.spf = 0.04  # second per frame
        self.sample_rate = 16000
        self.ins = instrument

    def process_roll(self):
        self.wo_Roll2Midi_data = []
        self.est_roll_files = glob.glob(self.est_roll_folder + "/*.npz")
        self.est_roll_files.sort(key=lambda x: int(x.split("/")[-1].split(".")[0].split("-")[0]))

        # Use the Roll prediction for Synthesis
        print("need to process {0} files".format(len(self.est_roll_folder)))
        for i in range(len(self.est_roll_files)):
            with np.load(self.est_roll_files[i]) as data:
                est_roll = data["roll"]
                if est_roll.shape[0] != self.frame:
                    target = np.zeros((self.frame, self.piano_keys))
                    target[: est_roll.shape[0], :] = est_roll
                    est_roll = target
                est_roll = np.where(est_roll > 0, 1, 0)
            self.wo_Roll2Midi_data.append(est_roll)
        self.complete_wo_Roll2Midi_midi = np.concatenate(self.wo_Roll2Midi_data)
        print(
            "Without Roll2MidiNet, the Roll result has shape:",
            self.complete_wo_Roll2Midi_midi.shape,
        )
        # compute onsets and offsets
        onset = np.zeros(self.complete_wo_Roll2Midi_midi.shape)
        offset = np.zeros(self.complete_wo_Roll2Midi_midi.shape)
        for j in range(self.complete_wo_Roll2Midi_midi.shape[0]):
            if j != 0:
                onset[j][
                    np.setdiff1d(
                        self.complete_wo_Roll2Midi_midi[j].nonzero(),
                        self.complete_wo_Roll2Midi_midi[j - 1].nonzero(),
                    )
                ] = 1
                offset[j][
                    np.setdiff1d(
                        self.complete_wo_Roll2Midi_midi[j - 1].nonzero(),
                        self.complete_wo_Roll2Midi_midi[j].nonzero(),
                    )
                ] = -1
            else:
                onset[j][self.complete_wo_Roll2Midi_midi[j].nonzero()] = 1
        onset += offset
        self.complete_wo_Roll2Midi_onset = onset.T
        print("Without Roll2MidiNet, the onset has shape:", self.complete_wo_Roll2Midi_onset.shape)

    def process_midi(self):
        self.w_Roll2Midi_data = []
        self.infer_out_files = glob.glob(self.midi_out_folder + "/*.npz")
        print(self.midi_out_folder)
        self.infer_out_files.sort(key=lambda x: int(x.split("/")[-1].split(".")[0].split("-")[0]))

        # Use the Midi prediction for Synthesis
        for i in range(len(self.infer_out_files)):
            with np.load(self.infer_out_files[i]) as data:
                est_midi = data["midi"]
                target = np.zeros((self.frame, self.piano_keys))
                target[: est_midi.shape[0], self.min_key : self.max_key + 1] = est_midi
                est_midi = target
                est_midi = np.where(est_midi > 0, 1, 0)
            self.w_Roll2Midi_data.append(est_midi)
        self.complete_w_Roll2Midi_midi = np.concatenate(self.w_Roll2Midi_data)
        print(
            "With Roll2MidiNet Midi, the Midi result has shape:",
            self.complete_w_Roll2Midi_midi.shape,
        )
        # compute onsets and offsets
        onset = np.zeros(self.complete_w_Roll2Midi_midi.shape)
        offset = np.zeros(self.complete_w_Roll2Midi_midi.shape)
        for j in range(self.complete_w_Roll2Midi_midi.shape[0]):
            if j != 0:
                onset[j][
                    np.setdiff1d(
                        self.complete_w_Roll2Midi_midi[j].nonzero(),
                        self.complete_w_Roll2Midi_midi[j - 1].nonzero(),
                    )
                ] = 1
                offset[j][
                    np.setdiff1d(
                        self.complete_w_Roll2Midi_midi[j - 1].nonzero(),
                        self.complete_w_Roll2Midi_midi[j].nonzero(),
                    )
                ] = -1
            else:
                onset[j][self.complete_w_Roll2Midi_midi[j].nonzero()] = 1
        onset += offset
        self.complete_w_Roll2Midi_onset = onset.T
        print("With Roll2MidiNet, the onset has shape:", self.complete_w_Roll2Midi_onset.shape)

    def GetNote(self):
        if self.midi:
            self.w_Roll2Midi_notes = {}
            for i in range(self.complete_w_Roll2Midi_onset.shape[0]):
                tmp = self.complete_w_Roll2Midi_onset[i]
                start = np.where(tmp == 1)[0]
                end = np.where(tmp == -1)[0]
                if len(start) != len(end):
                    end = np.append(end, tmp.shape)
                merged_list = [(start[i], end[i]) for i in range(0, len(start))]
                # 21 is the lowest piano key in the Midi note number (Midi has 128 notes)
                self.w_Roll2Midi_notes[21 + i] = merged_list
        else:
            self.wo_Roll2Midi_notes = {}
            for i in range(self.complete_wo_Roll2Midi_onset.shape[0]):
                tmp = self.complete_wo_Roll2Midi_onset[i]
                start = np.where(tmp == 1)[0]
                end = np.where(tmp == -1)[0]
                if len(start) != len(end):
                    end = np.append(end, tmp.shape)
                merged_list = [(start[i], end[i]) for i in range(0, len(start))]
                self.wo_Roll2Midi_notes[21 + i] = merged_list

    def Synthesize(self):
        if self.midi:
            wav, pm = self.generate_midi(self.w_Roll2Midi_notes, self.ins)
            path = self.create_output_dir()
            out_file = path + f"/Midi-{self.video_name}-{self.ins}.wav"
            out_file_midi = path + f"/Midi-{self.video_name}-{self.ins}.midi"
            sf.write(out_file, wav, self.sample_rate)
            pm.write(out_file_midi)
        else:
            wav, pm = self.generate_midi(self.wo_Roll2Midi_notes, self.ins)
            path = self.create_output_dir()
            out_file = path + f"/Roll-{self.video_name}-{self.ins}.wav"
            out_file_midi = path + f"/Midi-{self.video_name}-{self.ins}.midi"
            sf.write(out_file, wav, self.sample_rate)
            pm.write(out_file_midi)

    def generate_midi(self, notes, ins):
        pm = pretty_midi.PrettyMIDI(initial_tempo=80)
        piano_program = pretty_midi.instrument_name_to_program(ins)  # Acoustic Grand Piano
        piano = pretty_midi.Instrument(program=piano_program)
        for key in list(notes.keys()):
            values = notes[key]
            for i in range(len(values)):
                start, end = values[i]
                note = pretty_midi.Note(
                    velocity=100, pitch=key, start=start * self.spf, end=end * self.spf
                )
                piano.notes.append(note)
        pm.instruments.append(piano)
        wav = pm.fluidsynth(fs=16000)
        return wav, pm

    def create_output_dir(self):
        synth_out_dir = os.path.join(self.syn_dir, self.video_name)
        os.makedirs(synth_out_dir, exist_ok=True)
        return synth_out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--video_name", type=str)
    parser.add_argument(
        "--midi_path",
        type=str,
        default="./outputs_test/r2m_output/",
        help="default='./outputs_test/r2m_output/'",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./outputs_test/Midi_synth/",
        help="default='./outputs_test/Midi_synth/'",
    )
    parser.add_argument("--iter", action="store_true")

    args = parser.parse_args()

    # could select any instrument available in Midi
    instrument = "Acoustic Grand Piano"

    Midi_out_folder = args.midi_path  # Generated Midi output folder, change to your own path
    if args.iter:
        for video_name in os.listdir(Midi_out_folder):
            Synth = MIDISynth(
                Midi_out_folder, video_name, instrument, out_path=args.output_path, midi=True
            )
            Synth.GetNote()
            Synth.Synthesize()
    else:
        video_name = args.video_name
        Synth = MIDISynth(
            Midi_out_folder, video_name, instrument, out_path=args.output_path, midi=True
        )
        Synth.GetNote()
        Synth.Synthesize()
