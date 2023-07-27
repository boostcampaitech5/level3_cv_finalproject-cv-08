import numpy as np
import pretty_midi

# Synthesizing Audio using Fluid Synth
class MIDISynth():
    def __init__(self, roll, midi, frame, ins, is_midi=False):
        self.frame = frame
        self.piano_keys = 88
        self.spf = 0.04 # second per frame
        self.sample_rate = 16000
        self.ins = ins

        self.roll = roll
        self.midi = midi
        
        self.is_midi = is_midi

    def process_roll(self):
        self.wo_Roll2Midi_data = []

        est_roll = self.roll        
        if est_roll.shape[0] != self.frame:
            target = np.zeros((self.frame, self.piano_keys))
            target[:est_roll.shape[0], :] = est_roll
            est_roll = target
            
        est_roll = np.where(est_roll > 0, 1, 0)
            
        self.wo_Roll2Midi_data.append(est_roll)

        self.complete_wo_Roll2Midi_midi = np.concatenate(self.wo_Roll2Midi_data)
        
        # compute onsets and offsets
        onset = np.zeros(self.complete_wo_Roll2Midi_midi.shape)
        offset = np.zeros(self.complete_wo_Roll2Midi_midi.shape)
        
        for j in range(self.complete_wo_Roll2Midi_midi.shape[0]):
            if j != 0:
                onset[j][np.setdiff1d(self.complete_wo_Roll2Midi_midi[j].nonzero(),
                                      self.complete_wo_Roll2Midi_midi[j - 1].nonzero())] = 1
                offset[j][np.setdiff1d(self.complete_wo_Roll2Midi_midi[j - 1].nonzero(),
                                       self.complete_wo_Roll2Midi_midi[j].nonzero())] = -1
            else:
                onset[j][self.complete_wo_Roll2Midi_midi[j].nonzero()] = 1

        onset += offset
        self.complete_wo_Roll2Midi_onset = onset.T
        
        self.GetNote()
        wav, pm = self.generate_midi()
        return wav, pm

    def process_midi(self):
        self.w_Roll2Midi_data = []

        est_midi = self.midi
        if est_midi.shape[0] != self.frame:
            target = np.zeros((self.frame, self.piano_keys))
            target[:est_midi.shape[0], :] = est_midi
            est_midi = target
        
        est_midi = np.where(est_midi > 0, 1, 0)
        
        self.w_Roll2Midi_data.append(est_midi)
        
        self.complete_w_Roll2Midi_midi = np.concatenate(self.w_Roll2Midi_data)

        # compute onsets and offsets
        onset = np.zeros(self.complete_w_Roll2Midi_midi.shape)
        offset = np.zeros(self.complete_w_Roll2Midi_midi.shape)
        
        for j in range(self.complete_w_Roll2Midi_midi.shape[0]):
            if j != 0:
                onset[j][np.setdiff1d(self.complete_w_Roll2Midi_midi[j].nonzero(),
                                      self.complete_w_Roll2Midi_midi[j - 1].nonzero())] = 1
                offset[j][np.setdiff1d(self.complete_w_Roll2Midi_midi[j - 1].nonzero(),
                                       self.complete_w_Roll2Midi_midi[j].nonzero())] = -1
            else:
                onset[j][self.complete_w_Roll2Midi_midi[j].nonzero()] = 1
                
        onset += offset
        self.complete_w_Roll2Midi_onset = onset.T
        
        self.GetNote()
        wav, pm = self.generate_midi()
        return wav, pm

    
    def GetNote(self):
        if self.is_midi:
            self.w_Roll2Midi_notes = {}
            for i in range(self.complete_w_Roll2Midi_onset.shape[0]):
                tmp = self.complete_w_Roll2Midi_onset[i]
                start = np.where(tmp==1)[0]
                end = np.where(tmp==-1)[0]
                if len(start) != len(end):
                    end = np.append(end, tmp.shape)
                merged_list = [(start[i], end[i]) for i in range(0, len(start))]
                # 21 is the lowest piano key in the Midi note number (Midi has 128 notes)
                self.w_Roll2Midi_notes[39 + i] = merged_list
        else:
            self.wo_Roll2Midi_notes = {}
            for i in range(self.complete_wo_Roll2Midi_onset.shape[0]):
                tmp = self.complete_wo_Roll2Midi_onset[i]
                start = np.where(tmp==1)[0]
                end = np.where(tmp==-1)[0]
                if len(start)!=len(end):
                    end = np.append(end, tmp.shape)
                merged_list = [(start[i], end[i]) for i in range(0, len(start))]
                self.wo_Roll2Midi_notes[39 + i] = merged_list
            
        
    def generate_midi(self):
        if self.is_midi:
            notes, ins = self.w_Roll2Midi_notes, self.ins
        else:
            notes, ins = self.wo_Roll2Midi_notes, self.ins
        
        pm = pretty_midi.PrettyMIDI(initial_tempo=80)
        piano_program = pretty_midi.instrument_name_to_program(ins) # Acoustic Grand Piano
        piano = pretty_midi.Instrument(program=piano_program)
        for key in list(notes.keys()):
            values = notes[key]
            for i in range(len(values)):
                start, end = values[i]
                note = pretty_midi.Note(velocity=100, pitch=key, start=start * self.spf, end=end * self.spf)
                piano.notes.append(note)
        pm.instruments.append(piano)
        wav = pm.fluidsynth(fs=16000)
        
        return wav, pm