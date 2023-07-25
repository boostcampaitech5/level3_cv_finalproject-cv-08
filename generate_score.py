from music21 import converter, stream, chord, clef, instrument, note

def generate_two_hand_score(output_roll_midi_path):
    roll_score = converter.parse(output_roll_midi_path)
    pitch_threshold = 80

    right_hand_score = stream.Score()
    left_hand_score = stream.Score()
    for element in roll_score.chordify().recurse():

        if isinstance(element, chord.Chord):
            if element.quarterLength > 4.0:
                element.augmentOrDiminish(4.0 / (element.quarterLength + 1e-5), inPlace=True)                    
            notes = [pitch.midi for pitch in element.pitches]
            avg_note = sum(notes) // len(notes)
            if avg_note < pitch_threshold:
                left_hand_score.append(element)
                right_hand_score.append(note.Rest(quarterLength=element.quarterLength))
            else:
                right_hand_score.append(element)
                left_hand_score.append(note.Rest(quarterLength=element.quarterLength))

    left_hand_score.makeMeasures(inPlace=True)
    left_hand_score.insert(0, instrument.Piano())

    right_hand_score.makeMeasures(inPlace=True)
    right_hand_score.insert(0, instrument.Piano()) 

    right_hand_part = stream.Part()
    right_hand_part.append(clef.TrebleClef())
    right_hand_part.append(right_hand_score)

    left_hand_part = stream.Part()
    left_hand_part.append(clef.BassClef())
    left_hand_part.append(left_hand_score)

    combined_score = stream.Score()
    combined_score.insert(0, right_hand_part)
    combined_score.insert(0, left_hand_part)
    return combined_score

def generate_score(output_roll_midi_path):
    roll_score = converter.parse(output_roll_midi_path)
    score = stream.Score()
    for element in roll_score.chordify().recurse():
        if isinstance(element, chord.Chord):
            if element.quarterLength > 4.0:
                element.augmentOrDiminish(4.0 / (element.quarterLength + 1e-5), inPlace=True)            
        score.append(element)
    
    return score
