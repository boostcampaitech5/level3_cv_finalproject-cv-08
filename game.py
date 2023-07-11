import pygame
import mido
# import rtmidi
from midi_file import *
from mido import MidiFile
import numpy as np
from collections import deque
import streamlit as st
from PIL import Image

st.title("rhythm game demo")

pygame.init()

BLACK = [0, 0, 0]
WHITE = [255, 255, 255]

SIZE = [500, 200]
screen = pygame.display.set_mode(SIZE)
pygame.display.set_caption("Python MIDI Program by Wilson Chao")
clock = pygame.time.Clock()

done = False
midifile = MidiFile("classic.wav.midi", clip=True)
result_array = mid2arry(midifile)

note_list_on = deque()
note_list_off = deque()

(col,) = st.columns(1)
image_holder = st.empty()

for i in range(result_array.shape[0]):
    screen.fill(BLACK)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    for n in result_array[i]:
        if n > 0:
            note_list_on.append([n * (SIZE[0] // 88) + 10, 0])
        else:
            note_list_off.append([n * (SIZE[0] // 88) + 10, 0])
    note_list_on.append([-1, -1])
    note_list_off.append([-1, -1])
    step = 0

    while True:
        if note_list_on[step][0] == -1:
            del note_list_on[step]
            break
        if note_list_on[step][1] >= SIZE[1]:
            del note_list_on[step]
        pygame.draw.circle(screen, WHITE, note_list_on[step], 2)
        note_list_on[step][1] += 1
        step += 1

    image_array = pygame.surfarray.array3d(screen)
    image_surface = pygame.surfarray.make_surface(image_array)
    image_array_rgb = np.swapaxes(image_array, 0, 1)
    image = Image.fromarray(image_array_rgb)
    image_holder.image(image)
    pygame.display.flip()

    step = 0
    while True:
        if note_list_off[step][0] == -1:
            del note_list_off[step]
            break
        if note_list_off[step][1] >= SIZE[1]:
            del note_list_off[step]
        pygame.draw.circle(screen, BLACK, note_list_off[step], 2)
        note_list_off[step][1] += 1
        step += 1

    image_array = pygame.surfarray.array3d(screen)
    image_surface = pygame.surfarray.make_surface(image_array)
    image_array_rgb = np.swapaxes(image_array, 0, 1)
    image = Image.fromarray(image_array_rgb)
    image_holder.image(image)
    pygame.display.flip()

    clock.tick(500)

pygame.quit()
