import pygame
import mido
from midi_file import *
from mido import MidiFile
import numpy as np
from collections import deque
from PIL import Image
import cv2
import os

"""
# 120 beat per minute / temp0 : 50,000 microseconds per bit
# 1초당 2개 비트가 들어감
# 1beat 당 220개의 ticks
# 1초당 440개의 ticks가 들어감
"""

"""
# 웹을 통해서 들어오는 비디오 건 1초당 60 ticks 인듯하다.
# 
"""
"""
# 1초당 440개 ticks를 화면에 표시하면 음표 막대가 너무너무 길게 나오기 때문에
# 원래 1초당 440 ticks였는데, 1초당 22 ticks로 변경하였다.
"""

"""
# 음표가 바닦에 떨어지는게 화면에 음표가 보이는 것보다 느려야하기 때문에, 음악과 동시에 
# 보이는 효과가 나게하려면 음악을 1~2초 늦게 play하도록 하면 될 것이다.
"""

"""
# 검은 건반은 막대가 얇게, 하얀 건반은 막대 두껍게 영상에서는 했다....
# 음.. 이건 어떻게 처리..?
# [WBW ~~~~ W], WBWBWWBWBWBW*7 => 88개
# W=1, B=0
# 흰 건반은 15, 검은 건반은 5로 처리하자(가로 길이)
"""

"""
아직 음악 시간이랑 건반 떨어지는 속도를 맞추는 것은 해결하지 못함...    
"""

def video(midi):
    pygame.init()
    
    BLACK = [0, 0, 0]
    WHITE = [255, 255, 255]
    GREEN = [124, 252, 0]
    RED = [255, 160, 122]

    SIZE = [765, 380]
    IMAGE_SIZE = [975, 50]
    BAR_SIZE = [15, 10]
    keyboard = [1, 0, 1] + [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1] * 7 + [1]
    BAR_X = []
    BAR_Y = 15  # 15
    REFRESH_GAP = 15  # 10
    FRAME = 500
    TICKS_PER_SECOND = 1
    VIDEO_FRAME = 25
    
    tmp = -15
    for k in keyboard:
        if k == 1:
            tmp += BAR_SIZE[0]
            BAR_X.append(tmp)
        else:
            BAR_X.append(tmp + 10)

    background = pygame.image.load("./universe.png")
    screen = pygame.display.set_mode(SIZE, flags=pygame.HIDDEN)
    
    pygame.display.set_caption("Test")

    # 게임 tick설정하는 부분
    clock = pygame.time.Clock()

    # midi file 불러오고, numpy로 변경하는 부분
    done = False
    # midifile = MidiFile("classic.wav.midi", clip=True)
    # midifile = MidiFile("GT.midi", clip=True)
    # result_array = mid2arry(midifile)
    result_array = midi
    note_list_on = deque()

    # pathOut = "./video.mov"
    # out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*"mp4v"), 22, SIZE)

    pathOut = "./video.mov"
    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FRAME, SIZE)

    # pygame.mixer.music.load("GT.midi")
    # pygame.mixer.music.play()
    for i in range(0, result_array.shape[0] + VIDEO_FRAME*3, TICKS_PER_SECOND):
        screen.blit(background, (0, 0))
        start = -15
        for ii in keyboard:
            if ii == 1:
                start += 15
                pygame.draw.rect(
                    screen,
                    WHITE,
                    pygame.Rect(start, SIZE[1] - 50, BAR_SIZE[0], 50),
                    width=1,
                )
            else:
                pygame.draw.rect(
                    screen,
                    WHITE,
                    pygame.Rect(start + 10, SIZE[1] - 50, BAR_SIZE[1], 20, width=1),
                )
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                
        if i < result_array.shape[0]:
            for step, n in enumerate(result_array[i]):
                if n > 0:
                    # black key
                    if keyboard[step] == 0:
                        note_list_on.append(
                            # x, y, key_thick, color
                            [BAR_X[step], 0, BAR_SIZE[1], GREEN]
                        )
                    # white key
                    else:
                        note_list_on.append([BAR_X[step], 0, BAR_SIZE[0], WHITE])
        
        len_on = len(note_list_on)
        for idx in range(len_on):
            note = note_list_on.popleft()
            # 피아노 버튼이 눌렸을때 작동하는 로직
            if note[1] == SIZE[1] - IMAGE_SIZE[1] - BAR_Y:
                pygame.draw.rect(
                    screen,
                    # WHITE if note[2] == 15 else GREEN,
                    RED,
                    pygame.Rect(
                        note[0],
                        note[1] + BAR_Y,
                        note[2],
                        50 if note[2] == 15 else 20,
                    ),
                    width=2,
                )
            if note[1] < SIZE[1] - IMAGE_SIZE[1]:
                pygame.draw.rect(
                    screen, note[3], pygame.Rect(note[0], note[1], note[2], BAR_Y)
                )
                note_list_on.append([note[0], note[1] + REFRESH_GAP, note[2], note[3]])

        pygame.display.flip()
        image_array = pygame.surfarray.array3d(screen)
        # image shape : [*SIZE, 3]
        image = np.swapaxes(image_array, 0, 1)
        out.write(image)

        clock.tick(FRAME)
          
    out.release()
    pygame.quit()

    os.system("ffmpeg -i video.mov -vcodec libx264 video.mp4 -y")
    
if __name__=="__main__":
    result_array = np.load('./dump.npy')
    video(result_array)
        