import pygame
import mido
from mido import MidiFile
import numpy as np
from collections import deque
from PIL import Image
import cv2
import os
import random

# test url : https://youtu.be/_3qnL9ddHuw

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
        
class CONSTANT:
    BLACK = [0, 0, 0]
    WHITE = [255, 255, 255]
    GREEN = [124, 252, 0]
    RED = [255, 160, 122]
    BLUE = [4, 46, 255]
    SKY = [160, 211, 249]
    GRAY = [128, 128, 128]
    BASE_KEYBOARD_ALPHA = 255
    MATCH_KEYBOARD_ALPHA = 128
    
    SIZE = [765, 380]
    SCREEN = pygame.display.set_mode(SIZE, flags=pygame.HIDDEN)
    IMAGE_SIZE = [975, 50]
    BAR_SIZE = [15, 10]
    keyboard_coor = [1, 0, 1] + [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1] * 7 + [1]
    BAR_X = []
    BAR_Y = 15  # 15
    REFRESH_GAP = 15  # 10
    FRAME = 500
    TICKS_PER_SECOND = 1
    VIDEO_FRAME = 25
    
    WHITE_BAR_COLOR = WHITE
    BLACK_BAR_COLOR = BLACK
    BACKGROUND_COLOR = GRAY
    
tmp = -15
for k in CONSTANT.keyboard_coor:
    if k == 1:
        tmp += CONSTANT.BAR_SIZE[0]
        CONSTANT.BAR_X.append(tmp)
    else:
        CONSTANT.BAR_X.append(tmp + 10)
        
        
class Keyboard:
    color = {
        "RED": [255, 160, 122, CONSTANT.MATCH_KEYBOARD_ALPHA],
        "BLUE": [4, 46, 255, CONSTANT.MATCH_KEYBOARD_ALPHA],
        "BLACK": [0, 0, 0, CONSTANT.MATCH_KEYBOARD_ALPHA],
        "WHITE": [255, 255, 255, CONSTANT.MATCH_KEYBOARD_ALPHA],
        "GRAY": [128, 128, 128, CONSTANT.MATCH_KEYBOARD_ALPHA],
    }

    def __init__(self, x, y, color: str, kind: int):
        self.x = x
        self.y = y
        self.kind = kind
        self.max_surf_size = [[15, 50], [10, 20]]
        self.surf = pygame.Surface(self.max_surf_size[kind], pygame.SRCALPHA)
        self.color = Keyboard.color[color]

    def draw(self):
        pygame.draw.rect(
            self.surf,
            self.color,
            pygame.Rect(0, 0, self.surf.get_width(), self.surf.get_height())
        )
        CONSTANT.SCREEN.blit(self.surf, [self.x, self.y])

    def update(self):
        pass


class baseboard:
    alpha = 128
    color = {"WHITE": [255, 255, 255, CONSTANT.BASE_KEYBOARD_ALPHA], "BLACK": [0, 0, 0, CONSTANT.BASE_KEYBOARD_ALPHA]}

    def __init__(self):
        pass

    def draw(self):
        start = -15
        for k in CONSTANT.keyboard_coor:
            if k == 1:
                start += 15
                surf = pygame.Surface((CONSTANT.BAR_SIZE[0], 50), pygame.SRCALPHA)
                pygame.draw.rect(
                    surf,
                    baseboard.color["WHITE"],
                    pygame.Rect(0, 0, CONSTANT.BAR_SIZE[0], 50),
                )
                CONSTANT.SCREEN.blit(surf, [start, CONSTANT.SIZE[1] - 50])

                pygame.draw.rect(
                    surf,
                    CONSTANT.BLACK,
                    pygame.Rect(0, 0, surf.get_width(), surf.get_height()),
                    width=1,
                )
                CONSTANT.SCREEN.blit(surf, [start, CONSTANT.SIZE[1] - 50])
                
        start = -15
        for k in CONSTANT.keyboard_coor:
            if k == 1:
                start += 15
            else:
                surf = pygame.Surface((CONSTANT.BAR_SIZE[1], 20), pygame.SRCALPHA)
                pygame.draw.rect(
                    surf,
                    baseboard.color["BLACK"],
                    pygame.Rect(0, 0, CONSTANT.BAR_SIZE[1], 20),
                )
                CONSTANT.SCREEN.blit(surf, [start + 10, CONSTANT.SIZE[1] - 50])


def video(midi):
    pygame.init()    
    pygame.display.set_caption("Test")

    # 게임 tick설정하는 부분
    clock = pygame.time.Clock()

    # midi file 불러오고, numpy로 변경하는 부분
    done = False
    result_array = midi
    note_list_on = deque()

    pathOut = "./data/outputs/video.mov"
    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*"mp4v"), CONSTANT.VIDEO_FRAME, CONSTANT.SIZE)
    
    keyboard = baseboard()
    for i in range(0, result_array.shape[0] + CONSTANT.VIDEO_FRAME*3, CONSTANT.TICKS_PER_SECOND):
        CONSTANT.SCREEN.fill(CONSTANT.BACKGROUND_COLOR)
        
        keyboard.draw()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                
        if i < result_array.shape[0]:
            for step, n in enumerate(result_array[i]):
                if n > 0:
                    # black key
                    if CONSTANT.keyboard_coor[step] == 0:
                        note_list_on.append(
                            # x, y, key_thick, color
                            [CONSTANT.BAR_X[step], 0, CONSTANT.BAR_SIZE[1], CONSTANT.BLACK_BAR_COLOR]
                        )
                    # white key
                    else:
                        note_list_on.append([CONSTANT.BAR_X[step], 0, CONSTANT.BAR_SIZE[0], CONSTANT.WHITE_BAR_COLOR])
        
        len_on = len(note_list_on)
        for idx in range(len_on):
            note = note_list_on.popleft()
            # 피아노 버튼이 눌렸을때 작동하는 로직
            if note[1] == CONSTANT.SIZE[1] - CONSTANT.IMAGE_SIZE[1] - CONSTANT.BAR_Y:
                key = Keyboard(
                    note[0],
                    note[1] + CONSTANT.BAR_Y,
                    #"WHITE" if note[2] == 15 else "BLACK",
                    "GRAY",
                    0 if note[2] == 15 else 1,
                )
                key.draw()
            # 피아노 블록이 내려오는거 그려주기
            if note[1] < CONSTANT.SIZE[1] - CONSTANT.IMAGE_SIZE[1]:
                surf = pygame.Surface((note[2], CONSTANT.BAR_Y), pygame.SRCALPHA)
                pygame.draw.rect(
                    surf,
                    note[3] + [int((note[1] + 1) / (CONSTANT.SIZE[1] - CONSTANT.IMAGE_SIZE[1]) * 255)],
                    pygame.Rect(0, 0, note[2], CONSTANT.BAR_Y),
                    border_radius=1,
                )
                CONSTANT.SCREEN.blit(surf, (note[0], note[1]))
                note_list_on.append([note[0], note[1] + CONSTANT.REFRESH_GAP, note[2], note[3]])

        pygame.display.flip()
        image_array = pygame.surfarray.array3d(CONSTANT.SCREEN)
        image = np.swapaxes(image_array, 0, 1)
        out.write(image)

        clock.tick(CONSTANT.FRAME)
          
    out.release()
    pygame.quit()

    # os.system("ffmpeg -i video.mov -vcodec libx264 video.mp4 -y")
    os.system("ffmpeg -ss 0.8 -i ./data/outputs/video.mov -i ./data/outputs/sound.wav -vcodec libx264 ./data/outputs/video.mp4 -y -hide_banner -loglevel error")

if __name__=="__main__":
    result_array = np.load('./data/outputs/dump.npy')
    video(result_array)
        