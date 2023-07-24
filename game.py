import pygame
import mido
from midi_file import *
from mido import MidiFile
import numpy as np
from collections import deque
from PIL import Image
import cv2
import os
import random

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
    BLUE = [4, 46, 255]
    SKY = [160, 211, 249]

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

    WHITE_BAR_COLOR = RED
    BLACK_BAR_COLOR = BLACK
    BACKGROUND_COLOR = SKY

    for k in keyboard:
        if k == 1:
            tmp += BAR_SIZE[0]
            BAR_X.append(tmp)
        else:
            BAR_X.append(tmp + 10)

    # background = pygame.image.load("./universe.png")
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

    pathOut = "./data/outputs/video.mov"
    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FRAME, SIZE)

    # pygame.mixer.music.load("GT.midi")
    # pygame.mixer.music.play()

    # class FlameParticle:
    #     alpha_layer_qty = 2
    #     alpha_glow_difference_constant = 2

    #     def __init__(self, x=SIZE[0] // 2, y=SIZE[1] // 2, r=5):
    #         self.x = x
    #         self.y = y
    #         self.r = r
    #         self.original_r = r
    #         self.alpha_layers = FlameParticle.alpha_layer_qty
    #         self.alpha_glow = FlameParticle.alpha_glow_difference_constant
    #         max_surf_size = 2 * self.r * self.alpha_layers * self.alpha_layers * self.alpha_glow
    #         self.surf = pygame.Surface((max_surf_size, max_surf_size), pygame.SRCALPHA)
    #         self.burn_rate = 0.1 * random.randint(1, 4)

    #     def update(self):
    #         self.y -= 7 - self.r
    #         self.x += random.randint(-self.r, self.r)
    #         self.original_r -= self.burn_rate
    #         self.r = int(self.original_r)
    #         if self.r <= 0:
    #             self.r = 1

    #     def draw(self):
    #         max_surf_size = 2 * self.r * self.alpha_layers * self.alpha_layers * self.alpha_glow
    #         self.surf = pygame.Surface((max_surf_size, max_surf_size), pygame.SRCALPHA)
    #         for i in range(self.alpha_layers, -1, -1):
    #             alpha = 255 - i * (255 // self.alpha_layers - 5)
    #             if alpha <= 0:
    #                 alpha = 0
    #             radius = self.r * i * i * self.alpha_glow
    #             if self.r == 4 or self.r == 3:
    #                 r, g, b = (255, 0, 0)
    #             elif self.r == 2:
    #                 r, g, b = (255, 150, 0)
    #             else:
    #                 r, g, b = (50, 50, 50)
    #             # r, g, b = (0, 0, 255)  # uncomment this to make the flame blue
    #             color = (r, g, b, alpha)
    #             pygame.draw.circle(self.surf, color, (self.surf.get_width() // 2, self.surf.get_height() // 2), radius)
    #         screen.blit(self.surf, self.surf.get_rect(center=(self.x, self.y)))


    # class Flame:
    #     def __init__(self, x=SIZE[0] // 2, y=SIZE[1] // 2):
    #         self.x = x
    #         self.y = y
    #         self.flame_intensity = 2
    #         self.flame_particles = []
    #         for i in range(self.flame_intensity * 25):
    #             self.flame_particles.append(FlameParticle(self.x + random.randint(-5, 5), self.y, random.randint(1, 5)))

    #     def draw_flame(self):
    #         for i in self.flame_particles:
    #             if i.original_r <= 0:
    #                 self.flame_particles.remove(i)
    #                 self.flame_particles.append(FlameParticle(self.x + random.randint(-5, 5), self.y, random.randint(1, 5)))
    #                 del i
    #                 continue
    #             i.update()
    #             i.draw()

    # flame = Flame()

    for i in range(0, result_array.shape[0] + VIDEO_FRAME*3, TICKS_PER_SECOND):
        # screen.blit(background, (0, 0))
        # flame.draw_flame()
        screen.fill(BACKGROUND_COLOR)
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
                            [BAR_X[step], 0, BAR_SIZE[1], BLACK_BAR_COLOR]
                        )
                    # white key
                    else:
                        note_list_on.append([BAR_X[step], 0, BAR_SIZE[0], WHITE_BAR_COLOR])
        
        len_on = len(note_list_on)
        for idx in range(len_on):
            note = note_list_on.popleft()
            # 피아노 버튼이 눌렸을때 작동하는 로직
            if note[1] == SIZE[1] - IMAGE_SIZE[1] - BAR_Y:
                pygame.draw.rect(
                    screen,
                    WHITE_BAR_COLOR if note[2] == 15 else BLACK_BAR_COLOR,
                    # RED,
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
                    screen, note[3], pygame.Rect(note[0], note[1], note[2], BAR_Y), border_radius=1
                )
                note_list_on.append([note[0], note[1] + REFRESH_GAP, note[2], note[3]])

        pygame.display.flip()
        image_array = pygame.surfarray.array3d(screen)
        image = np.swapaxes(image_array, 0, 1)
        out.write(image)

        clock.tick(FRAME)
          
    out.release()
    pygame.quit()

    # os.system("ffmpeg -i video.mov -vcodec libx264 video.mp4 -y")
    os.system("ffmpeg -ss 0.8 -i ./data/outputs/video.mov -i ./data/outputs/sound.wav -vcodec libx264 ./data/outputs/video.mp4 -y")

if __name__=="__main__":
    result_array = np.load('./data/outputs/dump.npy')
    video(result_array)
        