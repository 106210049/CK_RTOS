import pygame
pygame.mixer.init()
pygame.mixer.music.load(r"D:\KY_8\TRUNG\T2_RTOS\CODE\alarm.wav")
pygame.mixer.music.play()
while pygame.mixer.music.get_busy():
    continue
