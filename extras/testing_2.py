from PIL import Image
#Image.open("C:/Users/dipbh/PycharmProjects/SMMVPython/data/my_img_1.png").save("sample1.bmp")



import pygame as pg
import numpy as np

pg.init()
screen = pg.display.set_mode((800, 800))
clock = pg.time.Clock()

colors = np.array([[0, 0, 0],[255, 255, 255]])
gridarray = np.random.randint(2, size=(2560, 1080))
#print(gridarray)
surface = pg.surfarray.make_surface(colors[gridarray])
surface = pg.transform.scale(surface, (2560, 1080))  # Scaled a bit.

running = True
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

    #screen.fill((30, 30, 30))
    screen.blit(surface, (100, 100))
    pg.display.flip()
    clock.tick(60)