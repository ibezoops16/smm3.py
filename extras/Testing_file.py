# from struct import pack
#
# class Bitmap():
#   def __init__(s, width, height):
#     s._bfType = 19778 # Bitmap signature
#     s._bfReserved1 = 0
#     s._bfReserved2 = 0
#     s._bcPlanes = 1
#     s._bcSize = 12
#     s._bcBitCount = 24
#     s._bfOffBits = 26
#     s._bcWidth = width
#     s._bcHeight = height
#     s._bfSize = 26+s._bcWidth*3*s._bcHeight
#     s.clear()
#
#
#   def clear(s):
#     s._graphics = [(0,0,0)]*s._bcWidth*s._bcHeight
#
#
#   def setPixel(s, x, y, color):
#     if isinstance(color, tuple):
#       if x<0 or y<0 or x>s._bcWidth-1 or y>s._bcHeight-1:
#         raise ValueError('Coords out of range')
#       if len(color) != 3:
#         raise ValueError('Color must be a tuple of 3 elems')
#       s._graphics[y*s._bcWidth+x] = (color[2], color[1], color[0])
#     else:
#       raise ValueError('Color must be a tuple of 3 elems')
#
#
#   def write(s, file):
#     with open(file, 'wb') as f:
#       f.write(pack('<HLHHL',
#                    s._bfType,
#                    s._bfSize,
#                    s._bfReserved1,
#                    s._bfReserved2,
#                    s._bfOffBits)) # Writing BITMAPFILEHEADER
#       f.write(pack('<LHHHH',
#                    s._bcSize,
#                    s._bcWidth,
#                    s._bcHeight,
#                    s._bcPlanes,
#                    s._bcBitCount)) # Writing BITMAPINFO
#       for px in s._graphics:
#         f.write(pack('<BBB', *px))
#       for i in range((4 - ((s._bcWidth*3) % 4)) % 4):
#         f.write(pack('B', 0))
#
#
#
# def main():
#   side = 520
#   b = Bitmap(side, side)
#   for j in range(0, side):
#     b.setPixel(j, j, (255, 0, 0))
#     b.setPixel(j, side-j-1, (255, 0, 0))
#     b.setPixel(j, 0, (255, 0, 0))
#     b.setPixel(j, side-1, (255, 0, 0))
#     b.setPixel(0, j, (255, 0, 0))
#     b.setPixel(side-1, j, (255, 0, 0))
#   b.write('file.bmp')
#
#
# if __name__ == '__main__':
#   main()




import numpy as np
from PIL import Image
import cv2
import matplotlib

# gradient between 0 and 1 for 2560*1080
from matplotlib import pyplot as plt
#matplotlib.use("wx")




array = np.linspace(0,1,2560*1080)

# reshape to 2d
mat = np.reshape(array,(2560,1080))
#print(mat)


# Creates PIL image
img = Image.fromarray( mat , 'L')
#cv2.imshow("image",mat)
#cv2.waitKey(0)
plt.imshow(img,cmap='gray')
#plt.axis('off')
plt.show()
#img.show()
mgr = plt.get_current_fig_manager()
mgr.window.SetPosition((0, 0))