import tkinter
from tkinter import *
from datetime import datetime
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import cv2

import numpy as np
from pypylon import pylon
import tkinter as tk

from extras.grab_live_image import GrabImage

class DigitalFringeProjection:

    def __init__(self):
        self.wrapped_img = []
        self.input_value = 0
        self.x_Proj = 0
        self.y_Proj = 0
        self.f_Pitch = 0
        self.f_Int = 0
        self.n_Shifts = 0
        self.H = 0.0
        self.L = 0.0
        self.L_theta = 0.0
        self.proj_Res = 0.0
        self.f_Theta1 = 0.0
        self.f_Theta2 = 0.0
        self.f_DTheta = 0.0
        self.fTheta = 0.0
        self.pitch_mm_PX = 0.0
        self.check = True
        self.f_Img = []
        #self.f_Img_2 = []
        #self.f_Img_3 = []
        #self.f_Img_4 = []



    def grab_image(self):
        grab_image_obj = GrabImage()
        grab_image_obj.grab_image()

    def set_f_Params(self):
        while True:
            try:
                setting_id = int(input("Input Fringe Parameters: (1) x_Proj (2) y_Proj (3) Pitch (4) Intensity (5) # of Shifts (0) Exit"))
                self.input_value = setting_id
            except ValueError:
                print("Please input integers only...")
                continue
            #self.input_value = setting_id

            if self.input_value == 1:
                try:
                    x_Dim = int(input("Set value for x_Proj "))
                    self.x_Proj = x_Dim
                except ValueError:
                    print("Please input integers only...")
                    continue
            elif self.input_value == 2:
                try:
                    y_Dim = int(input("Set value for y_Proj "))
                    self.y_Proj = y_Dim
                except ValueError:
                    print("Please input integers only...")
                    continue
            elif self.input_value == 3:
                try:
                    f_Pitch = int(input("Set fringe pitch "))
                    self.f_Pitch = f_Pitch
                except ValueError:
                    print("Please input integers only...")
                    continue
            elif self.input_value == 4:
                try:
                    f_Int = int(input("Set Intensity(255 is standard)"))
                    self.f_Int = f_Int
                except ValueError:
                    print("Please input integers only...")
                    continue
            elif self.input_value == 5:
                try:
                    n_Shifts = int(input("Set Number of Phase Shifts"))
                    self.n_Shifts = n_Shifts
                except ValueError:
                    print("Please input integers only...")
                    continue
            elif self.input_value == 0:
                break

    def set_p_Params(self):
        while True:
            try:
                setting_id = int(input("Input Projector Parameters: (1) H (2) L (3) L_theta (4) proj_Res (5) Calibrate Projector (0) Exit"))
                self.input_value = setting_id
            except ValueError:
                print("Please input integers only...")
                continue

            if self.input_value == 1:
                try:
                    h = float(input("Set value for H"))
                    self.H = h
                except ValueError:
                    print("Please input integers only...")
                    continue
            elif self.input_value == 2:
                try:
                    l = float(input("Set value for L"))
                    self.L = l
                except ValueError:
                    print("Please input integers only...")
                    continue
            elif self.input_value == 3:
                try:
                    l_t = float(input("Set value for L_theta"))
                    self.L_theta = l_t
                except ValueError:
                    print("Please input integers only...")
                    continue
            elif self.input_value == 4:
                try:
                    p_r = float(input("Set value for proj_Res"))
                    self.proj_Res = p_r
                except ValueError:
                    print("Please input integers only...")
                    continue
            elif self.input_value == 5:
                print()
                print("Projector Parameters:")
                print("Width: " + str(self.x_Proj))
                print("Height: " + str(self.y_Proj))
                print("Pitch: " + str(self.f_Pitch))
                print("Intensity: " + str(self.f_Int))
                print("Number of Shifts: " + str(self.n_Shifts))
                print()
                print("Calibration Parameters:")
                print("H: " + str(self.H))
                print("L: " + str(self.L))
                print("L_Theta: " + str(self.L_theta))
                print("Projector Resolution: " + str(self.proj_Res))
                print()

                try:
                    check = bool(input("Are the following parameters correct? (True or False)"))
                    self.check = check
                except TypeError:
                    print("Please input True or False...")
                    continue

                if self.check == True:
                    self.f_Theta1 = np.arctan(self.L_theta / self.H)
                    self.f_Theta2 = np.arctan((self.L_theta + self.L) / self.H)
                    self.f_DTheta = (self.f_Theta2 - self.f_Theta1) / self.proj_Res
                    print("Calibration Parameters:")
                    print("f_Theta1: " + str(self.f_Theta1))
                    print("f_Theta2: " + str(self.f_Theta2))
                    print("f_DTheta: " + str(self.f_DTheta))
                    break
                else:
                    break
            elif self.input_value == 0:
                break

    def convertPicthMMtoPx(self, Pitch):
        self.fTheta = self.f_Theta1 + self.f_DTheta * Pitch
        px = self.H * np.tan(self.fTheta) - self.L_theta
        return px

    def calculatePitchPx(self, y, pitch_MM):
        pitch_Px = (np.arctan(pitch_MM / self.H + np.tan(self.f_Theta1 + self.f_DTheta * y)) - self.f_Theta1 / self.f_DTheta - y)
        return pitch_Px

    def get_f_Value(self, y_Pos, y_Pitch, f_Shift):
        f_val = self.f_Int / 2 * np.sin(2 * np.pi * float(y_Pos / y_Pitch) - (2 * f_Shift * np.pi / self.n_Shifts)) + self.f_Int / 2
        return f_val


    def create_sine_Fringe(self, f_Shift):
        self.pitch_mm_PX = self.convertPicthMMtoPx(self.f_Pitch)
        #print(self.pitch_mm_PX)
        arr = np.zeros((self.y_Proj, self.x_Proj), dtype=np.uint8)
        for i in range(self.y_Proj):
            p_y = self.calculatePitchPx(self.y_Proj - i - 1, self.pitch_mm_PX)
            for j in range(self.x_Proj):
                arr[i][j] = float(self.get_f_Value(i, self.f_Pitch, f_Shift))
        return arr

    def dfp_proj_fringe(self):
        for i in range(self.n_Shifts):

            #create window
            root = tk.Tk()
            #generate fringe images
            self.f_Img = self.create_sine_Fringe(i)
            #set window attribute
            root.attributes('-fullscreen', True)
            #close window after 30ms
            root.after(300, lambda: root.destroy())

            canvas = tk.Canvas(root, width=self.x_Proj, height=self.y_Proj)
            canvas.pack()
            canvas.create_image(self.x_Proj/2, self.y_Proj/2, anchor="nw", image=ImageTk.PhotoImage(Image.fromarray(self.f_Img)))
            root.mainloop()
        print("\ndfp_fringe")

    def dfp_phase_wrapping(self):
        np.seterr(over='ignore')
        root = Tk()
        root.img_path = filedialog.askdirectory(initialdir="/", title="Select file")
        root.destroy()
        img_path = root.img_path

        lst = os.listdir(img_path)

        for i in range(0, len(lst), 4):
            img_1 = cv2.imread(str(img_path) + "/" + str(lst[i]), cv2.IMREAD_GRAYSCALE)
            img_2 = cv2.imread(str(img_path) + "/" + str(lst[i+1]), cv2.IMREAD_GRAYSCALE)
            img_3 = cv2.imread(str(img_path) + "/" + str(lst[i+2]), cv2.IMREAD_GRAYSCALE)
            img_4 = cv2.imread(str(img_path) + "/" + str(lst[i+3]), cv2.IMREAD_GRAYSCALE)

            img = [[0]*len(img_1[0]) for i in range(len(img_1))]
            a = [[0] * len(img_1[0]) for i in range(len(img_1))]
            b = [[0] * len(img_1[0]) for i in range(len(img_1))]

            for r in range(len(img_1)):
                for c in range(len(img_1[0])):

                    a[r][c] = float(img_4[r][c] - img_2[r][c])
                    b[r][c] = float(img_1[r][c] - img_3[r][c])

                    img[r][c] = float(np.arctan2(a[r][c], b[r][c]))

            ratio = float(256.00 / (np.amax(img)-np.amin(img)))
            img = (img - np.amin(img)) * ratio

            self.wrapped_img = img
            filepath = "%s + /Unwrapped_Img%s.bmp" % (str(img_path), str(i))
            cv2.imwrite(filepath, self.wrapped_img)
            print("\ndfp_phase_wrapping")

    def dfp_phase_unwrapping(self):

        print("\ndfp_phase_unwrapping")

    def dfp_demodulization(self):
        print("\ndfp_demodulization")

    def dfp_intensity_calibration(self):
        print("\ndfp_intensity_calibration")

    def dfp_coordinate_calibration(self):
        print("\ndfp_coordinate_calibration")

    def dfp_calibration_block(self):
        print("\ndfp_calibration_block")

    def dfp_displacement(self):
        print("\ndfp_displacement")

    def snap_image(self):
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        camera.Open()

        # demonstrate some feature access
        new_width = camera.Width.GetValue() - camera.Width.GetInc()
        if new_width >= camera.Width.GetMin():
            camera.Width.SetValue(new_width)

        numberOfImagesToGrab = 1
        camera.StartGrabbingMax(numberOfImagesToGrab)
        converter = pylon.ImageFormatConverter()

        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        count = 1
        while camera.IsGrabbing():
            grabResult = camera.RetrieveResult(100000000, pylon.TimeoutHandling_ThrowException)

            if grabResult.GrabSucceeded():
                # Access the image data.

                img = grabResult.Array
                image = converter.Convert(grabResult)
                image = image.GetArray()
                print("\nImage is saved successfully!!")
                print("Image Size X: ", grabResult.Width)
                print("Image Size Y: ", grabResult.Height)
                print("Image Displayed on Image Viewer, Press any key to continue the program execution.")

                now = datetime.now()
                string_tif = "../data/snap_" + now.strftime("%m%d%Y__%H_%M_%S") + ".tif"

                cv2.imwrite(string_tif, image)
                cv2.imshow("img" + str(count), image)
                count = count + 1
                k = cv2.waitKey(10000)

            grabResult.Release()

        camera.Close()

        return image

    def load_image(self):

        root = Tk()
        root.img_path = filedialog.askopenfilename(initialdir="/", title="Select file")
        root.destroy()
        img_path = root.img_path
        if img_path[-4:] == ".tif" or img_path[-4:] == ".bmp" or img_path[-4:] == ".png" or img_path[-4:] == ".jpg" \
                or img_path[4:] == ".TIF" or img_path[-4:] == ".BMP" or img_path[-4:] == ".PNG" or img_path[-4:] == ".JPG":
            original_image = cv2.imread(img_path)
            print("Image can be seen in Image viewer!! Press any key to continue execution!!")
            cv2.imshow("Loaded Image", original_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return original_image
        else:
            print("\nFile should be Image only!! bmp, tif, jpg and png are valid formats!!")
            print("Please select again!!")
            return 0

        pass
