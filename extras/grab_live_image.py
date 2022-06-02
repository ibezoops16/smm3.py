from datetime import time, datetime
import random
from tkinter import ttk

import PIL.Image, PIL.ImageTk
import os
os.environ["PYLON_CAMEMU"] = "3"
from pypylon import genicam
from pypylon import pylon
import cv2
import tkinter as tk

class cameraCapture(tk.Frame):

    def __init__(self, **kw):
        super().__init__(**kw)
        self.img0 = []
        self.windowName = 'title'

        try:
            # Create an instant camera object with the camera device found first.
            self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

            self.camera.Open()  # Need to open camera before can use camera.ExposureTime
            self.camera.Width = 800 #2412
            self.camera.Height = 950 #2024
            # Print the model name of the camera.
            print("Using device ", self.camera.GetDeviceInfo().GetModelName())

            # According to their default configuration, the cameras are
            # set up for free-running continuous acquisition.
            # Grabbing continuously (video) with minimal delay
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

            # converting to opencv bgr format
            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        except genicam.GenericException as e:
            # Error handling
            print("An exception occurred.", e.GetDescription())
            exitCode = 1

    def getFrame(self):
        try:
            self.grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            if self.grabResult.GrabSucceeded():
                image = self.converter.Convert(self.grabResult)  # Access the openCV image data
                self.img0 = image.GetArray()

            else:
                print("Error: ", self.grabResult.ErrorCode)

            self.grabResult.Release()
            # time.sleep(0.01)

            return self.img0

        except genicam.GenericException as e:
            # Error handling
            print("An exception occurred.", e.GetDescription())
            exitCode = 1


class Page(tk.Frame):

    def __init__(self, parent, window):
        tk.Frame.__init__(self, parent)
        self.window = window
        self.window.title = "Title"

        # Open camera source
        self.vid = cameraCapture()

        # Create a canvas that will fit the camera source
        self.canvas = tk.Canvas(window, width=1000, height=600)
        self.canvas.grid(row=0, column=0)

        # move
        menuFrame = ttk.Labelframe(window, text=("Menu"))
        menuFrame.grid(row=1, column=0, sticky="NSW",
            padx=5, pady=2)

        #Button that lets the user take a snapshot
        self.btnSaveImage = tk.Button(menuFrame, text="Grab", width = 25, command=window.destroy)
        self.btnSaveImage.grid(row=0, column=2, sticky="W", padx = 5, pady = 5)
        self.btnSaveImage = tk.Button(menuFrame, text="Snap", width = 25, command=self.snap_image)
        self.btnSaveImage.grid(row=0, column=3, sticky="W", padx = 5, pady = 5)
        self.btnSaveImage = tk.Button(menuFrame, text="Stop", width = 25, command=window.destroy)
        self.btnSaveImage.grid(row=0, column=4, sticky="W", padx = 5, pady = 5)
        self.destroyWindow = tk.Button(menuFrame, text="Save", width = 25, command=self.save_image)
        self.destroyWindow.grid(row=0, column=5, sticky="W", padx = 5, pady = 5)

        self.delay = 100
        self.update()
        # self.window.mainloop()



        # windows zoom
    def snap_image(self):
        pass


    def save_image(self):
        # Get a frame from the video source
        frame = self.vid.getFrame()
        now = datetime.now()
        string_tif = "../data/snapshot_" + now.strftime("%m%d%Y__%H_%M_%S") + ".tif"
        cv2.imwrite(string_tif,
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    def update(self):
        # Get a frame from cameraCapture
        frame = self.vid.getFrame()  # This is an array
        # https://stackoverflow.com/questions/48121916/numpy-resize-rescale-image/48121996
        frame = cv2.resize(frame, dsize=(1000, 600), interpolation=cv2.INTER_CUBIC)

        # OpenCV bindings for Python store an image in a NumPy array
        # Tkinter stores and displays images using the PhotoImage class
        # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
        self.canvas.create_image(500, 300, image=self.photo)

        self.window.after(self.delay, self.update)


class GrabImage:
    def grab_image(self):
        root = tk.Tk()
        testWidget = Page(root, root)
        testWidget.grid(row=0, column=0, sticky="W")
        root.mainloop()
