import os
from datetime import datetime
from tkinter import *
from tkinter import ttk
import tkinter as tk
import PIL.Image
import PIL.ImageTk
os.environ["PYLON_CAMEMU"] = "3"
from pypylon import genicam
from pypylon import pylon
import cv2



class cameraCapture():

    def __init__(self, **kw):
        super().__init__(**kw)
        self.img0 = []
        self.windowName = 'title'


    def setup_camera(self, img_width, img_height, exposureTime):

        try:
            # Create an instant camera object with the camera device found first.
            self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

            self.camera.Open()  # Need to open camera before can use camera.ExposureTime
            self.camera.Width = int(img_width)  # 2412
            self.camera.Height = int(img_height)  # 2024
            # Print the model name of the camera.
            print("Using device: ", self.camera.GetDeviceInfo().GetModelName())
            print("testing: ", exposureTime)
            #self.camera.ExposureTimeRaw.SetValue(100)
            #print("Camera Exposure Time: ",self.camera.ExposureTimeRaw.GetValue())

            #self.camera.AcquisitionFrameRateEnable.SetValue(True)
            #self.camera.AcquisitionFrameRate.SetValue(30.0)
            # self.camera.properties['AcquisitionFrameRateEnable'] = True
            # self.camera.properties['AcquisitionFrameRate'] = 1000
            #print("frame rate: ",self.camera.AcquisitionFrameRate.GetValue())


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
                self.image = self.converter.Convert(self.grabResult)  # Access the openCV image data
                self.img0 = self.image.GetArray()

            else:
                print("Error: ", self.grabResult.ErrorCode)

            self.grabResult.Release()
            # time.sleep(0.01)

            return self.img0

        except genicam.GenericException as e:
            # Error handling
            print("An exception occurred.", e.GetDescription())
            exitCode = 1


class GrabImage:

    def __init__(self):
        self.delay = 100
        self.keep_grabbing = 0
        self.img_width = 5472
        self.img_height = 3648
        self.exposureTime = 3500



    def grab_image(self):
        # Main self.root Window
        self.root = tk.Tk()

        self.vid = cameraCapture()
        self.vid.setup_camera(self.img_width, self.img_height, self.exposureTime)
        # Canvas to display Graphics
        self.canvas = tk.Canvas(self.root, width=1200, height=800)
        self.canvas.grid(row=0, column=0)
        self.camera_image_grabbing_menu()
        self.camera_pixel_setting_menu()
        self.Camera_menu_3()
        self.root.mainloop()


    def camera_image_grabbing_menu(self):
        # Designing Menu Frame to display buttons
        menuFrame1 = ttk.Labelframe(self.root, text=("Menu"))
        menuFrame1.grid(row=1, column=0, sticky="NSW",
                        padx=5, pady=2)
        self.btn_start_grabbing = tk.Button(menuFrame1, text="Start Grabbing", width=25, command=self.start_grabbing)
        self.btn_start_grabbing.grid(row=0, column=2, sticky="W", padx=5, pady=5)
        self.btn_stop_grabbing = tk.Button(menuFrame1, text="Stop Grabbing", width=25, command=self.restart)
        self.btn_stop_grabbing.grid(row=0, column=3, sticky="W", padx=5, pady=5)
        self.btn_snapshot = tk.Button(menuFrame1, text="Snapshot", width=25, command=self.snap_image)
        self.btn_snapshot.grid(row=0, column=4, sticky="W", padx=5, pady=5)
        self.btn_save_image = tk.Button(menuFrame1, text="Save Image", width=25, command=self.save_image)
        self.btn_save_image.grid(row=0, column=5, sticky="W", padx=5, pady=5)
        #self.scroll_bar.config(command=)

    def camera_pixel_setting_menu(self):
        # Designing Menu Frame to display buttons
        menuFrame2 = ttk.Labelframe(self.root, text=("Menu2"))
        menuFrame2.grid(row=2, column=0, sticky="NSW",
                        padx=5, pady=2)
        str_l1 = "Current Image Width: " + str(self.img_width)
        str_l2 = "Current Image Height: " + str(self.img_height)
        l1 = tk.Label(menuFrame2, text=str_l1).grid(row=0, column=0, padx=5, pady=5)
        l2 = tk.Label(menuFrame2, text=str_l2).grid(row=0, column=2, padx=5, pady=5)
        l1 = tk.Label(menuFrame2, text="Image Width: ").grid(row=1, column=0)
        l2 = tk.Label(menuFrame2, text="Image Height: ").grid(row=1, column=2)
        self.e1 = tk.Entry(menuFrame2)
        self.e1.insert(0, "800")
        self.e1.grid(row=1, column=1, padx=5, pady=5)
        self.e2 = tk.Entry(menuFrame2)
        self.e2.insert(0, "750")
        self.e2.grid(row=1, column=3, padx=5, pady=5)

    def Camera_menu_3(self):
        menuFrame3 = ttk.Labelframe(self.root,text=("Menu3"))
        menuFrame3.grid(row=3, column=0, sticky="NSW", padx=5, pady=5)
        image_lable = tk.Label(menuFrame3, text="Image size: ").grid(row=1, column=0)
        frame_rate_label= tk.Label(menuFrame3, text="Frame Rate: ").grid(row=1, column=2)
        exposure_time_label= tk.Label(menuFrame3, text="Exposure Rate: ").grid(row=1, column=4)

        self.image_entry = tk.Entry(menuFrame3)
        self.image_entry.insert(0, "")
        self.image_entry.grid(row=1, column=1, padx=(5,40), pady=5)
        self.frame_rate_entry = tk.Entry(menuFrame3)
        self.frame_rate_entry.insert(0, "")
        self.frame_rate_entry.grid(row=1, column=3, padx=(5,40), pady=5)
        self.exposure_time = tk.IntVar()
        self.exposure_time_entry = tk.Entry(menuFrame3)#,textvariable= self.exposure_time)
        self.exposure_time_entry.insert(0, self.exposureTime)
        self.exposure_time_entry.grid(row=1, column=5, padx=(5,40), pady=5)

    #     self.btn_set = tk.Button(menuFrame3, text="Set", width=20, command=self.set_exposure_time)
    #     self.btn_set.grid(row=1, column=6, sticky="W", padx=5, pady=5)
    #
    #
    # def set_exposure_time(self):
    #     self.exposureTime = self.exposure_time_entry.get()
    #     self.exposure_time_entry.delete(0,"end")
    #     self.exposure_time_entry.insert(0,self.exposureTime)
    #     #self.vid.setup_camera(self.img_width, self.img_height, self.exposureTime)
    #     print("updated exposure time: ",self.exposureTime)


    def restart(self):
        self.keep_grabbing = 0
        self.canvas.delete('all')
        return

    def save_image(self):
        # Get a frame from the video source
        frame = self.vid.getFrame()
        now = datetime.now()
        string_tif = "../data/snapshot_" + now.strftime("%m%d%Y__%H_%M_%S") + ".tif"
        cv2.imwrite(string_tif,
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return

    def snap_image(self):
        # Get a frame from the video source
        frame = self.vid.getFrame()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Image Snapshot", frame)
        return

    def start_grabbing(self):
        self.keep_grabbing = 1
        self.img_width = int(self.e1.get())
        self.img_height = int(self.e2.get())
        self.exposureTime = int(self.exposure_time_entry.get())
        self.vid.setup_camera(self.img_width, self.img_height, self.exposureTime)
        self.update()

    def update(self):
        # Get a frame from cameraCapture
        frame = self.vid.getFrame()  # This is an array
        # https://stackoverflow.com/questions/48121916/numpy-resize-rescale-image/48121996
        frame = cv2.resize(frame, dsize=(1000, 600), interpolation=cv2.INTER_CUBIC)

        # OpenCV bindings for Python store an image in a NumPy array
        # Tkinter stores and displays images using the PhotoImage class
        # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
        if (self.keep_grabbing == 1):
            self.canvas.create_image(500, 300, image=self.photo)
            self.root.after(self.delay, self.update)
