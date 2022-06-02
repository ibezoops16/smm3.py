import cv2
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt, exp
import tkinter as tk
from tkinter import filedialog
import math


def image_viewer():
    # create window
    # display
    # clsoing instructions
    pass


class ImageProcessing:

    def load_image(self, convert_to_gray):

        root = tk.Tk()
        root.img_path = filedialog.askopenfilename(initialdir="/", title="Select file")
        root.destroy()
        img_path = root.img_path
        if img_path[-4:] == ".tif" or img_path[-4:] == ".bmp" or img_path[-4:] == ".png" or img_path[-4:] == ".jpg" \
                or img_path[4:] == ".TIF" or img_path[-4:] == ".BMP" or img_path[-4:] == ".PNG" or img_path[-4:] == ".JPG":
            input_img = cv2.imread(img_path)

            if convert_to_gray:
                # convert to gray scale
                input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
                cv2.imshow('Gray Scaled Input Image ', input_img)
                print("Gray Scaled Image Displayed on Image Viewer, Press any key to continue the program execution")
            else:
                cv2.imshow("Original Input Image", input_img)
                print("Original Input Image Displayed on Image Viewer, Press any key to continue the program execution")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return input_img
        else:
            print("\nFile should be Image only!! bmp, tif, jpg and png are valid formats!!")
            print("Please select again!!")
            return 0

        pass

    def get_input_image(self, convert_to_gray):
        image_path = "G:\Pycharm\SMMVpy2\data\my_img_1.png"
        input_img = cv2.imread(image_path)
        cv2.imshow("Original Input Image", input_img)
        print("Original Input Image Displayed on Image Viewer, Press any key to continue the program execution")
        if convert_to_gray:
            # convert to gray scale
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            cv2.imshow('Gray Scaled Input Image ', input_img)
            print("Gray Scaled Image Displayed on Image Viewer, Press any key to continue the program execution")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return input_img

    def show_image_processing_submenu(self):
        pass

    def distance(self, point1, point2):
        return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def ideal_filter_lowpass(self, D0, imgShape):
        base = np.zeros(imgShape[:2])
        rows, cols = imgShape[:2]
        center = (rows / 2, cols / 2)
        for x in range(cols):
            for y in range(rows):
                if self.distance((y, x), center) < D0:
                    base[y, x] = 1
        return base

    ## 1. Load Image
    ## 2. Morphological IP: (1) dilation, (2) erosion, (3) open, (4) close
    ## 3. Filters: (1) Gaussian, (2) Average, (3) Sharpening, ... --> User inputs filter size
    ## 4. Frequency-domain IP: (1) FFP, (2) Apply Mask (Band-pass filter), (3) Inverse FFT, ...
    ## 5. Edge detection: (1) Canny, (2) Hough, ...

    def morphological_dilation(self, input_img):
        """
        Dilation:
            In cases like noise removal, erosion is followed by dilation.
            Because, erosion removes white noises, but it also shrinks our object.
            So we dilate it.
            Since noise is gone, they won’t come back, but our object area increases.
            It is also useful in joining broken parts of an object.
        """
        kernel_width = int(input("Enter width of kernel >>>"))
        kernel_height = int(input("Enter height of kernel >>>"))
        kernel = np.ones((kernel_width, kernel_height), np.uint8)
        img_dilation = cv2.dilate(input_img, kernel, iterations=1)
        cv2.imshow('Dilation Image', img_dilation)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def morphological_erosion(self, input_img):
        """
        Erosion:
            It is useful for removing small white noises.
            Used to detach two connected objects etc.
        """
        kernel_width = int(input("Enter width of kernel >>>"))
        kernel_height = int(input("Enter height of kernel >>>"))
        kernel = np.ones((kernel_width, kernel_height), np.uint8)
        img_erosion = cv2.erode(input_img, kernel, iterations=1)
        cv2.imshow('Erosion Image', img_erosion)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def morphological_open(self):
        pass

    def morphological_close(self):
        pass

    def filters_gaussian(self, input_img):
        kernel_width = int(input("Enter width of kernel >>>"))
        kernel_height = int(input("Enter height of kernel >>>"))
        gaussian_img = cv2.GaussianBlur(input_img,(kernel_width, kernel_height),0)
        cv2.imshow('Gaussian Image', gaussian_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def filters_Average(self, input_img):
        kernel_width = int(input("Enter width of kernel >>>"))
        kernel_height = int(input("Enter height of kernel >>>"))
        gaussian_img = cv2.blur(input_img, (kernel_width, kernel_height), 0)
        cv2.imshow('Gaussian Image', gaussian_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def filters_Sharpening(self, input_img):
        kernel_width = int(input("Enter width of kernel >>>"))
        kernel_height = int(input("Enter height of kernel >>>"))
        gaussian_img = cv2.GaussianBlur(input_img, (kernel_width, kernel_height), 0)
        sharpened_img = input_img - gaussian_img
        cv2.imshow('Sharpened Image', sharpened_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def fq_domain_ffp(self, input_img):
        plt.figure(figsize=(6.4 * 5, 4.8 * 5), constrained_layout=False)
        plt.subplot(151), plt.imshow(input_img, "gray"), plt.title("Original Image")
        original = np.fft.fft2(input_img)
        plt.subplot(152), plt.imshow(np.log(1 + np.abs(original)), "gray"), plt.title("Spectrum")
        center = np.fft.fftshift(original)
        plt.subplot(153), plt.imshow(np.log(1 + np.abs(center)), "gray"), plt.title("Centered Spectrum")
        inv_center = np.fft.ifftshift(center)
        plt.subplot(154), plt.imshow(np.log(1 + np.abs(inv_center)), "gray"), plt.title("Decentralized")
        processed_img = np.fft.ifft2(inv_center)
        plt.subplot(155), plt.imshow(np.abs(processed_img), "gray"), plt.title("Processed Image")
        plt.show()

    def fq_domain_bandpass_filter(self, input_img):
        radius = int(input("Enter radius >>>"))
        n = int(input("Enter n >>>"))
        w = int(input("Enter w >>>"))
        fft = cv2.dft(np.float32(input_img), flags=cv2.DFT_COMPLEX_OUTPUT)
        # Centralize fft, the generated dshift is still a three-dimensional array
        dshift = np.fft.fftshift(fft)

        # Get the center pixel
        rows, cols = input_img.shape[:2]
        mid_row, mid_col = int(rows / 2), int(cols / 2)

        # Build mask, 256 bits, two channels
        mask = np.zeros((rows, cols, 2), np.float32)
        for i in range(0, rows):
            for j in range(0, cols):
                # Calculate the distance from (i, j) to the center
                d = math.sqrt(pow(i - mid_row, 2) + pow(j - mid_col, 2))
                if radius - w / 2 < d < radius + w / 2:
                    mask[i, j, 0] = mask[i, j, 1] = 1
                else:
                    mask[i, j, 0] = mask[i, j, 1] = 0

            # Multiply the Fourier transform result by a mask
        fft_filtering = dshift * np.float32(mask)
        # Inverse Fourier transform
        ishift = np.fft.ifftshift(fft_filtering)
        image_filtering = cv2.idft(ishift)
        image_filtering = cv2.magnitude(image_filtering[:, :, 0], image_filtering[:, :, 1])
        # Normalize the inverse transform results (generally normalize the last step of image processing, except in special cases)
        cv2.normalize(image_filtering, image_filtering, 0, 1, cv2.NORM_MINMAX)
        cv2.imshow('Bandpass Filtered Image', image_filtering)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def fq_domain_inverse_fft(self, input_img):
        print("Inverse FFT")

    def edge_detection_canny(self, input_img):
        lower_limit = int(input("Enter lower_limit >>>"))
        upper_limit = int(input("Enter upper_limit >>>"))
        edges = cv2.Canny(input_img, lower_limit, upper_limit)
        cv2.imshow('Edge Detected Image', edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def edge_detection_hough(self):
        print("Hough Transform")

    def fq_domain_highpass_filter(self, input_img):
        radius = int(input("Enter radius >>>"))
        n = int(input("Enter n >>>"))
        fft = cv2.dft(np.float32(input_img), flags=cv2.DFT_COMPLEX_OUTPUT)
        # Centralize fft, the generated dshift is still a three-dimensional array
        dshift = np.fft.fftshift(fft)

        # Get the center pixel
        rows, cols = input_img.shape[:2]
        mid_row, mid_col = int(rows / 2), int(cols / 2)

        # Build ButterWorth high-pass filter mask

        mask = np.zeros((rows, cols, 2), np.float32)
        for i in range(0, rows):
            for j in range(0, cols):
                # Calculate the distance from (i, j) to the center
                d = math.sqrt(pow(i - mid_row, 2) + pow(j - mid_col, 2))
                try:
                    mask[i, j, 0] = mask[i, j, 1] = 1 / (1 + pow(radius / d, 2 * n))
                except ZeroDivisionError:
                    mask[i, j, 0] = mask[i, j, 1] = 0
            # Multiply the Fourier transform result by a mask
        fft_filtering = dshift * mask
        # Inverse Fourier transform
        ishift = np.fft.ifftshift(fft_filtering)
        image_filtering = cv2.idft(ishift)
        image_filtering = cv2.magnitude(image_filtering[:, :, 0], image_filtering[:, :, 1])
        # Normalize the inverse transform results (generally normalize the last step of image processing, except in special cases)
        cv2.normalize(image_filtering, image_filtering, 0, 1, cv2.NORM_MINMAX)

        cv2.imshow('Highpass Filtered Image', image_filtering)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def fq_domain_lowpass_filter(self, input_img):
        radius = int(input("Enter radius >>>"))

        fft = cv2.dft(np.float32(input_img), flags=cv2.DFT_COMPLEX_OUTPUT)
        # Centralize fft, the generated dshift is still a three-dimensional array
        dshift = np.fft.fftshift(fft)

        # Get the center pixel
        rows, cols = input_img.shape[:2]
        mid_row, mid_col = int(rows / 2), int(cols / 2)

        # Build mask, 256 bits, two channels
        mask = np.zeros((rows, cols, 2), np.float32)
        mask[mid_row - radius:mid_row + radius, mid_col - radius:mid_col + radius] = 1

        # Multiply the Fourier transform result by a mask
        fft_filtering = dshift * mask
        # Inverse Fourier transform
        ishift = np.fft.ifftshift(fft_filtering)
        image_filtering = cv2.idft(ishift)
        image_filtering = cv2.magnitude(image_filtering[:, :, 0], image_filtering[:, :, 1])
        # Normalize the inverse transform results (generally normalize the last step of image processing, except in special cases)
        cv2.normalize(image_filtering, image_filtering, 0, 1, cv2.NORM_MINMAX)
        cv2.imshow('Lowpass Filtered Image', image_filtering)
        cv2.waitKey(0)
        cv2.destroyAllWindows()