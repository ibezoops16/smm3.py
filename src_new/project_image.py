from tkinter import *
from tkinter import ttk
import tkinter as tk

class ProjectImage:

    #def __init__(self):



    def project_image_UI(self):
        print("testing for project image")
        self.root = tk.Tk()
        self.root.title("Projection Setting")
        self.monitor_X1 = tk.StringVar()
        self.seeting_menu()
        self.projection_menu()
        self.root.mainloop()

    def set_button_fucntion(self):
        result = self.monitor_X1.get()
        self.entry1.delete(0, END)
        self.monitor_X1.set(result)
         # deletes the current value
       # self.entry1.insert(0, result)  #
        print("set button function is working")

    def seeting_menu(self):
        menuFrame1 = ttk.Labelframe(self.root, text=("Setting Menu"))
        menuFrame1.grid(row=0, column=0, sticky="NSW", padx=5, pady=5)
        #Monitor
        self.entry1 = tk.Entry(menuFrame1,width=10,textvariable = self.monitor_X1)
        self.entry1.insert(0,"2560")
        self.entry1.grid(row=0, column=0, padx=5, pady=5)
        self.entry2 = tk.Entry(menuFrame1,width=10)
        self.entry2.insert(0, "1080")
        self.entry2.grid(row=0, column=1, padx=(5,40), pady=5)
        monitor_lable = tk.Label(menuFrame1, text="Monitor").grid(row=1, column=0)

        #projector-1
        self.entry3 = tk.Entry(menuFrame1,width=10)
        self.entry3.insert(0, "3840")
        self.entry3.grid(row=0, column=2, padx=5, pady=5)
        self.entry4 = tk.Entry(menuFrame1,width=10)
        self.entry4.insert(0, "2160")
        self.entry4.grid(row=0, column=3, padx=(5,40), pady=5)
        projector_1_lable = tk.Label(menuFrame1, text="Projector 1").grid(row=1, column=2)

        #projector-2
        self.entry5 = tk.Entry(menuFrame1,width=10)
        self.entry5.insert(0, "3840")
        self.entry5.grid(row=0, column=4, padx=5, pady=5)
        self.entry6 = tk.Entry(menuFrame1,width=10)
        self.entry6.insert(0, "2160")
        self.entry6.grid(row=0, column=5, padx=(5,40), pady=5)
        projector_2_lable = tk.Label(menuFrame1, text="Projector 2").grid(row=1, column=4)

        self.btn_set = tk.Button(menuFrame1, text="set", width=10, command= lambda:self.set_button_fucntion())
        self.btn_set.grid(row=0, column=6, sticky="W", padx=5, pady=5)



    def projection_menu(self):
        # this menu will show 2 checkboxes to select projector
        menuFrame2 = ttk.Labelframe(self.root, text=("Projection Menu"))
        menuFrame2.grid(row=3, column=0, sticky="NSW", padx=5, pady=2)
        self.projector1 = IntVar()
        Checkbutton(menuFrame2, text="Projector 1", variable=self.projector1).grid(row=0, column=0, padx=5, pady=5, sticky=W)
        self.projector2 = IntVar()
        Checkbutton(menuFrame2, text="Projector 2", variable=self.projector2).grid(row=0, column=1, padx=5, pady=5, sticky=W)

        # # Pattern Selection
        ttk.Label(menuFrame2, text="Pattern Direction :",font=("Times New Roman", 10)).grid(column=0,row=1, padx=5, pady=5)
        self.p_selection_text = tk.StringVar()
        self.pattern_direction = ttk.Combobox(menuFrame2, width=15, textvariable= self.p_selection_text)
        self.pattern_direction['values'] = (' Horizontal',' Vertical')
        self.pattern_direction.grid(column=1, row=1)
        self.pattern_direction.current()

        # # Pattern color
        ttk.Label(menuFrame2, text="Pattern Color :", font=("Times New Roman", 10)).grid(column=0, row=2, padx=5, pady=5)
        self.p_color_text = tk.StringVar()
        pattern_color = ttk.Combobox(menuFrame2, width=15, textvariable=self.p_color_text)
        pattern_color['values'] = ('Grayscale', 'Green', 'Blue', 'Red')
        pattern_color.grid(column=1, row=2)
        pattern_color.current()
        #pattern_color.bind('<<ComboboxSelected>>', pattern_color)

        ## Edit boxes
        amplitude_lable = tk.Label(menuFrame2, text="Amplitude: ").grid(row=1, column=2)
        pitch_lable = tk.Label(menuFrame2, text="Pitch(px) : ").grid(row=2, column=2)
        self.amplitude = tk.Entry(menuFrame2)
        self.amplitude.grid(row=1, column=3, padx=5, pady=5)
        self.pitch = tk.Entry(menuFrame2)
        self.pitch.grid(row=2, column=3, padx=5, pady=5)

        ##buttons
        self.btn_binary = tk.Button(menuFrame2, text="Binary Pattern", width=20)
        self.btn_binary.grid(row=3, column=0, sticky="W", padx=5, pady=(20,5))
        self.btn_sinusoidal = tk.Button(menuFrame2, text="Sinusoidal Pattern", width=20)
        self.btn_sinusoidal.grid(row=3, column=2, sticky="W", padx=5, pady=(20,5))