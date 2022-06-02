from image_processing_functions import ImageProcessing
from src_new.digital_fringe_projection import DigitalFringeProjection
from machine_learning import UnsupervisedML, SupervisedML, MIMO_Model
from camera_viewer import GrabImage
from project_image import ProjectImage


class GUIInterface:

    def __init__(self):
        self.image_processing_obj = ImageProcessing()
        self.dfp_obj = DigitalFringeProjection()
        self.grab_obj = GrabImage()
        self.usml = UnsupervisedML()
        self.sml = SupervisedML()
        self.pro_image = ProjectImage()
        self.input_img = [[0],[1]]
        self.mimoml = MIMO_Model()


    def start_gui(self):
        print("SMMV\n")
        user_ip = self.show_main_menu()
        self.process_main_menu(user_ip)

    def show_main_menu(self):
        print("Choose options: (1) Open Cam (2) Image Processing (3) Machine Learning (4) DFP (0) Exit")
        while True:
            try:
                temp_ip = int(input("Enter values from given options: "))
                return temp_ip
            except ValueError:
                print("Please input integer only...")
                continue

    def process_main_menu(self, user_ip):
        if user_ip == 1:
            self.grab_obj.grab_image()
            self.start_gui()
        elif user_ip == 2:
            print("\nImage Processing Module Started!!!")
            # convert_to_gray = True
            # self.input_img = self.image_processing_obj.get_input_image(convert_to_gray)
            self.image_processing_module()
        elif user_ip == 3:
            print("\nMachine Learning Module Started!!!")
            self.machine_learning_module()
        elif user_ip == 4:
            print("\nDigital Fringe Projection Module Started!!!")
            self.digital_fringe_projection_module()
        elif user_ip == 0:
            exit()
        else:
            print("\nInvalid!!! Select value only from available inputs.. ")
            self.start_gui()

    def image_processing_module(self):
        self.show_image_processing_menu()

    def show_image_processing_menu(self):
        print(
            "\nImage Processing options: (1) Load Image (2) Morphological IP (3) Filters (4) Frequency-domain IP "
            "(5) Edge detection (-1) Main Menu (-2) Previous Menu (0) Exit")
        while True:
            try:
                temp_ip = int(input("Enter values from given options: "))
                self.input_value = temp_ip
                self.process_image()
            except ValueError:
                print("Please input integer only...")
                continue

    ## Image Processing Module
    ## 1. Load Image
    ## 2. Morphological IP: (1) dilation, (2) erosion, (3) open, (4) close
    ## 3. Filters: (1) Gaussian, (2) Average, (3) Sharpening, ... --> User inputs filter size
    ## 4. Frequency-domain IP: (1) FFP, (2) Apply Mask (Band-pass filter), (3) Inverse FFT, ...
    ## 5. Edge detection: (1) Canny, (2) Hough, ...
    def process_image(self):
        if self.input_value == 1:
            convert_to_gray = True
            self.input_img = self.image_processing_obj.load_image(convert_to_gray)
            ## Debug image viewer problem
            self.show_image_processing_menu()
        elif self.input_value == 2:
            if len(self.input_img)>2 and len(self.input_img[0])>2:
                self.show_morphological_ip_menu()
            else:
                print("please load input image")
        elif self.input_value == 3:
            if len(self.input_img)>2 and len(self.input_img[0])>2:
                self.show_image_filters_menu()
            else:
                print("please load input image")
        elif self.input_value == 4:
            if len(self.input_img)>2 and len(self.input_img[0])>2:
                self.show_frequency_domain_ip_menu()
            else:
                print("please load input image")
        elif self.input_value == 5:
            if len(self.input_img)>2 and len(self.input_img[0])>2:
                self.show_edge_detection_menu()
            else:
                print("please load input image")
        elif self.input_value == -1:
            self.start_gui()
        elif self.input_value == -2:
            self.start_gui()
        elif self.input_value == 0:
            exit()
        else:
            print("\nInvalid!!! Select value only from available inputs.. ")
            self.image_processing_module()

    def show_morphological_ip_menu(self):
        print("Morphological IP Options: (1) dilation, (2) erosion, (3) open, (4) close")
        while True:
            try:
                temp_ip = int(input("Enter values from given options: "))
                self.input_value = temp_ip
                self.process_morphological_ip_menu()
            except ValueError:
                print("Please input integer only...")
                continue

    def process_morphological_ip_menu(self):
        if self.input_value == 1:
            self.image_processing_obj.morphological_dilation(self.input_img)
            print("Dilation")
            self.show_morphological_ip_menu()
        elif self.input_value == 2:
            self.image_processing_obj.morphological_erosion(self.input_img)
            print("Erosion")
            self.show_morphological_ip_menu()
        elif self.input_value == 3:
            print("Open")
            self.show_morphological_ip_menu()
        elif self.input_value == 4:
            print("Close")
            self.show_morphological_ip_menu()
        elif self.input_value == -1:
            self.start_gui()
        elif self.input_value == -2:
            self.start_gui()
        elif self.input_value == 0:
            exit()
        else:
            print("\nInvalid!!! Select value only from available inputs.. ")
            self.show_morphological_ip_menu()

    def show_image_filters_menu(self):
        print("Filters Options: (1) Gaussian, (2) Average, (3) Sharpening")
        while True:
            try:
                temp_ip = int(input("Enter values from given options: "))
                self.input_value = temp_ip
                self.process_image_filters_menu()
            except ValueError:
                print("Please input integer only...")
                continue

    def process_image_filters_menu(self):
        if self.input_value == 1:
            print("Gaussian")
            self.show_image_filters_menu()
        elif self.input_value == 2:
            print("Average")
            self.show_image_filters_menu()
        elif self.input_value == 3:
            print("Sharpening")
            self.show_image_filters_menu()
        elif self.input_value == -1:
            self.start_gui()
        elif self.input_value == -2:
            self.start_gui()
        elif self.input_value == 0:
            exit()
        else:
            print("\nInvalid!!! Select value only from available inputs.. ")
            self.show_image_filters_menu()

    def show_frequency_domain_ip_menu(self):
        print("Frequency Domain IP Options: (1) FFP, (2) High Pass Filter, (3) Low Pass Filter,"
              "(4) Apply Mask (Band-pass filter), (5) Inverse FFT")
        while True:
            try:
                temp_ip = int(input("Enter values from given options: "))
                self.input_value = temp_ip
                self.process_frequency_domain_ip_menu()
            except ValueError:
                print("Please input integer only...")
                continue

    def process_frequency_domain_ip_menu(self):
        if self.input_value == 1:
            self.image_processing_obj.fq_domain_ffp(self.input_img)
            self.show_frequency_domain_ip_menu()
        elif self.input_value == 2:
            self.image_processing_obj.fq_domain_highpass_filter(self.input_img)
            self.show_frequency_domain_ip_menu()
        elif self.input_value == 3:
            self.image_processing_obj.fq_domain_lowpass_filter(self.input_img)
            self.show_frequency_domain_ip_menu()
        elif self.input_value == 4:
            self.image_processing_obj.fq_domain_bandpass_filter(self.input_img)
            self.show_frequency_domain_ip_menu()
        elif self.input_value == 5:
            self.image_processing_obj.fq_domain_inverse_fft(self.input_img)
            self.show_frequency_domain_ip_menu()
        elif self.input_value == -1:
            self.start_gui()
        elif self.input_value == -2:
            self.start_gui()
        elif self.input_value == 0:
            exit()
        else:
            print("\nInvalid!!! Select value only from available inputs.. ")
            self.show_frequency_domain_ip_menu()

    def show_edge_detection_menu(self):
        print("Edge Detection Options: (1) Canny, (2) Hough")
        while True:
            try:
                temp_ip = int(input("Enter values from given options: "))
                self.input_value = temp_ip
                self.process_edge_detection_menu()
            except ValueError:
                print("Please input integer only...")
                continue

    def process_edge_detection_menu(self):
        if self.input_value == 1:
            print("Canny Edge Detection")
            self.show_edge_detection_menu()
        elif self.input_value == 2:
            print("Hough Transform")
            self.show_edge_detection_menu()
        elif self.input_value == -1:
            self.start_gui()
        elif self.input_value == -2:
            self.start_gui()
        elif self.input_value == 0:
            exit()
        else:
            print("\nInvalid!!! Select value only from available inputs.. ")
            self.show_edge_detection_menu()

    def digital_fringe_projection_module(self):
        self.input_value = self.show_digital_fringe_projection_menu()
        self.process_dfp_menu()
        pass

    def process_dfp_menu(self):
        """
        if self.input_value == 11:
            self.dfp_obj.grab_image()
            self.digital_fringe_projection_module()
        elif self.input_value == 12:
            self.dfp_obj.snap_image()
            self.digital_fringe_projection_module()
        elif self.input_value == 13:
            original_image = self.dfp_obj.load_image()
            self.digital_fringe_projection_module()
        """

        if self.input_value == 1:
            self.dfp_obj.set_f_Params()
            self.dfp_obj.set_p_Params()
            self.dfp_obj.dfp_proj_fringe()
            self.digital_fringe_projection_module()

        elif self.input_value == 2:
            self.dfp_obj.dfp_phase_wrapping()
            self.digital_fringe_projection_module()

        elif self.input_value == 3:
            self.dfp_obj.dfp_phase_unwrapping()
            self.digital_fringe_projection_module()

        elif self.input_value == 4:
            self.dfp_obj.dfp_demodulization()
            self.digital_fringe_projection_module()

        elif self.input_value == 5:
            self.dfp_obj.dfp_intensity_calibration()
            self.digital_fringe_projection_module()

        elif self.input_value == 6:
            self.dfp_obj.dfp_coordinate_calibration()
            self.digital_fringe_projection_module()

        elif self.input_value == 7:
            self.dfp_obj.dfp_calibration_block()
            self.digital_fringe_projection_module()

        elif self.input_value == 8:
            self.dfp_obj.dfp_displacement()
            self.digital_fringe_projection_module()

        elif self.input_value == -1:
            self.start_gui()
        elif self.input_value == -2:
            self.start_gui()
        elif self.input_value == 0:
            exit()
        else:
            print("\nInvalid!!! Select value only from available inputs.. ")
            self.digital_fringe_projection_module()
        pass

    def show_digital_fringe_projection_menu(self):
        print("\nDFP Module: (1) Setting, (2) Calibration, (3) Fringe Projection, (4) Run DFP, "
              "(-1) Return to Main Menu , (-2) Return to Previous Menu (0) Exit")
        ## 1. Setting: (1) Display Size, (2) Fringe Pitch, (3) Fringe Amplitude, (4) # of Shifts
        ## 2. Calibration: (1) Intensity Calibration, (2) Camera Lens Calibration
        ## 3. Fringe Projection: Project (1) Hor Sine Fringe, (2) Hor Binary Fringe, (3) Ver Sine Fringe, (4) Ver Binary Fringe
        ## 4. Run DFP: (1) Obtain Fringe Images, (2) Wrap Phase, (3) Unwrap Phase, (4) Get Heights, (4) Run All
        while True:
            try:
                temp_ip = int(input("Enter values from given options: "))
                return temp_ip
            except ValueError:
                print("Please input integer only...")
                continue

    def machine_learning_module(self):
        self.input_value = self.show_machine_learning_menu()
        self.process_machine_learning_menu()

    def show_machine_learning_menu(self):
        print("\nMachine Learning Options: (1) Supervised Machine Learning, (2) Unsupervised Machine"
              "Learning, (3) MIMO, (-1) Return to Main Menu, (-2) Return to previous Menu, (0) Exit")

        while True:
            try:
                temp_ip = int(input("Enter values from given options: "))
                return temp_ip
            except ValueError:
                print("Please input integer only...")
                continue

    def process_machine_learning_menu(self):
        if self.input_value == 1:
            print("\nSupervised Machine Learning Module Started!!!")
            self.supervised_ml_module()
        elif self.input_value == 2:
            print("\nUnsupervised Machine Learning Module Started!!!")
            self.unsupervised_ml_module()
        elif self.input_value == 3:
            print("\nUnsupervised Machine Learning Module Started!!!")
            self.mimo_ml_module()
        elif self.input_value == -1:
            self.start_gui()
        elif self.input_value == -2:
            self.start_gui()
        elif self.input_value == 0:
            exit()
        else:
            print("\nInvalid!!! Select value only from available inputs.. ")
            self.machine_learning_module()

    def unsupervised_ml_module(self):
        self.input_value = self.show_unsupervised_ml_menu()
        self.process_unsupervised_ml_menu()

    def show_unsupervised_ml_menu(self):
        print("\nUnsupervised ML Options: (1) Load Training Data, (2) Train USML Model 1, (3) Train USML Model 2"
              "(4) Train USML Model 3, (-1) Return to Main Menu, (-2) Return to Previous Menu, (0) Exit")
        while True:
            try:
                temp_ip = int(input("Enter values from given options: "))
                return temp_ip
            except ValueError:
                print("Please input integer only...")
                continue

    def process_unsupervised_ml_menu(self):
        if self.input_value == 1:
            self.usml.load_training_data()
            self.unsupervised_ml_module()
        elif self.input_value == 2:
            self.usml.train_usml_model_1()
            self.unsupervised_ml_module()
        elif self.input_value == 3:
            self.usml.train_usml_model_2()
            self.unsupervised_ml_module()
        elif self.input_value == 4:
            self.usml.train_usml_model_3()
            self.unsupervised_ml_module()
        elif self.input_value == -1:
            self.start_gui()
        elif self.input_value == -2:
            self.machine_learning_module()
        elif self.input_value == 0:
            exit()
        else:
            print("\nInvalid!!! Select value only from available inputs.. ")
            self.machine_learning_module()

    def supervised_ml_module(self):
        self.input_value = self.show_supervised_ml_menu()
        self.process_supervised_ml_menu()

    def show_supervised_ml_menu(self):
        print("\nUnsupervised ML Options: (1) Load Training Data, 2. Load Testing Data 3. Load Two Image Datasets"
              "4. Train SML Support Vector Machine 5. Train SML Random Forest Classifier"
              "6. Train SML Neural Network 7. Train SML Naive Bayes Classification 8. Train SML Decision Tree Classifier "
              "9. Train SML K-nearest Neighbors Classifier 10. Train SML Logistic Regression "
              "11. Train SML Gradient Boosting Classifier Regression 12. Train All SML Models "
              "13. Test All SML Models -1. Return to Main Menu -2. Return to Previous Menu "
              "0. Exit")
        while True:
            try:
                temp_ip = int(input("Enter values from given options: "))
                return temp_ip
            except ValueError:
                print("Please input integer only...")
                continue

    def process_supervised_ml_menu(self):
        if self.input_value == 1:
            self.sml.load_training_data()
            self.supervised_ml_module()
        elif self.input_value == 2:
            self.sml.load_testing_data()
            self.supervised_ml_module()
        elif self.input_value == 3:
            self.sml.load_multiple_training_data()
            self.supervised_ml_module()
        elif self.input_value == 4:
            self.sml.train_sml_suport_vector_machine()
            self.supervised_ml_module()
        elif self.input_value == 5:
            self.sml.train_sml_random_forest()
            self.supervised_ml_module()
        elif self.input_value == 6:
            self.sml.train_sml_neural_network()
            self.supervised_ml_module()
        elif self.input_value == 7:
            self.sml.train_sml_naive_bayes()
            self.supervised_ml_module()
        elif self.input_value == 8:
            self.sml.train_sml_decision_tree()
            self.supervised_ml_module()
        elif self.input_value == 9:
            self.sml.train_sml_knearest_neighbors()
            self.supervised_ml_module()
        elif self.input_value == 10:
            self.sml.train_sml_logistic_regression()
            self.supervised_ml_module()
        elif self.input_value == 11:
            self.sml.train_sml_gradient_boosting()
            self.supervised_ml_module()
        elif self.input_value == 12:
            self.sml.train_all_sml_models()
            self.supervised_ml_module()
        elif self.input_value == 13:
            self.sml.predict_all_sml_models()
            self.supervised_ml_module()
        elif self.input_value == -1:
            self.start_gui()
        elif self.input_value == -2:
            self.machine_learning_module()
        elif self.input_value == 0:
            exit()
        else:
            print("\nInvalid!!! Select value only from available inputs.. ")
            self.machine_learning_module()
    # MIMO Module

    def mimo_ml_module(self):
        self.input_value = self.show_mimo_ml_menu()
        self.process_mimo_ml_menu()

    def show_mimo_ml_menu(self):
        print("\nMIMO options: (1) Load Training Data (2) Load Testing Data (3) Train Linear Regression  (4) Train K nearest neighbors "
              "(5) Train Decision Tree\n (6) Train Artificial Neural Network (7) Train all models"
              "(8) Predict all models (9) Save all models (-1) Return to Main Menu"
              " (-2) Return to Previous Menu (0) Exit")
        while True:
            try:
                temp_ip = int(input("Enter values from given options: "))
                return temp_ip
            except ValueError:
                print("Please input integer only...")
                continue

    def process_mimo_ml_menu(self):
        if self.input_value == 1:
            self.mimoml.load_training_data()
            self.mimo_ml_module()
        elif self.input_value == 2:
            self.mimoml.load_testing_data()
            self.mimo_ml_module()
        elif self.input_value == 3:
            self.mimoml.mimo_linear_regression_model(0)
            self.mimo_ml_module()
        elif self.input_value == 4:
            self.mimoml.mimo_k_nearest_neighbor(0)
            self.mimo_ml_module()
        elif self.input_value == 5:
            self.mimoml.mimo_decision_tree(0)
            self.mimo_ml_module()
        elif self.input_value == 6:
            self.mimoml.mimo_ann_model(0)
            self.mimo_ml_module()
        elif self.input_value == 7:
            self.mimoml.train_all_models()
            self.mimo_ml_module()
        elif self.input_value == 8:
            self.mimoml.predict_all_models()
            self.mimo_ml_module()
        elif self.input_value == 9:
            self.mimoml.save_all_models()
            self.mimo_ml_module()
        elif self.input_value == -1:
            self.start_gui()
        elif self.input_value == -2:
            self.machine_learning_module()
        elif self.input_value == 0:
            exit()
        else:
            print("\nInvalid!!! Select value only from available inputs.. ")
            self.machine_learning_module()