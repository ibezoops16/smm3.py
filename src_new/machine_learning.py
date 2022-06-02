import glob
import os
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
import seaborn as sns
from sklearn import metrics
import cv2
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


import pandas as pd
import pickle
from keras.models import Sequential
from keras.layers import Dense

class SupervisedML:

    def __init__(self):
        self.data_train_X = pd.DataFrame()
        self.data_train_Y = pd.DataFrame()
        self.data_test_X = pd.DataFrame()
        self.data_test_Y = pd.DataFrame()
        self.image_set = []
        self.image_set_1 = []
        self.image_set_2 = []
        # pass argument 1 for image data, 0 for csv data
        self.image_data = 0

    def read_csv_data(self, csv_file):
        self.sml_data = pd.read_csv(csv_file)
        feature_set_x = self.sml_data.iloc[:, :-1]
        label_set_y = self.sml_data.iloc[:, -1]
        return feature_set_x, label_set_y

    def load_images_from_folder(self, folder):
        cv_img = []
        for img in glob.glob(str(folder + "/*.bmp")):
            n = cv2.imread(img)
            cv_img.append(n)
        for img in glob.glob(str(folder + "/*.tif")):
            n = cv2.imread(img)
            cv_img.append(n)
        for img in glob.glob(str(folder + "/*.jpg")):
            n = cv2.imread(img)
            cv_img.append(n)
        for img in glob.glob(str(folder + "/*.png")):
            n = cv2.imread(img)
            cv_img.append(n)
        for img in glob.glob(str(folder + "/*.jpeg")):
            n = cv2.imread(img)
            cv_img.append(n)
        return cv_img

    def getFolderPath_trainingData(self):
        self.reset_all_sml_model_flag()
        training_data_path = ""
        if self.image_data == 1:
            folder_selected = filedialog.askdirectory()
            self.string_path.set(folder_selected)
            self.image_set = self.load_images_from_folder(folder_selected)
            print(len(self.image_set), "Images successfully read!!")

        else:
            self.file_selected = filedialog.askopenfilename(initialdir="C:/Users/dipbh/PycharmProjects/SMMVPython/data/", title="Select file")
            self.string_path.set(self.file_selected)

            self.data_train_X, self.data_tarin_Y = self.read_csv_data(self.file_selected)
            print("CSV Data successfully read!!")
            print("Features train set X: \n", self.data_train_X)
            print("Labels train set Y: \n", self.data_tarin_Y)

    def getFolderPath_testingData(self):
        training_data_path = ""
        if self.image_data == 1:
            folder_selected = filedialog.askdirectory()
            self.string_path.set(folder_selected)
            self.image_set = self.load_images_from_folder(folder_selected)
            print(len(self.image_set), "Images successfully read!!")

        else:
            self.file_selected = filedialog.askopenfilename(
                initialdir="C:/Users/dipbh/PycharmProjects/SMMVPython/data/", title="Select file")
            self.string_path.set(self.file_selected)

            self.data_test_X, self.data_test_Y = self.read_csv_data(self.file_selected)
            print("CSV Data successfully read!!")
            print("Features test set X: \n", self.data_test_X)
            print("Labels test set Y: \n", self.data_test_Y)

    def load_training_data(self):
        gui = Tk()

        gui.geometry("550x100")
        gui.title("Select Training Data")

        self.string_path = StringVar()
        a = Label(gui, text="Enter name")
        a.grid(row=0, column=0, ipadx=5, ipady=5, padx=5, pady=5)
        E = Entry(gui, textvariable=self.string_path, width=40)
        E.grid(row=0, column=1, ipadx=5, ipady=5, padx=5, pady=5)
        btnFind = ttk.Button(gui, text="Browse", command=self.getFolderPath_trainingData)
        btnFind.grid(row=0, column=2, ipadx=5, ipady=5, padx=5, pady=5)

        c = ttk.Button(gui, text="Confirm Path", command=gui.destroy)
        c.grid(row=4, column=1, ipadx=5, ipady=5, padx=5, pady=5)
        gui.mainloop()

    def load_testing_data(self):
        gui = Tk()

        gui.geometry("550x100")
        gui.title("Select Testing Data")

        self.string_path = StringVar()
        a = Label(gui, text="Enter name")
        a.grid(row=0, column=0, ipadx=5, ipady=5, padx=5, pady=5)
        E = Entry(gui, textvariable=self.string_path, width=40)
        E.grid(row=0, column=1, ipadx=5, ipady=5, padx=5, pady=5)
        btnFind = ttk.Button(gui, text="Browse", command=self.getFolderPath_testingData)
        btnFind.grid(row=0, column=2, ipadx=5, ipady=5, padx=5, pady=5)

        c = ttk.Button(gui, text="Confirm Path", command=gui.destroy)
        c.grid(row=4, column=1, ipadx=5, ipady=5, padx=5, pady=5)
        gui.mainloop()

    def load_multiple_training_data(self):
        gui = Tk()

        gui.geometry("700x100")
        gui.title("Select Image Set 1")

        self.string_path = StringVar()
        a = Label(gui, text="Enter name")
        a.grid(row=0, column=0, ipadx=5, ipady=5, padx=5, pady=5)
        E = Entry(gui, textvariable=self.string_path, width=40)
        E.grid(row=0, column=1, ipadx=5, ipady=5, padx=5, pady=5)
        btnFind = ttk.Button(gui, text="Select Imageset 1", command=self.select_image_set_1)
        btnFind.grid(row=0, column=2, ipadx=5, ipady=5, padx=5, pady=5)
        btnFind = ttk.Button(gui, text="Select Imageset 2", command=self.select_image_set_2)
        btnFind.grid(row=0, column=3, ipadx=5, ipady=5, padx=5, pady=5)
        c = ttk.Button(gui, text="Confirm Path", command=gui.destroy)
        c.grid(row=4, column=1, ipadx=5, ipady=5, padx=5, pady=5)
        gui.mainloop()

    def select_image_set_1(self):
        folder_selected = filedialog.askdirectory()
        self.string_path.set(folder_selected)
        self.image_set_1 = self.load_images_from_folder(folder_selected)
        print(len(self.image_set_1), "Images successfully read!!")

    def select_image_set_2(self):
        folder_selected = filedialog.askdirectory()
        self.string_path.set(folder_selected)
        self.image_set_2 = self.load_images_from_folder(folder_selected)
        print(len(self.image_set_2), "Images successfully read!!")

    def reset_all_sml_model_flag(self):
        self.svm_flag = 0
        self.rfc_flag = 0
        self.nn_flag = 0
        self.nbc_flag = 0
        self.dtc_flag = 0
        self.knn_flag = 0
        self.lr_flag = 0
        self.gbcr_flag = 0

    def train_sml_suport_vector_machine(self):
        self.svm_clf = SVC(kernel='linear')  # Linear Kernel
        self.svm_clf.fit(self.data_train_X, self.data_tarin_Y)
        self.svm_flag = 1
        print("Support Vector Machine Model Trained Successfully!!")

    def train_sml_random_forest(self):
        self.rfc_clf = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=5, min_samples_split=2,
                                              min_samples_leaf=1, max_features='auto', bootstrap=False, oob_score=False,
                                              n_jobs=1, verbose=0)
        self.rfc_clf.fit(self.data_train_X, self.data_tarin_Y)
        self.rfc_flag = 1
        print("Random Forest Classifier Model Trained Successfully!!")

    def train_sml_neural_network(self):
        self.nn_clf = MLPClassifier(hidden_layer_sizes=(8, 8, 8), activation='relu', solver='adam', max_iter=500)
        self.nn_clf.fit(self.data_train_X, self.data_tarin_Y)
        self.nn_flag = 1
        print("Neural Network model Trained Successfully!!")

    def train_sml_naive_bayes(self):
        self.nbc_clf = GaussianNB()
        self.nbc_clf.fit(self.data_train_X, self.data_tarin_Y)
        self.nbc_flag = 1
        print("Naive Bayes Classification Trained Successfully!!")

    def train_sml_decision_tree(self):
        self.dtc_clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
        self.dtc_clf.fit(self.data_train_X, self.data_tarin_Y)
        self.dtc_flag = 1
        print("Decision Tree Classifier Trained Successfully!!")

    def train_sml_knearest_neighbors(self):
        self.knn_clf = KNeighborsClassifier(n_neighbors=5)
        self.knn_clf.fit(self.data_train_X, self.data_tarin_Y)
        self.knn_flag = 1
        print("K-nearest Neighbors Classifier Model Trained Successfully!!")

    def train_sml_logistic_regression(self):
        self.lr_clf = LogisticRegression(solver='liblinear')
        self.lr_clf.fit(self.data_train_X, self.data_tarin_Y)
        self.lr_flag = 1
        print("Logistic Regression Model Trained Successfully!!")

    def train_sml_gradient_boosting(self):
        self.gbcr_clf = GradientBoostingClassifier()
        self.gbcr_clf.fit(self.data_train_X, self.data_tarin_Y)
        self.gbcr_flag = 1
        print("Gradient Boosting Classifier Regression Model Trained Successfully!!")

    def train_all_sml_models(self):
        self.train_sml_suport_vector_machine()
        self.train_sml_random_forest()
        self.train_sml_neural_network()
        self.train_sml_naive_bayes()
        self.train_sml_decision_tree()
        self.train_sml_knearest_neighbors()
        self.train_sml_logistic_regression()
        self.train_sml_gradient_boosting()

    def predict_all_sml_models(self):
        self.load_testing_data()
        # add an empty row to the dataframe
        self.sml_data = self.sml_data.append(pd.Series(), ignore_index=True)

        # support vector machine
        if self.svm_flag == 1:
            self.svm_y_pred = self.svm_clf.predict(self.data_test_X)
            accuracy_svm = metrics.accuracy_score(self.data_test_Y, self.svm_y_pred)
            print("Accuracy of Support Vector Machine:", accuracy_svm)
            print("Report of Support Vector Machine:", classification_report(self.data_test_Y, self.svm_y_pred))
            print("****************************************************")
            self.sml_data['SVM Predicted Value'] = pd.Series(self.svm_y_pred)
            self.sml_data.loc[[len(self.sml_data) - 1], 'SVM Predicted Value'] = accuracy_svm

        # random forest
        if self.rfc_flag == 1:
            self.rfc_y_pred = self.rfc_clf.predict(self.data_test_X)
            accuracy_rfc = metrics.accuracy_score(self.data_test_Y, self.rfc_y_pred)
            print("Accuracy of Random Forest Classifier:", accuracy_rfc)
            print("Report of Random Forest Classifier:", classification_report(self.data_test_Y, self.rfc_y_pred))
            print("****************************************************")
            self.sml_data['Random Forest Predicted Value'] = pd.Series(self.rfc_y_pred)
            self.sml_data.loc[[len(self.sml_data) - 1], 'Random Forest Predicted Value'] = accuracy_rfc

        # neural network
        if self.nn_flag == 1:
            self.nn_y_pred = self.nn_clf.predict(self.data_test_X)
            accuracy_nn = metrics.accuracy_score(self.data_test_Y, self.nn_y_pred)
            print("Accuracy of Neural Network:", accuracy_nn)
            print("Report of Neural Network:", classification_report(self.data_test_Y, self.nn_y_pred))
            print("****************************************************")
            self.sml_data['Neural Network Predicted Value'] = pd.Series(self.nn_y_pred)
            self.sml_data.loc[[len(self.sml_data) - 1], 'Neural Network Predicted Value'] = accuracy_nn

        # naive bayes classification
        if self.nbc_flag == 1:
            self.nbc_y_pred = self.nbc_clf.predict(self.data_test_X)
            accuracy_nbc = metrics.accuracy_score(self.data_test_Y, self.nbc_y_pred)
            print("Accuracy of Naive Bayes Classification:", accuracy_nbc)
            print("Report of Naive Bayes Classification:", classification_report(self.data_test_Y, self.nbc_y_pred))
            print("****************************************************")
            self.sml_data['Naive Bayes Predicted Value'] = pd.Series(self.nbc_y_pred)
            self.sml_data.loc[[len(self.sml_data) - 1], 'Naive Bayes Predicted Value'] = accuracy_nbc

        # decision tree classifier
        if self.dtc_flag == 1:
            self.dtc_y_pred = self.dtc_clf.predict(self.data_test_X)
            accuracy_dtc = metrics.accuracy_score(self.data_test_Y, self.dtc_y_pred)
            print("Accuracy od Decision Tree Classifier:", accuracy_dtc)
            print("Report of Decision Tree Classifier:", classification_report(self.data_test_Y, self.dtc_y_pred))
            print("****************************************************")
            self.sml_data['Decision Tree Predicted Value'] = pd.Series(self.dtc_y_pred)
            self.sml_data.loc[[len(self.sml_data) - 1], 'Decision Tree Predicted Value'] = accuracy_dtc

        # k-nearest neighbors
        if self.knn_flag == 1:
            self.knn_y_pred = self.knn_clf.predict(self.data_test_X)
            accuracy_knn = metrics.accuracy_score(self.data_test_Y, self.knn_y_pred)
            print("Accuracy of K-nearest Neighbors Classifier:", accuracy_knn)
            print("Report of K-nearest Neighbors Classifier:", classification_report(self.data_test_Y, self.knn_y_pred))
            print("****************************************************")
            self.sml_data['Knearest Neighnors Predicted Value'] = pd.Series(self.knn_y_pred)
            self.sml_data.loc[[len(self.sml_data) - 1], 'Knearest Neighnors Predicted Value'] = accuracy_knn

        # logistic regression
        if self.lr_flag == 1:
            self.lr_y_pred = self.lr_clf.predict(self.data_test_X)
            accuracy_lr = metrics.accuracy_score(self.data_test_Y, self.lr_y_pred)
            print("Accuracy of Logistic Regression:", accuracy_lr)
            print("Report of Logistic Regression:", classification_report(self.data_test_Y, self.lr_y_pred))
            print("****************************************************")
            self.sml_data['Logistic Regression Predicted Value'] = pd.Series(self.lr_y_pred)
            self.sml_data.loc[[len(self.sml_data) - 1], 'Logistic Regression Predicted Value'] = accuracy_lr

        # gradient boosting classifier regression
        if self.gbcr_flag == 1:
            self.gbcr_y_pred = self.gbcr_clf.predict(self.data_test_X)
            accuracy_gbcr = metrics.accuracy_score(self.data_test_Y, self.gbcr_y_pred)
            print("Accuracy of Gradient Boosting Classifier Regression:", accuracy_gbcr)
            print("Report of Gradient Boosting Classifier Regression:",classification_report(self.data_test_Y, self.gbcr_y_pred))
            print("****************************************************")
            self.sml_data['Gardient Boosting Predicted Value'] = pd.Series(self.gbcr_y_pred)
            self.sml_data.loc[[len(self.sml_data) - 1], 'Gardient Boosting Predicted Value'] = accuracy_gbcr

        self.sml_data.to_csv("G:/Pycharm/SMMVpy2/data/predicted csv file/csv_with_predicted_value_SML.csv")


class MIMO_Model:
    # button 7 for train all models
    # button 8 for predict trained models - which generates csv for comparison including featureset
    # button 9 to save trained models: filename = <ml_algo_name>_<date&time>.h5/.json

    def __init__(self):
        self.data_X = pd.DataFrame()
        self.data_Y = pd.DataFrame()
        self.test_data_X = pd.DataFrame()
        self.test_data_Y = pd.DataFrame()
        X, Y = make_regression(n_samples=2, n_features=4, n_informative=5, n_targets=1, random_state=1, noise=0.5)
        self.data_X = pd.DataFrame(X, columns=["a", "b", "c", "d"])
        self.data_Y = pd.DataFrame(Y, columns=["e"])
        X, y = make_regression(n_samples=2, n_features=4, n_informative=5, n_targets=1, random_state=1, noise=0.5)
        self.test_data_X = pd.DataFrame(X, columns=["a", "b", "c", "d"])
        self.test_data_Y = pd.DataFrame(Y, columns=["e"])

    def read_csv_data(self, csv_file):
        data = pd.read_csv(csv_file)
        feature_set_x = data.iloc[:, :-1]
        label_set_y = data.iloc[:, -1]
        return feature_set_x, label_set_y

    def getFolderPath(self):
        training_data_path = ""

        file_selected = filedialog.askopenfilename(initialdir="/", title="Select file")
        self.string_path.set(file_selected)

        self.data_X, self.data_Y = self.read_csv_data(file_selected)
        print("CSV Data successfully read!!")
        print("Features set X: \n", self.data_X)
        print("Labels set Y: \n", self.data_Y)

    def load_training_data(self):
        gui = Tk()

        gui.geometry("550x100")
        gui.title("Select Training Data")

        self.string_path = StringVar()
        a = Label(gui, text="Enter name")
        a.grid(row=0, column=0, ipadx=5, ipady=5, padx=5, pady=5)
        E = Entry(gui, textvariable=self.string_path, width=40)
        E.grid(row=0, column=1, ipadx=5, ipady=5, padx=5, pady=5)
        btnFind = ttk.Button(gui, text="Browse", command=self.getFolderPath)
        btnFind.grid(row=0, column=2, ipadx=5, ipady=5, padx=5, pady=5)

        c = ttk.Button(gui, text="Confirm Path", command=gui.destroy)
        c.grid(row=4, column=1, ipadx=5, ipady=5, padx=5, pady=5)
        gui.mainloop()

    def getTestFolderPath(self):
        training_data_path = ""

        file_selected = filedialog.askopenfilename(initialdir="/", title="Select file")
        self.string_path.set(file_selected)

        self.test_data_X, self.test_data_Y = self.read_csv_data(file_selected)
        print("CSV Data successfully read!!")
        print("Features set X: \n", self.test_data_X)
        print("Labels set Y: \n", self.test_data_Y)

    def load_testing_data(self):
        gui = Tk()

        gui.geometry("550x100")
        gui.title("Select Training Data")

        self.string_path = StringVar()
        a = Label(gui, text="Enter name")
        a.grid(row=0, column=0, ipadx=5, ipady=5, padx=5, pady=5)
        E = Entry(gui, textvariable=self.string_path, width=40)
        E.grid(row=0, column=1, ipadx=5, ipady=5, padx=5, pady=5)
        btnFind = ttk.Button(gui, text="Browse", command=self.getTestFolderPath)
        btnFind.grid(row=0, column=2, ipadx=5, ipady=5, padx=5, pady=5)

        c = ttk.Button(gui, text="Confirm Path", command=gui.destroy)
        c.grid(row=4, column=1, ipadx=5, ipady=5, padx=5, pady=5)
        gui.mainloop()
        pass

    def mimo_linear_regression_model(self, save_model):
        if self.data_X.shape[0] <= 2:
            print("Please select training and testing data")
        else:
            print("Number of Input Features: ", len(self.data_X.columns))

            # Train Model
            model = LinearRegression()
            model.fit(self.data_X, self.data_Y)
            if save_model != 0:
                pickle.dump(model, open("G:/Pycharm/SMMVpy2/data/Models/mimo_linear_regression_model.sav", 'wb'))

            # Make prediction
            predict_data_Y = model.predict(self.test_data_X)
            predict_data_Y = pd.Series(predict_data_Y)
            df = pd.concat([self.test_data_Y, predict_data_Y, self.test_data_Y - predict_data_Y], axis=1)
            df = df.rename(columns={"E": "Actual Labels", 0: "Predicted Labels", 1: "Difference"})
            print(df)

            print("\nmimo_linear_regression_model Trained Successfully!!")

    def mimo_k_nearest_neighbor(self, save_model):
        if self.data_X.shape[0] <= 2:
            print("Please select training and testing data")
        else:
            print("Number of Input Features: ", len(self.data_X.columns))

            # Train Model
            model = KNeighborsRegressor()
            model.fit(self.data_X, self.data_Y)
            if save_model != 0:
                pickle.dump(model, open("G:/Pycharm/SMMVpy2/data/Models/mimo_k_nearest_neighbor.sav", 'wb'))

            # Make prediction
            predict_data_Y = model.predict(self.test_data_X)
            predict_data_Y = pd.Series(predict_data_Y)
            df = pd.concat([self.test_data_Y, predict_data_Y, self.test_data_Y - predict_data_Y], axis=1)
            df = df.rename(columns={"E": "Actual Labels", 0: "Predicted Labels", 1: "Difference"})
            print(df)

            print("\nmimo_k_nearest_neighbor Trained Successfully!!")

    def mimo_decision_tree(self, save_model):
        if self.data_X.shape[0] <= 2:
            print("Please select training and testing data")
        else:
            print("Number of Input Features: ", len(self.data_X.columns))

            # Train Model
            model = DecisionTreeRegressor()
            model.fit(self.data_X, self.data_Y)
            if save_model != 0:
                pickle.dump(model, open("G:/Pycharm/SMMVpy2/data/Models/mimo_decision_tree.sav", 'wb'))

            # Make prediction
            predict_data_Y = model.predict(self.test_data_X)
            predict_data_Y = pd.Series(predict_data_Y)
            df = pd.concat([self.test_data_Y, predict_data_Y, self.test_data_Y - predict_data_Y], axis=1)
            df = df.rename(columns={"E": "Actual Labels", 0: "Predicted Labels", 1: "Difference"})
            print(df)

            print("\nMIMO Decision Tree Trained Successfully!!")

    def mimo_ann_model(self, save_model):
        if self.data_X.shape[0] <= 2:
            print("Please select training and testing data")
        else:
            print("Number of Input Features: ", len(self.data_X.columns))

            # define the keras model
            model = Sequential()
            model.add(Dense(12, input_dim=len(self.data_X.columns), activation='relu'))
            model.add(Dense(8, activation='relu'))
            model.add(Dense(8, activation='relu'))
            model.add(Dense(8, activation='relu'))
            model.add(Dense(1, activation='relu'))

            # compile and train the keras model
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(self.data_X, self.data_Y, epochs=150, batch_size=10)
            if save_model != 0:
                #pickle.dump(model, open("G:/Pycharm/SMMVpy2/data/Models/mimo_ann_model.sav", 'wb'))
                pass
            # Make prediction
            predict_data_Y = model.predict(self.test_data_X)
            predict_data_Y = predict_data_Y.reshape((len(predict_data_Y),))
            difference = self.test_data_Y - predict_data_Y
            predict_data_Y = pd.DataFrame(predict_data_Y)
            df = pd.concat([self.test_data_Y, predict_data_Y, difference], axis=1)
            df = df.rename(columns={"E": "Actual Labels", 0: "Predicted Labels", 1: "Difference"})
            print(df)

            print("\nMIMO Artificial Neural Network Trained Successfully!!")

    def train_all_models(self):
        if self.data_X.shape[0] <= 2:
            print("Please select training and testing data")
        else:
            self.mimo_linear_regression_model(0)
            self.mimo_k_nearest_neighbor(0)
            self.mimo_decision_tree(0)
            self.mimo_ann_model(0)

    def save_all_models(self):
        if self.data_X.shape[0] <= 2:
            print("Please select training and testing data")
        else:
            self.mimo_linear_regression_model(1)
            self.mimo_k_nearest_neighbor(1)
            self.mimo_decision_tree(1)
            self.mimo_ann_model(1)

    def predict_all_models(self):
        if self.data_X.shape[0] <= 2:
            print("Please select training and testing data")
        else:
            mimo_lr_model = pickle.load(
                open("G:/Pycharm/SMMVpy2/data/Models/mimo_linear_regression_model.sav", 'rb'))
            # Make prediction
            predict_data_Y = mimo_lr_model.predict(self.test_data_X)
            print("\n\nTesting Input: \n", self.test_data_X, "\n\nPredicted Output: \n", predict_data_Y)

            mimo_knn_model = pickle.load(
                open("G:/Pycharm/SMMVpy2/data/Models/mimo_k_nearest_neighbor.sav.sav", 'rb'))
            # Make prediction
            predict_data_Y = mimo_knn_model.predict(self.test_data_X)
            print("\n\nTesting Input: \n", self.test_data_X, "\n\nPredicted Output: \n", predict_data_Y)

            mimo_dt_model = pickle.load(open("G:/Pycharm/SMMVpy2/data/Models/mimo_decision_tree.sav.sav", 'rb'))
            # Make prediction
            predict_data_Y = mimo_dt_model.predict(self.test_data_X)
            print("\n\nTesting Input: \n", self.test_data_X, "\n\nPredicted Output: \n", predict_data_Y)

            mimo_ann_model = pickle.load(open("G:/Pycharm/SMMVpy2/data/Models/mimo_ann_model.sav.sav", 'rb'))
            # Make prediction
            predict_data_Y = mimo_ann_model.predict(self.test_data_X)
            print("\n\nTesting Input: \n", self.test_data_X, "\n\nPredicted Output: \n", predict_data_Y)


class UnsupervisedML:

    def __init__(self):
        self.data_X = pd.DataFrame()
        self.data_Y = pd.DataFrame()
        self.image_set = []
        # pass argument 1 for image data, 0 for csv data
        self.image_data = 1

    def read_csv_data(self, csv_file):
        data = pd.read_csv(csv_file)
        feature_set_x = data.iloc[:, :-1]
        label_set_y = data.iloc[:, -1]
        return feature_set_x, label_set_y

    def load_images_from_folder(self, folder):
        cv_img = []
        for img in glob.glob(str(folder + "/*.bmp")):
            n = cv2.imread(img)
            cv_img.append(n)
        return cv_img

    def getFolderPath(self):
        training_data_path = ""
        if self.image_data == 1:
            folder_selected = filedialog.askdirectory()
            self.string_path.set(folder_selected)
            self.image_set = self.load_images_from_folder(folder_selected)
            print(len(self.image_set), " Images successfully read!!")

        else:
            file_selected = filedialog.askopenfilename(initialdir="/", title="Select file")
            self.string_path.set(file_selected)
            self.data_X, self.data_Y = self.read_csv_data(file_selected)
            print("CSV Data successfully read!!")
            print("Features set X: \n", self.data_X)
            print("Labels set Y: \n", self.data_Y)

    def load_training_data(self):
        gui = Tk()
        gui.geometry("550x100")
        gui.title("Select Training Data Path")

        self.string_path = StringVar()
        a = Label(gui, text="Enter name")
        a.grid(row=0, column=0, ipadx=5, ipady=5, padx=5, pady=5)
        E = Entry(gui, textvariable=self.string_path, width=40)
        E.grid(row=0, column=1, ipadx=5, ipady=5, padx=5, pady=5)
        btnFind = ttk.Button(gui, text="Browse", command=self.getFolderPath)
        btnFind.grid(row=0, column=2, ipadx=5, ipady=5, padx=5, pady=5)

        c = ttk.Button(gui, text="Confirm Path", command=gui.destroy)
        c.grid(row=4, column=1, ipadx=5, ipady=5, padx=5, pady=5)
        gui.mainloop()

    def train_usml_model_1(self):
        print("USML Model 1 Trained Successfully!!")

    def train_usml_model_2(self):
        print("USML Model 2 Trained Successfully!!")

    def train_usml_model_3(self):
        print("USML Model 3 Trained Successfully!!")
