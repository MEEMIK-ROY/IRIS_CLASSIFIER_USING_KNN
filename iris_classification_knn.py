# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 12:51:29 2021

@author: Admin
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn import metrics
import numpy as np
import tkinter as tk

#Load and check dataset andb target and features
iris=load_iris()
#Splitting datset into test and train dataset
x_train, x_test, y_train, y_test = train_test_split(iris['data'], iris['target'], test_size=0.20, random_state=1)
#Training the model using Knn Algorithm
try:
    k=knn(n_neighbors=11)
    k.fit(x_train,y_train)
    print("Model Training succesful")
except:
    print("Training unsuccessful, check for errors")
#Testing the model and finding accuracy
y_pred=k.predict(x_test)
accuracy=metrics.accuracy_score(y_test, y_pred)
print("Accuracy is : {}".format(accuracy))


#Creating an app for classification
class App:
    def __init__(self,window):
        self.head=tk.Label(window, text="IRIS SPECIES CLASSIFIER", font=("Times New Roman", 12))
        self.head.place(x=100, y= 10)
        self.lbl1=tk.Label(window, text="Enter Sepal Length (in cm)", font=("Arial", 10))
        self.lbl1.place(x=0, y= 50)
        self.lbl2=tk.Label(window, text="Enter Sepal Width (in cm)", font=("Arial", 10))
        self.lbl2.place(x=0, y= 100)
        self.lbl3=tk.Label(window, text="Enter Petal Length (in cm)", font=("Arial", 10))
        self.lbl3.place(x=0, y= 150)
        self.lbl4=tk.Label(window, text="Enter Petal Width (in cm)", font=("Arial", 10))
        self.lbl4.place(x=0, y= 200)
        self.lbl5=tk.Label(window,text="Classified Species", font=("Arial", 10))
        self.lbl5.place(x=0, y=400)
        self.t1=tk.Entry(window, bd=5)
        self.t1.place(x=180, y=50)
        self.t2=tk.Entry(window, bd=5)
        self.t2.place(x=180, y=100)
        self.t3=tk.Entry(window, bd=5)
        self.t3.place(x=180, y=150)
        self.t4=tk.Entry(window, bd=5)
        self.t4.place(x=180, y=200)
        self.btn=tk.Button(window, text="CLASSIFY", font=("Arial",10), command=self.classify)
        self.btn.place(x=100, y=300)
        self.res=tk.Entry(window,bd=5)
        self.res.place(x=180,y=400)
    
    def classify(self):
        self.res.delete(0, 'end')
        sepal_length=float(self.t1.get())
        sepal_width=float(self.t2.get())
        petal_length=float(self.t3.get())
        petal_width=float(self.t4.get())
        r=np.array([[sepal_length,sepal_width,petal_length,petal_width]])
        y=k.predict(r)
        s=str(iris['target_names'][y])
        s=s[2:-2]
        self.res.insert(0, s)
        
    
window=tk.Tk()
window.geometry("400x500")
window.title("Iris Flower Classification using knn")
app=App(window)
window.mainloop()   
    
   
    
    
    
    
    
    
