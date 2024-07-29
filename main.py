import pandas as pd
#from pandas import DataFrame
import numpy as np
#classification
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
#from  sklearn.datasets import load_breast_cancer 
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectPercentile,chi2,f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
#from  sklearn.tree  import DecisionTreeRegressor
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import six
import sys
sys.modules['sklearn.externals.six'] = six
#from id3 import Id3Estimator

MinMaxScale=MinMaxScaler(feature_range=(1,10))
dataset=pd.read_csv("CustomersDataset(1).csv")
dataset=pd.DataFrame(dataset)
#print(dataset.columns.values)

maxtenure=dataset['tenure'].max()
minTenure=dataset['tenure'].min()
maxMonthly=dataset['MonthlyCharges'].max()
minMonthly=dataset['MonthlyCharges'].min()
maxTotal=dataset['TotalCharges'].max()
minTotal=dataset['TotalCharges'].min()

column_scale=['tenure','MonthlyCharges','TotalCharges']
#scale
dataset[column_scale]=MinMaxScale.fit_transform(dataset[column_scale])

#--------------------
dataset['PhoneService'].replace(['Yes','No'],[1,0],inplace= True)

dataset['InternetService'].replace(['Fiber optic','DSL','No'],[2,1,0],inplace= True)

dataset['OnlineSecurity'].replace(['Yes','No','No internet service'],[2,1,0],inplace= True)

dataset['OnlineBackup'].replace(['Yes','No','No internet service'],[2,1,0],inplace= True)

dataset['DeviceProtection'].replace(['Yes','No','No internet service'],[2,1,0],inplace= True)

dataset['TechSupport'].replace(['Yes','No','No internet service'],[2,1,0],inplace= True)

dataset['StreamingTV'].replace(['Yes','No','No internet service'],[2,1,0],inplace= True)

dataset['StreamingMovies'].replace(['Yes','No','No internet service'],[2,1,0],inplace= True)

dataset['Contract'].replace(['Month-to-month','One year','Two year'],[0,1,2],inplace= True)

dataset['PaperlessBilling'].replace(['Yes','No'],[1,0],inplace= True)

dataset['PaymentMethod'].replace(['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)'],[1,2,3,4],inplace= True)

dataset['Churn'].replace(['Yes','No'],[1,0],inplace= True)

dataset['gender'].replace(['Female','Male'],[1,2],inplace= True)

dataset['Partner'].replace(['Yes','No'],[1,0],inplace= True)

dataset['MultipleLines'].replace(['Yes','No','No phone service'],[2,1,0],inplace=True)

dataset['Dependents'].replace(['Yes','No'],[1,0],inplace= True)
#dataset['MonthlyCharges'].

#dataset['TotalCharges'].

#dataset['tenure'].
#/////////////////////
#print(dataset['Partner'])
#print('----------------------')
#print(dataset['Contract'])
#/////////////////////
dataset.drop(['customerID'],axis=1,inplace=True)
#dataset.drop(['PhoneService'],axis=1,inplace=True)
#X=dataset.iloc[:,0:17].copy()
Y_tree=dataset['Churn'].copy()
X_tree=dataset.iloc[:,4:19].copy()
X_tree.drop(['PhoneService'],axis=1,inplace=True)
X_tree.drop(['StreamingTV'],axis=1,inplace=True)
#X.drop(['PhoneService'],axis=1,inplace=True)
#X.drop(['Contract'],axis=1,inplace=True)
#X.drop(['StreamingMovies'],axis=1,inplace=True)
model = ExtraTreesClassifier(n_estimators=150)
model.fit(X_tree, Y_tree)
#print("extra tree",model.feature_importances_)

#id3 module
#clf = Id3Estimator(min_samples_split=32,min_entropy_decrease=23,is_repeating=False)

'''
sel = RandomForestClassifier(n_estimators = 30) 
sel.fit(X,Y)
importance=sel.feature_importances_
print(importance)
'''
#FeatureSelection = GenericUnivariateSelect(score_func= f_classif, mode= 'fdr', param=10) 
# score_func can = f_classif
#FeatureSelection = SelectKBest(score_func= f_classif ,k=4) 

#knn
x_knn=dataset.iloc[:,0:19].copy()
y_knn=dataset['Churn'].copy()
#-----svm
x_svm=dataset.iloc[:,0:19].copy()
y_svm=dataset['Churn'].copy()
#X = FeatureSelection.fit_transform(X, Y)
#------------for logistic
Y_l=dataset['Churn'].copy()
X_l=dataset.iloc[:,3:19]
#knn
X_train_knn,X_test_knn,Y_train_knn,Y_test_knn=train_test_split(x_knn,y_knn,test_size=0.30,random_state=200,shuffle=True)
classifier = KNeighborsClassifier(n_neighbors=10)
classifier.fit(X_train_knn, Y_train_knn)
y_pred_knn = classifier.predict(X_test_knn)
print("---------------------------")
print("KNN classification report \n",classification_report(Y_test_knn, y_pred_knn))
print("KNN confusion matrix \n",confusion_matrix(Y_test_knn, y_pred_knn))
#-------for svm

X_train_svm,X_test_svm,Y_train_svm,Y_test_svm=train_test_split(x_svm,y_svm,test_size=0.30,random_state=200,shuffle=True)

#------------------------dor logistic
X_train_l,X_test_l,Y_train_l,Y_test_l=train_test_split(X_l,Y_l,test_size=0.30,random_state=200,shuffle=True)

#---for id3
X_train,X_test,Y_train,Y_test=train_test_split(X_tree,Y_tree,test_size=0.30,random_state=200,shuffle=True)

#logistic regression

LogisticRegressionModel = LogisticRegression(penalty='l1',solver='liblinear',C=0.1,random_state=200,tol=0.00000001)
LogisticRegressionModel.fit(X_train_l, Y_train_l)
Y_pred_L=LogisticRegressionModel.predict(X_test_l)

print("logistic report: \n",classification_report(Y_test_l,Y_pred_L))
#print('LogisticRegressionModel Train Score is : ' , LogisticRegressionModel.score(X_train, Y_train))
#print('LogisticRegressionModel Test Score is : ' , LogisticRegressionModel.score(X_test, Y_test))
#print('LogisticRegressionModel Classes are : ' , LogisticRegressionModel.classes_)

#x21=np.array([49,	2,	2,	1,	2,	2,	1,	2,	0,	1,	3,	103.7,	5036.3]).reshape(1,-1)

SVRModel = SVC(kernel='rbf',C=40,tol=0.0000000001,max_iter=100000,probability=True,random_state=10) 
SVRModel.fit(X_train_svm , Y_train_svm) 
Y_pred_S=SVRModel.predict(X_test_svm)
print("---------------------------")
print("SVC Classification report \n",classification_report(Y_test_svm,Y_pred_S))
#DecisionTreeModel=DecisionTreeRegressor(max_depth=18,random_state=80)

DecisionTreeModel = DecisionTreeClassifier(criterion='entropy',min_samples_leaf=15,splitter='best',min_samples_split=4,random_state=10)
DecisionTreeModel.fit(X_train, Y_train)
#clf.fit(X_train,Y_train)
Y_pred = DecisionTreeModel.predict(X_test)
#Y_pred1=clf.predict(X_test)
#clf.predict(X_train)
#AccScore = accuracy_score(Y_test, Y_pred1, normalize=False)
#ClassificationReport = classification_report(Y_test,Y_pred1)
#print('Classification Report is : ', ClassificationReport )
#print(DecisionTreeModel.predict(x21))
#print('Accuracy Score is : ', AccScore)
#print("clf 1:",clf.n_features)
#print("clf train score:",metrics.accuracy_score(Y_train, Y_pred2))
#print("clf test score:",metrics.accuracy_score(Y_test, Y_pred1))
print("----------------------------------")
#print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
print("decition tree train score : ",DecisionTreeModel.score(X_train, Y_train))
print("decition tree test score : ",DecisionTreeModel.score(X_test, Y_test))
print('DecisionTreeClassifierModel feature importances are : ' , DecisionTreeModel.feature_importances_)
print(classification_report(Y_test, Y_pred))
print("decision tree prediction: ",DecisionTreeModel.predict(X_tree.iloc[:10]))
print("Real Y: \n",Y_tree.iloc[:10])
print ("SVm prediction",SVRModel.predict(x_svm.iloc[:10]))
print("Logistic prediction",LogisticRegressionModel.predict(X_l.iloc[:10]))
print("KNN prediction: ",classifier.predict(x_knn.iloc[:10]))
#-----------functions-----------------------------------------------------------------------
def train_button():
    if(var1.get()==1):
        regression_train()
    if(var2.get()==1):
        svm_train()
    if(var3.get()==1):
        decisionTree_train()
    if(var4.get()==1):
        knn_train()
def test_button():
    if(var1.get()==1):
        regression_test()             
    if(var2.get()==1):
        svm_test()
    if(var3.get()==1):                  
        decisionTree_test()
    if(var4.get()==1):
        knn_test()
     
def regression_predict(gen_pred1,senior_pred1,partenter_pred1,Dependent_pred1,tenure_pred1,phone_pred1,Multiple_pred1,internet_pred1,Security_pred1,Online_pred1,Device_pred1,Tech_pred1,Streaming_pred1,Movies_pred1,Contract_pred1,Paperless_pred1,Payment_pred1,Monthly_pred1,Total_pred1):
    x_reg_prediction=np.array([Dependent_pred1,tenure_pred1,phone_pred1,Multiple_pred1,internet_pred1,Security_pred1,Online_pred1,Device_pred1,Tech_pred1,Streaming_pred1,Movies_pred1,Contract_pred1,Paperless_pred1,Payment_pred1,Monthly_pred1,Total_pred1]).reshape(1,-1)
    y_reg_prediction=LogisticRegressionModel.predict(x_reg_prediction)
    if y_reg_prediction==1:
        print("Yes")    
    elif y_reg_prediction==0:
        print("No")

def svm_predict(gen_pred1,senior_pred1,partenter_pred1,Dependent_pred1,tenure_pred1,phone_pred1,Multiple_pred1,internet_pred1,Security_pred1,Online_pred1,Device_pred1,Tech_pred1,Streaming_pred1,Movies_pred1,Contract_pred1,Paperless_pred1,Payment_pred1,Monthly_pred1,Total_pred1):
    x_svm_prediction=np.array([gen_pred1,senior_pred1,partenter_pred1,Dependent_pred1,tenure_pred1,phone_pred1,Multiple_pred1,internet_pred1,Security_pred1,Online_pred1,Device_pred1,Tech_pred1,Streaming_pred1,Movies_pred1,Contract_pred1,Paperless_pred1,Payment_pred1,Monthly_pred1,Total_pred1]).reshape(1,-1)
    y_svm_prediction=SVRModel.predict(x_svm_prediction)
    if y_svm_prediction==1:
        print("Yes")    
    elif y_svm_prediction==0:
        print("No")


def decisionTree_predict(gen_pred1,senior_pred1,partenter_pred1,Dependent_pred1,tenure_pred1,phone_pred1,Multiple_pred1,internet_pred1,Security_pred1,Online_pred1,Device_pred1,Tech_pred1,Streaming_pred1,Movies_pred1,Contract_pred1,Paperless_pred1,Payment_pred1,Monthly_pred1,Total_pred1):
    x_tree_prediction=np.array([tenure_pred1,Multiple_pred1,internet_pred1,Security_pred1,Online_pred1,Device_pred1,Tech_pred1,Movies_pred1,Contract_pred1,Paperless_pred1,Payment_pred1,Monthly_pred1,Total_pred1]).reshape(1,-1)
    y_tree_prediction=DecisionTreeModel.predict(x_tree_prediction)
    print("there is error in decision tree")
    if y_tree_prediction==1:
        print("Yes")    
    elif y_tree_prediction==0:
        print("No")
def knn_predict(gen_pred1,senior_pred1,partenter_pred1,Dependent_pred1,tenure_pred1,phone_pred1,Multiple_pred1,internet_pred1,Security_pred1,Online_pred1,Device_pred1,Tech_pred1,Streaming_pred1,Movies_pred1,Contract_pred1,Paperless_pred1,Payment_pred1,Monthly_pred1,Total_pred1):
     x_knn_prediction=np.array([gen_pred1,senior_pred1,partenter_pred1,Dependent_pred1,tenure_pred1,phone_pred1,Multiple_pred1,internet_pred1,Security_pred1,Online_pred1,Device_pred1,Tech_pred1,Streaming_pred1,Movies_pred1,Contract_pred1,Paperless_pred1,Payment_pred1,Monthly_pred1,Total_pred1]).reshape(1,-1)
     y_knn_prediction=classifier.predict(x_knn_prediction)
     if y_knn_prediction==1:
         print("Yes")    
     elif y_knn_prediction==0:
         print("No")
        
def regression_test():
    print('LogisticRegressionModel Test Score is : ' , LogisticRegressionModel.score(X_test_l, Y_test_l))
def regression_train():
    print('LogisticRegressionModel Train Score is : ' , LogisticRegressionModel.score(X_train_l, Y_train_l))

def svm_test():
    print('SVRModel Test Score is :',SVRModel.score(X_test_svm,Y_test_svm))

def svm_train():
    print('SVRModel Train Score is :',SVRModel.score(X_train_svm,Y_train_svm))
def knn_test():
       print('KNN classifier Test Score is :',classifier.score(X_test_knn,Y_test_knn))

def knn_train():
       print('KNN classifier Train Score is :',classifier.score(X_train_knn,Y_train_knn))



def decisionTree_test():
   #print("decision tree (id3 module) test score:",metrics.accuracy_score(Y_test, Y_pred1))
   print("decision tree test score : ",DecisionTreeModel.score(X_test, Y_test))
def decisionTree_train():
    #print("decision tree(id3 module) train score:",metrics.accuracy_score(Y_train, Y_pred2))
    print("decision tree train score : ",DecisionTreeModel.score(X_train, Y_train))
'''
gen_pred=Gender_txt.get("1.0","end")
senior_pred=Senior_txt.get("1.0","end")
     partenter_pred=Partner_txt.get("1.0","end")
     Dependent_pred=Dependent_txt.get("1.0","end")
     tenure_pred=Tenure_txt.get("1.0","end")
     phone_pred=Phone_txt.get("1.0","end")
     Multiple_pred=Multiple_txt.get("1.0","end")
     internet_pred=Internet_txt.get("1.0","end")
     Security_pred=Security_txt.get("1.0","end")
     Online_pred=Online_txt.get("1.0","end")
     Device_pred=Device_txt.get("1.0","end")
     Tech_pred=Tech_txt.get("1.0","end")
     Streaming_pred=Streaming_txt.get("1.0","end")
     Movies_pred=Movies_txt.get("1.0","end")
     Contract_pred=Contract_txt.get("1.0","end")
     Paperless_pred=Paperless_txt.get("1.0","end")
     Payment_pred=Payment_txt.get("1.0","end")
     Monthly_pred=Monthly_txt.get("1.0","end")
     Total_pred=Total_txt.get("1.0","end")
'''   
#gen_pred,senior_pred,partenter_pred,Dependent_pred,tenure_pred,phone_pred,Multiple_pred,internet_pred,Security_pred,Online_pred,Device_pred,Tech_pred,Streaming_pred,Movies_pred,Contract_pred,Paperless_pred,Payment_pred,Monthly_pred,Total_pred  
def prediction():
    gen_pred1=gen_pred.get()
    senior_pred1=senior_pred.get()
    partenter_pred1=partenter_pred.get()
    Dependent_pred1=Dependent_pred.get()
    tenure_pred1=tenure_pred.get()
    phone_pred1=phone_pred.get()
    Multiple_pred1=Multiple_pred.get()
    internet_pred1=internet_pred.get()
    Security_pred1=Security_pred.get()
    Online_pred1=Online_pred.get()
    Device_pred1=Device_pred.get()
    Tech_pred1=Tech_pred.get()
    Streaming_pred1=Streaming_pred.get()
    Movies_pred1=Movies_pred.get()
    Contract_pred1=Contract_pred.get()
    Paperless_pred1=Paperless_pred.get()
    Payment_pred1=Payment_pred.get()
    #Monthly_pred1=Monthly_pred.get()
    Total_pred1=Total_pred.get()
    Monthly_pred1=(((Monthly_pred.get()-minMonthly)/(maxMonthly-minMonthly))*9+1)
    Total_pred1=(((Total_pred.get()-minTotal)/(maxTotal-minTotal))*9+1)
    tenure_pred1=((((tenure_pred.get()-minTenure)/(maxtenure-minTenure))*9)+1)
   #gender
    print("prediction clicked")
    if (gen_pred1=="Female"):
        gen_pred11=1
    elif (gen_pred1=="Male"):
        gen_pred11=2
    #partener
    if (partenter_pred1=='Yes'):
        partenter_pred11=1
    elif (partenter_pred1=='No'):
        partenter_pred11=0
    #dependent
    if (Dependent_pred1=='Yes'):
        Dependent_pred11=1
    elif (Dependent_pred1=='No'):
        Dependent_pred11=0
    #phone
    if (phone_pred1=='Yes'):
        phone_pred11=1
    elif (phone_pred1=='No'):
        phone_pred11=0
    #multiple    
    if (Multiple_pred1=='Yes'):
        Multiple_pred11=2
    elif (Multiple_pred1=='No'):
        Multiple_pred11=1   
    elif (Multiple_pred1=='No internet service'):
        Multiple_pred11=0
     #internet   
    if (internet_pred1=='Fiber optic'):
        internet_pred11=2
    elif (internet_pred1=='DSL'):
        internet_pred11=1   
    elif (internet_pred1=='No'):
        internet_pred11=0
      #secuirity  
    if (Security_pred1=='Yes'):
        Security_pred11=2
    elif (Security_pred1=='No'):
        Security_pred11=1   
    elif (Security_pred1=='No internet service'):
        Security_pred11=0   
        #backup
    if (Online_pred1=='Yes'):
        Online_pred11=2
    elif (Online_pred1=='No'):
        Online_pred11=1   
    elif (Online_pred1=='No internet service'):
        Online_pred11=0  
        #device    
    if (Device_pred1=='Yes'):
        Device_pred11=2
    elif (Device_pred1=='No'):
        Device_pred11=1   
    elif (Device_pred1=='No internet service'):
        Device_pred11=0
        #tech
    if (Tech_pred1=='Yes'):
        Tech_pred11=2
    elif (Tech_pred1=='No'):
        Tech_pred11=1   
    elif (Tech_pred1=='No internet service'):
        Tech_pred11=0
        #streaming 
    if (Streaming_pred1=='Yes'):
        Streaming_pred11=2
    elif (Streaming_pred1=='No'):
        Streaming_pred11=1   
    elif (Streaming_pred1=='No internet service'):
        Streaming_pred11=0
        #movies
    if (Movies_pred1=='Yes'):
        Movies_pred11=2
    elif (Movies_pred1=='No'):
        Movies_pred11=1   
    elif (Movies_pred1=='No internet service'):
        Movies_pred11=0
        #contract
    if (Contract_pred1=='Two year'):
        Contract_pred11=2
    elif (Contract_pred1=='One year'):
        Contract_pred11=1   
    elif (Contract_pred1=='Month-to-month'):
        Contract_pred11=0  
        #paperless
    if (Paperless_pred1=='Yes'):
        Paperless_pred11=1
    elif( Paperless_pred1=='No'):
        Paperless_pred11=0    
        #payment
    if (Payment_pred1=='Electronic check'):
        Payment_pred11=1
    elif (Payment_pred1=='Mailed check'):
        Payment_pred11=2   
    elif (Payment_pred1=='Bank transfer (automatic)'):
        Payment_pred11=3    
    elif (Payment_pred1=='Credit card (automatic)'):
        Payment_pred11=4
    #logistic var1.get()==1
    #print("Monthly_pred1",Monthly_pred1)
    #print("total pred 1", Total_pred1)
    #print("tenure ",tenure_pred1)
    if(LogisticRegressionModel.score(X_test_l, Y_test_l)>=DecisionTreeModel.score(X_test, Y_test) and LogisticRegressionModel.score(X_test_l, Y_test_l)>=SVRModel.score(X_test_svm,Y_test_svm) and LogisticRegressionModel.score(X_test_l, Y_test_l)>=classifier.score(X_test_knn,Y_test_knn)):
        print("logisticRegresssion Prediction")
        # print("tenure pred",tenure_pred1)
        regression_predict(gen_pred11,senior_pred1,partenter_pred11,Dependent_pred11,tenure_pred1,phone_pred11,Multiple_pred11,internet_pred11,Security_pred11,Online_pred11,Device_pred11,Tech_pred11,Streaming_pred11,Movies_pred11,Contract_pred11,Paperless_pred11,Payment_pred11,Monthly_pred1,Total_pred1)
    #svm var2.get()==1
    if(SVRModel.score(X_test_svm,Y_test_svm)>=LogisticRegressionModel.score(X_test_l, Y_test_l) and SVRModel.score(X_test_svm,Y_test_svm)>=DecisionTreeModel.score(X_test, Y_test) and SVRModel.score(X_test_svm,Y_test_svm) >=classifier.score(X_test_knn,Y_test_knn)):
        print("svm prediction")
        svm_predict(gen_pred11,senior_pred1,partenter_pred11,Dependent_pred11,tenure_pred1,phone_pred11,Multiple_pred11,internet_pred11,Security_pred11,Online_pred11,Device_pred11,Tech_pred11,Streaming_pred11,Movies_pred11,Contract_pred11,Paperless_pred11,Payment_pred11,Monthly_pred1,Total_pred1)
    #tree var3.get()==1
    if(DecisionTreeModel.score(X_test, Y_test)>=SVRModel.score(X_test_svm,Y_test_svm) and DecisionTreeModel.score(X_test, Y_test)>=LogisticRegressionModel.score(X_test_l, Y_test_l) and DecisionTreeModel.score(X_test, Y_test)>=classifier.score(X_test_knn,Y_test_knn)):
        print("decision tree prediction")
        decisionTree_predict(gen_pred11,senior_pred1,partenter_pred11,Dependent_pred11,tenure_pred1,phone_pred11,Multiple_pred11,internet_pred11,Security_pred11,Online_pred11,Device_pred11,Tech_pred11,Streaming_pred11,Movies_pred11,Contract_pred11,Paperless_pred11,Payment_pred11,Monthly_pred1,Total_pred1)
    #knn  var4.get()==1   
    if(classifier.score(X_test_knn,Y_test_knn)>=LogisticRegressionModel.score(X_test_l, Y_test_l) and classifier.score(X_test_knn,Y_test_knn)>=SVRModel.score(X_test_svm,Y_test_svm) and classifier.score(X_test_knn,Y_test_knn)>=DecisionTreeModel.score(X_test, Y_test)):
        print("knn prediction")
        knn_predict(gen_pred11,senior_pred1,partenter_pred11,Dependent_pred11,tenure_pred1,phone_pred11,Multiple_pred11,internet_pred11,Security_pred11,Online_pred11,Device_pred11,Tech_pred11,Streaming_pred11,Movies_pred11,Contract_pred11,Paperless_pred11,Payment_pred11,Monthly_pred1,Total_pred1)
       #

#------------tkinter-------    
from tkinter import *
from tkinter import ttk
import tkinter as tk
root =Tk()
root.geometry("1000x500")
var1 = IntVar()
var2 = IntVar()
var3 = IntVar()
var4 =IntVar()
'''
gen_pred=Gender_txt.get("1.0","end")
senior_pred=Senior_txt.get("1.0","end")
partenter_pred=Partner_txt.get("1.0","end")
Dependent_pred=Dependent_txt.get("1.0","end")
tenure_pred=Tenure_txt.get("1.0","end")
phone_pred=Phone_txt.get("1.0","end")
Multiple_pred=Multiple_txt.get("1.0","end")
internet_pred=Internet_txt.get("1.0","end")
Security_pred=Security_txt.get("1.0","end")
Online_pred=Online_txt.get("1.0","end")
Device_pred=Device_txt.get("1.0","end")
Tech_pred=Tech_txt.get("1.0","end")
Streaming_pred=Streaming_txt.get("1.0","end")
Movies_pred=Movies_txt.get("1.0","end")
Contract_pred=Contract_txt.get("1.0","end")
Paperless_pred=Paperless_txt.get("1.0","end")
Payment_pred=Payment_txt.get("1.0","end")
Monthly_pred=Monthly_txt.get("1.0","end")
Total_pred=Total_txt.get("1.0","end")
'''

gen_pred=tk.StringVar()
senior_pred=tk.IntVar()
partenter_pred=tk.StringVar()
Dependent_pred=tk.StringVar()
tenure_pred=tk.IntVar()
phone_pred=tk.StringVar()
Multiple_pred=tk.StringVar()
internet_pred=tk.StringVar()
Security_pred=tk.StringVar()
Online_pred=tk.StringVar()
Device_pred=tk.StringVar()
Tech_pred=tk.StringVar()
Streaming_pred=tk.StringVar()
Movies_pred=tk.StringVar()
Contract_pred=tk.StringVar()
Paperless_pred=tk.StringVar()
Payment_pred=tk.StringVar()
#Monthly_pred=tk.StringVar()
#Total_pred=tk.StringVar()
Monthly_pred=tk.DoubleVar()
Total_pred=tk.DoubleVar()

root.title("Service Cancellation Predictor")
btn1=tk.Button(root,text="Train",width="20",height="1",command=train_button)
btn1.pack()
btn1.place(x=0,y=70)
btn2=tk.Button(root,text="Test",width="20",height="1",command=test_button)
btn2.pack()
btn2.place(x=150,y=70)
x1=tk.Label(root,text="Mehtology")
x1.pack()
x1.place(x=50,y=5)
x1=tk.Checkbutton(root,text="Logistic Regression",variable=var1) #############
y1=tk.Checkbutton(root,text="SVM",variable=var2)
z1=tk.Checkbutton(root,text="ID3",variable=var3)
k1=tk.Checkbutton(root,text="knn",variable=var4)
x1.pack()
x1.place(x=20,y=30)
y1.pack()
y1.place(x=200,y=30)
z1.pack()
z1.place(x=300,y=30)
k1.pack()
k1.place(x=400,y=30)
x1=tk.Label(root,text="Customer Data")
x1.pack()
x1.place(x=50,y=100)
#-----------------------------------------
x1=tk.Label(root,text="Customer ID")
x1.pack()
x1.place(x=50,y=120)

ID_txt=tk.Entry(root)
ID_txt.pack()
ID_txt.place(x=170,y=120)
#--------------------------
x1=tk.Label(root,text="Partner")
x1.pack()
x1.place(x=50,y=150)

Partner_txt=tk.Entry(root,textvariable=partenter_pred)
Partner_txt.pack()
Partner_txt.place(x=170,y=150)
#--------------------------------
x1=tk.Label(root,text="Phone Service")
x1.pack()
x1.place(x=50,y=180)

Phone_txt=tk.Entry(root,textvariable=phone_pred)
Phone_txt.pack()
Phone_txt.place(x=170,y=180)
#--------------------------------------
x1=tk.Label(root,text="Online Security")
x1.pack()
x1.place(x=50,y=210)
Security_txt=tk.Entry(root,textvariable=Security_pred)
Security_txt.pack()
Security_txt.place(x=170,y=210)
#------------------------------------
x1=tk.Label(root,text="Tech support")
x1.pack()
x1.place(x=50,y=240)
Tech_txt=tk.Entry(root,textvariable=Tech_pred)
Tech_txt.pack()
Tech_txt.place(x=170,y=240)
#-------------------------------
x1=tk.Label(root,text="Contract")
x1.pack()
x1.place(x=50,y=270)
Contract_txt=tk.Entry(root,textvariable=Contract_pred)
Contract_txt.pack()
Contract_txt.place(x=170,y=270)
#---------------------------------
x1=tk.Label(root,text="Monthly Charges")
x1.pack()
x1.place(x=50,y=300)
Monthly_txt=tk.Entry(root,textvariable=Monthly_pred)
Monthly_txt.pack()
Monthly_txt.place(x=170,y=300)
#-----------------------------------------------------------------------------------------------------------------
x=tk.Label(root,text="Customer Data")
x.pack()
x.place(x=50,y=100)
#-----------------------------------------
x1=tk.Label(root,text="Gender")
x1.pack()
x1.place(x=350,y=120)

Gender_txt=tk.Entry(root,textvariable=gen_pred)
Gender_txt.pack()
Gender_txt.place(x=450,y=120)
#--------------------------
x1=tk.Label(root,text="Dependent")
x1.pack()
x1.place(x=350,y=150)

Dependent_txt=tk.Entry(root,textvariable=Dependent_pred)
Dependent_txt.pack()
Dependent_txt.place(x=450,y=150)
#--------------------------------
x1=tk.Label(root,text="Multiple lines")
x1.pack()
x1.place(x=350,y=180)

Multiple_txt=tk.Entry(root,textvariable=Multiple_pred)
Multiple_txt.pack()
Multiple_txt.place(x=450,y=180)
#--------------------------------------
x1=tk.Label(root,text="Online Backup")
x1.pack()
x1.place(x=350,y=210)
Online_txt=tk.Entry(root,textvariable=Online_pred)
Online_txt.pack()
Online_txt.place(x=450,y=210)
#------------------------------------
x1=tk.Label(root,text="Streaming TV")
x1.pack()
x1.place(x=350,y=240)
Streaming_txt=tk.Entry(root,textvariable=Streaming_pred)
Streaming_txt.pack()
Streaming_txt.place(x=450,y=240)
#-------------------------------
x1=tk.Label(root,text="Paperless Billing")
x1.pack()
x1.place(x=350,y=270)
Paperless_txt=tk.Entry(root,textvariable=Paperless_pred)
Paperless_txt.pack()
Paperless_txt.place(x=450,y=270)
#---------------------------------
x1=tk.Label(root,text="Total Charges")
x1.pack()
x1.place(x=350,y=300)
Total_txt=tk.Entry(root,textvariable=Total_pred)
Total_txt.pack()
Total_txt.place(x=450,y=300)
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
x1=tk.Label(root,text="Senior Citizen")
x1.pack()
x1.place(x=600,y=120)

Senior_txt=tk.Entry(root,textvariable=senior_pred)
Senior_txt.pack()
Senior_txt.place(x=700,y=120)
#--------------------------
x=tk.Label(root,text="Tenure")
x.pack()
x.place(x=600,y=150)

Tenure_txt=tk.Entry(root,textvariable=tenure_pred)
Tenure_txt.pack()
Tenure_txt.place(x=700,y=150)
#--------------------------------
x=tk.Label(root,text="Internet Service")
x.pack()
x.place(x=600,y=180)

Internet_txt=tk.Entry(root,textvariable=internet_pred)
Internet_txt.pack()
Internet_txt.place(x=700,y=180)
#--------------------------------------
x=tk.Label(root,text="Device Protection")
x.pack()
x.place(x=600,y=210)
Device_txt=tk.Entry(root,textvariable=Device_pred)
Device_txt.pack()
Device_txt.place(x=700,y=210)
#------------------------------------
x=tk.Label(root,text="Streaming Movies")
x.pack()
x.place(x=600,y=240)
Movies_txt=tk.Entry(root,textvariable=Movies_pred)
Movies_txt.pack()
Movies_txt.place(x=700,y=240)
#-------------------------------
x=tk.Label(root,text="Payment Method")
x.pack()
x.place(x=600,y=270)
Payment_txt=tk.Entry(root,textvariable=Payment_pred)
Payment_txt.pack()
Payment_txt.place(x=700,y=270)
#-------------------------------------------------------------
btn3=tk.Button(root,text="Predict",width="20",height="1",command=prediction)
btn3.pack()
btn3.place(x=350,y=330)
root.mainloop()

    