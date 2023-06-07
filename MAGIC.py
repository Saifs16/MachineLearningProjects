#Progress "Machine Learning for Everybody" 55:00Min
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score # To test the accuracy of the ML

#Naming the columns
cols = ["fLength", "fWidth", "fSize", "fConc", "fConcl", "fAsym",
         "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
#Reading the datasheet, the columns have been named 'cols'
df = pd.read_csv('magic04.data', names=cols)

#use print() when printing on the terminal 
#print(df.head())

# g = gamma | h = hadron
df["class"] = (df["class"] == "g").astype(int)
#print(df["class"]) # Only prints 'class' column
#print(df.head())
#
for label in cols[:-1]:
    #Drawing a histogram charts for each column(11 total) 
    plt.hist(df[df["class"]==1][label], color='blue', label='gamma', alpha=0.7, density=True)
    plt.hist(df[df["class"]==0][label], color='red', label='hadron', alpha=0.7, density=True)
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    #plt.show() #Comment and uncomment to show or hide graphs

""" The dataset is split into 3 portions
Train: a set used to train the Machine Learning system
Validation: to tune and pick optimal point for the ML performance
Test: is for giving an unbiased final evaluation for the ML 
"""
train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])
""" 
dataframe refers to the three groups of data created above
oversample is default to False, you can change to True when function is called
turn to True when database is imbalanced
""" 
def scale_dataset(dataframe, oversample=False):
    #[:-1] takes all but the last column
    x = dataframe[dataframe.columns[:-1]].values
    #[-1] only takes last column
    y = dataframe[dataframe.columns[-1]].values

    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    # Only when oversample is True in function
    # balances data gamma and hadron for unbia performance
    if oversample:
        ros = RandomOverSampler()
        x, y = ros.fit_resample(x,y)

    data = np.hstack((x,np.reshape(y,(-1,1))))

    return data, x, y

#print(len(train[train["class"]==1])) # gamma
#print(len(train[train["class"]==0])) 
""" 
The variable starting with 'x_' contain 
the coordonates 'y_' variables contain 
the labels the other variable contains both
"""
train, x_train, y_train = scale_dataset(train, oversample=True)
#Oversampling is off for valid and test database because we want to see the ML performace
valid, x_valid, y_valid = scale_dataset(valid, oversample=False)  
test, x_test, y_test = scale_dataset(test, oversample=False)

# The below is the K-nearest Neighbor method to achieve the Supervised Machine Learning Output
knn_model = KNeighborsClassifier(n_neighbors=1)
#The 'fit' takes all the vaules x-axis and y-axis of 'train' and trains its self  
knn_model.fit(x_train,y_train)
# here we make the ML predict the results of 
y_pred = knn_model.predict(x_test) 
print(y_pred)
print(y_test) 

""" 
Below code is to test the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
"""