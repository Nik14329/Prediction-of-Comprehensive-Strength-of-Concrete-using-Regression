
In [14]:
# Important libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
import itertools as it
from sklearn.linear_model import LinearRegression # Linear regression
from sklearn.metrics import mean_squared_error	# Compute mean square error
from sklearn.model_selection import train_test_split	# Splitting dataset into training and test da
from sklearn.linear_model import Lasso	#Lasso Regression
from sklearn.neighbors import KNeighborsRegressor	#KNN Neighbor
from sklearn.svm import SVR	# SVM
from sklearn import metrics
%matplotlib inline
from sklearn.preprocessing import PolynomialFeatures from sklearn.preprocessing import PolynomialFeatures import statsmodels.api as sm


















In [15]:# Loading of dataset
df=pd.read_excel('C:/Users/Asus/Desktop/Test@2.xlsx')
print('Shape before deleting duplicate values:', df.shape) df.head(300)	#Reading of first 5 rows




Shape before deleting duplicate values: (228, 7)

Out[15]:	
		cement	blast_furnace_slag	water	coarse_aggregate	fine_aggregate	age _in_days	compressive_strength
	0	190.0	236.0	192.0	1026.6	781.2	28	32.817778
	1	210.7	173.0	203.5	958.2	825.5	28	31.560000
	2	190.0	190.0	171.0	1096.0	920.0	28	30.000000
	3	190.0	190.0	171.0	1096.0	920.0	28	29.955000
	4	190.0	190.0	171.0	1096.0	920.0	28	29.933000
	...	...	...	...	...	...	...	...
	223	190.0	190.0	171.0	1096.0	920.0	7	17.422000
	224	190.0	190.0	171.0	1096.0	920.0	7	17.418000
	225	190.0	190.0	171.0	1096.0	920.0	7	17.409000
	226	190.0	190.0	171.0	1096.0	920.0	6	17.260000
	227	190.0	190.0	171.0	1096.0	920.0	7	17.315000

228 rows × 7 columns
 
In [16]:# Data Structuring
print('Number of rows',df.shape[0])
print('Number of columns',df.shape[1]) print(df.info())




Number of rows 228 Number of columns 7
<class 'pandas.core.frame.DataFrame'> RangeIndex: 228 entries, 0 to 227
Data columns (total 7 columns):
#	Column	Non-Null Count Dtype
0	cement	228 non-null	float64
1	blast_furnace_slag	228 non-null	float64
2	water	228 non-null	float64
3	coarse_aggregate	228 non-null	float64
4	fine_aggregate	228 non-null	float64
5	age _in_days	228 non-null	int64
6	compressive_strength 228 non-null	float64 dtypes: float64(6), int64(1)
memory usage: 12.6 KB None

In [17]:
# Missing Values
print('Number of missing values', df.isnull().sum()) 'The dataset contains no missing values'



Number of missing values cement	0
blast_furnace_slag	0
water	0
coarse_aggregate	0
fine_aggregate	0
age _in_days	0
compressive_strength	0
dtype: int64

Out[17]: 'The dataset contains no missing values'
 
In [18]:
# Data visualization #1 Correlation Matrix import seaborn as sns
sns.heatmap(df.corr(), annot=True, linewidth=2) plt.title("Correlation between variables")
plt.show()

#2 Pair plot
sns.pairplot(df,markers="h") plt.show()

#3 Distribution plot
sns.distplot(df[' compressive_strength'], bins=10, color='b') plt.ylabel("Frequency")
plt.title('Distribution of concrete strength')













 
 

E:\Anaconda\lib\site-packages\seaborn\distributions.py:2619: FutureWarning: `distplot` is a depreca ted function and will be removed in a future version. Please adapt your code to use either `displot
` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for hist ograms).
warnings.warn(msg, FutureWarning)

Out[18]: Text(0.5, 1.0, 'Distribution of concrete strength')
 
 

In [19]:
# Distribution of components of concrete
cols = [i for i in df.columns if i not in 'compressive_strength'] length = len(cols)
cs = ["b","r","g","c","m","k","lime","c"] fig = plt.figure(figsize=(13,25))

for i,j,k in it.zip_longest(cols,range(length),cs): plt.subplot(4,2,j+1)
ax = sns.distplot(df[i],color=k,rug=True) ax.set_facecolor("w")
plt.axvline(df[i].mean(),linestyle="dashed",label="mean",color="k") plt.legend(loc="best")
plt.title(i,color="navy") plt.xlabel("")













 
In [20]:
# Scatterplot between components fig = plt.figure(figsize=(13,8)) ax = fig.add_subplot(111)
plt.scatter(df["water"],df["cement"],
c=df[" compressive_strength"],s=df[" compressive_strength"]*3, linewidth=1,edgecolor="k",cmap="viridis")
ax.set_facecolor("w") ax.set_xlabel("water") ax.set_ylabel("cement") lab = plt.colorbar()
lab.set_label(" compressive_strength") plt.title("cement vs water")
plt.show()













In [21]:
# Data Splitting
# The dataset is divided into a 70 to 30 splitting between training data and test data

train,test = train_test_split(df,test_size =.3,random_state = 0)
train_X = train[[x for x in train.columns if x not in [" compressive_strength"] + ["age_in_days"]]] train_Y = train[" compressive_strength"]
test_X = test[[x for x in test.columns if x not in [" compressive_strength"] + ["age_in_days"]]] test_Y = test[" compressive_strength"]

 
In [26]:
#Model 1= Multiple linear regression # fit a model
lm = LinearRegression()
model = lm.fit(train_X, train_Y) predictions = lm.predict(test_X) m1=model.score(test_X, test_Y)
RMSE1=round(np.sqrt(metrics.mean_squared_error(test_Y, predictions)),3) print('Accuracy of model is', round(model.score(test_X, test_Y),3))
print('Mean Absolute Error:', round(metrics.mean_absolute_error(test_Y, predictions),3)) print('Mean Squared Error:', round(metrics.mean_squared_error(test_Y, predictions),3)) print('Root Mean Squared Error:', RMSE1)











Accuracy of model is 0.817 Mean Absolute Error: 2.702 Mean Squared Error: 9.962
Root Mean Squared Error: 3.156

In [27]:#Plot of true value vs. predicted values
dat = pd.DataFrame({'Actual': test_Y, 'Predicted': predictions}) dat1=dat.head(25) #just a sample which shows top 25 columns
dat1.plot(kind='bar',figsize=(7,7))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green') plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black') plt.show()









In [28]:
#initiate linear regression model
model = LinearRegression()

 
In [32]:
#define predictor and response variables
X = df[['cement', 'blast_furnace_slag', 'water', 'coarse_aggregate','fine_aggregate','age _in_days'] y = df.iloc[:,-1]
model.fit(X, y)
#print regression coefficients
z=pd.DataFrame(zip(X.columns, model.coef_)) print("The Coefficents are as follows: ")
print(z)









The Coefficents are as follows:
0	1
0	cement 0.008055
1 blast_furnace_slag 0.009700 2	water 0.062254
3	coarse_aggregate 0.012303
4	fine_aggregate -0.025220
5	age _in_days 0.091122

In [34]:
#print intercept value
print(f"The Intercept is {round(model.intercept_,3)}")

 




In [36]:
 cement=float(input("enter cement quantity : "))
blast_furnace_slag=float(input("enter blast_furnace quantity : ")) water=float(input("enter water quantity : "))
coarse_aggregate=float(input("enter coarse_aggregate quantity : ")) fine_aggregate=float(input("enter fine_aggregate quantity : "))
age_in_days=float(input("enter number of days : "))
compressive_strength= model.coef_[0]*cement+model.coef_[1]*blast_furnace_slag+model.coef_[2]*water+m print(f"The compressive strength of concrete of {age_in_days} is {round(compressive_strength,3)}Mpa"




cement=float(input("enter cement quantity : "))
blast_furnace_slag=float(input("enter blast_furnace quantity : ")) water=float(input("enter water quantity : "))
coarse_aggregate=float(input("enter coarse_aggregate quantity : ")) fine_aggregate=float(input("enter fine_aggregate quantity : "))
age_in_days=float(input("enter number of days : "))
compressive_strength= model.coef_[0]*cement+model.coef_[1]*blast_furnace_slag+model.coef_[2]*water+m print(f"The compressive strength of concrete of {age_in_days} is {round(compressive_strength,3)}Mpa"

 enter cement quantity : 190
enter blast_furnace quantity : 190 enter water quantity : 171
enter coarse_aggregate quantity : 1096 enter fine_aggregate quantity : 920
enter number of days : 28
The compressive strength of concrete of 28.0 is 23.365Mpa

	
 

