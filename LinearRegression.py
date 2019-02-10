#%matplotlib inline
from IPython import get_ipython
ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Data visualisation libraries
import seaborn as sns


#Dataset location: https://github.com/kkleintank/LinearRegressionTest
USAhousing = pd.read_csv(
    'https://raw.githubusercontent.com/kkleintank/LinearRegressionTest/master/USA_Housing.csv')
USAhousing.head()
#USAhousing.info()
#USAhousing.describe()
#USAhousing.columns()

sns.set(style="ticks", color_codes=True)

#sns.pairplot(USAhousing)
#sns.distplot(USAhousing['Price'])
#sns.pairplot(USAhousing)


# Correlation: The correlation coefficient, or simply the correlation, is an index that ranges from -1 to 1. 
# When the value is near zero, there is no linear relationship. 
# As the correlation gets closer to plus or minus one, the relationship is stronger. 
# A value of one (or negative one) indicates a perfect linear relationship between two variables.
# corr = USAhousing.corr()
# sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns)


# Training the model: Let’s now begin to train out regression model! 
# We will need to first split up our data into an X array that contains the features to train on, and a y array with the target variable, in this case the Price column. 
# We will toss out the Address column because it only has text info that the linear regression model can’t use.
X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms', 'Area Population']]

#X = USAhousing[['Avg. Area Income']]
y = USAhousing['Price']


#Our goal is to create a model that generalises well to new data. 
# Our test set serves as a proxy for new data.
# Trained data is the data on which we apply the linear regression algorithm. 
# And finally we test that algorithm on the test data.The code for splitting is as follows:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101) #40% of the set is used to train the model 

#The below code fits the linear regression model on the training data.
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)

#Let’s grab predictions off the test set and see how well it did!
predictions = lm.predict(X_test)

#visualise the prediction
plt.scatter(y_test,predictions)
plt.show()



