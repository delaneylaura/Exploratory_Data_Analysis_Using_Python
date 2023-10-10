import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import calmap
# from pandas_profiling import ProfileReport

# ** Task 1: Initial Data Exploration ** #
# https://www.kaggle.com/datasets/aungpyaeap/supermarket-sales?resource=download
df = pd.read_csv("C:\\Coursera\\Exploratory_Data_Analysis_Using_Python\\supermarket_sales.csv")
print(df.head())  # have a glimpse at the dataset
print(df.tail(10))  # view last ten rows
print(df.shape)  # view DataFrame dimensions
print(df.columns)
print(df.dtypes)
df['Date'] = pd.to_datetime(df['Date'])
print(df['Date'])
print(df.dtypes)  # check Date has been converted
df.set_index('Date', inplace=True)  # setting the index to be Date. Setting inplace=True for it to be a permanent change
print(df.head())

# calculate quick summary statistics (for every numeric (float) column)
print(df.describe())

# ** Task 2: Univariate Analysis ** #
# Question 1: What does the distribution of customer ratings looks like? Is it skewed?
sns.displot(df['Rating'])
plt.axvline(x=np.mean(df['Rating']), c='red', ls='--', label='mean')
plt.axvline(x=np.percentile(df['Rating'], 25), c='green', ls='--', label='25-75th percentile')
plt.axvline(x=np.percentile(df['Rating'], 75), c='green', ls='--')
plt.legend()
plt.show()
# Answer: Distribution of user ratings looks relatively uniform. Doesn't seem to be skewed in left or right direction

# Plotting distributions of all the numerical variables
df.hist(figsize=(10, 10))
plt.show()
# tax, total, cogs and gross income all have a right skew. Unit price and quantity - uniform distribution

# Question 2: Do aggregate sales numbers differ by much between branches?
sns.countplot(x=df['Branch'])
plt.show()
print(df['Branch'].value_counts())
sns.countplot(x=df['Payment'])
plt.show()

# Answer:
# They do not differ by much

# ** Task 3: Bivariate Analysis ** #
# Question 3: Is there a relationship between gross income and customer ratings?
sns.scatterplot(x=df['Rating'], y=df['gross income'])
sns.regplot(x=df['Rating'], y=df['gross income'])  # regression plot
plt.show()  # flat - no relationship between variables

sns.boxplot(x=df['Branch'], y=df['gross income'])
plt.show()
# there doesn't seem to be much variation in the gross income between branches, at the average level

sns.boxplot(x=df['Gender'], y=df['gross income'])
plt.show()
# men and women spend about the same. At 75th percentile women spend higher than men, but on average they are similar

# Question 4: Is there a noticeable time trend in gross income?
print(df.groupby(df.index).mean())  # so no duplicate dates

print(df.groupby(df.index).mean().index)
sns.lineplot(x=df.groupby(df.index).mean().index,
             y=df.groupby(df.index).mean()['gross income'])
plt.show()  # no particular trend

# plot all the bivariate relationships possible
sns.pairplot(df)  # not recommended for large datasets. time consuming to run
plt.show()  # shows univariate, pairwise distributions

# ** Task 4: Dealing With Duplicate Rows and Missing Values ** #
print(df.duplicated())
print(df.duplicated().sum())
print(df[df.duplicated()==True])  # to view the duplicated rows
df.drop_duplicates(inplace=True)  # permanently remove the duplicate rows

# missing values
print(df.isna().sum())
print(df.isna().sum() / len(df))  # percentage

# heatmap to visualise missings
sns.heatmap(df.isnull(), cbar=False)
plt.show()

# fill in missings with mean for numerical columns, and maintain update
df.fillna(df.mean(), inplace=True)

# fill in missings with the mode for each (categorical) column (for the remaining categorical variables with missings)
print(df.mode().iloc[0])  # the mode for each column
df.fillna(df.mode().iloc[0], inplace=True)

# pandas_profiling organises EDA very succinctly - but not feasible for a large dataset
# dataset = pd.read_csv("C:\\Coursera\\Exploratory_Data_Analysis_Using_Python\\supermarket_sales.csv")
# prof = ProfileReport(dataset)
# prof

# ** Task 5: Correlation Analysis ** #
print(np.corrcoef(df['gross income'], df['Rating']))

print(round(np.corrcoef(df['gross income'], df['Rating'])[1][0], 2))

# correlation matrix - calculates the correlation for every pairwise numeric column
print(df.corr())
print(np.round(df.corr(), 2))

sns.heatmap(np.round(df.corr(), 2), annot=True)
plt.show()
# Rating has low correlation with every other variable. It seems that the amount a customer spends on an item isn't
# related to their overall shopping experience.
