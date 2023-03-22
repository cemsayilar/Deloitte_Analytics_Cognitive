## Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
from helpers import check_df, grab_col_names
import sweetviz as sv

pd.set_option('display.max_columns',100)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', 100)

df = pd.read_excel('/Users/buraksayilar/Desktop/82210-1.xlsx')
df.head()
check_df(df)
df.drop(index= 1780, axis=0, inplace=True)


df = df.replace('-', np.nan) # Now checking nan values again.
check_df(df) # Almost %90 of the VG feature is nan. So I dropped it.
df.drop(['VG'], axis=1, inplace=True)
# There are high amount of nan values in SLP and STP features, especially in SLP.
# But I may be able to fill those values. Will see.

df.fillna(method='bfill', inplace=True) # I will use dates to fill nan values with back and
df[['VM', 'VG', 'PP', 'H', 'STP', 'SLP']] = df[['VM',
                                                'VG',
                                                'PP',
                                                'H',
                                                'STP',
                                                'SLP']].astype('float') # Made all features num.

df['Date'] = df['Y'].astype('str') + '-' + df['M'].astype('str') + '-' + df['D'].astype('str')
df['Date'] = pd.to_datetime(df['Date'])
df.drop(['Y', 'M', 'D'], axis=1, inplace=True)



################################# Visualisation #################################
df['Year'] = df['Date'].dt.year
grouped = df.groupby(['Year', 'Season'])[['T',
                                          'TM',
                                          'Tm',
                                          'SLP',
                                          'STP',
                                          'H',
                                          'PP',
                                          'VV',
                                          'V',
                                          'VM',
                                          'FG',
                                          'RA',
                                          'SN',
                                          'GR',
                                          'TS',
                                          'TR']].mean().reset_index()



# Splitting each year into seasons
df['Season'] = pd.cut(df['Date'].dt.month,
                      [0, 2, 5, 8, 11, 12],
                      labels=['Winter', 'Spring', 'Summer', 'Fall', 'Winter'],
                      ordered=False
                      )

# Creating a function to plot correlation heatmap
def plot_heatmap(df):
    corr = df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")

# Creating a function to plot time series line plot
def plot_time_series(df, x, y, hue=None):
    plt.figure(figsize=(16, 6))
    sns.lineplot(data=df, x=x, y=y, hue=hue, linewidth=2.5)

# Creating a function to plot scatter plot
def plot_scatter(df, x, y, hue=None):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x, y=y, hue=hue)

# Plotting correlation heatmap
plot_heatmap(df.drop(['Date'], axis=1))
plt.show()

# Plotting time series line plot for temperature and humidity
plot_time_series(df, x='Date',
                 y='T',
                 hue='Season'
                 )
plot_time_series(df, x='Date',
                 y='H',
                 hue='Season'
                 )

sns.barplot(x='Year',
            y='PP',
            hue='Season',
            data=grouped
            )
sns.despine()
plt.title('Total Rainfall and/or Snowmelt by Year and Season')
plt.ylabel('Total Rainfall and/or Snowmelt (mm)')

# Show the plots
plt.show()



```python
## Imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
from helpers import check_df, grab_col_names
import sweetviz as sv

pd.set_option('display.max_columns',100)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', 100)

df = pd.read_excel('/Users/buraksayilar/Desktop/82210-1.xlsx')
df.head()
check_df(df)
df.drop(index= 1780, axis=0, inplace=True)
```

```python

df = df.replace('-', np.nan) # Now checking nan values again.
check_df(df) # Almost %90 of the VG feature is nan. So I dropped it.
df.drop(['VG'], axis=1, inplace=True)
```

```python
df.fillna(method='bfill', inplace=True) # I will use dates to fill nan values with back and
df[['VM', 'VG', 'PP', 'H', 'STP', 'SLP']] = df[['VM', 'VG', 'PP', 'H', 'STP', 'SLP']].astype('float') # Made all features num.

df['Date'] = df['Y'].astype('str') + '-' + df['M'].astype('str') + '-' + df['D'].astype('str')
df['Date'] = pd.to_datetime(df['Date'])
df.drop(['Y', 'M', 'D'], axis=1, inplace=True)



################################# Visualisation #################################
df['Year'] = df['Date'].dt.year
grouped = df.groupby(['Year', 'Season'])[['T', 
                                          'TM', 
                                          'Tm', 
                                          'SLP', 
                                          'STP', 
                                          'H', 
                                          'PP', 
                                          'VV', 
                                          'V', 
                                          'VM', 
                                          'FG', 
                                          'RA', 
                                          'SN', 
                                          'GR', 
                                          'TS', 
                                          'TR']].mean().reset_index()



# Splitting each year into seasons
df['Season'] = pd.cut(df['Date'].dt.month, 
                      [0, 2, 5, 8, 11, 12], 
                      labels=['Winter', 'Spring', 'Summer', 'Fall', 'Winter'], 
                      ordered=False
                      )

# Creating a function to plot correlation heatmap
def plot_heatmap(df):
    corr = df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")

# Creating a function to plot time series line plot
def plot_time_series(df, x, y, hue=None):
    plt.figure(figsize=(16, 6))
    sns.lineplot(data=df, x=x, y=y, hue=hue, linewidth=2.5)

# Creating a function to plot scatter plot
def plot_scatter(df, x, y, hue=None):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x, y=y, hue=hue)

# Plotting correlation heatmap
plot_heatmap(df.drop(['Date'], axis=1))
plt.show()

# Plotting time series line plot for temperature and humidity
plot_time_series(df, x='Date', 
                 y='T', 
                 hue='Season'
                 )
plot_time_series(df, x='Date', 
                 y='H',
                 hue='Season'
                 )

sns.barplot(x='Year', 
            y='PP', 
            hue='Season', 
            data=grouped
            )
sns.despine()
plt.title('Total Rainfall and/or Snowmelt by Year and Season')
plt.ylabel('Total Rainfall and/or Snowmelt (mm)')

# Show the plots
plt.show()
```
#### Graph-2 Total Rainfall and/or Snowmelt (PP) by Year and Season

![image](https://user-images.githubusercontent.com/96774646/227050430-bd022200-73ba-47e8-9e8b-775c24d13359.png)

#### Graph-3 Temparature (T) Changes by Year and Season
![image](https://user-images.githubusercontent.com/96774646/227050703-8b58d86b-551b-430c-afb5-f31f593d1ebb.png)

#### Graph-4 Average Relative Humidity (H) by Year and Season
![image](https://user-images.githubusercontent.com/96774646/227050898-8136981c-0ff6-47c2-bcb6-3b705e37b27b.png)


```

