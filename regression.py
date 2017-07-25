
import matplotlib.pyplot as plt
from sklearn import linear_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
batting_df = pd.read_csv('Batting.csv')
salaries_df = pd.read_csv('Salaries.csv')
teams_df = pd.read_csv('Teams.csv')

team_salaries_df_full =pd.merge(salaries_df.groupby(['yearID','teamID'],as_index=False).sum(),teams_df,on=['yearID','teamID'],how = 'left')
team_salaries_df = team_salaries_df_full[['yearID','teamID','salary','W']].assign (costperwin = lambda x: x.salary/x.W)


# get data frame for a team in each year with salary and wins, fieldnames: yearID teamID salary W costperwin.
year = 2016
def get_year_removena(y, df = team_salaries_df):
    """
        return data in chosen year, which is 2016 here.
    """
    return df[(df['yearID']==y) & (np.isfinite(df['W']))].reset_index()

def adjusted_r(r_squared,featurelist):
    """
        input score of the regression (r square) and independent variables
        return adjusted r squared of the regression model.
    """
    num_variables = len(featurelist)
    num_observations = len(featurelist[0])

    for l in featurelist:
        if num_observations != len(l) :
            warnings.warn("Varaiables do not have the same lenths.")

    return 1 - (1-r_squared)*(num_observations-1)/(num_observations-num_variables-1)

def outlierCleaner(predictions, features, wins):
    """
        Clean away the 15% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).
        Return a list of tuples named cleaned_data where
        each tuple is of the form (predictions, features, wins).
    """
    cleaned_data = []
    ### your code goes here
    error = (wins- predictions)**2
    cleaned_data = zip(features, wins, error)
    cleaned_data = sorted(cleaned_data,key = lambda x: x[2],reverse =True)
    # print map(lambda x : x[2], cleaned_data)
    cleaned_data = cleaned_data[int((len(cleaned_data)*0.15)):]
    return cleaned_data

# get data in year 2016
team_runs_df = teams_df.dropna(axis = 0, how = 'any').reset_index()[['yearID','teamID','R']]
team_salaries_runs = pd.merge(team_salaries_df,team_runs_df,on = ['yearID','teamID'],how = 'inner')
team_salaries_runs = get_year_removena(2016,team_salaries_runs)

#load data
salary_cost = np.reshape(np.array(team_salaries_runs['salary']), (len(team_salaries_runs['salary']), 1))
wins = np.reshape(np.array(team_salaries_runs['W']), (len(team_salaries_runs['W']), 1))
runs = np.reshape(np.array(team_salaries_runs['R']), (len(team_salaries_runs['R']), 1))

clf = linear_model.LinearRegression()
clf.fit(salary_cost,wins)
print 'The coefficient with outliers is ', clf.score(salary_cost,wins)

#Create plots
fig = plt.figure(1)
ax = fig.add_subplot(111)
# print spots
ax.scatter(x = salary_cost, y =wins, label = 'Wins (outliers)', color = 'red')

# print the linear function
ax.plot(salary_cost , clf.predict(salary_cost),label = 'Wins Regression (with outliers)', color = 'red')



# Clean data and remove outliers
cleaned_data =  outlierCleaner(clf.predict(salary_cost), salary_cost, wins)
salary_cost, wins, errors = zip(*cleaned_data)
salary_cost_rp       = np.reshape( np.array(salary_cost), (len(salary_cost), 1))
wins_rp = np.reshape( np.array(wins), (len(wins), 1))

### refit your cleaned data!
clf.fit(salary_cost_rp, wins_rp)
print 'The coefficient without outliers is ', clf.score(salary_cost_rp,wins_rp)
print 'The adjusted R square is', adjusted_r(clf.score(salary_cost,wins),[zip(*salary_cost)[0]])
ax.scatter(x = salary_cost, y = wins, label = 'Wins (no outliers)', color = 'g')
# print the linear function
ax.plot(salary_cost , clf.predict(salary_cost),label = 'Wins Regression (no outliers)', color = 'g')


plt.xlabel('Sum of salaries')
plt.ylabel('Wins')
plt.title('Wins v.s salaries for each team in ' + str(year))
handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02,0.5),borderaxespad=0.)
fig.savefig('Regression', bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.tight_layout()
plt.show()

# Add runs into the table for regression

#
# print team_salaries_runs.head()
# # print [team_salaries_runs[['salary']]].apply(lambda x: np.array(x))
# team_salaries_runs = get_year_removena(2016,team_salaries_runs)
# x_features = np.array(team_salaries_runs[['salary','R']])
#
# clf.fit(x_features,team_salaries_runs[['W']])
# print clf.score(x_features,team_salaries_runs[['W']])
#
#
#
# # find the error of each team
# team_salaries_runs = team_salaries_runs.assign(error = lambda x: abs(x.W - list(zip(*clf.predict(x_features))[0])))
#
# # Sort and find two outliers and remove them
# team_salaries_runs = team_salaries_runs.sort_values('error',ascending = False)[2:]
# print team_salaries_runs
#
# x_features = np.array(team_salaries_runs[['salary','R']])
#
# clf.fit(x_features,team_salaries_runs[['W']])
# r_squared =  clf.score(x_features,team_salaries_runs[['W']])
# print adjusted_r(r_squared,team_salaries_runs[['salary','R']])
