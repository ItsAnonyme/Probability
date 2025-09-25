import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson
from AvgGoalMethods import *
import matplotlib as plt
import math


# Choose your teams
HomeTeam = "Chelsea"
AwayTeam = "Arsenal"

# Where we get our data
Data = pd.read_csv("Dataset-small.csv", skiprows=1, skipfooter=196, engine='python')

if 'Date' in Data.columns:
    Data['Date'] = pd.to_datetime(Data['Date'], dayfirst=True)
else:
    raise ValueError("Your dataset must contain a 'Date' column to compute time weights.")

# Function to calculate weights
def calculate_weights(dates, x):
    latest_date = dates.max()
    diff_half_weeks = ((latest_date - dates).dt.days) / 3.5
    return np.exp(-x * diff_half_weeks)

def prediction_poisson(HomeTeam, AwayTeam, x):
    #print("Predicting for:", HomeTeam, "vs", AwayTeam)
    max_goals = 10

    # Rewriting the data
    home_df = pd.DataFrame(data={"team": Data.HomeTeam, "opponent": Data.AwayTeam, "goals": Data.FTHG, "home": 1, "Date": Data.Date})
    away_df = pd.DataFrame(data={"team": Data.AwayTeam, "opponent": Data.HomeTeam, "goals": Data.FTAG, "home": 0, "Date": Data.Date})

    full_df = pd.concat([home_df, away_df]).reset_index(drop=True)

    # Apply time-based weights
    full_df['weights'] = calculate_weights(full_df['Date'], x)

    # Applying Poisson Distribution
    model_Poisson = smf.glm(data=pd.concat([home_df, away_df]), family=sm.families.Poisson(),
                            formula="goals ~ home * team + opponent", freq_weights=full_df['weights']).fit()
    #print(model_Poisson.summary())
    # Goals prediction
    home_goals = \
    (model_Poisson.predict(pd.DataFrame(data={"team": HomeTeam, "opponent": AwayTeam, "home": 1}, index=[1])).values[0])
    away_goals = \
    model_Poisson.predict(pd.DataFrame(data={"team": AwayTeam, "opponent": HomeTeam, "home": 0}, index=[1])).values[0]

    # The matrix containing the probability of each outcome up to max_goals
    Probability_matrix = [[poisson.pmf(i, team_avg) for i in range(0, max_goals)]
                          for team_avg in [home_goals, away_goals]]

    matrix = np.outer(np.array(Probability_matrix[0]), np.array(Probability_matrix[1]))

    # Summing the probability of each outcome
    Home_Win = np.sum(np.tril(matrix, -1))
    Draw = np.sum(np.diag(matrix))
    Away_Win = np.sum(np.triu(matrix, 1))
    if Home_Win > Away_Win and Home_Win > Draw:
        return "H"
    elif Away_Win > Draw and Away_Win > Home_Win:
        return "A"
    else:
        return "D"

if __name__ == "__main__":
    print(prediction_poisson(HomeTeam, AwayTeam))



























