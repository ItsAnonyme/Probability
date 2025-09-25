import pandas as pd
import math

HomeTeam = "Brondby"
AwayTeam = "Aalborg"

Data = pd.read_csv("Dataset-small.csv", skiprows=1, skipfooter=216, engine="python")

AvgAwayGoals = sum(Data.iloc[:, 8])/len(Data)
AvgHomeGoals = sum(Data.iloc[:, 7])/len(Data)

def relative_homefield_advantage(h):
    home_goals = 0
    away_goals = 0
    matches_home = 0
    matches_away = 0
    for i in range(0, len(Data)):
        if h == Data.iloc[i, 2]:
            home_goals += Data.iloc[i, 4]
            matches_home += 1
        if h == Data.iloc[i, 3]:
            away_goals += Data.iloc[i, 5]
            matches_away += 1
    return home_goals/matches_home - away_goals/matches_away

def avg_home_goals_scored(h):
    home_goals = 0
    matches = 0
    for i in range(0, len(Data)):
        if h in Data.iloc[i, 5]:
            home_goals += Data.iloc[i, 7]
            matches += 1
        if h in Data.iloc[i, 6]:
            home_goals += Data.iloc[i, 8]
            matches += 1
    return home_goals/matches

def avg_home_goals_conceded(h):
    home_goals = 0
    matches = 0
    for i in range(0, len(Data)):
        if h in Data.iloc[i, 5]:
            home_goals += Data.iloc[i, 8]
            matches += 1
        if h in Data.iloc[i, 6]:
            home_goals += Data.iloc[i, 7]
            matches += 1
    return home_goals/matches

def avg_away_goals_scored(a):
    away_goals = 0
    matches = 0
    for i in range(0, len(Data)):
        if a in Data.iloc[i, 5]:
            away_goals += Data.iloc[i, 7]
            matches += 1
        if a in Data.iloc[i, 6]:
            away_goals += Data.iloc[i, 8]
            matches += 1
    return away_goals/matches

def avg_away_goals_conceded(a):
    away_goals = 0
    matches = 0
    for i in range(0, len(Data)):
        if a in Data.iloc[i, 5]:
            away_goals += Data.iloc[i, 8]
            matches += 1
        if a in Data.iloc[i, 6]:
            away_goals += Data.iloc[i, 7]
            matches += 1
    return away_goals/matches

def method1_exp_home(h, a):
    expected_home_goals = ((avg_home_goals_scored(h) - avg_away_goals_scored(a))
                           + (avg_away_goals_conceded(a) - avg_home_goals_conceded(h))
                           + relative_homefield_advantage(h))
    return math.exp(expected_home_goals)

def method1_exp_away(h, a):
    expected_away_goals = ((avg_away_goals_scored(a) - avg_home_goals_scored(h))
                           + (avg_home_goals_conceded(h) - avg_away_goals_conceded(a)))
    return math.exp(expected_away_goals)

def method1_2_exp_home(h, a):
    expected_home_goals = ((avg_home_goals_scored(h) - avg_away_goals_scored(a))
                           + (avg_away_goals_conceded(a) - avg_home_goals_conceded(h))
                           + (AvgHomeGoals - AvgAwayGoals))
    return math.exp(expected_home_goals)

def method2_exp_home(h):
    expected_home_goals = ((avg_home_goals_scored(h) - AvgHomeGoals)
                           + (avg_away_goals_conceded(h) - AvgAwayGoals)
                           + (AvgHomeGoals - AvgAwayGoals))
    return math.exp(expected_home_goals)

def method2_exp_away(a):
    expected_away_goals = ((avg_away_goals_scored(a) - AvgAwayGoals)
                           + (avg_home_goals_conceded(a) - AvgHomeGoals))
    return math.exp(expected_away_goals)

def method3_exp_home(h, a):
    expected_home_goals = (avg_away_goals_conceded(a)/avg_home_goals_scored(h)
                           + avg_away_goals_scored(a)/avg_away_goals_conceded(h)
                           + relative_homefield_advantage(h))
    return expected_home_goals

def method3_exp_away(h, a):
    expected_home_goals = (avg_home_goals_conceded(a)/avg_away_goals_scored(h)
                           + avg_home_goals_scored(a)/avg_away_goals_conceded(h))
    return expected_home_goals

if __name__ == '__main__':
    print('Method 1: We expect', method1_exp_home(HomeTeam, AwayTeam), 'from', HomeTeam)
    print('Method 1: We expect', method1_exp_away(HomeTeam, AwayTeam), 'from', AwayTeam)
    print('Method 1.2: We expect', method1_2_exp_home(HomeTeam, AwayTeam), 'from', HomeTeam)
    print('Method 2: We expect', method2_exp_home(HomeTeam), 'from', HomeTeam)
    print('Method 2: We expect', method2_exp_away(AwayTeam), 'from', AwayTeam)
    print('Method 3: We expect', method3_exp_home(HomeTeam, AwayTeam), 'from', HomeTeam)
    print('Method 3: We expect', method3_exp_away(AwayTeam, HomeTeam), 'from', AwayTeam)