import requests
from bs4 import BeautifulSoup
import pandas as pd
import math

List = pd.read_csv("https://www.football-data.co.uk/new/DNK.csv")

# To get predictions for any upcoming matches, please fill the variables below
HomeTeam = "Brondby"
AwayTeam = "Aarhus"

def home_attack_strength(HomeTeam):
    Attack = 0
    Matches_HomeAttack = 0
    for i in range(len(List)):
        if HomeTeam in List.iloc[i, 5]:
            Attack += List.iloc[i, 7]
            Matches_HomeAttack += 1
        #if HomeTeam in List.iloc[i, 6]:
            #Attack += List.iloc[i, 8]
            #Matches_HomeAttack += 1
    return Attack/Matches_HomeAttack

def home_defence_strength(HomeTeam):
    Defence = 0
    Matches_HomeDefence = 0
    for i in range(len(List)):
        if HomeTeam in List.iloc[i, 5]:
            Defence += List.iloc[i, 8]
            Matches_HomeDefence += 1
        if HomeTeam in List.iloc[i, 6]:
            Defence += List.iloc[i, 7]
            Matches_HomeDefence += 1
    return Defence/Matches_HomeDefence

def away_attack_strength(AwayTeam):
    Attack = 0
    Matches_AwayAttack = 0
    for i in range(len(List)):
        if AwayTeam in List.iloc[i, 5]:
            Attack += List.iloc[i, 7]
            Matches_AwayAttack += 1
        if AwayTeam in List.iloc[i, 6]:
            Attack += List.iloc[i, 8]
            Matches_AwayAttack += 1
    return Attack / Matches_AwayAttack

def away_defence_strength(AwayTeam):
    Defence = 0
    Matches_AwayDefence = 0
    for i in range(len(List)):
        if AwayTeam in List.iloc[i, 5]:
            Defence += List.iloc[i, 8]
            Matches_AwayDefence += 1
        if AwayTeam in List.iloc[i, 6]:
            Defence += List.iloc[i, 7]
            Matches_AwayDefence += 1
    return Defence / Matches_AwayDefence

def expected_goals(HomeTeam, AwayTeam):
    HomeGoals = math.exp(sum(List.iloc[:, 7])-sum(List.iloc[:, 8])
                         + (home_attack_strength(HomeTeam) - away_defence_strength(AwayTeam))
                         + (home_defence_strength(HomeTeam) - away_attack_strength(AwayTeam))
                         + (sum(List.iloc[:, 7])/len(List) - sum(List.iloc[:, 8])/len(List)))
    return HomeGoals

# The matrix containing the probability of each outcome up to max_goals
#Probabilities = [[BivariatePoisson(i, j, home_goals, away_goals, team_difference) for i in range(0, max_goals)] for j in range(0, max_goals)]
#matrix = np.array(Probabilities)

if __name__ == '__main__':
    print(home_attack_strength(HomeTeam))
    print(home_defence_strength(HomeTeam))
    print(away_attack_strength(AwayTeam))
    print(away_defence_strength(AwayTeam))
    print(expected_goals(HomeTeam, AwayTeam))
    print(sum(List.iloc[:, 7])-sum(List.iloc[:, 8]))
    print((sum(List.iloc[:, 7])/len(List) - sum(List.iloc[:, 8])/len(List)))