import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

#USER INPUT FOR BET
pFirstName = input("Player First Name: ").upper()
pLastName = input("Player Last Name: ").upper()
print()

while True:
    stat = input('Which Stat? (type: PTS, AST, STL, BLK, TRB): ').upper()
    if stat == 'PTS':
        index = 13
        break
    elif stat == 'AST':
        index = 8
        break
    elif stat == 'STL':
        index = 9
        break
    elif stat == 'BLK':
        index = 10
        break
    elif stat == 'TRB':
        index = 7
        break
    else:
        print('Invalid stat please enter again')
    print()

line = int(input(f"Input {stat} line for {pFirstName} {pLastName}: "))
print()

teams = {'HAWKS': 1, 'CELTICS': 2, 'NETS': 3, 'HORNETS': 4, 'BULLS': 5,
         'CAVALIERS': 6, 'MAVERICKS': 7, 'NUGGETS': 8, 'PISTONS': 9, 'WARRIORS': 10,
         'ROCKETS': 11, 'PACERS': 12, 'CLIPPERS': 13, 'LAKERS': 14, 'GRIZZLIES': 15,
         'HEAT': 16, 'BUCKS': 17, 'WOLVES': 18, 'PELICANS': 19, 'KNICKS': 20,
         'THUNDER': 21, 'MAGIC': 22, '76ERS': 23, 'SUNS': 24, 'BLAZERS': 25,
         'KINGS': 26, 'SPURTS': 27, 'RAPTORS': 28, 'JAZZ': 29, 'WIZARDS': 30}

teamIn = {1: 'ATL', 2: 'BOS', 3: 'BRK', 4: 'CHO', 5: 'CHI',
         6: 'CLE', 7: 'DAL', 8: 'DEN', 9: 'DET', 10: 'GSW',
         11: 'HOU', 12: 'IND', 13: 'LAC', 14: 'LAL', 15: 'MEM',
         16: 'MIA', 17: 'MIL', 18: 'MIN', 19: 'NOP', 20: 'NYK',
         21: 'OKC', 22: 'ORL', 23: 'PHI', 24: 'PHO', 25: 'POR',
         26: 'SAC', 27: 'SAS', 28: 'TOR', 29: 'UTA', 30: 'WAS'}

while True:
    teamName = input('Enter Opposing Team Name (Lakers, Celtics, Nets, etc.): ').upper()
    if teamName in teams:
        team = teamIn[teams[teamName]]
        break
    else:
        print('Team not found please try again.')
    print()

while True:
    home = input('Home or Away?: ').upper()
    if home == 'HOME':
        print()
        break
    elif home == 'AWAY':
        print()
        break
    else:
        print('Invalid: try again')
    print()

#GRABS PLAYER AVERAGES FROM LAST 5 GAMES
pFirstName = pFirstName.lower()
pLastName = pLastName.lower()
link = f"https://www.basketball-reference.com/players/{pLastName[0:1]}/{pLastName[0:5]}{pFirstName[0:2]}01.html"
lastGames = pd.read_html(link)[0]

mp = lastGames['MP'].mean()
fg = lastGames['FG'].mean()
fga = lastGames['FGA'].mean()
thr = lastGames['3P'].mean()
thra = lastGames['3PA'].mean()
ft = lastGames['FT'].mean()
fta = lastGames['FTA'].mean()
trb = lastGames['TRB'].mean()
ast = lastGames['AST'].mean()
stl = lastGames['STL'].mean()
blk = lastGames['BLK'].mean()
tov = lastGames['TOV'].mean()
pf = lastGames['PF'].mean()
pts = lastGames['PTS'].mean()
pm = lastGames['+/-'].mean()

#COLLECTING ALL PREVIOUS CAREER STATS
playerPage = requests.get(link)
soup = BeautifulSoup(playerPage.content, 'html.parser')
info = soup.find(id = "meta")
tags = info.find_all('p')
n = len(tags)
exp = int(tags[n - 1].text[12:15].strip())
currYear = 2024
year = currYear - (exp - 1)
all = pd.DataFrame()
while year <= currYear:
    prevGamesLink = f"https://www.basketball-reference.com/players/{pLastName[0:1]}/{pLastName[0:5]}{pFirstName[0:2]}01/gamelog/{year}"
    currSeason = pd.read_html(prevGamesLink)[7]
    all = pd.concat([all, currSeason], ignore_index=True)
    year += 1

#DATA CLEANING
def convert(x):
    value = x[0:2]
    if value[-1] == ':':
        value = value[0]
    return float(value)

def convertHomeGame(x):
    if x == '@':
        return 'AWAY'
    else:
        return 'HOME'

all = all[~all['MP'].str.contains('Did Not Play')]
all = all[~all['MP'].str.contains('Did Not Dress')]
all = all[~all['MP'].str.contains('Not With Team')]
all = all[~all['MP'].str.contains('MP')]
all = all[~all['MP'].str.contains('Inactive')]
all = all.drop_duplicates()
all = all[['MP', 'FG', 'FGA', '3P', '3PA', 'FT', 'FTA', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', '+/-', 'PTS',
           'Unnamed: 5', 'Opp']]
all['MP'] = all['MP'].apply(convert)
all['Unnamed: 5'] = all['Unnamed: 5'].apply(convertHomeGame)
all['FG'] = all['FG'].astype(float)
all['FGA'] = all['FGA'].astype(float)
all['3P'] = all['3P'].astype(float)
all['3PA'] = all['3PA'].astype(float)
all['FT'] = all['FT'].astype(float)
all['FTA'] = all['FTA'].astype(float)
all['TRB'] = all['TRB'].astype(float)
all['AST'] = all['AST'].astype(float)
all['STL'] = all['STL'].astype(float)
all['BLK'] = all['BLK'].astype(float)
all['TOV'] = all['TOV'].astype(float)
all['PF'] = all['PF'].astype(float)
all['PTS'] = all['PTS'].astype(float)
all['+/-'] = all['+/-'].astype(float)
all.dropna(how='any', axis=0, inplace = True)

#CREATING MODEL AND DATA TRAINING
teamEncoder = LabelEncoder()
homeEncoder = LabelEncoder()
X = all.drop(stat, axis = 1)
X['Unnamed: 5'] = homeEncoder.fit_transform(X['Unnamed: 5'])
X['Opp'] = teamEncoder.fit_transform(X['Opp'])
y = all[stat]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

LR = LinearRegression()
LR.fit(X_train.values, y_train.values)

#PREDICTION
n = [mp, fg, fga, thr, thra, ft, fta, trb, ast, stl, blk, tov, pf, pts, pm, homeEncoder.transform([home])[0], teamEncoder.transform([team])[0]]
n.pop(index)
avgs = np.array(n)
statPred = round(LR.predict(avgs.reshape(1, -1))[0], 3)

if statPred > line:
    print('Take the Over!')
    print(f'{pFirstName.upper()} {pLastName.upper()} is predicted to get {statPred} {stat}S')
else:
    print(f'Take the Under!')
    print(f'{pFirstName.upper()} {pLastName.upper()} is predicted to get {statPred} {stat}S')

