import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#USER INPUT FOR BET
pFirstName = input("Player first name: ").upper()
pLastName = input("Player last name: ").upper()
stat = input('Which stat? (type: PTS, AST, STL, BLK, TRB): ').upper()
line = int(input(f"Input {stat} line for {pFirstName} {pLastName}: "))
print()

if stat == 'PTS':
    index = 13
elif stat == 'AST':
    index = 8
elif stat == 'STL':
    index = 9
elif stat == 'BLK':
    index = 10
elif stat == 'TRB':
    index = 7

MAKE DICTIONARY OF TEAMS AND CREATE COLUMN TO FACTOR IN TEAM PLAYING AGAINST
ALSO CAN USE AT SYMBOL OF 1 AND 0 AT @ COLUMN

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

n = [mp, fg, fga, thr, thra, ft, fta, trb, ast, stl, blk, tov, pf, pts, pm]
n.pop(index)
avgs = np.array(n)

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

all = all[~all['MP'].str.contains('Did Not Play')]
all = all[~all['MP'].str.contains('Did Not Dress')]
all = all[~all['MP'].str.contains('Not With Team')]
all = all[~all['MP'].str.contains('MP')]
all = all[~all['MP'].str.contains('Inactive')]
all = all.drop_duplicates()
all = all[['MP', 'FG', 'FGA', '3P', '3PA', 'FT', 'FTA', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', '+/-']]
all['MP'] = all['MP'].apply(convert)
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
all = all.dropna(how='any',axis=0)
all.to_csv('file.csv')

#CREATING MODEL AND DATA TRAINING
x = all.drop(stat, axis = 1)
y = all[stat]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

LR = LinearRegression()
LR.fit(X_train.values, y_train.values)

#PREDICTION
statPred = LR.predict(avgs.reshape(1, -1))[0]

if statPred > line:
    print(f'Over: {pFirstName.upper()} {pLastName.upper()} is predicted to get {statPred} {stat}S')
else:
    print(f'Under: {pFirstName.upper()} {pLastName.upper()} is predicted to get {statPred} {stat}S')