import pickle
import pandas as pd

file = open('scrapers/ScrapedData/basic_stats_dict.txt', 'rb')

data = pickle.load(file)

df = pd.read_csv('AuctionId.csv')

pd.set_option('mode.chained_assignment', None)

#removing irrelevant basic stats
del_keys = ['Opened Batting','Balls Faced','Ducks','Maidens','Runs Conceded','Tosses Won','Matches/Won/Lost','Stumpings', 'Most Dismissals in Match',
       'Most Catches in Innings', 'Most Dismissals in Innings', 'Won/Lost',
       '5 Wickets in  Innings', '10 Wickets in Match', 'Best - Innings',
       'Best - Match','Catches',
       'Most Catches in Match']

for i in data:
    for j in del_keys:
        if (j in i.keys()):
            del i[j]
    
#saving basic data to file
for i in data:
    id = i['id']
    for key in i:
        df.loc[df['id']==id, key] = i[key]


file.close()

file = open('scrapers/ScrapedData/progressive_battingcollection.txt', 'rb')
progressing_batting = pickle.load(file)

for i in progressing_batting:
    id = i['id']
    year = int(i['auction_year'])
    for key in i:
       df.loc[(df['id']==id) & (df['Year']==year), key] = i[key]

file.close()

file = open('scrapers/ScrapedData/progressive_bowlingollection.txt', 'rb')
progressive_bowling = pickle.load(file)

for i in progressive_bowling:
    id = i['id']
    year = int(i['auction_year'])
    for key in i:
       df.loc[(df['id']==id) & (df['Year']==year), key] = i[key]

file.close()

file = open('scrapers/ScrapedData/last_bat.txt', 'rb')
last_bat = pickle.load(file)

for i in last_bat:
    id = i['id']
    year = int(i['auction_year'])
    for key in i:
       df.loc[(df['id']==id) & (df['Year']==year), key] = i[key]


file.close()

file = open('scrapers/ScrapedData/last_bowl.txt', 'rb')
last_bowl = pickle.load(file)

for i in last_bowl:
    id = i['id']
    year = int(i['auction_year'])
    for key in i:
       df.loc[(df['id']==id) & (df['Year']==year), key] = i[key]

df.to_csv('AuctionId.csv')




