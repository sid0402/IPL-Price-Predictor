import pandas as pd
from bs4 import BeautifulSoup 
import requests

df = pd.read_csv('IPLPlayerAuctionData.csv')
a= lambda x: x.strip()
df['Player'] = df['Player'].apply(a)
df['id'] = 0

unique_last = []

for name in set(df['Player']):
    df_last = name.split()[-1]
    if not(df_last in unique_last):
        unique_last.append(df_last)
    else:
        unique_last.remove(df_last)

def player_id():
    req = requests.get("http://www.howstat.com/cricket/Statistics/IPL/PlayerList.asp?s=XXXX")
    soup = BeautifulSoup(req.content, 'html.parser')

    players = soup.find_all('a',class_='LinkTable')

    for player in players:
        #print(player.text)
        #print(player['href'])
        last_name = player.text.split()[-1]
        for name in df['Player']:
            df_last = name.split()[-1]
            if (df_last == last_name and df_last in unique_last):
                id = player['href'].split('=')[-1]
                df.loc[df['Player']==name,'id'] = id

#player_id()
print(df.head)
df.to_csv('AuctionId.csv')