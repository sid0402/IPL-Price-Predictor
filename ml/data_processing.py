import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("FinalAuctionData.csv")

mega_years = [2014, 2018, 2022]
normal_years = [2013, 2015, 2016, 2017, 2019, 2020, 2021]
years = [2013, 2015, 2016, 2017, 2019, 2020, 2021,2014, 2018, 2022]

df = df.fillna(0)

#Number of players sold each year
#print(df.groupby("Year")["Player"].count())

#ADJUSTING FOR PRICE
largest = 5
year_df = df.groupby(["Mega","Year"])
df1 = year_df['Amount'].apply(lambda grp: grp.nlargest(largest).mean())
mega_max = year_df.get_group((1,2022))['Amount'].nlargest(largest).mean()
normal_max = year_df.get_group((0,2021))['Amount'].nlargest(largest).mean()
#df1[0] = df1[0].apply(lambda x: (x/normal_max)*100)
df1[0] = df1[0].apply(lambda x: (x/normal_max)*100)
df1[1] = df1[1].apply(lambda x: (x/mega_max)*100)
cpi = {'0':{},'1':{}}

for i in mega_years:
    cpi['1'][i] = df1[1][i]/100
for i in normal_years:
    cpi['0'][i] = df1[0][i]/100

print(cpi)

df2 = year_df['Amount'].apply(lambda grp: grp.nlargest(largest).mean())

for i in cpi:
    for j in cpi[i]:
        df.loc[df['Year']==j,'Amount'] = df[df['Year']==j]['Amount'].apply(lambda x: x/cpi[i][j])

print(df['Amount'].describe())

#Making columns numeric and getting dummy variables
df['last_hs'] = df['last_hs'].apply(lambda x: float(str(x).split('*')[0]))

df['Top Scored %'] = df['Top Scored %'].apply(lambda x: float(x))

df['Scoring Rate'] = df['Scoring Rate'].apply(lambda x: float(x))

df.drop(['Team','Player','100s','5 Wickets in Innings'], inplace=True, axis=1)

df = pd.get_dummies(df)
df = df*1

df = df[df['prog_matches']>3]

#df_check = df[df['Role_Batsman']==1]
#print(df_check[df['prog_matches']==0].sort_values(by="Amount",ascending=False)[['id','Amount']])

cat_columns = ['Role_All-Rounder', 'Role_Batsman',
       'Role_Bowler', 'Role_Wicket Keeper', 'Nation_Indian',
       'Nation_Overseas', 'Mega']

def skew_high(df):
    skew_col = []
    for i in df.columns:
        if (i != 'Amount' and not(i in cat_columns) and df[i].skew() > 1.5):
            skew_col.append(i)
    return skew_col

skew_col = skew_high(df)
print(skew_col)

def transform_log(df):
    for i in skew_col:
        df[i] = np.log(df[i]+0.5)
    return df

#df = transform_log(df)

print(df.skew().sort_values(ascending=False))

bowl_col = ['overs', 'last_wkts', 'last_bowlavg', 'last_bowlsr',
       'last_bowler', 'last_bowlpercent', 'birthyear','Overs', 'Balls Bowled', 'Wickets',
       '3 Wickets in Innings', '4 Wickets in Innings', 'Economy Rate',
       'Strike Rate','prog_bowlsr', 'prog_er', 'prog_wkts', 'prog_bowlavg']

plt.figure(figsize=(12,10))
corr = df.drop(bowl_col, axis=1).corr()
#print(corr)
sns.heatmap(corr, annot=False)
#plt.show()

third = np.percentile(df['Amount'], 75)
first = np.percentile(df['Amount'], 25)
iqr = third-first
rang = third + (1.5*iqr)
df = df[df['Amount']<=15]
#df.loc[df['Amount']>third+1.5*iqr,'Amount'] = df.loc[df['Amount']>third+1.5*iqr,'Amount']/1.75

#REMOVING PADDIKAL, INGRAM, PABHSIMRAN
df = df.drop(df[df['id'].isin([4948,3832,4937])].index)

print(df.columns)

df.to_csv("ml/Preprocessed.csv", index=False)
