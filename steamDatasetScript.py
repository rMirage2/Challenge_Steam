# -*- coding: utf-8 -*-
"""
Created on Sat May  4 23:40:11 2019

Initial analysis on the steam dataset.  Need classification problems from it, see website on what's
available (https://steam.internet.byu.edu/), to see what has been done with the dataset, see the paper at 
https://steam.internet.byu.edu/oneill-condensing-steam.pdf or things that cite it

@author: Rachael Purta
"""



from io import StringIO
import re, shutil
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

# for whatever reason, they give the dataset as a mysql dump file, so we need to separate it into
# tables and make it a csv.  I use the code at https://stackoverflow.com/questions/27584405/how-to-import-a-mysqldump-into-pandas
# to do this, it is not originally my own, though I debugged the pieces I marked
def read_dump(dump_filename, target_table):
    sio = StringIO()

    fast_forward = True
    with open(dump_filename, 'r', encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line.lower().startswith('insert') and target_table in line:
                fast_forward = False
            else:
                fast_forward = True #added: after we found table, keep going
            if fast_forward:
                continue
            data = re.findall('\([^\)]*\)', line)
            try:
                # need to make this a for loop, bc this mysql dump does not separate onto lines
                for i in range(0,len(data)):
                    newline = data[i]
                    newline = newline.strip(' ()')
                    newline = newline.replace('`', '')
                    sio.write(newline)
                    sio.write("\n")
            except IndexError:
                pass
            # added - write to csv, avoids keeping everything in memory...
            if len(data) > 0:
                #sio.pos = 0 #bug for later versions of python
                sio.seek(0)
                with open (target_table+'.csv', 'a', encoding="utf8") as fd:
                    shutil.copyfileobj(sio, fd,-1)
                sio = StringIO() # faster to just make new object than clear?
            #if line.endswith(';'): # comment this out - our sql does multiple inserts per table
                #break
    return
    #return data
    
# for whatever reason, they give the dataset as a mysql dump file, so we need to separate it into
# tables and make it a csv.  I use the code at https://stackoverflow.com/questions/27584405/how-to-import-a-mysqldump-into-pandas
# to do this, it is not originally my own, though I debugged the pieces I marked - shorter dataset
def read_dump_short(dump_filename, target_table):
    sio = StringIO()

    fast_forward = True
    with open(dump_filename, 'r', encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line.lower().startswith('insert') and target_table in line:
                fast_forward = False
            else:
                fast_forward = True #added: after we found table, keep going
            if fast_forward:
                continue
            data = re.findall('\([^\)]*\)', line)
            try:
                # need to make this a for loop, bc this mysql dump does not separate onto lines
                for i in range(0,len(data)):
                    newline = data[i]
                    newline = newline.strip(' ()')
                    newline = newline.replace('`', '')
                    sio.write(newline)
                    sio.write("\n")
            except IndexError:
                pass
            # added - write to csv, avoids keeping everything in memory...
            if len(data) > 0:
                #sio.pos = 0 #bug for later versions of python
                sio.seek(0)
                with open (target_table+'_short.csv', 'a', encoding="utf8") as fd:
                    shutil.copyfileobj(sio, fd,-1)
                sio = StringIO() # faster to just make new object than clear?
            if line.endswith(';'): # comment this out for long version
                break
    return

if __name__ == "__main__":
    headerSet=['steamid', 'appid','playtime_2weeks','playtime_forever','dateretrieved']
    
    """ list of tables in sql file: ['achievement_percentages','app_id_info','app_id_info_old','games_developers_old',
    'games_genres_old','games_publishers_old','friends','games_1','games_2','games_developers','games_genres',
    'games_publishers','groups','player_summaries','games_daily']
    """
    
    #read_dump('D:/Rachael/Documents/spring2019Chall/steam.sql','Games_2')
    # shorter version of dataset - runs faster
    read_dump_short('D:/Rachael/Documents/spring2019Chall/steam.sql','Games_2')
    dfGames2 = pd.read_csv('D:/Rachael/Documents/spring2019Chall/Games_2_short.csv',names=headerSet)
    print('games2 length before: ',len(dfGames2))
    
    # get rid of users/games that have a NaN or 0 playtime_forever
    dfGames2 = dfGames2.dropna(subset=['playtime_forever'])
    dfGames2 = dfGames2[(dfGames2['playtime_forever'] > 0)]
    print('games2 length after: ',len(dfGames2))
    
    # problem 1: lets first identify the types of users according to how much they play the games
    # they have, then cluster.  Use the table with more users, though it's only one snapshot of their game library
    # I think there should be 2 groups of users: those who play lots of games a little, and those
    # who play a few games a lot
    user_avgs = dfGames2.groupby('steamid')['playtime_forever'].mean()
    user_med = dfGames2.groupby('steamid')['playtime_forever'].median()
    user_std = dfGames2.groupby('steamid')['playtime_forever'].std()
    user_gameCount = dfGames2.groupby('steamid')['playtime_forever'].count()
    
    #make NaNs 0 for computation
    user_avgs = user_avgs.fillna(0)
    user_med = user_med.fillna(0)
    user_std = user_std.fillna(0)
    user_gameCount = user_gameCount.fillna(0)
    
    # useful for visualization - are there distinct clusters already?  No
    #plt.scatter(user_avgs,user_std)
    #plt.show()
    
    X = scale(np.asarray(list(zip(user_avgs,user_med,user_std,user_gameCount))))
    reduced_data = PCA(n_components=2).fit_transform(X)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(reduced_data)
    plt.scatter(reduced_data[:,0],reduced_data[:,1], c=kmeans.labels_, cmap='rainbow')
    plt.savefig('userGroupsByPlaytime.pdf', format='pdf')
    plt.show()
    
    # problem 2: further characterization of users by creating a normalized vector of play times
    # across all possible games, for each user, then cluster again
    dfGames2Filt = dfGames2.drop(['playtime_2weeks','dateretrieved'], axis=1)
    dfGames2Filt = pd.pivot_table(dfGames2Filt, index=['steamid'], columns=['appid'], aggfunc=np.sum, fill_value=0).reset_index()
    dfGames2Filt['playtime_forever'] = dfGames2Filt['playtime_forever'].apply(lambda x: x.map(lambda s: s/x.sum() if(s>0) else 0))
    
    reduced_data2 = PCA(n_components=2).fit_transform(np.asarray(dfGames2Filt))
    kmeans = KMeans(n_clusters=2, random_state=0).fit(reduced_data2)
    plt.scatter(reduced_data2[:,0],reduced_data[:,1], c=kmeans.labels_, cmap='rainbow')
    plt.savefig('userGroupsByPlaytimeSimilarApps.pdf', format='pdf')
    plt.show()
    
    # for later - can't get this table to read properly for some reason
    #read_dump('D:/Rachael/Documents/spring2019Chall/steam.sql','Games_Daily')
    #dfGames3 = pd.read_csv('D:/Rachael/Documents/spring2019Chall/games_daily.csv',names=headerSet)
    #print('games3 length before: ',len(dfGames3))
    #dfGames3 = dfGames3.dropna(subset=['playtime_forever'])
    #dfGames3 = dfGames3[(dfGames3['playtime_forever'] > 0)]
    #print('games3 length after: ',len(dfGames3))