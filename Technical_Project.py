# Databricks notebook source
import math
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')

%matplotlib inline





# COMMAND ----------


song_data_df = (spark.read
  .format("csv")
  .option("header", "true")
  .option("inferSchema", "true")
  .load("/FileStore/tables/song_data.csv")
)





# COMMAND ----------

# get a row count

print(song_data_df.count())

# print the schema (shape of your df)
print(song_data_df.printSchema())

# get the columns as a list
print(song_data_df.columns)
 
# get the columns and types as tuples in a list
print(song_data_df.dtypes)




# COMMAND ----------

print(song_data_df.printSchema())


# COMMAND ----------

song_data_df.show()


# COMMAND ----------

song_data_df.count()


# COMMAND ----------

print(song_data_df)
display(song_data_df)


# COMMAND ----------

song_data_df.head()


# COMMAND ----------

triplets_file_df = (spark.read
  .format("csv")
  .option("header", "true")
  .option("inferSchema", "true")
  .load("/FileStore/tables/triplets_file.csv")
)

# COMMAND ----------

# get a row count

print(triplets_file_df.count())

# print the schema (shape of your df)
print(triplets_file_df.printSchema())

# get the columns as a list
print(triplets_file_df.columns)
 
# get the columns and types as tuples in a list
print(triplets_file_df.dtypes)



# COMMAND ----------

print(triplets_file_df)
display(triplets_file_df)


# COMMAND ----------

song_data_df.show()


# COMMAND ----------

triplets_file_df.show()


# COMMAND ----------

#Convert PySpark Dataframe to Pandas DataFrame

triplets_file_df_pd = triplets_file_df.toPandas()
print(triplets_file_df_pd)

# COMMAND ----------

#Convert PySpark Dataframe to Pandas DataFrame

song_data_df_pd = song_data_df.toPandas()
print(song_data_df_pd)

# COMMAND ----------


#Find Pandas DataFrame Size, Shape, and Dimensions Properties


print("size is song_data" ,song_data_df_pd.shape)
print("total data count in song_data" ,song_data_df_pd.size)



# COMMAND ----------


#Find Pandas DataFrame Size, Shape, and Dimensions Properties


print("size is triplets_file" ,triplets_file_df_pd.shape)
print("total data count in triplets_file" ,triplets_file_df_pd.size)



# COMMAND ----------

song_data_df_pd

# COMMAND ----------

triplets_file_df_pd

# COMMAND ----------

##Size of the two datasets for a reference
print(len(triplets_file_df_pd),len(song_data_df_pd))

# COMMAND ----------

####Joining Data Frame


join_song_df = pd.merge(triplets_file_df_pd, song_data_df_pd.drop_duplicates(['song_id']), on='song_id', how='left')
join_song_df.head()


# COMMAND ----------

print(len(join_song_df))

# COMMAND ----------

join_song_df

# COMMAND ----------

#Standardising the missing data values
#Sometimes missing data is coded as ‘NO DATA’, ‘0’, ‘N/A’ or just an empty string. For ease of cleaning, convert all these into np.nan.

clean_song_df = join_song_df.replace([ 0, ''],np.nan)


# COMMAND ----------


clean_song_df

# COMMAND ----------

#to check the datatypes
clean_song_df.dtypes


# COMMAND ----------

len(join_song_df)


# COMMAND ----------

# Preprocessing the data

join_song_df['song'] = join_song_df['title']+' - '+join_song_df['artist_name']
join_song_df.head()

# COMMAND ----------

# Taking a small sample for refrence
join_song_df = join_song_df.head(5000)
join_song_df.head(10000)

# COMMAND ----------

grouped_sum = song_grouped['listen_count'].sum()
song_grouped['percentage'] = (song_grouped['listen_count'] / grouped_sum ) * 100
song_grouped.sort_values(['listen_count', 'song'], ascending=[0,1])

# COMMAND ----------

#Class initialisation based on Popularity Recommendation 
class pop_rec_py():
    def __init__(self):
        self.item_id = None
        self.trained_data = None
        self.pop_rec = None
        self.user_id = None
    #Creating the popularity based recommender model
    def create(self, trained_data, user_id, item_id):
        self.trained_data = trained_data
        self.item_id = item_id
        self.user_id = user_id
        #Get a count of userids for each distinct song as recommendation score
        trained_data_group = trained_data.groupby([self.item_id]).agg({self.user_id: 'count'}).reset_index()
        trained_data_group.rename(columns = {'user_id': 'score'},inplace=True)
    
        #Sorting the tracks based upon suggestion score
        trained_data_sorting = trained_data_group.sort_values(['score', self.item_id], ascending = [0,1])
    
        #Generate a suggetion  based upon score
        trained_data_sorting['Rank'] = trained_data_sorting['score'].rank(ascending=0, method='first')
        
        
        self.pop_rec = trained_data_sorting.head(10)  #Get the top 10 recommendations
        
    #Use the popularity based recommender system model for makeing suggestions
    
    def rec(self, user_id):    
        user_rec = self.pop_rec
        
       
        user_rec['user_id'] = user_id
    
        
        cols = user_rec.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        user_rec = user_rec[cols]
        
        return user_rec

#Class initialisation for similarity  suggestion model
class item_sim_rec_py():
    def __init__(self):
        self.trained_data = None
        self.user_id = None
        self.item_id = None
        self.cooccur_mat = None
        self.songs_dict = None
        self.rev_songs_dict = None
        self.item_sim_rec = None
        
    #Getting unique songs corresponding to a given user
    def get_user_items(self, user):
        user_data = self.trained_data[self.trained_data[self.user_id] == user]
        user_items = list(user_data[self.item_id].unique())
        
        return user_items
        
    
    def get_item_users(self, item):  #Get unique users for a given  song
        item_data = self.trained_data[self.trained_data[self.item_id] == item]
        item_users = set(item_data[self.user_id].unique())
            
        return item_users
        
    
    def get_all_items_trained_data(self):  #Get unique songs in the training data
        all_items = list(self.trained_data[self.item_id].unique())
            
        return all_items
        
    
    def construct_cooccur_mat(self, usersongs, all_songs):   #Constructing cooccurence matrix
            
        
        #Get listeners for all songs in dataset.
        
        user_songs_users = []        
        for i in range(0, len(usersongs)):
            user_songs_users.append(self.get_item_users(usersongs[i]))
            
       
        #Initialize the item cooccurence matrix of size 
        
        cooccur_mat = np.matrix(np.zeros(shape=(len(usersongs), len(all_songs))), float)
           
        
        #Calculating the similarity between user songs and personalised songs in the training data
        
    
        for i in range(0,len(all_songs)):
            #Calculating unique listeners  of songs
            songs_i_data = self.trained_data[self.trained_data[self.item_id] == all_songs[i]]
            users_i = set(songs_i_data[self.user_id].unique())
            
            for j in range(0,len(usersongs)):       
                    
                
                users_j = user_songs_users[j]   #Getting unique listeners  of song 
                    
                #Calculating intersection of listeners of songs i and j
                users_intersect = users_i.intersection(users_j)
                
                #Calculate cooccurence_matrix[i,j] as Jaccard Index
                if len(users_intersect) != 0:
                    
                    users_union = users_i.union(users_j)  #generating  union of listeners of songs i and j
                    
                    cooccur_mat[j,i] = float(len(users_intersect))/float(len(users_union))
                else:
                    cooccur_mat[j,i] = 0
                    
        
        return cooccur_mat

    
    #Use the cooccurence matrix for making top recommendations
    def generate_top_rec(self, user, cooccur_mat, all_songs, usersongs):
        print("Non zero values in cooccurence_matrix :%d" % np.count_nonzero(cooccur_mat))
        
        #Calculating a weighted average of the scores in cooccurence matrix for all user songs.
        user_sim_scores = cooccur_mat.sum(axis=0)/float(cooccur_mat.shape[0])
        user_sim_scores = np.array(user_sim_scores)[0].tolist()
 
        #Sort the indices of user_sim_scores based upon their value and maintaining the corresponding score
    
        sort_index = sorted(((e,i) for i,e in enumerate(list(user_sim_scores))), reverse=True)
    
        #Creating  a new dataframe
        columns = ['user_id', 'song', 'score', 'rank']
       
        df = pd.DataFrame(columns=columns)
         
        #Filling the dataframe with top 15 item based recommendations
        rank = 1 
        for i in range(0,len(sort_index)):
            if ~np.isnan(sort_index[i][0]) and all_songs[sort_index[i][1]] not in usersongs and rank <= 15:
                df.loc[len(df)]=[user,all_songs[sort_index[i][1]],sort_index[i][0],rank]
                rank = rank+1
        
        #if no recommendations
        if df.shape[0] == 0:
            print("The current user has no songs for training the item similarity based recommendation model.")
            return -1
        else:
            return df
 
    #Creating the item similarity based recommender  model
    def create(self, trained_data, user_id, item_id):
        self.trained_data = trained_data
        self.user_id = user_id
        self.item_id = item_id

    
    def rec(self, user):
        
        
        # Get all unique songs for this user
        
        usersongs = self.get_user_items(user)    
            
        print("No. of unique songs for the user: %d" % len(usersongs))
        
        
         #Get all unique items (songs) in the training data
        
        all_songs = self.get_all_items_trained_data()
        
        print("no. of unique songs in the training set: %d" % len(all_songs))
         
        
        # Construct item cooccurence matrix of size 
      
        cooccur_mat = self.construct_cooccur_mat(usersongs, all_songs)
        
        
        # Use the cooccurence matrix to make recommendations
        
        df_rec = self.generate_top_rec(user, cooccur_mat, all_songs, usersongs)
                
        return df_rec
    
    #similar items to given items
    def get_similar_items(self, item_list):
        
        usersongs = item_list
        
        
        # Get all unique songs in the training data
        
        all_songs = self.get_all_items_trained_data()
        
        print("no. of unique songs in the training set: %d" % len(all_songs))
         
        
        
        cooccur_mat = self.construct_cooccur_mat(usersongs, all_songs)        # Construct item cooccurence matrix of size 

        
        
        # Use the cooccurence matrix to make recommendations
        
        user = ""
        df_rec = self.generate_top_rec(user, cooccur_mat, all_songs, usersongs)
         
        return df_rec
    
    
    



# COMMAND ----------

pr = pop_rec_py()


# COMMAND ----------

pr.create(join_song_df, 'user_id', 'song')


# COMMAND ----------

# display the top 10 popular songs
pr.rec(join_song_df['user_id'][1250])


# COMMAND ----------

pr.rec(join_song_df['user_id'][1700])


# COMMAND ----------

pr.rec(join_song_df['user_id'][1025])


# COMMAND ----------

#Item Similarity Recommendation

ir = item_sim_rec_py()
ir.create(join_song_df, 'user_id', 'song')

# COMMAND ----------

user_items = ir.get_user_items(join_song_df['user_id'][5])


# COMMAND ----------

# display user songs history
for user_item in user_items:
    print(user_item)

# COMMAND ----------

# give song recommendation for user
ir.rec(join_song_df['user_id'][130])


# COMMAND ----------

# give song recommendation for that user
ir.rec(join_song_df['user_id'][1240])


# COMMAND ----------

1# give related songs based on the words
ir.get_similar_items(['Oliver James - Fleet Foxes', 'The End - Pearl Jam'])

# COMMAND ----------

1# give related songs based on the words
ir.get_similar_items(['Your Protector - Fleet Foxes', 'Misled - Céline Dion	'])

# COMMAND ----------


