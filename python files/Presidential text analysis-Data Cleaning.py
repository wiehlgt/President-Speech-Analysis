# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 14:57:58 2020

@author: gwhy5
"""

#!/usr/bin/env python
# coding: utf-8

# # Final Project DS5001
# ## An exploratory text analysis of Presidential Speeches
# ### Gavin Wiehl (gtw4vx)

# ## Introduction
# For my final project I have chosen to use a dataset of presidential speeches obtained through [kaggle.com](https://www.kaggle.com/kboghe/presidentialspeeches?select=2presidential_speeches_with_metadata.csv) to understand and potentially find interesting insights into presidential speeches. Are there some presidents who sound similar in their speeches? Are there certain topics that come up most frequently? Which word choices stick out among the speech corpus? These are among the questions I would be interested in answering.

# In[225]:
## Import libraries

import os
import pandas as pd

# In[226]:
## Read the CSV file

#work_dir = os.chdir('Desktop\\UVA DS\\DS 5001\\final_project\\President-Speech-NLP\\deliverables')

speeches_path = './presidential_speeches_with_metadata1.csv'
OHCO = ['pres_id', 'speech_id','sent_num', 'token_num'] # we will be breaking things into president, speeches, sentences, and tokens

# In[227]:
# certain rows will have all nans, we need to removed these and then reset the index

speeches = pd.read_csv(speeches_path,encoding='cp1252').dropna(how='all').reset_index(drop=True)
speeches.head()

# In[229]:

speeches.shape

# In[230]:
# ## Section 1: Data Cleaning
# Here I will do some quick data cleaning in order for the tokenization to work better. At the end I will save a checkpoint just in case, but the reader does not have to run that cell

speeches = speeches[~speeches.title.str.contains("[Dd]ebate")] # remove the debates, makes it easier
speeches.shape

# In[232]:

speeches.speech.replace(r'\(.*?\)', "", regex=True, inplace = True) # remove (laughter) and (Applause.)
speeches.speech.replace(r"\xa0","", regex=True, inplace=True) # remove \xa0
speeches.speech.replace(r'^THE PRESIDENT',"", regex=True, inplace=True) # remove beginning with THE PRESIDENT
speeches.speech.replace(r'^PRESIDENT TRUMP:', "", regex=True, inplace=True) # remove beginning with PRESIDENT TRUMP
speeches.speech.replace(r"(Q\s+.+?:)", "", regex=True, inplace=True) # remove the questions to get most of the presidents words
speeches.speech.replace(r"(Q:\s*.+?:)","", regex=True, inplace=True) # remove more questions
speeches.speech.replace(r"(Q.\s+.+?\?)","", regex=True, inplace=True) # remove even more questions
speeches.speech.replace(r'(PRESIDENT.*?[:.])',"", regex=True, inplace=True) # remove PRESIDENT:
speeches.speech.replace(r'(\[[Ll]aughter\])',"", regex=True, inplace=True) # remove the [laughter]
speeches.speech.replace(r"—", " ", regex=True, inplace=True) # remove unwanted characters
speeches.speech.replace(r":"," ", regex=True, inplace=True) # remove more unwanted characters
speeches.speech.replace(r'(\')', "", regex=True, inplace=True) # remove even more unwanted characters


# In[233]:

# create a series that is goes by each president and all of his speeches

pres_group = speeches.groupby(['President'],sort=False).speech

# In[234]:

# We need to add the president's number, as in Trump being the 45th president
# First, a new dataframe is created of the presidents in order, then their number is added
# this dataframe will then be joined together on the Presidents name

group_df = pres_group.first().to_frame()
group_df['pres_id'] = [i for i in range(45,0,-1)]
pres_num_df = group_df['pres_id'].to_frame()

# In[237]:

# the president's number dataframe is joined on the speeches dataframe
# this is in order to add the presidents number in the correct place
# for all of the speeches of each president

speeches = speeches.join(pres_num_df,on='President')

# In[238]:

# We also need to add a number to the speeches, in order to better navigate
# the speeches. The latest speech for President Trump that we have is given '1'
# while the first speech for President Washington is the '981'

speeches['speech_id'] = [i for i in range(1,speeches.shape[0] + 1)]
speeches.head()

# In[239]:

# Lets create a bar plot of the speeches by president.

speeches.President.value_counts().plot.barh(figsize=(10,10),title='Distribution of speeches by President');

# We can see hear that LBJ, Ronald Reagan, and Barack Obama have the greatest number of speeches in the dataset. William Harrison, James A. Garfield, and Zachary Taylor have the least amount of speeches. 

# In[14]:

# the meta list will include the columns that can be considered metadata. Much of these will not be relevant to later analysis
# (however some will). These will be the columns of the LIB (or library) dataframe.

meta = ['President', 'Party', 'from', 'until', 'Vice President', 'title', 'date', 'info', 'pres_id', 'speech_id']
LIB = speeches[meta] # create the LIB dataframe

# In[15]:

# Next the DOC (or document) dataframe will be created. The each speech by each president will be considered a document for now.

doc_fields = ['pres_id', 'speech_id', 'speech']
DOC = speeches[doc_fields].copy().set_index(['pres_id', 'speech_id'])
DOC.head()

# In[17]:
# Time to save the necessary dataframes for the next section

#speeches.to_csv('speeches_checkpoint.csv',encoding='cp1252') #saving the checkpoint, can comment 
DOC.to_pickle('./DOC.pkl')
LIB.to_pickle('./LIB.pkl')
group_df.to_pickle('./group_df.pkl')
speeches.to_pickle('./speeches.pkl')

## End of Section 1: Data Cleaning