# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 14:58:49 2020

@author: gwhy5
"""

# In[225]:
## Import libraries

import pandas as pd
import numpy as np
import nltk
import seaborn as sns
import plotly_express as px 

# In[18]:
### Section 2: Tokenization and Data Models
# Load in the dataframes and create the correct variables

DOC = pd.read_pickle('./DOC.pkl')
LIB = pd.read_pickle('./LIB.pkl')
group_df = pd.read_pickle('./group_df.pkl')
speeches = pd.read_pickle('./speeches.pkl')

OHCO = ['pres_id', 'speech_id','sent_num', 'token_num'] # we will be breaking things into president, speeches, sentences, and tokens

#%%
## Tokenize the documents
# Download the necessary datasets in order to tokenize

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('tagsets')

# In[19]:

def tokenize(doc_df, OHCO=OHCO, remove_pos_tuple=False, ws=False):
    
    # Speech to Sentences
    df = doc_df.speech.apply(lambda x: pd.Series(nltk.sent_tokenize(x))).stack().to_frame().rename(columns={0:'sent_str'})
    
    # Sentences to Tokens
    # Local function to pick tokenizer
    def word_tokenize(x):
        if ws:
            s = pd.Series(nltk.pos_tag(nltk.WhitespaceTokenizer().tokenize(x)))
        else:
            s = pd.Series(nltk.pos_tag(nltk.word_tokenize(x)))
        return s
            
    df = df.sent_str.apply(word_tokenize).stack().to_frame().rename(columns={0:'pos_tuple'})
    
    # Grab info from tuple
    df['pos'] = df.pos_tuple.apply(lambda x: x[1])
    df['token_str'] = df.pos_tuple.apply(lambda x: x[0])
    if remove_pos_tuple:
        df = df.drop('pos_tuple', 1)
    
    # Add index
    df.index.names = OHCO
    
    return df

# In[20]:

TOKEN = tokenize(DOC, ws=True)
TOKEN

# In[22]:

TOKEN['term_str'] = TOKEN['token_str'].str.lower().str.replace('[\W_]', '')

# In[23]:

VOCAB = TOKEN.term_str.value_counts().to_frame().rename(columns={'index':'term_str', 'term_str':'n'})    .sort_index().reset_index().rename(columns={'index':'term_str'})
VOCAB.index.name = 'term_id'

# In[24]:

VOCAB['num'] = VOCAB.term_str.str.match("\d+").astype('int')
VOCAB.sample(10)

# In[26]:
## Simple Unigram Model

n_tokens = VOCAB.n.sum()
VOCAB['p'] = VOCAB['n'] / n_tokens
VOCAB['log_p'] = np.log2(VOCAB['p'])

# In[27]:

VOCAB.sort_values('p', ascending=False).head(10)

# In[28]:

smooth = VOCAB['p'].min()
def predict_sentence(sent_str):
    
    # Parse sentence into tokens and normalize string
    tokens = pd.DataFrame(sent_str.lower().split(), columns=['term_str'])
    
    # Link the tokens with model vocabulary
    tokens = tokens.merge(VOCAB, on='term_str', how='left') # Left join is key
    
    # Add minimum values where token is not in our vocabulary
    tokens.loc[tokens['p'].isna(), 'p'] = [smooth]
    
    # Compute probability of sentence by getting product of token probabilities
    p = tokens['p'].product()
        
    # Print results
    print("p('{}') = {}".format(sent_str, p))

# In[29]:

predict_sentence('I love you')
predict_sentence('The economy is great')
predict_sentence("Turkey and Egypt")
predict_sentence("Four Score and Seven years")
predict_sentence("We are keeping in contact")
predict_sentence('Fake news')

# In[30]:
## Annotate (VOCAB)
### Add Stopwords

sw = pd.DataFrame(nltk.corpus.stopwords.words('english'), columns=['term_str'])
sw = sw.reset_index().set_index('term_str')
sw.columns = ['dummy']
sw.dummy = 1

# In[31]:

VOCAB['stop'] = VOCAB.term_str.map(sw.dummy)
VOCAB['stop'] = VOCAB['stop'].fillna(0).astype('int')

# In[32]:

VOCAB[VOCAB.stop == 1].sample(10)

# In[33]:
### Add Stems

from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()
VOCAB['p_stem'] = VOCAB.term_str.apply(stemmer.stem)
VOCAB.sample(10)

# In[35]:
# Below we save the dataframes to csv, not necessary
## Save

#DOC.to_csv('DOC.csv',encoding = 'cp1252')
#LIB.to_csv('LIB.csv',encoding = 'cp1252')
#VOCAB.to_csv('VOCAB.csv',encoding = 'cp1252')
#TOKEN.to_csv('TOKEN.csv',encoding = 'cp1252')

# In[36]:
## TDIF and Vector Space Models

count_method = 'n' # 'c' or 'n' # n = n tokens, c = distinct token (term) count
tf_method = 'sum' # sum, max, log, double_norm, raw, binary
tf_norm_k = .5 # only used for double_norm
idf_method = 'standard' # standard, max, smooth
gradient_cmap = 'GnBu' 

# In[37]:

sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')

# In[38]:

TOKEN['term_id'] = TOKEN.term_str.map(VOCAB.reset_index().set_index('term_str').term_id)
TOKEN.head()

# In[40]:
## Add Max Pos to VOCAB

VOCAB['pos_max'] = TOKEN.groupby(['term_id', 'pos']).count().iloc[:,0].unstack().idxmax(1)
VOCAB.sample(10)

# In[42]:
## A look at TOKEN POS graph

POS = TOKEN.pos.value_counts().to_frame().rename(columns={'pos':'n'})
POS.index.name = 'pos_id'

# In[43]:

POS.sort_values('n').plot.bar(y='n', figsize=(15,5), rot=45);

# The most common POS is the noun (which makes sense since they would be common in speeches pertaining to real world events) followed by preposition, and then determiners.

# In[44]:
## Adding Term rank to VOCAB

if 'term_rank' not in VOCAB.columns:
    VOCAB = VOCAB.sort_values('n', ascending=False).reset_index()
    VOCAB.index.name = 'term_rank'
    VOCAB = VOCAB.reset_index()
    VOCAB = VOCAB.set_index('term_id')
    VOCAB['term_rank'] = VOCAB['term_rank'] + 1
    
VOCAB.head()

# In[46]:
## Compute Zipf's K

VOCAB['zipf_k'] = VOCAB.n * VOCAB.term_rank

# In[47]:

px.histogram(VOCAB, 'zipf_k', marginal='box')

# This looks about right, however there is a fat tail at the end in which the count jumps up at around 300k

# In[48]:
## Demo Rank Index

rank_index = [1, 2, 3, 15, 30, 45, 60, 75, 90, 105, 120, 235, 350, 400, 550, 675, 775, 875, 975, 1075, 2075, 3075, 4075, 5075, 6075, 7075, 8075]

# In[49]:

demo = VOCAB.loc[VOCAB.term_rank.isin(rank_index), ['term_str', 'term_rank', 'n', 'zipf_k', 'pos_max']]
demo.style.background_gradient(cmap=gradient_cmap, high=.5)

# In[51]:
## Compute VOCAB ENTROPY

VOCAB['p'] = VOCAB.n / VOCAB.n.sum()

# In[52]:

VOCAB['h'] = VOCAB.p * np.log2(1/VOCAB.p) # Self entropy of each word 
H = VOCAB.h.sum()
N_v = VOCAB.shape[0]
H_max = np.log2(N_v)
R = round(1 - (H/H_max), 2) * 100

# In[53]:
## Time to create the Bag-of-words

SENTS = OHCO[:3]
SPEECH = OHCO[:2]
PRES = OHCO[:1]

# In[54]:

bag = SPEECH # choose speech as bag

# In[55]:

BOW = TOKEN.groupby(bag+['term_id']).term_id.count().to_frame().rename(columns={'term_id':'n'})
BOW['c'] = BOW.n.astype('bool').astype('int')
BOW.head()

# In[58]:
## Document-Term Matrix
# We create a document-term count matrix. 

DTCM = BOW[count_method].unstack().fillna(0).astype('int')
DTCM.head(10)

# In[60]:
## Compute TF

print('TF method:', tf_method)
if tf_method == 'sum':
    TF = DTCM.T / DTCM.T.sum()
elif tf_method == 'max':
    TF = DTCM.T / DTCM.T.max()
elif tf_method == 'log':
    TF = np.log10(1 + DTCM.T)
elif tf_method == 'raw':
    TF = DTCM.T
elif tf_method == 'double_norm':
    TF = DTCM.T / DTCM.T.max()
    TF = tf_norm_k + (1 - tf_norm_k) * TF[TF > 0]
elif tf_method == 'binary':
    TF = DTCM.T.astype('bool').astype('int')
TF = TF.T
TF.head()

# In[62]:
## Compute DF

DF = DTCM[DTCM > 0].count()

# In[63]:
## Compute IDF

N = DTCM.shape[0]

# In[64]:

print('IDF method:', idf_method)
if idf_method == 'standard':
    IDF = np.log10(N / DF)
elif idf_method == 'max':
    IDF = np.log10(DF.max() / DF) 
elif idf_method == 'smooth':
    IDF = np.log10((1 + N) / (1 + DF)) + 1

# In[65]:
## Compute TFIDF

TFIDF = TF * IDF
TFIDF.head()

# In[67]:
# Move it into VOCAB

VOCAB['df'] = DF
VOCAB['idf'] = IDF
VOCAB.head()

# In[69]:

BOW['tf'] = TF.stack()
BOW['tfidf'] = TFIDF.stack()
BOW.head()

# In[71]:
## Apply aggregates to VOCAB and visualize

VOCAB['tfidf_mean'] = TFIDF[TFIDF > 0].mean().fillna(0) # EXPLAIN
VOCAB['tfidf_sum'] = TFIDF.sum()
VOCAB['tfidf_median'] = TFIDF[TFIDF > 0].median().fillna(0) # EXPLAIN
VOCAB['tfidf_max'] = TFIDF.max()

# In[72]:
## Make a VOCAB without stopwords

VOCAB_nostop = VOCAB[VOCAB['stop'] == 0] # VOCAB table with no stop words
VOCAB_nostop.head(10)

# In[73]:

VOCAB.sort_values('tfidf_sum', ascending=False).head(50).style.background_gradient(cmap=gradient_cmap)

# In[74]:

VOCAB_nostop.sort_values('tfidf_sum', ascending=False).head(50).style.background_gradient(cmap=gradient_cmap)

# In[75]:

VOCAB[['term_rank','term_str','pos_max','tfidf_sum']].sort_values('tfidf_sum', ascending=False).head(50).style.background_gradient(cmap=gradient_cmap)

# In[76]:

VOCAB_nostop[['term_rank','term_str','pos_max','tfidf_sum']].sort_values('tfidf_sum', ascending=False).head(50).style.background_gradient(cmap=gradient_cmap)

# I am suprised by how many terms do not make an impact on the tfidf_sum. "we' and 'you' have the greatest impact here, both words high term ranks

# In[77]:

BOW = BOW.join(VOCAB[['term_str','pos_max']], on='term_id')
BOW.sort_values('tfidf', ascending=False).head(20).style.background_gradient(cmap=gradient_cmap, high=.5)

# The highest tfidf terms are freedman, deck, and statue

# In[79]:
## Rank and TFIDF mean

px.scatter(VOCAB, x='term_rank', y='tfidf_mean', hover_name='term_str', hover_data=['n'], color='pos_max', 
           log_x=False, log_y=False)

# The highest POS on the tfidf_mean scale are the NNP (Proper Nouns). This makes sense since certain speeches will revolve around certain historical characters/places/things.

# In[80]:
## Rank and TFIDF sum

px.scatter(VOCAB, x='term_rank', y='tfidf_sum', hover_name='term_str', hover_data=['n'], color='pos_max')

# The NN (Noun singular or mass) seems to be consistantly on top of the slope, with NNP's around there as well. The tfidf_sum looks like it accounts for the difference in singular (or mass) and proper nouns

# In[81]:
## Show Demo Table with TFIDF

demo2 = VOCAB_nostop.loc[VOCAB.term_rank.isin(rank_index), ['term_str', 'pos_max', 'term_rank', 'n', 'zipf_k', 'tfidf_mean', 'tfidf_sum', 'tfidf_max']]
demo2.style.background_gradient(cmap=gradient_cmap)

# It is interesting that the country of panama shows up prominently in the above table. It is clearly important in US world affairs judging by it's tfidf scores

# In[83]:

px.scatter(demo2, x='term_rank', y='tfidf_sum', log_x=True, log_y=True, text='term_str', color='pos_max', size='n')

# When using the VOCAB with no stopwords, the tfidf_sum decreases as term rank increases, which makes sense. This is not the case with tfidf_mean measure, which is high for some terms that are have a high rank

# In[84]:
### Save new files , can comment out

#VOCAB.to_csv('VOCAB2.csv',encoding = 'cp1252')
#TOKEN.to_csv('TOKEN2.csv',encoding = 'cp1252')
#BOW.to_csv('DOC2.csv',encoding = 'cp1252')
#DTCM.to_csv('DTCM.csv',encoding = 'cp1252')
#TFIDF.to_csv('TFIDF.csv',encoding = 'cp1252')

# In[85]:
## Similarity and Distance Measures
# Here we will group by speeches to better analyze/visualize the data

sns.set(style="ticks")
get_ipython().run_line_magic('matplotlib', 'inline')

# In[86]:

OHCO_src = ['pres_id', 'speech_id']
TFIDF1 = TFIDF.reset_index().set_index(OHCO_src)

TFIDF1.head()

# In[88]:
## Create a new DOC table
 
# We want to create a new table that maps the OHCO levels to a single doc_id. We do this so that when we create a table to store pairs of docs and their distances, we can use a single-valued ID for each docs. 
# This table will also be used to store cluster assignments.

DOC2 = TFIDF1.reset_index().set_index('speech_id')[PRES] # We create a table from the OHCO in our TFIDF table

DOC2.head()

# In[89]:

title_df = speeches[['speech_id','title']].copy().set_index('speech_id') # a dataframe of the speech titles
TFIDF1 = TFIDF1.merge(title_df,on='speech_id')

# In[91]:
## Add a meaningful Title to DOC index

presidents = pd.Series(LIB.groupby('pres_id')['President'].unique(),dtype=str) # capture presidents names by pres_id
DOC2['president'] = DOC2.pres_id.map(presidents) # add in presidents names
speech_titles = TFIDF1.groupby('speech_id')['title'].first().to_frame() # capture the speech titles by speech_id
DOC2 = DOC2.merge(speech_titles, on='speech_id') # add in the speech titles
DOC2['president_speech'] = DOC2['president'].str.cat(DOC2['title'],sep=" ") # concatinate president and title col
TFIDF1 = TFIDF.reset_index().set_index(OHCO_src) # return the TFIDF1 table back to original for processing
DOC2 = DOC2.reset_index().set_index('speech_id').drop(['president','title'],axis=1) # remove president, title col

DOC2.head()

#%%
## Time to save the necessary dataframes for the next section

DOC2.to_pickle('./DOC2.pkl')
group_df.to_pickle('./group_df.pkl')
speeches.to_pickle('./speeches.pkl')
TFIDF.to_pickle('./TFIDF.pkl')
TFIDF1.to_pickle('./TFIDF1.pkl')
TOKEN.to_pickle('./TOKEN.pkl')
VOCAB.to_pickle('./VOCAB.pkl')
VOCAB_nostop.to_pickle('./VOCAB_nostop.pkl')

## End of Section 2: Tokenization and Data Models