# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 14:58:50 2020

@author: gwhy5
"""
# In[225]:
## Import libraries

import pandas as pd
import numpy as np
from numpy.linalg import norm
import plotly_express as px 
from scipy.spatial.distance import pdist
from scipy.linalg import eigh as eig
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.manifold import TSNE
from gensim.models import word2vec
from IPython.core.display import display, HTML

# In[93]:
### Section 3: Analysis
## Read in necessary dataframes

DOC2 = pd.read_pickle('./DOC2.pkl')
group_df = pd.read_pickle('./group_df.pkl')
speeches = pd.read_pickle('./speeches.pkl')
TFIDF = pd.read_pickle('./TFIDF.pkl')
TFIDF1 = pd.read_pickle('./TFIDF1.pkl')
TOKEN = pd.read_pickle('./TOKEN.pkl')
VOCAB = pd.read_pickle('./VOCAB.pkl')
VOCAB_nostop = pd.read_pickle('./VOCAB_nostop.pkl')

#%%
## Create Normalized Tables

L0 = TFIDF1.astype('bool').astype('int')
L1 = TFIDF1.apply(lambda x: x / x.sum(), 1)
L2 = TFIDF1.apply(lambda x: x / norm(x), 1)

# In[94]:
## Create Doc Pair Table
# Create a table to store our results.

PAIRS = pd.DataFrame(index=pd.MultiIndex.from_product([DOC2.index.tolist(), DOC2.index.tolist()])).reset_index()
PAIRS = PAIRS[PAIRS.level_0 < PAIRS.level_1].set_index(['level_0','level_1'])
PAIRS.index.names = ['doc_a', 'doc_b']
PAIRS.head()

# In[96]:
## Compute Distances

PAIRS['cityblock'] = pdist(TFIDF1, 'cityblock')
PAIRS['euclidean'] = pdist(TFIDF1, 'euclidean')
PAIRS['cosine'] = pdist(TFIDF1, 'cosine')
PAIRS.head()

# In[100]:
## Create Hierarchical Clusters

def hca(name,sims, linkage_method='ward', color_thresh=.3, figsize=(15, 100)):
    tree = sch.linkage(sims, method=linkage_method)
    labels = DOC2.president_speech.values
    plt.figure()
    fig, axes = plt.subplots(figsize=figsize)
    dendrogram = sch.dendrogram(tree, 
                                labels=labels, 
                                orientation="left", 
                                count_sort=True,
                                distance_sort=True,
                                above_threshold_color='.75',
                                color_threshold=color_thresh
                               )
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.savefig(name)

# In[101]:

hca('cityblock_pairs.jpeg',PAIRS.cityblock, color_thresh=1) #adjust size as needed here

# In[102]:

hca("euclidean_pairs.jpeg",PAIRS.euclidean, color_thresh=1)

# In[103]:

hca('cosine_pairs.jpeg',PAIRS.cosine, color_thresh=1)

# There are very interesting things going on in those graphs, cityblock and cosine in particular. A lot of the times since presidents use similar styles and word choices, and are talking about similar things throughout their presidency, presidential speeches bunch up with the particular president. When different presidents get bunched up is when it gets interesting. For instance, the model can pick up the inagural addresses of different presidents, and things like foreign policy speeches (speeches by Carter, Bush Sr, Obama, etc bunch up in foreign policy speeches. One group in the cosine model looks like they are bunching up around Middle East topics in particular). I can also see certain speeches on drugs or crime bunch up together as well.

# In[104]:
## K-Means Clustering

# this variable is the number of clusters
n_clusters = 17

# In[105]:

DOC2['y_raw'] = KMeans(n_clusters).fit_predict(TFIDF)
DOC2['y_L0'] = KMeans(n_clusters).fit_predict(L0)
DOC2['y_L1'] = KMeans(n_clusters).fit_predict(L1)
DOC2['y_L2'] = KMeans(n_clusters).fit_predict(L2)

# In[106]:

DOC2.sort_values('y_L0').style.background_gradient(cmap='viridis')

# We see much of the same behavior in K-means clustering

# In[107]:
## PCA

OHCO = ['pres_id', 'speech_id','sent_num', 'token_num']
SPEECH = OHCO[:2]
PRES = OHCO[:1]

# In[108]:

TFIDF_s = TFIDF.reset_index().set_index(SPEECH) # group by speeches
TFIDF_p = TFIDF.reset_index().set_index(PRES) # group by presidents

# In[109]:

TFIDF_p = TFIDF_p.groupby('pres_id').mean() # get the mean, by presidents

# In[110]:

top_4000_terms = VOCAB_nostop.sort_values('tfidf_sum', ascending=False).iloc[:4000].index.tolist()

# In[111]:

TFIDF_s = TFIDF_s[top_4000_terms] # select only the top 4000 terms
TFIDF_p = TFIDF_p[top_4000_terms]

# In[112]:
# don't need to do this
#TFIDF_s = TFIDF_s - TFIDF_s.mean() 
#TFIDF_p = TFIDF_p - TFIDF_p.mean()

# In[113]:
## Compute Covariance Matrix

COV_s = TFIDF_s.T.dot(TFIDF_s) 
COV_p = TFIDF_p.T.dot(TFIDF_p)  

# In[114]:

COV_s.iloc[:5,:10].style.background_gradient()

# In[115]:
## Decompose the Matrix

eig_vals_s, eig_vecs_s = eig(COV_s)
eig_vals_p, eig_vecs_p = eig(COV_p)

# In[116]:
## Convert eigen data to dataframes

TERM_IDX = COV_p.index # We could use other tables as well, e.g. TFIDF_b, TFIDF_c, or COV_c

# In[117]:

EIG_VEC_s = pd.DataFrame(eig_vecs_s, index=TERM_IDX, columns=TERM_IDX)
EIG_VEC_p = pd.DataFrame(eig_vecs_p, index=TERM_IDX, columns=TERM_IDX)

# In[118]:

EIG_VAL_s = pd.DataFrame(eig_vals_s, index=TERM_IDX, columns=['eig_val'])
EIG_VAL_s.index.name = 'term_id'

EIG_VAL_p = pd.DataFrame(eig_vals_p, index=TERM_IDX, columns=['eig_val'])
EIG_VAL_p.index.name = 'term_id'

# In[119]:

EIG_VEC_p.iloc[:5, :10].style.background_gradient()

# In[120]:
## Select Principal Components
### Combine eigenvalues and eignvectors

EIG_PAIRS_s = EIG_VAL_s.join(EIG_VEC_s.T)
EIG_PAIRS_p = EIG_VAL_p.join(EIG_VEC_p.T)

# In[121]:
## Compute and Show Explained Variance

EIG_PAIRS_p['exp_var'] = np.round((EIG_PAIRS_p.eig_val / EIG_PAIRS_p.eig_val.sum()) * 100, 2)
EIG_PAIRS_s['exp_var'] = np.round((EIG_PAIRS_s.eig_val / EIG_PAIRS_s.eig_val.sum()) * 100, 2)

# In[122]:

EIG_PAIRS_p.exp_var.sort_values(ascending=False).head().plot.bar(rot=45)

# In[123]:

EIG_PAIRS_s.exp_var.sort_values(ascending=False).head().plot.bar(rot=45)

# In[124]:
## Pick Top K (10) Components

TOPS_p = EIG_PAIRS_p.sort_values('exp_var', ascending=False).head(10).reset_index(drop=True)
TOPS_p.index.name = 'comp_id'
TOPS_p.index = ["PC{}".format(i) for i in TOPS_p.index.tolist()]

# In[125]:

TOPS_s = EIG_PAIRS_s.sort_values('exp_var', ascending=False).head(10).reset_index(drop=True)
TOPS_s.index.name = 'comp_id'
TOPS_s.index = ["PC{}".format(i) for i in TOPS_s.index.tolist()]

# In[126]:
## Show Loadings

LOADINGS_p = TOPS_p[TERM_IDX].T
LOADINGS_p.index.name = 'term_id'

# In[127]:

LOADINGS_p.head().style.background_gradient()

# In[128]:

LOADINGS_s = TOPS_s[TERM_IDX].T
LOADINGS_s.index.name = 'term_id'

# In[129]:

LOADINGS_s.head().style.background_gradient()

# In[130]:

LOADINGS_p['term_str'] = LOADINGS_p.apply(lambda x: VOCAB.loc[int(x.name)].term_str, 1)
LOADINGS_s['term_str'] = LOADINGS_s.apply(lambda x: VOCAB.loc[int(x.name)].term_str, 1)

# In[131]:

lp0_pos = LOADINGS_p.sort_values('PC0', ascending=False).head(10).term_str.str.cat(sep=' ')
lp0_neg = LOADINGS_p.sort_values('PC0', ascending=True).head(10).term_str.str.cat(sep=' ')
lp1_pos = LOADINGS_p.sort_values('PC1', ascending=False).head(10).term_str.str.cat(sep=' ')
lp1_neg = LOADINGS_p.sort_values('PC1', ascending=True).head(10).term_str.str.cat(sep=' ')
lp2_pos = LOADINGS_p.sort_values('PC2', ascending=False).head(10).term_str.str.cat(sep=' ')
lp2_neg = LOADINGS_p.sort_values('PC2', ascending=True).head(10).term_str.str.cat(sep=' ')
ls0_pos = LOADINGS_s.sort_values('PC0', ascending=False).head(10).term_str.str.cat(sep=' ')
ls0_neg = LOADINGS_s.sort_values('PC0', ascending=True).head(10).term_str.str.cat(sep=' ')
ls1_pos = LOADINGS_s.sort_values('PC1', ascending=False).head(10).term_str.str.cat(sep=' ')
ls1_neg = LOADINGS_s.sort_values('PC1', ascending=True).head(10).term_str.str.cat(sep=' ')
ls2_pos = LOADINGS_s.sort_values('PC2', ascending=False).head(10).term_str.str.cat(sep=' ')
ls2_neg = LOADINGS_s.sort_values('PC2', ascending=True).head(10).term_str.str.cat(sep=' ')

# In[132]:
## Project Docs onto New Subspace

DCM_s = TFIDF_s.dot(TOPS_s[TERM_IDX].T)
DCM_p = TFIDF_p.dot(TOPS_p[TERM_IDX].T)

# In[133]:

DCM_s = DCM_s.merge(speeches[['pres_id','speech_id','President','title']], on=['pres_id','speech_id']).set_index(SPEECH)
DCM_p = DCM_p.merge(group_df.reset_index()[['President','pres_id']], on='pres_id').set_index(PRES)
DCM_p['title'] = DCM_p['President'] # need to set the title to president to visualize it

# In[134]:

DCM_s.head().style.background_gradient()

# In[135]:

DCM_p.head().style.background_gradient()

# In[136]:
## Visualize

def vis_pcs(M, a, b, prefix='PC'):

    fig = px.scatter(M, prefix + str(a), prefix + str(b), 
                        color='President', 
                        hover_name='title')
    fig.show()

# In[137]:
## President PC0, PC1

print('President PC0+', lp0_pos)
print('President PC0-', lp0_neg)
print('President PC1+', lp1_pos)
print('President PC1-', lp1_neg)

# In[138]:

vis_pcs(DCM_p, 0, 1)

# In[139]:
## President PC1, PC2

print('President PC1+', lp1_pos)
print('President PC1-', lp1_neg)
print('President PC2+', lp2_pos)
print('President PC2-', lp2_neg)

# In[140]:
vis_pcs(DCM_p, 1, 2)

# In[141]:
## Speeches PC0, PC1

print('Speeches PC0+', ls0_pos)
print('Speeches PC0-', ls0_neg)
print('Speeches PC1+', ls1_pos)
print('Speeches PC1-', ls1_neg)

# In[142]:

vis_pcs(DCM_s, 0, 1)

# In[143]:
## Speeches PC1, PC2

print('Speeches PC1+', ls1_pos)
print('Speeches PC1-', ls1_neg)
print('Speeches PC2+', ls2_pos)
print('Speeches PC2-', ls2_neg)

# In[144]:

vis_pcs(DCM_s, 1, 2)

# In[145]:
## Save

#DCM_p.to_csv('PCA_DCM_president.csv')
#TOPS_p.to_csv('PCA_TCM_president.csv')
#DCM_s.to_csv('PCA_DCM_speeches.csv')
#TOPS_s.to_csv('PCA_TCM_speeches.csv')

# In[146]:
## LDA with SciKit Learn
### Configs

n_terms = 5000
n_topics = 10
max_iter = 5

# In[147]:
## Convert TOKENS to table of paragraphs

PARAS = TOKEN[TOKEN.pos.str.match(r'^NNS?$')].groupby(SPEECH).term_str.apply(lambda x: ' '.join(x)).to_frame().rename(columns={'term_str':'para_str'})

# In[148]:
## Create Vector Space

tfv = CountVectorizer(max_features=n_terms, stop_words='english')
tf = tfv.fit_transform(PARAS.para_str)
TERMS = tfv.get_feature_names()

# In[149]:
## Generate Model

lda = LDA(n_components=n_topics, max_iter=max_iter, learning_offset=50., random_state=0)

# In[150]:
## THETA

THETA = pd.DataFrame(lda.fit_transform(tf), index=PARAS.index)
THETA.columns.name = 'topic_id'

# In[151]:

THETA.sample(20).style.background_gradient()

# In[152]:
## PHI

PHI = pd.DataFrame(lda.components_, columns=TERMS)
PHI.index.name = 'topic_id'
PHI.columns.name  = 'term_str'

# In[153]:

PHI.T.head().style.background_gradient()

# In[154]:
## Inspect Results
### Get Top Terms per Topic

TOPICS = PHI.stack().to_frame().rename(columns={0:'weight'})    .groupby('topic_id')    .apply(lambda x: 
           x.weight.sort_values(ascending=False)\
               .head(10)\
               .reset_index()\
               .drop('topic_id',1)\
               .term_str)
TOPICS

# In[156]:

TOPICS['label'] = TOPICS.apply(lambda x: str(x.name) + ' ' + ' '.join(x), 1)

# In[157]:
## Sort Topics by Doc Weight

TOPICS['doc_weight_sum'] = THETA.sum()
TOPICS.sort_values('doc_weight_sum', ascending=True).plot.barh(y='doc_weight_sum', x='label', figsize=(5,10)) 

# In[159]:
## Explore Topics by President

topic_cols = [t for t in range(n_topics)]
PRESIDENTS = THETA.groupby('pres_id')[topic_cols].mean().T                                            
PRESIDENTS.index.name = 'topic_id'

# In[161]:

PRESIDENTS['topterms'] = TOPICS[[i for i in range(10)]].apply(lambda x: ' '.join(x), 1)
president_dict = group_df.reset_index().set_index('pres_id')['President'].to_dict() # to rename the columns in terms of president names
PRESIDENTS = PRESIDENTS.rename(columns = president_dict)

# In[162]:

PRESIDENTS[['Barack Obama','Donald Trump','topterms']].sort_values('Barack Obama', ascending=False).style.background_gradient()

# In[241]:

PRESIDENTS[['Barack Obama','Donald Trump','Theodore Roosevelt', 'George Washington','topterms']].sort_values('Donald Trump', ascending=False).style.background_gradient()

# In[165]:

PRESIDENTS.sort_index(ascending=False).style.background_gradient()
 
# According to the LDA Model, Trump and Obama talk mostly about the same two topics. The top terms of the first topic suggest some typical president talk; government, jobs, economy, world. These are broad things presidents like to talk about. The second has terms of world affairs like; world, peace, war, nations, men, freedom. Adding in George Washington, we see a different set of topics prevelent. Washington talked about topics including; country, power, citizens, treaty, subject, government etc. These words seem slightly similar to the second topic (topic_id 1) but phrased in different words and different contexts. Obama and Trump spoke in large ideals of peace, war freedom, nations. Washington speaks in seemingly lower/simpler concepts of citizenship, treaties, subjects, authority, etc. Looking at the whole picture, you can see that the topics change through time. Specifically, by the FDR administration the topics change completely to topic_id one throughout all following presidents. There could be a couple reasons for this, the great influence of Roosevelt on American governance, as well as the changing role the president (and the country) has after WW2, since topic_id one has mostly to do with world affairs of war and peace.

# In[166]:
## Word2Vec
### Configuration

window = 5
BAG = SPEECH # speeches bag

# In[167]:

corpus = TOKEN[~TOKEN.pos.str.match('NNPS?')].groupby(BAG).term_str.apply(lambda  x:  x.tolist()).reset_index()['term_str'].tolist()

# In[168]:
## Generate word embeddings with Gensim's library

model = word2vec.Word2Vec(corpus, size=246, window=window, min_count=200, workers=4)

# In[169]:
## Visualize with tSNE

coords = pd.DataFrame(index=range(len(model.wv.vocab)))
coords['label'] = [w for w in model.wv.vocab]
coords['vector'] = coords['label'].apply(lambda x: model.wv.get_vector(x))

# In[171]:
## Use ScikitLearn's TSNE library

tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
tsne_values = tsne_model.fit_transform(coords['vector'].tolist())

# In[172]:

coords['x'] = tsne_values[:,0]
coords['y'] = tsne_values[:,1]

# In[174]:
## Plot the coordinates

px.scatter(coords, 'x', 'y', text='label', height=1000).update_traces(mode='text')

# Here we can see certain terms bunch up together. For instance, in the lower area (around (-10,-30)) we can see terms dealing with finance and economics (deficit, balance, fiscal).

# In[175]:
## Semantic Algebra

def complete_analogy(A, B, C, n=2):
    try:
        return model.wv.most_similar(positive=[B, C], negative=[A])[0:n]
    except KeyError as e:
        print('Error:', e)
        return None

# In[176]:

complete_analogy('democracy', 'prosperity', 'soviet')

# In[177]:

complete_analogy('criminal', 'violence', 'judge')

# The analogies don't make a ton of sense here. Perhaps criminal:violence is judge:fight means judges fight criminals? Not sure

# In[178]:

model.wv.most_similar('democracy')

# In[179]:

model.wv.most_similar('terrorists')

# Both of these similarity models produce good results. 'Terrorists' certainly are related to 'violence', 'terror', 'enemies' etc. and 'democracy' is related to 'freedom', 'civilization', 'liberty', etc.

# In[180]:

model.wv.most_similar(['soviet'],['hostile'])

# This suggests that there was a lot of emphasis on negotions and discussions with the 'hostile' 'soviets' in these speeches. We see 'cooperation' and 'agreements', as well  as 'historic'.

# In[181]:

model.wv.most_similar(['terrorists'],['war'])

# Here we some form of discussion on our dealings with 'terrorists' and 'war'. We see 'agents', 'local', 'individual', 'groups', etc. This may be presidents talking about how they see these groups (ie as 'local', 'individual', or 'groups') or how they aim to deal with the war on terror (ie use 'local' help to 'police' or 'enforce').

# In[182]:
## Sentiment Analysis of Novels 

OHCO = ['pres_id', 'speech_id','sent_num', 'token_num'] 
SPEECH = OHCO[1:2]
SENTS = OHCO[1:3]
salex_csv = 'salex_nrc.csv'
nrc_cols = "nrc_negative nrc_positive nrc_anger nrc_anticipation nrc_disgust nrc_fear nrc_joy nrc_sadness nrc_surprise nrc_trust".split()
emo = 'polarity'
trump = 45 #'Donald Trump'
obama = 44 #'Barack Obama'

# In[183]:
## Get Lexicon

salex = pd.read_csv(salex_csv).set_index('term_str')
salex.columns = [col.replace('nrc_','') for col in salex.columns]

# In[184]:

salex['polarity'] = salex.positive - salex.negative

# In[185]:
## Get lexicon columns

emo_cols = "anger anticipation disgust fear joy sadness surprise trust polarity".split()


# In[186]:
### Get tokens by president

TOKEN = TOKEN.reset_index().set_index(PRES).sort_index()

# In[187]:
# append the emotions to the token strings

TOKEN = TOKEN.join(salex, on='term_str', how='left')
TOKEN[emo_cols] = TOKEN[emo_cols].fillna(0)

# In[188]:

TOKEN[salex.columns].sample(10)

# In[189]:

TOKEN[['term_str'] + emo_cols].sample(10)

# In[190]:

TOKEN[emo_cols] = TOKEN[emo_cols].fillna(0)
TOKEN.head()

# In[192]:

TRUMP = TOKEN.loc[trump].copy()
OBAMA = TOKEN.loc[obama].copy()

# In[193]:

TRUMP[emo_cols].mean().sort_values().plot.barh()

# In[194]:

OBAMA[emo_cols].mean().sort_values().plot.barh()

# Both Trump and Obama have 'trust' and 'joy' as their top sentiments. Below that is where they differ. Trump is higher on 'fear', which is third, versus Obama, in which 'anticipation' is third ('fear' is fourth for Obama). Trump is also higher on 'anger' and 'sadness'.

# In[195]:
## Sentiment by Speech

TRUMP_speech = TRUMP.groupby(SPEECH)[emo_cols].mean()

# In[196]:

OBAMA_speech = OBAMA.groupby(SPEECH)[emo_cols].mean()

# In[197]:

def plot_sentiments(df, emo='polarity'):
    FIG = dict(figsize=(25, 5), legend=True, fontsize=14, rot=45)
    df[emo].plot(**FIG)

# In[198]:

plot_sentiments(TRUMP_speech, ['trust','joy','fear','anticipation','polarity'])

# Most speeches for Trump have positive polarity, however there is a speech recently that is very negative and high on fear. It looks as though it is probably speech_id 3, where President Trump announces the death of Abu Bakr al-Baghdadi, the founder and leader of ISIS. This speech would have a lot of negative words, which would explain the negative polarity here.

# In[199]:

plot_sentiments(TRUMP_speech, ['polarity'])

# In[200]:

plot_sentiments(OBAMA_speech, ['trust','joy','fear','anticipation','polarity'])

# Obama's speeches are much more over the place in terms of polarity and emotion. There is a lot of spikes in his sentiment plot. You can see where the polarity spikes downward, the fear spikes up. Obama's final speech largely ends up where his first speech is. 

# In[201]:

plot_sentiments(OBAMA_speech, ['polarity'])

# In[202]:
## Explore Sentiment in Texts

TRUMP['html'] =  TRUMP.apply(lambda x: "<span class='sent{}'>{}</span>".format(int(np.sign(x[emo])), x.token_str), 1)
OBAMA['html'] =  OBAMA.apply(lambda x: "<span class='sent{}'>{}</span>".format(int(np.sign(x[emo])), x.token_str), 1)

# In[203]:

TRUMP_sents = TRUMP.groupby(SENTS)[emo_cols].mean()
OBAMA_sents = OBAMA.groupby(SENTS)[emo_cols].mean()

# In[205]:

TRUMP_sents['sent_str'] = TRUMP.groupby(SENTS).term_str.apply(lambda x: x.str.cat(sep=' '))
TRUMP_sents['html_str'] = TRUMP.groupby(SENTS).html.apply(lambda x: x.str.cat(sep=' '))

# In[206]:

OBAMA_sents['sent_str'] = OBAMA.groupby(SENTS).term_str.apply(lambda x: x.str.cat(sep=' '))
OBAMA_sents['html_str'] = OBAMA.groupby(SENTS).html.apply(lambda x: x.str.cat(sep=' '))

# In[207]:

def sample_sentences(df):
    rows = []
    for idx in df.sample(10).index:

        valence = round(df.loc[idx, emo], 4)     
        t = 0
        if valence > t: color = '#ccffcc'
        elif valence < t: color = '#ffcccc'
        else: color = '#f2f2f2'
        z=0
        rows.append("""<tr style="background-color:{0};padding:.5rem 1rem;font-size:110%;">
        <td>{1}</td><td>{3}</td><td width="400" style="text-align:left;">{2}</td>
        </tr>""".format(color, valence, df.loc[idx, 'html_str'], idx))

    display(HTML('<style>#sample1 td{font-size:120%;vertical-align:top;} .sent-1{color:red;font-weight:bold;} .sent1{color:green;font-weight:bold;}</style>'))
    display(HTML('<table id="sample1"><tr><th>Sentiment</th><th>ID</th><th width="600">Sentence</th></tr>'+''.join(rows)+'</table>'))

# In[208]:

sample_sentences(TRUMP_sents)

# In[209]:

sample_sentences(OBAMA_sents)

# Along with the general problem of how these words are classified, some of these sentences don't look correct. This is something that needs to be improved upon. It may be a problem with the tokenizer and the dataset in particular.