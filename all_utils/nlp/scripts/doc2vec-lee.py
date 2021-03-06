
# coding: utf-8

# # Doc2Vec Tutorial on the Lee Dataset

# In[1]:


import gensim
import os
import collections
import smart_open
import random


# ## What is it?
# 
# Doc2Vec is an NLP tool for representing documents as a vector and is a generalizing of the Word2Vec method. This tutorial will serve as an introduction to Doc2Vec and present ways to train and assess a Doc2Vec model.

# ## Resources
# 
# * [Word2Vec Paper](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
# * [Doc2Vec Paper](https://cs.stanford.edu/~quocle/paragraph_vector.pdf)
# * [Dr. Michael D. Lee's Website](http://faculty.sites.uci.edu/mdlee)
# * [Lee Corpus](http://faculty.sites.uci.edu/mdlee/similarity-data/)
# * [IMDB Doc2Vec Tutorial](doc2vec-IMDB.ipynb)

# ## Getting Started

# To get going, we'll need to have a set of documents to train our doc2vec model. In theory, a document could be anything from a short 140 character tweet, a single paragraph (i.e., journal article abstract), a news article, or a book. In NLP parlance a collection or set of documents is often referred to as a <b>corpus</b>. 
# 
# For this tutorial, we'll be training our model using the [Lee Background Corpus](https://hekyll.services.adelaide.edu.au/dspace/bitstream/2440/28910/1/hdl_28910.pdf) included in gensim. This corpus contains 314 documents selected from the Australian Broadcasting
# Corporation’s news mail service, which provides text e-mails of headline stories and covers a number of broad topics.
# 
# And we'll test our model by eye using the much shorter [Lee Corpus](https://hekyll.services.adelaide.edu.au/dspace/bitstream/2440/28910/1/hdl_28910.pdf) which contains 50 documents.

# In[2]:


# Set file names for train and test data
test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data'])
lee_train_file = test_data_dir + os.sep + 'lee_background.cor'
lee_test_file = test_data_dir + os.sep + 'lee.cor'


# ## Define a Function to Read and Preprocess Text

# Below, we define a function to open the train/test file (with latin encoding), read the file line-by-line, pre-process each line using a simple gensim pre-processing tool (i.e., tokenize text into individual words, remove punctuation, set to lowercase, etc), and return a list of words. Note that, for a given file (aka corpus), each continuous line constitutes a single document and the length of each line (i.e., document) can vary. Also, to train the model, we'll need to associate a tag/number with each document of the training corpus. In our case, the tag is simply the zero-based line number.

# In[3]:


def read_corpus(fname, tokens_only=False):
    with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            if tokens_only:
                d= gensim.utils.simple_preprocess(line)
                print type(d)
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                d=gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])
                print type(d)
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])


# In[4]:


train_corpus = list(read_corpus(lee_train_file))
test_corpus = list(read_corpus(lee_test_file, tokens_only=True))


# Let's take a look at the training corpus

# In[5]:


train_corpus[:2]


# And the testing corpus looks like this:

# In[6]:


print(test_corpus[:2])


# Notice that the testing corpus is just a list of lists and does not contain any tags.

# ## Training the Model

# ### Instantiate a Doc2Vec Object 

# Now, we'll instantiate a Doc2Vec model with a vector size with 50 words and iterating over the training corpus 55 times. We set the minimum word count to 2 in order to give higher frequency words more weighting. Model accuracy can be improved by increasing the number of iterations but this generally increases the training time. Small datasets with short documents, like this one, can benefit from more training passes.

# In[7]:


model = gensim.models.doc2vec.Doc2Vec(documents=train_corpus, size=50, min_count=2, iter=55)
model.save_word2vec_format('doc2vec_model.txt')

# ### Build a Vocabulary

# In[8]:


## -- model.build_vocab(train_corpus)


# Essentially, the vocabulary is a dictionary (accessible via `model.wv.vocab`) of all of the unique words extracted from the training corpus along with the count (e.g., `model.wv.vocab['penalty'].count` for counts for the word `penalty`).

# ### Time to Train
# 
# If the BLAS library is being used, this should take no more than 3 seconds.
# If the BLAS library is not being used, this should take no more than 2 minutes, so use BLAS if you value your time.

# In[9]:


## -- model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)


# ### Inferring a Vector

# One important thing to note is that you can now infer a vector for any piece of text without having to re-train the model by passing a list of words to the `model.infer_vector` function. This vector can then be compared with other vectors via cosine similarity.

# In[10]:


model.infer_vector(['only', 'you', 'can', 'prevent', 'forrest', 'fires'])


# ## Assessing Model

# To assess our new model, we'll first infer new vectors for each document of the training corpus, compare the inferred vectors with the training corpus, and then returning the rank of the document based on self-similarity. Basically, we're pretending as if the training corpus is some new unseen data and then seeing how they compare with the trained model. The expectation is that we've likely overfit our model (i.e., all of the ranks will be less than 2) and so we should be able to find similar documents very easily. Additionally, we'll keep track of the second ranks for a comparison of less similar documents. 

# In[11]:


ranks = []
second_ranks = []
for doc_id in range(len(train_corpus)):
    inferred_vector = model.infer_vector(train_corpus[doc_id].words)
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    rank = [docid for docid, sim in sims].index(doc_id)
    ranks.append(rank)
    
    second_ranks.append(sims[1])
print zip(ranks,second_ranks)

# Let's count how each document ranks with respect to the training corpus 

# In[12]:


print collections.Counter(ranks)  # Results vary due to random seeding and very small corpus


# Basically, greater than 95% of the inferred documents are found to be most similar to itself and about 5% of the time it is mistakenly most similar to another document. the checking of an inferred-vector against a training-vector is a sort of 'sanity check' as to whether the model is behaving in a usefully consistent manner, though not a real 'accuracy' value.
# 
# This is great and not entirely surprising. We can take a look at an example:

# In[15]:


print('Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))


# Notice above that the most similar document is has a similarity score of ~80% (or higher). However, the similarity score for the second ranked documents should be significantly lower (assuming the documents are in fact different) and the reasoning becomes obvious when we examine the text itself

# In[14]:


# Pick a random document from the test corpus and infer a vector from the model
doc_id = random.randint(0, len(train_corpus))

# Compare and print the most/median/least similar documents from the train corpus
print('Train Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
sim_id = second_ranks[doc_id]
print('Similar Document {}: «{}»\n'.format(sim_id, ' '.join(train_corpus[sim_id[0]].words)))


# ## Testing the Model

# Using the same approach above, we'll infer the vector for a randomly chosen test document, and compare the document to our model by eye.

# In[15]:


# Pick a random document from the test corpus and infer a vector from the model
doc_id = random.randint(0, len(test_corpus))
inferred_vector = model.infer_vector(test_corpus[doc_id])
sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

# Compare and print the most/median/least similar documents from the train corpus
print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(test_corpus[doc_id])))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))


# ### Wrapping Up
# 
# That's it! Doc2Vec is a great way to explore relationships between documents.
