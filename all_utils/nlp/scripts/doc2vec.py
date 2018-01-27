#!/usr/bin/env python
# coding: utf-8
import gensim
import os
import collections
import smart_open
import random
import argparse

def main(input_file, vec_file, size, iter_nums, min_count, workers=12):

    def read_corpus(fname, tokens_only=False):
        with open(fname) as f:
            for i, line in enumerate(f):
                if tokens_only:
                    yield line.strip().split()
                else:
                    # For training data, add tags
                    yield gensim.models.doc2vec.TaggedDocument(line.strip().split(), [i])

    train_corpus = list(read_corpus(input_file))
    model = gensim.models.doc2vec.Doc2Vec(documents=train_corpus, size=size, min_count=min_count, iter=iter_nums, workers= workers)
    model.save_word2vec_format(vec_file)

    print " Train FINISH！"
    ranks = []
    second_ranks = []
    for doc_id in range(len(train_corpus)):
        inferred_vector = model.infer_vector(train_corpus[doc_id].words)
        sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        rank = [docid for docid, sim in sims].index(doc_id)
        ranks.append(rank)
    
        second_ranks.append(sims[1])
    print " RANKS", collections.Counter(ranks)  # Results vary due to random seeding and very small corpus
    '''
    print('Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
    print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
    for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
        print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))
    doc_id = random.randint(0, len(train_corpus))

    print('Train Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
    sim_id = second_ranks[doc_id]
    print('Similar Document {}: «{}»\n'.format(sim_id, ' '.join(train_corpus[sim_id[0]].words)))
    '''

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--train_file')
    parser.add_argument('-o','--vec_path')
    parser.add_argument('-l', '--size', type=int, default=300)
    parser.add_argument('-n', '--iter_nums', type=int, default=5)
    parser.add_argument('-m', '--min_count', type=int, default=5)
    args = parser.parse_args()
    main(args.train_file, args.vec_path, args.size, args.iter_nums, args.min_count)

