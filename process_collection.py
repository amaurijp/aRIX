#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import os
diretorio = os.getcwd()
import sys
sys.path.append(diretorio + '/Modules')

from DB import database # type: ignore
from TEXTOBS import text_objects # type: ignore
from CLASSIFIER import lda, lsa # type: ignore
from ML import train_machine # type: ignore
from WV import WV # type: ignore


def main():

    #basic input
    article_file_type = 'webscience_csv_report' #('webscience_csv_report', 'pdf', 'xml')
    min_token_appereance_in_corpus = 10
    tfidf_batch_size = 1000
    words_to_test_wv_models = ['nanoparticles', 'nanosheets', 'aunps', 'gold', 'nps', 'cu', 'copper', 'molybdenum']

    ######################
    time_begin=time.time()

    #corpus update
    datab = database(diretorio=diretorio, article_file_type = article_file_type)
    datab.update()
    
    #documents processing
    t_objects = text_objects(article_file_type = article_file_type, diretorio = diretorio)
    t_objects.get_text_objects()
    t_objects.find_sent_stats(mode = 'raw_sentences')
    t_objects.filter_sentences()
    t_objects.find_sent_stats(mode = 'filtered_sentences')
    t_objects.find_text_sections()
    t_objects.get_ngrams_appearance()
    t_objects.filter_2grams()
    t_objects.find_idf(min_token_appereance_in_corpus = min_token_appereance_in_corpus)
    t_objects.set_tfidf_log_and_sent_stats(file_batch_size = tfidf_batch_size)
    t_objects.find_tfidf()
    t_objects.pos_filter(apply = False)

    #getting matrices doc_tokens, doc_topics and training WVs with SVD (LSA)
    wv = WV(wv_model = 'svd', diretorio = diretorio) #(w2vec or gensim or svd))
    wv.set_svd_parameters(mode = 'truncated_svd')
    wv.get_LSA_wv_matrixes(n_dimensions = 100, word_list_to_test = words_to_test_wv_models)
    wv.get_LSA_sent_topic_matrix()
    wv.get_LSA_article_topic_matrix()
    wv.get_LSA_topic_vector_stats()
    wv.get_wv_stats()

    #LDA classification
    lda_class = lda(diretorio = diretorio)
    lda_class.set_classifier(doc_type = 'articles', n_topics = 100, alpha = 0.01, beta = 0.01, file_batch_size = 10)
    lda_class.start_lda(iterations = 10)
    lda_class.run_lda()
    
    lda_class = lda(diretorio = diretorio)
    lda_class.set_classifier(doc_type = 'sents', n_topics = 100, alpha = 0.01, beta = 0.01, file_batch_size = 10)
    lda_class.start_lda(iterations = 10)
    lda_class.run_lda()
    
    #LSA classification
    lsa_class = lsa(diretorio = diretorio)
    lsa_class.export_token_topics_to_csv()

    #training WVs with DNNs and determining semantically similar tokens
    wv = WV(wv_model = 'w2vec', diretorio = diretorio) #(w2vec or gensim or svd))
    wv.set_w2vec_parameters(mode = 'cbow')
    wv.get_W2Vec(wv_dim = 300, word_list_to_test = words_to_test_wv_models)
    wv.get_wv_stats()
    wv.find_terms_sem_similarity(ner_classes = [])
    
    #training the section filters with CNNs
    mc = train_machine(machine_type = 'conv1d', wv_matrix_name ='wv_sents_w2vec_cbow_WS_3', diretorio = diretorio)
    mc.set_train_sections(section_name = 'introduction',
                          article_batch_size = 30,
                          sent_batch_size = 4,
                          sent_stride = 1,
                          n_epochs = 100,
                          load_architecture_number = False)
    mc.train_on_sections()

    mc = train_machine(machine_type = 'conv1d', wv_matrix_name ='wv_sents_w2vec_cbow_WS_3', diretorio = diretorio)
    mc.set_train_sections(section_name = 'methodology',
                          article_batch_size = 30,
                          sent_batch_size = 4,
                          sent_stride = 1,
                          n_epochs = 100,
                          load_architecture_number = False)
    mc.train_on_sections()

    mc = train_machine(machine_type = 'conv1d', wv_matrix_name ='wv_sents_w2vec_cbow_WS_3', diretorio = diretorio)
    mc.set_train_sections(section_name = 'results',
                          article_batch_size = 30,
                          sent_batch_size = 4,
                          sent_stride = 1,
                          n_epochs = 100,
                          load_architecture_number = False)
    mc.train_on_sections()
      
    time_end = time.time()
    print('\nTempo de processamento: ', round(time_end - time_begin, 2))



if __name__ == '__main__':
    main()