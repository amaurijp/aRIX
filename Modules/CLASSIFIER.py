#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd # type: ignore
import spacy # type: ignore
from nltk.corpus import stopwords # type: ignore
import numpy as np # type: ignore
import numba # type: ignore
from multiprocessing import get_context
import h5py # type: ignore
import os
import time
import regex as re # type: ignore
import psutil # type: ignore
from sklearn.manifold import TSNE # type: ignore
import matplotlib.pyplot as plt # type: ignore
import matplotlib.gridspec as gridspec # type: ignore
from matplotlib.colors import to_hex
import matplotlib.cm as cm
import seaborn as sns # type: ignore
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, to_tree
from scipy.cluster.hierarchy import fcluster
import networkx as nx # type: ignore
from sklearn.manifold import MDS

from FUNCTIONS import get_filenames_from_folder
from FUNCTIONS import save_dic_to_json
from FUNCTIONS import get_tag_name
from FUNCTIONS import get_file_batch_index_list
from FUNCTIONS import load_dic_from_json
from FUNCTIONS import update_log
from FUNCTIONS import load_log_info

from functions_TOKENS import get_tokens_indexes
from functions_TOKENS import get_tokens_from_sent

from functions_VECTORS import load_dense_matrix, load_sparse_csr_matrix
from functions_VECTORS import save_dense_matrix, save_sparse_csr_matrix


class lda(object):

    def __init__(self, diretorio = None):

        print('\n( Class: LDA classifier )')

        self.diretorio = diretorio
        
        if not os.path.exists(self.diretorio + '/Outputs/models'):
            print('Creating folder ', self.diretorio + '/Outputs/models')
            os.makedirs(self.diretorio + '/Outputs/models')



    def set_classifier(self, doc_type: str = None, n_topics: int = 100, alpha:float = 0.0, beta:float = 0.0, 
                       file_batch_size: int = 0):


        self.nlp = spacy.load('en_core_web_sm')
        self.stopwords_list = stopwords.words('english')
        self.n_topics = n_topics
        self.doc_type = doc_type.lower()
        self.alpha = alpha
        self.beta = beta

        #definições do modelo
        self.lda_definitions = f'{self.doc_type}_ntopics_{self.n_topics}_a_{self.alpha}_b_{self.beta}'
        self.path_doc_topic_counts = self.diretorio + f'/Outputs/models/lda_doc_topic_counts_{self.lda_definitions}'
        self.path_token_topic_counts = self.diretorio + f'/Outputs/models/lda_token_topic_counts_{self.lda_definitions}'

        #checando os caminhos das pastas das matrizes
        if not os.path.exists(self.diretorio + f'/Outputs/models/lda_{self.doc_type}_token_topic'):
            os.makedirs(self.diretorio + f'/Outputs/models/lda_{self.doc_type}_token_topic')

        #carregando o df dos IDF para pegar os tokens
        IDF = pd.read_csv(self.diretorio + f'/Outputs/tfidf/idf.csv', index_col = 0)
        self.all_tokens_array = IDF.index.values
        self.n_tokens = len(self.all_tokens_array)
        print('Tokens dictionary size: ', self.n_tokens)

        #getting the number of articles
        self.articles_filenames = get_filenames_from_folder(self.diretorio + '/Outputs/sents_filtered', file_type = 'csv')
        self.n_articles = len(self.articles_filenames)
        print('Número de artigos: ', self.n_articles)

        #carregando o LOG file
        create_new_log_entry = False
        if os.path.exists(self.diretorio + '/Outputs/log/lda_batches_log.json'):
            self.log_batches = load_dic_from_json(self.diretorio + '/Outputs/log/lda_batches_log.json')
            try:
                self.log_batches[self.lda_definitions]['batches']
                print('O LDA_LOG para essa configuração já foi extraído.')
                print('Carregando o arquivo em ' + self.diretorio + '/Outputs/log/lda_batches_log.json')
            
            except KeyError:
                print('Adding new LOG for LDA batches...')
                create_new_log_entry = True
        else:
            print('  Getting LOG for LDA batches...')
            self.log_batches = {}
            create_new_log_entry = True

        if create_new_log_entry is True:

            self.log_batches[self.lda_definitions] = {}
            self.log_batches[self.lda_definitions]['batches'] = {}
            
            #caso o valor de file_batch_size seja incorreto
            if file_batch_size > self.n_articles:
                print('ERRO!')
                print('O valor inserido para o "file_batch_size" é maior que o número total de arquivos em ~/Outputs/sents_filtered')
                return
            
            #caso o processamento não for feito em batch
            elif file_batch_size == 0:
                file_batch_size = self.n_articles
            
            #criando os batches de arquivos
            batch_indexes = get_file_batch_index_list(self.n_articles, file_batch_size)
            
            #varrendo todos os slices
            #número do batch
            batch_counter_number = 0
            cum_sents = 0
            for slice_begin, slice_end in batch_indexes:

                #dicionário para cada batch
                batch_counter_number += 1
                tagged_batch_counter_number = get_tag_name(batch_counter_number, prefix = '')
                self.log_batches[self.lda_definitions]['batches'][tagged_batch_counter_number] = {}
                print('  Processing batch: ', tagged_batch_counter_number, '; indexes: ', slice_begin, slice_end, '; files: ', self.articles_filenames[slice_begin], self.articles_filenames[slice_end])
                
                #contador de sentenças do batch
                count_sents_batch = 0
                for i in range(slice_begin, slice_end + 1):
                    
                    filename = self.articles_filenames[i]
                    DFsents = pd.read_csv(self.diretorio + f'/Outputs/sents_filtered/{filename}.csv', index_col = 0)
                    count_sents_batch += len(DFsents.index)
                    del DFsents
                
                #coletando o doc_index acumulado
                cum_doc_index = slice_begin if self.doc_type == 'articles' else cum_sents
                
                self.log_batches[self.lda_definitions]['batches'][tagged_batch_counter_number]['first_file_index'] = slice_begin
                self.log_batches[self.lda_definitions]['batches'][tagged_batch_counter_number]['last_file_index'] = slice_end
                self.log_batches[self.lda_definitions]['batches'][tagged_batch_counter_number]['n_sents'] = count_sents_batch            
                self.log_batches[self.lda_definitions]['batches'][tagged_batch_counter_number]['n_articles'] = slice_end - slice_begin + 1
                self.log_batches[self.lda_definitions]['batches'][tagged_batch_counter_number][f'cum_doc_index'] = cum_doc_index

                cum_sents += count_sents_batch

            self.log_batches[self.lda_definitions]['total_sents'] = load_dic_from_json(self.diretorio + '/Outputs/log/filtered_sents.json')['sentence_counter']
            self.log_batches[self.lda_definitions]['total_articles'] = self.n_articles
            
            #salvando o log de slices para calcular a matriz TFIDF
            save_dic_to_json(self.diretorio + '/Outputs/log/lda_batches_log.json', self.log_batches)



    def get_initial_doc_token_topic_m(self):


        print('> function get_initial_doc_token_topic_m')

        #varrendo os batches e atribuindo tópicos iniciais na matrix doc_token_topic
        for batch in self.log_batches[self.lda_definitions]['batches'].keys():

            path_doc_token_topic = self.diretorio + f'/Outputs/models/lda_{self.doc_type}_token_topic/doc_{self.doc_type}_token_topic_ntopics_{self.n_topics}_a_{self.alpha}_b_{self.beta}_{batch}'

            #checando se a matriz token_topic_count já foi criada para esse batch
            if not os.path.exists(path_doc_token_topic + '.npz'):
                
                #matriz com atribuição de tópico para cada doc e cada token 
                n_matrix_lines = self.log_batches[self.lda_definitions]['batches'][batch][f'n_{self.doc_type}']
                doc_token_topic_m = np.zeros((n_matrix_lines, self.n_tokens), dtype=int)

                #varrendos os artigos
                row_n = 0
                article_token_list = []
                for filename in self.articles_filenames[ self.log_batches[self.lda_definitions]['batches'][batch]['first_file_index'] : self.log_batches[self.lda_definitions]['batches'][batch]['last_file_index'] + 1 ]:
                    
                    #abrindo o csv com as sentenças do artigo
                    sentDF = pd.read_csv(self.diretorio + '/Outputs/sents_filtered/' + f'{filename}.csv', index_col = 0)

                    #analisando cada sentença
                    for index in sentDF.index:
                        
                        sent = sentDF.loc[index, 'Sentence']
                        sent_tokens = get_tokens_from_sent(sent.lower(), 
                                                           tokens_list_to_filter = self.all_tokens_array, 
                                                           stopwords_list_to_remove = self.stopwords_list, 
                                                           filter_unique = True,
                                                           spacy_tokenizer = self.nlp)
                        
                        if self.doc_type == 'sents':
                            
                            #coletando os indexes de todos os tokens considerando a ordem da IDF.csv
                            n_sent_tokens = len(sent_tokens)
                            #print('\nsent_tokens ', sent_tokens)
                            sent_tokens_indexes = get_tokens_indexes(sent_tokens, all_token_array = self.all_tokens_array)
                            #print('sent_tokens_indexes ', sent_tokens_indexes)
                            #atribuindo um valor de um tópico randômico para cada token de cada doc
                            row_t = np.zeros(self.n_tokens, dtype=int)
                            random_topics = get_initial_random_topics_for_tokens(n_sent_tokens, self.n_topics)
                            row_t[sent_tokens_indexes] = random_topics
                            #print('row_t ', row_t)
                            doc_token_topic_m[ row_n, : ] = row_t
                            #print('doc_token_topic_m[ row_n, : ] ', doc_token_topic_m[ row_n, : ])
                            row_n += 1
                            
                        if self.doc_type == 'articles':                    
                            
                            #coletando todos os tokens que estão no artigo
                            for token in sent_tokens:
                                if token not in article_token_list:
                                    article_token_list.append(token)
                    
                    if self.doc_type == 'articles':            
                        
                        #coletando os indexes de todos os tokens considerando a ordem da IDF.csv
                        n_articles_tokens = len(article_token_list)
                        #print('\nn_articles_tokens ', n_articles_tokens)
                        articles_tokens_indexes = get_tokens_indexes(article_token_list, all_token_array = self.all_tokens_array)
                        #print('articles_tokens_indexes ', articles_tokens_indexes)
                        #atribuindo um valor de um tópico randômico para cada token de cada doc
                        row_t = np.zeros(self.n_tokens, dtype=int)
                        row_t[articles_tokens_indexes] = get_initial_random_topics_for_tokens(n_articles_tokens, self.n_topics)
                        doc_token_topic_m[ row_n, : ] = row_t
                        #print('random_topics', random_topics)
                        #print('articles_tokens_indexes', articles_tokens_indexes)
                        #print('row_t', row_t)
                        #print(f'doc_token_topic_m[ {row_n}, : ]', doc_token_topic_m[ row_n, : ])
                        #time.sleep(1)
                        row_n += 1

                #salvando a matrix doc_token_topic
                save_sparse_csr_matrix(path_doc_token_topic, doc_token_topic_m)
                print(f'  Criada matrix doc_{self.doc_type}_token_topic com dimensão: ', doc_token_topic_m.shape)
                        
                #apagando a matrix
                del doc_token_topic_m



    def count_doc_topics(self):
        
        import time

        print('> count_doc_topics')

        #matriz com contagem de tópico para cada doc
        n_matrix_lines = self.log_batches[self.lda_definitions][f'total_{self.doc_type}']
        doc_topic_counts_m = np.zeros((n_matrix_lines, self.n_topics), dtype=int)

        #varrendo os batches
        line_counter = 0
        for batch in self.log_batches[self.lda_definitions]['batches'].keys():

            #carregando a matrix doc_token_topic
            path_doc_token_topic = self.diretorio + f'/Outputs/models/lda_{self.doc_type}_token_topic/doc_{self.doc_type}_token_topic_ntopics_{self.n_topics}_a_{self.alpha}_b_{self.beta}_{batch}'
            doc_token_topic_m = load_sparse_csr_matrix(path_doc_token_topic)#, matrix_name = 'doc_token_topic_m')

            #varrendo os docs
            for i in range(doc_token_topic_m.shape[0]):
                
                #contando o número de tópicos para cada document
                doc_token_topic_array = np.asarray(doc_token_topic_m[ i , : ]).ravel()

                doc_topic_array_with_topics = doc_token_topic_array[ np.nonzero( doc_token_topic_array )[0] ]
                topics_found, counts = np.unique(doc_topic_array_with_topics, return_counts=True)
                #faz-se essa subtração pois o primeiro tópico (1) tem index 0
                topics_index = topics_found.astype(int) - 1    
                doc_topic_counts_m[ ( [line_counter] * len(topics_index) , topics_index )] = counts
                line_counter += 1
                #time.sleep(1)

        #salvando as matrizes em scipy sparse
        save_dense_matrix(self.path_doc_topic_counts, doc_topic_counts_m)
        #print(f'  Criada matrix doc_{self.doc_type}_topic_counts com dimensão: ', doc_topic_counts_m.shape)
        
        del doc_topic_counts_m



    def count_token_topics(self):

        import time

        print('> count_token_topics')
    
        #matriz com contagem de tópico para cada token
        token_topic_counts_m = np.zeros((self.n_tokens, self.n_topics), dtype=int)

        #varrendo os batches        
        for batch in self.log_batches[self.lda_definitions]['batches'].keys():

            #carregando a matrix doc_token_topic
            path_doc_token_topic = self.diretorio + f'/Outputs/models/lda_{self.doc_type}_token_topic/doc_{self.doc_type}_token_topic_ntopics_{self.n_topics}_a_{self.alpha}_b_{self.beta}_{batch}'
            doc_token_topic_m = load_sparse_csr_matrix(path_doc_token_topic) #, matrix_name = 'doc_token_topic_m')

            #varrendo os tokens
            for j in range(doc_token_topic_m.shape[1]):

                #contando o número de tópicos para cada token
                doc_token_topic_array = np.asarray(doc_token_topic_m[ : , j ]).ravel()
                
                token_topic_array_with_topics = doc_token_topic_array[ np.nonzero( doc_token_topic_array )[0] ]
                topics_found, counts = np.unique(token_topic_array_with_topics, return_counts=True)
                #faz-se essa subtração pois o primeiro tópico (1) tem index 0
                topics_index = topics_found.astype(int) - 1
                token_topic_counts_m[ ([j] * len(topics_index) , topics_index)] += counts
                #time.sleep(1)

        #salvando a matrix token_topic_counts concatenada
        save_dense_matrix(self.path_token_topic_counts, token_topic_counts_m)
        #print(f'  Criada matrix token_{self.doc_type}_topic_counts com dimensão: ', token_topic_counts_m.shape)
            
        del token_topic_counts_m



    def start_lda(self, iterations = 10):
        
        #os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        #os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
        #os.environ.setdefault("OMP_NUM_THREADS", "1")
        #os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        #os.environ.setdefault("MKL_NUM_THREADS", "1")

        #creating temp folders
        #if not os.path.exists(self.path_doc_topic_counts + '_temp'):
        #    os.makedirs(self.path_doc_topic_counts + '_temp')
        
        #if not os.path.exists(self.path_token_topic_counts + '_temp'):
        #    os.makedirs(self.path_token_topic_counts + '_temp')

        self.iter_n = iterations

        #carregando o número da iteração
        #carregando o número da iteração
        n = load_log_info(log_name = f'iter_n_{self.lda_definitions}', logpath = self.diretorio + '/Outputs/log/lda_iter_log.json')
        if n is not None: 
            self.iter_n_done = int(n)
            print('\n> Iteração já realizadas: ', self.iter_n_done)
            
            if self.iter_n_done < self.iter_n:
                self.count_doc_topics()
                self.count_token_topics()
            else: pass

        else:
            #gerando as matrizes iniciais
            self.get_initial_doc_token_topic_m()
            self.count_doc_topics()
            self.count_token_topics()
            self.iter_n_done = 0
            print('\n> Primeira iteração: ', self.iter_n_done)



    def run_lda(self):

        #loading topic matrices in read only
        doc_topic_counts_m = load_dense_matrix(self.path_doc_topic_counts)
        token_topic_counts_m = load_dense_matrix(self.path_token_topic_counts)

        erase_temp_matrix = False
        while self.iter_n_done < self.iter_n:
            erase_temp_matrix = True    

            for i in self.log_batches[self.lda_definitions]['batches'].keys():
                doc_topic_counts_m, token_topic_counts_m = self.gibbs_sampling(i, doc_topic_counts_m, token_topic_counts_m)

            self.update_iter_n()

        #salvando
        save_dense_matrix(self.path_doc_topic_counts, doc_topic_counts_m)
        save_dense_matrix(self.path_token_topic_counts, token_topic_counts_m)

        self.export_token_topics_to_csv()

        if erase_temp_matrix is True:
            self.final_matrices_processing()



    def plots(self, topn_topics = 10, alpha_combination = 0.5, hplot_figsize = 10, hplot_fontsize = 10, hplot_palette = 'flare'):

        print('> Plotting LDA topics...')
        
        #open the list of representative tokens
        tagged_niter = get_tag_name(self.iter_n_done, prefix='')
        #topic_tokens_df = pd.read_csv(self.diretorio + f'/Outputs/models/lda_token_topic_{self.lda_definitions}_niter_{tagged_niter}.csv', index_col = 0)
        
        #loading the token_topics matrices
        token_topic_counts_m = load_dense_matrix(self.path_token_topic_counts)
        #token_topic_counts_m = token_topic_counts_m[ : 2000]

        #loading the doc_topics matrices
        doc_topic_counts_m = load_dense_matrix(self.path_doc_topic_counts)
        
        #plot_scatter_token_topics(token_topic_counts_m, self.lda_definitions, tagged_niter, self.diretorio,)
        plot_hierarchical_topics(token_topic_counts_m, doc_topic_counts_m, tagged_niter, vocab = self.all_tokens_array, topn_topics = topn_topics, 
                                 alpha_combination = alpha_combination, figure_height = hplot_figsize, fontsize = hplot_fontsize,
                                 pallete = hplot_palette, lda_definitions = self.lda_definitions, folder = self.diretorio)




    def update_iter_n(self):

        #salvando o número da iteração
        self.iter_n_done += 1
        print('\n> Iteração número: ', self.iter_n_done, ' realizada.')
        update_log(log_names = [f'iter_n_{self.lda_definitions}'], entries = [self.iter_n_done], logpath = self.diretorio + '/Outputs/log/lda_iter_log.json')



    def export_token_topics_to_csv(self, n_word_to_represent_topic = 15):


        df = pd.DataFrame(index = range(1, self.n_topics + 1), columns = ['tokens'], dtype=object)

        #carregando a matrix token_topic_counts concatenada
        token_topic_counts_m = load_dense_matrix(self.path_token_topic_counts) #, matrix_name='token_topic_counts_m')

        print('> Exportando a relação token_topics em csv...')
        for topic_i in range(self.n_topics):
            
            #print('\n')
            reversed_arg_sorted_counts_array = np.ravel(token_topic_counts_m[ : , topic_i]).argsort()[ :: -1]

            word_count = 0
            tokens_str = ''
            for i in reversed_arg_sorted_counts_array[ : n_word_to_represent_topic]:
                token = self.all_tokens_array[i]
                tokens_str += f'{token}, '
                word_count +=1

            tokens_str = tokens_str[ :-2]
            #soma-se 1 no topic_i pois na matrix doc_token_topic o primeiro tópico é o 1
            df.loc[ topic_i + 1 , 'tokens'] = tokens_str
        
        tagged_niter = get_tag_name(self.iter_n_done, prefix='')
        df.to_csv(self.diretorio + f'/Outputs/models/lda_token_topic_{self.lda_definitions}_niter_{tagged_niter}.csv')



    def final_matrices_processing(self):
        
        print('> Final processing')
        print(f'  Concatenando a matriz {self.doc_type}_topics_full (H5 file)...')
        h5_doc_topic = h5py.File(self.diretorio + f'/Outputs/models/lda_{self.lda_definitions}_full_matrix.h5', 'w')
        h5_doc_topic.create_dataset('data', shape=(self.log_batches[self.lda_definitions][f'total_{self.doc_type}'], self.n_topics), dtype=int)
        doc_topic_counts_m_full = h5_doc_topic['data']

        #carregando as matrizes doc_topic por batch
        doc_topic_counts_m = load_dense_matrix(self.path_doc_topic_counts) #, matrix_name='doc_topic_counts_m')
        doc_topic_counts_m_full[ : ] = doc_topic_counts_m

        h5_doc_topic.close()
        del doc_topic_counts_m_full



    def gibbs_sampling(self, batch, doc_topic_counts_m, token_topic_counts_m):

        print('> processin batch: ', batch, flush = True)
        
        #carregando as matrizes
        path_doc_token_topic = self.diretorio + f'/Outputs/models/lda_{self.doc_type}_token_topic/doc_{self.doc_type}_token_topic_ntopics_{self.n_topics}_a_{self.alpha}_b_{self.beta}_{batch}'
        doc_token_topic_m = load_sparse_csr_matrix(path_doc_token_topic)

        #start = time.time()
        doc_token_topic_m, doc_topic_counts_m, token_topic_counts_m = operate_matrices(doc_token_topic_m, 
                                                                                    doc_topic_counts_m, 
                                                                                    token_topic_counts_m,
                                                                                    cum_doc_index = self.log_batches[self.lda_definitions]['batches'][batch]['cum_doc_index'],
                                                                                    n_topics = self.n_topics, 
                                                                                    n_tokens = self.n_tokens, 
                                                                                    alpha = self.alpha, 
                                                                                    beta = self.beta)
        
        #end = time.time()
        #print('> batch runtime: ', end-start, '\n')

        #salvando a matrix doc_token_topic
        save_sparse_csr_matrix(path_doc_token_topic, doc_token_topic_m)

        #check_memory()
        del doc_token_topic_m

        return doc_topic_counts_m, token_topic_counts_m



class lsa(object):
    
    def __init__(self, diretorio = None):

        print('\n( Class: LSA classifier )')

        self.diretorio = diretorio
        
        if not os.path.exists(self.diretorio + '/Outputs/models'):
            print('Creating folder ', self.diretorio + '/Outputs/models')
            os.makedirs(self.diretorio + '/Outputs/models')
        
        #abrindo as matrizes doc_topics
        if not os.path.exists(self.diretorio + f'/Outputs/wv/wv_sents_svd_truncated_svd.csv'):
            print('ERRO!')
            print('Matrix lsa_token_sent_topic não encontrada!')
            print('Abortando a classe...')
            return
        
        else:
            self.lsa_token_sent_topic_m = pd.read_csv(self.diretorio + f'/Outputs/wv/wv_sents_svd_truncated_svd.csv', index_col = 0)

        if not os.path.exists(self.diretorio + f'/Outputs/wv/wv_articles_svd_truncated_svd.csv'):
            print('ERRO!')
            print('Matrix lsa_token_article_topic não encontrada!')
            print('Abortando a classe...')
            return
        
        else:
            self.lsa_token_article_topic_m = pd.read_csv(self.diretorio + f'/Outputs/wv/wv_articles_svd_truncated_svd.csv', index_col = 0)



    def export_token_topics_to_csv(self, n_word_to_represent_topic = 15):

        print('> Exportando a relação lsa_token_topics em csv...')

        n_topics = self.lsa_token_sent_topic_m.shape[1]

        lsa_token_sent_topic_pos = pd.DataFrame(index = range(1, n_topics + 1), columns = ['tokens'], dtype=object)
        lsa_token_article_topic_pos = pd.DataFrame(index = range(1, n_topics + 1), columns = ['tokens'], dtype=object)
        lsa_token_sent_topic_neg = pd.DataFrame(index = range(1, n_topics + 1), columns = ['tokens'], dtype=object)
        lsa_token_article_topic_neg = pd.DataFrame(index = range(1, n_topics + 1), columns = ['tokens'], dtype=object)


        #eliminando os valores negativos das matrizes (para encontrar os tokens com influcencia positivas nos tópicos)
        self.lsa_token_sent_topic_m_copy = self.lsa_token_sent_topic_m.values.copy()
        self.lsa_token_article_topic_m_copy = self.lsa_token_article_topic_m.values.copy()
        
        self.lsa_token_sent_topic_m_copy[ self.lsa_token_sent_topic_m_copy < 0 ] = 0
        self.lsa_token_article_topic_m_copy[ self.lsa_token_article_topic_m_copy < 0 ] = 0

        for topic_i in range(n_topics):
            
            #arg sorting para os maiores valores na matrix sent_topic
            reversed_arg_sorted_array_sent = self.lsa_token_sent_topic_m_copy[ :, topic_i].argsort()[ :: -1]

            word_count = 0
            tokens_str = ''
            for i in reversed_arg_sorted_array_sent[ : n_word_to_represent_topic]:
                token = self.lsa_token_sent_topic_m.index[i]
                tokens_str += f'{token}, '
                word_count +=1

            tokens_str = tokens_str[ :-2]
            #soma-se 1 no topic_i pois na matrix doc_token_topic o primeiro tópico é o 1
            lsa_token_sent_topic_pos.loc[ topic_i + 1 , 'tokens'] = tokens_str


            reversed_sorted_array_article = self.lsa_token_article_topic_m_copy[ :, topic_i].argsort()[ :: -1]

            word_count = 0
            tokens_str = ''
            for i in reversed_sorted_array_article[ : n_word_to_represent_topic]:
                token = self.lsa_token_article_topic_m.index[i]
                tokens_str += f'{token}, '
                word_count +=1

            tokens_str = tokens_str[ :-2]
            #soma-se 1 no topic_i pois na matrix doc_token_topic o primeiro tópico é o 1
            lsa_token_article_topic_pos.loc[ topic_i + 1 , 'tokens'] = tokens_str


        #eliminando os valores positivos das matrizes (para encontrar os tokens com influcencia negativa nos tópicos)
        self.lsa_token_sent_topic_m_copy = self.lsa_token_sent_topic_m.values.copy()
        self.lsa_token_article_topic_m_copy = self.lsa_token_article_topic_m.values.copy()

        self.lsa_token_sent_topic_m_copy[ self.lsa_token_sent_topic_m_copy > 0 ] = 0
        self.lsa_token_article_topic_m_copy[ self.lsa_token_article_topic_m_copy > 0 ] = 0
        
        #convertendo os valores negativos para positivos
        self.lsa_token_sent_topic_m_copy = self.lsa_token_sent_topic_m_copy * -1
        self.lsa_token_article_topic_m_copy = self.lsa_token_article_topic_m_copy * 1

        for topic_i in range(n_topics):
            
            #arg sorting para os maiores valores na matrix sent_topic
            reversed_arg_sorted_array_sent = self.lsa_token_sent_topic_m_copy[ :, topic_i].argsort()[ :: -1]

            word_count = 0
            tokens_str = ''
            for i in reversed_arg_sorted_array_sent[ : n_word_to_represent_topic]:
                token = self.lsa_token_sent_topic_m.index[i]
                tokens_str += f'{token}, '
                word_count +=1

            tokens_str = tokens_str[ :-2]
            #soma-se 1 no topic_i pois na matrix doc_token_topic o primeiro tópico é o 1
            lsa_token_sent_topic_neg.loc[ topic_i + 1 , 'tokens'] = tokens_str


            reversed_sorted_array_article = self.lsa_token_article_topic_m_copy[ :, topic_i].argsort()[ :: -1]

            word_count = 0
            tokens_str = ''
            for i in reversed_sorted_array_article[ : n_word_to_represent_topic]:
                token = self.lsa_token_article_topic_m.index[i]
                tokens_str += f'{token}, '
                word_count +=1

            tokens_str = tokens_str[ :-2]
            #soma-se 1 no topic_i pois na matrix doc_token_topic o primeiro tópico é o 1
            lsa_token_article_topic_neg.loc[ topic_i + 1 , 'tokens'] = tokens_str

        
        lsa_token_sent_topic_pos.to_csv(self.diretorio + f'/Outputs/models/lsa_token_topic_sent_ntopics_{n_topics}_pos.csv')
        lsa_token_article_topic_pos.to_csv(self.diretorio + f'/Outputs/models/lsa_token_topic_articles_ntopics_{n_topics}_pos.csv')
        lsa_token_sent_topic_neg.to_csv(self.diretorio + f'/Outputs/models/lsa_token_topic_sent_ntopics_{n_topics}_neg.csv')
        lsa_token_article_topic_neg.to_csv(self.diretorio + f'/Outputs/models/lsa_token_topic_articles_ntopics_{n_topics}_neg.csv')



    def plots(self, model = 'svd', mode = 'truncated_svd'):

        print('> Plotting LSA topics...')
        
        #carregando o df dos IDF para pegar os tokens
        IDF = pd.read_csv(self.diretorio + f'/Outputs/tfidf/idf.csv', index_col = 0)
        self.all_tokens_array = IDF.index.values
        self.n_tokens = len(self.all_tokens_array)
        print('Tokens dictionary size: ', self.n_tokens)

        #loading the token_topics matrices for articles
        if os.path.exists(self.diretorio + f'/Outputs/wv/wv_articles_{model}_{mode}.csv'):
            lsa_token_topic_m_articles = pd.read_csv(self.diretorio + f'/Outputs/wv/wv_articles_{model}_{mode}.csv', index_col = 0)
            lsa_token_topic_m_articles = lsa_token_topic_m_articles.values
        else:
            print('Error! LSA token topic matrix for articles not found at: ', self.diretorio + f'/Outputs/wv/wv_articles_{model}_{mode}.csv')
        
        tsne = TSNE(n_components=2, perplexity=100, random_state=42)
        tokens_topic_tsne = tsne.fit_transform(lsa_token_topic_m_articles)
        
        #plotting
        fig = plt.figure(figsize=(8,8))
        plot_grid = gridspec.GridSpec(15, 15, figure = fig)
        plot_grid.update(wspace=0.1, hspace=0.1, left = 0.1, right = 0.9, top = 0.9, bottom = 0.1)

        #coloring topic tokens
        n_colors = lsa_token_topic_m_articles.shape[1]
        palette = sns.color_palette("Spectral", n_colors)

        ax1 = fig.add_subplot(plot_grid[ : , : ])
        for i, xy in enumerate(tokens_topic_tsne):
            ax1.scatter(xy[0], xy[1], s = 30, c=np.array([palette[ np.argmax(lsa_token_topic_m_articles[i]) ]]), alpha=0.5)

        ax1.spines['top'].set_linewidth(1)
        ax1.spines['bottom'].set_linewidth(1)
        ax1.spines['left'].set_linewidth(1)
        ax1.spines['right'].set_linewidth(1)

        ax1.xaxis.set_tick_params(labelbottom=False)
        ax1.yaxis.set_tick_params(labelleft=False)
        ax1.set_xticks([])
        ax1.set_yticks([])

        print('Saving LSA topic (articles) plot to: ', self.diretorio + f'/Outputs/models/lsa_topics_articles_{model}_{mode}.png')
        plt.savefig(self.diretorio + f'/Outputs/models/lsa_topics_articles_{model}_{mode}.png', dpi=200)
        #plt.show()

        #loading the token_topics matrices for sents
        if os.path.exists(self.diretorio + f'/Outputs/wv/wv_sents_{model}_{mode}.csv'):
            lsa_token_topic_m_sents = pd.read_csv(self.diretorio + f'/Outputs/wv/wv_sents_{model}_{mode}.csv', index_col = 0)
            lsa_token_topic_m_sents = lsa_token_topic_m_sents.values
        else:
            print('Error! LSA token topic matrix for sents not found at: ', self.diretorio + f'/Outputs/wv/wv_sents_{model}_{mode}.csv')

        tsne = TSNE(n_components=2, perplexity=100, random_state=42)
        tokens_topic_tsne = tsne.fit_transform(lsa_token_topic_m_sents)
        
        #plotting
        fig = plt.figure(figsize=(8,8))
        plot_grid = gridspec.GridSpec(15, 15, figure = fig)
        plot_grid.update(wspace=0.1, hspace=0.1, left = 0.1, right = 0.9, top = 0.9, bottom = 0.1)

        #coloring topic tokens
        n_colors = lsa_token_topic_m_sents.shape[1]
        palette = sns.color_palette("Spectral", n_colors)

        ax1 = fig.add_subplot(plot_grid[ : , : ])
        for i, xy in enumerate(tokens_topic_tsne):
            ax1.scatter(xy[0], xy[1], s = 30, c=np.array([palette[ np.argmax(lsa_token_topic_m_sents[i]) ]]), alpha=0.5)

        ax1.spines['top'].set_linewidth(1)
        ax1.spines['bottom'].set_linewidth(1)
        ax1.spines['left'].set_linewidth(1)
        ax1.spines['right'].set_linewidth(1)

        ax1.xaxis.set_tick_params(labelbottom=False)
        ax1.yaxis.set_tick_params(labelleft=False)
        ax1.set_xticks([])
        ax1.set_yticks([])

        print('Saving LSA topic (sentsarticles) plot to: ', self.diretorio + f'/Outputs/models/lsa_topics_sents_{model}_{mode}.png')
        plt.savefig(self.diretorio + f'/Outputs/models/lsa_topics_sents_{model}_{mode}.png', dpi=200)
        #plt.show()



def get_initial_random_topics_for_tokens(n_tokens, n_topics):

    #np.random.seed(42)
    return np.random.randint(1, n_topics + 1, size = n_tokens)



@numba.jit(nopython=True, fastmath=True)
def operate_matrices(doc_token_topic_m, doc_topic_counts_m, token_topic_counts_m,
                     cum_doc_index = 0, n_topics = None, n_tokens = None, alpha = None, beta = None):    

    #varrendo todos os elementos nonzero da matriz doc_token
    for i, j in zip(*np.nonzero(doc_token_topic_m)):

        #o index do tópico é o tópico - 1
        sampling_topic_index = doc_token_topic_m[ i , j ] - 1
        #print('> varendo doc_token_topic_m em ', i, j)
        doc_topic_counts_m[ i + cum_doc_index , sampling_topic_index ] -= 1
        #print('  varendo doc_topic_counts_m em ', i + cum_doc_index , sampling_topic_index)
        token_topic_counts_m[ j , sampling_topic_index ] -= 1
        #print('  varendo token_topic_counts_m em ', j , sampling_topic_index)

        #CDT é o vetor contendo o counts de tópico para o doc "i"
        row_doc = doc_topic_counts_m[ i + cum_doc_index , : ]
        doc_topic_counts_array = np.asarray(row_doc).ravel()
        
        cDT = doc_topic_counts_array + alpha
        denominator_pDT = doc_topic_counts_array.sum() + (n_topics * alpha)
        #print(cDT.shape, denominator_pDT)
        pDT = cDT / denominator_pDT

        #CTT é o vetor contendo o counts de tópico para o token "j"
        row_tok = token_topic_counts_m[ j , : ]
        token_topic_counts_array = np.asarray(row_tok).ravel()
        
        cTT = token_topic_counts_array + beta
        sum_tok = token_topic_counts_m.sum(axis=0)
        token_topic_counts_sum = np.asarray(sum_tok).ravel()
        denominator_pTT = token_topic_counts_sum + (n_tokens * beta)
        #print(cTT.shape, denominator_pTT.shape)
        pTT = cTT / denominator_pTT
        
        #determinando p_t
        p_t = pDT * pTT

        #normalizando a distribuição
        p_t /= p_t.sum()
        #obtendo o novo topic index a ser atribuido
        new_t_index = np.random.multinomial(1, p_t).argmax()
        
        #soma-se 1 pois o o topic_index começa em 0 e o primeiro tópico é 1
        doc_token_topic_m[ i , j ] = new_t_index + 1

        #atualizando a contagem do novo tópico para o doc "i", token "j" 
        doc_topic_counts_m[ i + cum_doc_index , new_t_index ] += 1
        token_topic_counts_m[ j , new_t_index ] += 1
        #print(f'doc_topic_counts_m[ {i} , : ]', doc_topic_counts_m[ i , : ])
        #print(f'token_topic_counts_m[ {j} , : ]', token_topic_counts_m[ j , : ])
        #time.sleep(1)
    
    return doc_token_topic_m, doc_topic_counts_m, token_topic_counts_m



def check_memory():

    # Get the memory usage in bytes
    memory_info = psutil.virtual_memory()

    # Print memory details
    print(f"Total Memory: {memory_info.total / (1024 ** 3):.2f} GB")
    print(f"Available Memory: {memory_info.available / (1024 ** 3):.2f} GB")
    print(f"Used Memory: {memory_info.used / (1024 ** 3):.2f} GB")
    print(f"Percentage Used: {memory_info.percent}%")



def plot_scatter_token_topics(token_topic_counts_matrix, lda_definitions, niter, folder):
        
        tsne = TSNE(n_components=2, perplexity=100, random_state=42)
        tokens_topic_tsne = tsne.fit_transform(token_topic_counts_matrix)

        #plotting
        fig = plt.figure(figsize=(8,8))
        plot_grid = gridspec.GridSpec(15, 15, figure = fig)
        plot_grid.update(wspace=0.1, hspace=0.1, left = 0.1, right = 0.9, top = 0.9, bottom = 0.1)

        #coloring topic tokens
        n_colors = token_topic_counts_matrix.shape[1]
        palette = sns.color_palette("Spectral", n_colors)

        ax1 = fig.add_subplot(plot_grid[ : , : ])
        for i, xy in enumerate(tokens_topic_tsne):
            ax1.scatter(xy[0], xy[1], s = 30, c=np.array([palette[ np.argmax(token_topic_counts_matrix[i]) ]]), alpha=0.5)

        ax1.spines['top'].set_linewidth(1)
        ax1.spines['bottom'].set_linewidth(1)
        ax1.spines['left'].set_linewidth(1)
        ax1.spines['right'].set_linewidth(1)

        ax1.xaxis.set_tick_params(labelbottom=False)
        ax1.yaxis.set_tick_params(labelleft=False)
        ax1.set_xticks([])
        ax1.set_yticks([])

        print('Saving LDA topic plot to: ', folder + f'/Outputs/models/lda_topics_splot_{lda_definitions}_niter_{niter}.png')
        plt.savefig(folder + f'/Outputs/models/lda_topics_splot_{lda_definitions}.png', dpi=200)
        #plt.show()'''



def plot_hierarchical_topics(token_topic_counts_matrix, doc_topic_counts_matrix, niter, vocab = None, topn_topics = 10,
                             alpha_combination = 0.5, lda_definitions = None, figure_height = 10, pallete = 'viridis',
                             fontsize = 10, folder = ''):


    if pallete.lower() == 'hot':
        cmap = cm.hot
    elif pallete.lower() == 'magma':
        cmap = cm.magma
    elif pallete.lower() == 'ocean':
        cmap = cm.ocean
    elif pallete.lower() == 'plasma':
        cmap = cm.plasma
    elif pallete.lower() == 'viridis':
        cmap = cm.viridis
    else:
        print('ERRO! Inserir uma palette válida:')
        print('Palettes: "hot", "magma", "ocean", "plasma", "viridis"')

    #matrix token_topic    
    #1. substituindo os zeros na matrix
    #2. convertendo para matrix token_topic_counts (V, K) > topic_token_counts (K, V)
    #3. normalizando as linhas da matrix topic_token_counts (K, V)
    token_topic_counts_matrix = np.maximum(token_topic_counts_matrix, 1e-12)
    topic_token_counts_matrix = token_topic_counts_matrix.T / token_topic_counts_matrix.T.sum(axis=1, keepdims = True)

    #listando os top n tokens do topico
    K, V = topic_token_counts_matrix.shape
    terms = []
    for k in range(K):
        idx = np.argsort(topic_token_counts_matrix[k])[::-1][:topn_topics]
        terms.append([vocab[j] for j in idx])
    
    topic_labels = [f"T{1 + idx:02d}: " + ", ".join(ts[:topn_topics])
                    for idx, ts in enumerate(terms)]

    #encontrando as distâncias entre os vetores da matrix topic_token_counts
    def jsd(u, v):
        m = 0.5 * (u + v)
        kl = lambda a, b: np.sum(a * (np.log(a) - np.log(b)))
        return np.sqrt(0.5 * kl(u, m) + 0.5 * kl(v, m))

    #calculate dist vector with dimension K*(K-1)/2 based on tokens
    topic_token_dist =  pdist(topic_token_counts_matrix, metric=lambda u, v: jsd(u, v))

    #matrix doc_topic
    #1. substituindo os zeros na matrix
    #2. convertendo para matrix doc_topic_counts (D, K) > topic_doc_counts (K, D)
    #3. normalizando as linhas da matrix topic_doc_counts (K, D)
    doc_topic_counts_matrix = np.maximum(doc_topic_counts_matrix, 1e-12)
    topic_doc_dist = pdist(doc_topic_counts_matrix.T / doc_topic_counts_matrix.T.sum(axis=1, keepdims=True), metric='correlation')
    
    # Combine (convex combination of distances)
    dist_comb = ( (1 - alpha_combination) * topic_doc_dist ) + ( alpha_combination * topic_token_dist )

    #linkage
    Z = linkage(dist_comb, method='average')

    #color function
    K = Z.shape[0] + 1
    # Map internal node id (K..2K-2) -> merge height
    height_by_id = {K+i: Z[i, 2] for i in range(Z.shape[0])}
    hmin = min(height_by_id.values())
    hmax = max(height_by_id.values())

    def link_color_func(node_id: int):
        if node_id < K:
            return "#444444"
        h = height_by_id[node_id]
        t = (h - hmin) / (hmax - hmin + 1e-12)  # normalize to [0,1]
        return to_hex(cmap(t))

    # Dendrogram (quick visualization)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 16))
    
    dendrogram(Z, labels=topic_labels, orientation = 'right', link_color_func=link_color_func, ax=ax2)
    
    # tick labels (numbers along the axis)
    ax2.tick_params(axis='y', labelsize=fontsize)
    ax2.tick_params(axis='x', labelsize=fontsize)

    # axis label text
    ax2.set_ylabel("Topics", fontsize=fontsize)
    ax2.set_xlabel("Distance", fontsize=fontsize)

    #bubble plot
    mds = MDS(n_components=2, dissimilarity="euclidean", random_state=0, n_init=4, max_iter=600)
    XY = mds.fit_transform(np.sqrt(topic_token_counts_matrix))
    #XY = mds.fit_transform(squareform(dist_comb))

    #sizes
    # Topic mass (importance): total tokens across docs (or normalize to %) token_docs
    mass = doc_topic_counts_matrix.T.sum(axis=1)
    mass = (mass - mass.min()) / (mass.max() - mass.min())
    sizes = 200 + 500 * mass                        # bubble areas

    #color
    #cl = fcluster(Z, t=5, criterion="maxclust")

    ax1.scatter(XY[:,0], XY[:,1], s=sizes, c='gray', cmap="tab10", alpha=0.85, edgecolors="k", linewidths=0.5)

    for (x,y), lab in zip(XY, topic_labels):
        ax1.text(x, y, lab[ : 3], ha="center", va="center", fontsize=8, color="white", weight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.4))

    ax1.xaxis.set_tick_params(labelbottom=False)
    ax1.yaxis.set_tick_params(labelleft=False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_linewidth(False)
    ax1.spines['right'].set_visible(False)

    fig.tight_layout()
    plt.savefig(folder + f'/Outputs/models/lda_topics_hplot_{lda_definitions}_niter_{niter}.png', dpi=200)
    #plt.show()        
