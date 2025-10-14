#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import os
import pandas as pd # type: ignore
import numpy as np # type: ignore
#np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
import regex as re # type: ignore
import random
import math
import itertools as itt

#from sklearn.cluster import KMeans
#from sklearn.metrics.pairwise import pairwise_distances_argmin

import matplotlib.pyplot as plt # type: ignore
import matplotlib.gridspec as gridspec # type: ignore
import matplotlib.patches as patches

import seaborn as sns # type: ignore
import statsmodels.stats.multicomp as mc # type: ignore
from scipy.stats import gaussian_kde # type: ignore

from FUNCTIONS import get_filenames_from_folder
from FUNCTIONS import save_dic_to_json
from FUNCTIONS import load_dic_from_json
from FUNCTIONS import get_tag_name

from functions_STATS import diff_of_welch_t_means, permutation_test_welch_anova, permutation_test_AB, p_holm_bh_correction, trim_array

#from sklearn.linear_model import LinearRegression
#from sklearn.decomposition import PCA

#from bokeh.layouts import gridplot, row
#from bokeh.plotting import figure, show
#from bokeh.palettes import Category20, turbo, plasma
#from bokeh.models import Range1d, Label, LinearColorMapper, ColorBar
#from bokeh.models import BasicTicker, ColorBar, LinearColorMapper, ColumnDataSource
#from bokeh.models import Arrow, VeeHead, Text
#from bokeh.io import export_png

#import networkx as nx
#from networkx.algorithms import community

#import holoviews as hv
#from holoviews import opts, dim


class results(object):
    
    def __init__(self, DF_input_name = 'None', diretorio = None, color_pat = "flare"):
                
        print('\n( Class: results )')
        print('> opening consolidated DF...')

        self.diretorio = diretorio

        #checando os diretorios
        if not os.path.exists(self.diretorio + '/Outputs/analyses'):
            os.makedirs(self.diretorio + '/Outputs/analyses')

        #importando a DF consolidada        
        self.results_DF = pd.read_csv(self.diretorio + f'/Outputs/dataframes/{DF_input_name}.csv', index_col=[0,1])
        
        #copiando a DF original
        self.results_DF_copy = self.results_DF.copy()
        
        print('> features presentes na DF:')
        for column in self.results_DF_copy:
            print(f'  {column}')


        #checando o dicionário com as index_lists dos grupos
        if not os.path.exists(self.diretorio + '/Outputs/log/index_lists.json'):
            self.index_lists = dict()
        else:
            self.index_lists = load_dic_from_json(self.diretorio + '/Outputs/log/index_lists.json')

        #econtrando o ultimo numerador dos arquivos de figuras
        filenames = get_filenames_from_folder(folder = self.diretorio + '/Outputs/analyses', file_type='png')
        if filenames is not None and len(filenames) > 0:
            filenames.sort()
            self.last_fig_counter = int(filenames[-1])
        else:
            self.last_fig_counter = 0

        #abrindo dic com os setups das figuras
        if not os.path.exists(self.diretorio + '/Outputs/analyses/dic_plotsetup.json'):
            self.plotsetup = dict()
        else:
            self.plotsetup = load_dic_from_json(self.diretorio + '/Outputs/analyses/dic_plotsetup.json')

        #abrindo dic com os outputs dos stats_results
        if not os.path.exists(self.diretorio + '/Outputs/analyses/dic_stats_results.json'):
            self.stats_results = dict()
        else:
            self.stats_results = load_dic_from_json(self.diretorio + '/Outputs/analyses/dic_stats_results.json')

        #abrindo dic com os outputs dos cats_len
        if not os.path.exists(self.diretorio + '/Outputs/analyses/dic_cat_len_results.json'):
            self.cat_len_results = dict()
        else:
            self.cat_len_results = load_dic_from_json(self.diretorio + '/Outputs/analyses/dic_cat_len_results.json')

        #escolhendo uma paleta de cores
        self.color_palette = color_pat
        if color_pat == 'flare':
            self.contrast_text_color_to_plot = 'black'
        elif color_pat == 'viridis':
            self.contrast_text_color_to_plot = 'white'



    def merge_str_columns(self, new_column_name: str = '', columns: list = [], sep: str = ' '):
        
        self.results_DF_copy[new_column_name] = ''
        for column in columns:
            self.results_DF_copy[new_column_name] = self.results_DF_copy[new_column_name] + f' {sep} ' + self.results_DF_copy[column]



    def remove_terms_in_DF_with_json(self, columns = []):

        print('> removing terms from cats in ~/Inputs/ngrams_to_remove.json...')
        
        #abrindo o dic
        dic_to_remove = load_dic_from_json(self.diretorio + '/Inputs/ngrams_to_remove.json')
        
        #removendo termos presentes no dic ngrams_to_remove
        for column in columns:
            if (column in self.results_DF_copy.columns) and (column in dic_to_remove.keys()):
                
                #contando quantas aparições o termo term
                unique, counts = np.unique(self.results_DF_copy[column].values.astype(str), return_counts=True)
                dic_counter = dict(zip(unique, counts))
                
                for term in dic_to_remove[column]:
                    try:
                        #replacing
                        self.results_DF_copy = self.results_DF_copy.reset_index().set_index(column).drop(index = term).reset_index().set_index(['Filename', 'Counter'])
                        print(f'  removing term {term} in cat: {column} ({dic_counter[term]} removals)')
                    except KeyError:
                        continue
            
            else:
                print(f'  Erro! cat: {column} não encontrada ou no DF ou no arquivo /Inputs/ngrams_to_remove.json')



    def change_terms_in_DF_with_json(self, column_json_key : dict = {'None':'None'}):

        print('> replacing terms from cats in ~/Inputs/ngrams_to_replace.json...')

        #abrindo o dic
        dic_to_replace = load_dic_from_json(self.diretorio + '/Inputs/ngrams_to_replace.json')

        for column, key in column_json_key.items():
            if (key in dic_to_replace.keys()) and (column in self.results_DF_copy.columns):
                print(f'  replacing {column} -> {key}')
                self.results_DF_copy.replace({f'{column}': dic_to_replace[key]}, inplace = True)
                
            else:
                print(f'  Erro! cat {column} ou val {key} não encontrado ou no DF consolidado.')



    def replace_terms_in_DF(self, parameters_terms : dict = {'column' : {'none': 'none'} }):

        for param in parameters_terms.keys():

            self.results_DF_copy[param] = self.results_DF_copy[param].replace( parameters_terms[param] )



    def filter_cat_term(self, term = 'None', cat = 'None'):
        
        print('> function: set_DF_by_category')
        grouped = self.results_DF_copy.copy().groupby(cat)
        #após o filtro, usar a instancia "self.results_DF_processed"
        self.results_DF_processed = grouped.get_group(term)
        print(f'  filtering term {term} in cat {cat}')



    def extract_vals_from_df_str(self, indexes = None, num_col = None, mode = 'mean', interval_to_filter = None,
                                axis_multiplier = None, remove_extreme_vals = None):

        nums = np.array([], dtype = object)
        for index in indexes:

            text_to_extract = str(self.results_DF_processed.loc[ index , num_col ])

            str_vals = re.findall(r'[\-e0-9\.]+', text_to_extract)
            vals = [ float(str_val) for str_val in str_vals ]
            vals = np.array(vals, dtype=float)

            if mode.lower() == 'min':
                val = vals.min()
            elif mode.lower() == 'max':
                val = vals.max()
            elif mode.lower() == 'mean':
                val = vals.mean()
            elif mode.lower() == 'median':
                val = np.median(vals)

            if axis_multiplier is not None and type(axis_multiplier) in (int, float):
                val *= axis_multiplier

            if interval_to_filter is not None and type(interval_to_filter) == list and len(interval_to_filter) == 2:
                if float(interval_to_filter[0]) <= val <= float(interval_to_filter[1]):
                    nums = np.hstack( ( nums, val) )
                else:
                    nums = np.hstack( ( nums, np.nan) )
            else:
                nums = np.hstack( ( nums, val) )

        if remove_extreme_vals is not None and type(remove_extreme_vals) == float:            
            nums = trim_array(nums, remove_extreme_vals)
        
        return nums



    def export_grouped_cats_num(self, cat_col_to_group='', num_cols_range_to_cluster={'': [], '': []},
                                cats_terms_to_filter = { 'col1' : ['term1', 'term2'] }, terms_to_remove = ['term1', 'term2'], terms_to_replace = {}):
        
        #format num_cols_range_to_cluster = {'num_col': [multiplier, stat_function_to_get_vals, [min_val, max_val], remove_extreme_vals, min_num_vals_occur]]} 
        
        print(f'\n> function: export_grouped_cats_num: {cat_col_to_group} - {str(num_cols_range_to_cluster.keys())} - {str(cats_terms_to_filter)}')
        
        #copiando a DF para processar
        columns_checked_in_DF = check_columns_in_DF( list(cats_terms_to_filter.keys()) , self.results_DF_copy )
        if columns_checked_in_DF is True:
            self.results_DF_processed = self.results_DF_copy[ [ cat_col_to_group ] + list(num_cols_range_to_cluster.keys()) + list(cats_terms_to_filter.keys()) ].copy()
        else:
            self.results_DF_processed = self.results_DF_copy[ [ cat_col_to_group ] + list(num_cols_range_to_cluster.keys()) ].copy()

        #removendo termos presentes na lista terms
        for term in terms_to_remove:
            if term in self.results_DF_processed[cat_col_to_group].values:
                self.results_DF_processed = self.results_DF_processed.reset_index().set_index(cat_col_to_group).drop(index = term).reset_index().set_index(['Filename', 'Counter'])

        #filtrando cats em colunas (quando filtra se determina um plot_title para mostrar o que foi filtrado)
        self.results_DF_processed, _ = filter_categories(self.results_DF_processed, 
                                                         columns_checked_in_DF, 
                                                         cats_terms_to_filter)
        
        #extraindo os valores numéricos para cada grupo
        stats_col_cat_val = {}
        
        for num_col in num_cols_range_to_cluster.keys():
            
            if num_col in self.results_DF_processed.columns:
                
                #Eliminando os NaN
                results_DF_processed_ = self.results_DF_processed[ [cat_col_to_group, num_col] ].copy()
                results_DF_processed_ = remove_nan(results_DF_processed_)
            
                #removendo as duplicatas
                results_DF_processed_ = remove_duplicates(results_DF_processed_, columns_to_consider = [cat_col_to_group, num_col])

                #agrupando os termos
                groups_index_dic = results_DF_processed_.groupby(by=cat_col_to_group).groups
                
                stats_col_cat_val[num_col + ' ' + num_cols_range_to_cluster[num_col][1] + ' x ' + str(num_cols_range_to_cluster[num_col][0])] = {}
                
                for cat in groups_index_dic.keys():
                    vals = self.extract_vals_from_df_str(indexes = groups_index_dic[ cat ], 
                                                    num_col = num_col,
                                                    axis_multiplier = num_cols_range_to_cluster[num_col][0],
                                                    mode = num_cols_range_to_cluster[num_col][1],
                                                    interval_to_filter = num_cols_range_to_cluster[num_col][2],
                                                    remove_extreme_vals = num_cols_range_to_cluster[num_col][3])
                    
                    vals = vals[~np.isnan(vals.astype(float))]
                    if len(vals) >= num_cols_range_to_cluster[num_col][4]:

                        #Q1 = np.percentile(vals, 25)
                        #Q3 = np.percentile(vals, 75)
                        #IQR = Q3 - Q1
                        #lower_bound = Q1 - 1.5 * IQR
                        #upper_bound = Q3 + 1.5 * IQR
                        filtered_vals = vals #[(vals >= lower_bound) & (vals <= upper_bound)]
                        stats_col_cat_val[num_col + ' ' + num_cols_range_to_cluster[num_col][1] + ' x ' + str(num_cols_range_to_cluster[num_col][0])][cat] = str(np.round(np.mean(filtered_vals), 2)) + \
                                                                                                                                                            f' ({np.round(np.std(filtered_vals), 2)})' + \
                                                                                                                                                            f' [{np.round(np.median(filtered_vals), 2)}]'

        df_to_export = pd.DataFrame.from_dict(stats_col_cat_val, orient='index').T
        df_to_export = df_to_export.reset_index().sort_values(by=['index']).set_index('index')
        
        #windows does not allow character ':' from python dic
        cats_terms_to_filter_ = re.sub(r':', '-', str(cats_terms_to_filter))

        print('  exporting dataframe with values to ', f'/Outputs/analyses/df_mean_vals_{cat_col_to_group}_filter_{cats_terms_to_filter_}.csv')
        df_to_export.to_csv(self.diretorio + f'/Outputs/analyses/df_mean_vals_{cat_col_to_group}_filter_{cats_terms_to_filter_}.csv')
        print('  exporting latex table with values to ', f'/Outputs/analyses/table_mean_vals_{cat_col_to_group}_filter_{cats_terms_to_filter_}.txt')
        self.export_df_to_latex_table(df_to_export, self.diretorio + f'/Outputs/analyses/df_mean_vals_{cat_col_to_group}_filter_{cats_terms_to_filter_}.txt',
                                      terms_to_replace = terms_to_replace)



    def export_df_to_latex_table(self, df, path, terms_to_replace = {}):

        df = df.copy().reset_index()

        with open(path, 'w') as file:
            c_i = 0
            for i in range(len(df.index)):
                c_j = 0
                for j in range(len(df.columns)):

                    if type(df.iloc[ i , j ]) == float:
                        file.write( '' )
                    
                    elif re.match(r'[$\s\+]*[0-9]*[$A-Za-z\-\+]+[$\s\+]*?', df.iloc[ i , j ]):
                        try:
                            r_term = terms_to_replace[df.iloc[ i , j ]]
                        except KeyError:
                            r_term = df.iloc[ i , j ]
                        
                        file.write( r_term )

                    else:                        
                        stat = re.findall(r'(?<!\()\-?[0-9]+\.?[0-9]*(?=\s)', df.iloc[ i , j])[0]
                        std = re.findall(r'(?<=\()\-?[0-9]+\.?[0-9]*(?=\))', df.iloc[ i , j])[0]
                        median = re.findall(r'(?<=\[)\-?[0-9]+\.?[0-9]*(?=\])', df.iloc[ i , j])[0]
                        file.write( '\makecell{' + f'${stat}$' + '\\\\' + f'$({str(std)})$' + '\\\\$' + f'[{str(median)}]' + '$}' )
                    
                    if c_j != len(df.columns) - 1:
                        file.write(' & ')
                    else:
                        file.write(' \\\\ \hline \n')
                    
                    c_j += 1
                c_i += 1
            
            file.close()



    def get_plot_counter(self, plot_name):

        try:
            plot_counter = int(self.plotsetup[plot_name])
        except KeyError:            
            self.last_fig_counter += 1
            plot_counter = self.last_fig_counter
            self.plotsetup[plot_name] = plot_counter
        
        return plot_counter



    def plot_cat_barh(self, cat = None, font_scale = None, x_label = None, y_label = None, max_nbars = 10, 
                      log_scale = False, terms_to_remove = ['term1', 'term2'], x_label_rotation = 70,
                      replace_terms_to_plot = { 'term1' : 'term_to_replace'}):
        
        print('\n> function: plot_cat_barh: ', cat)
                
        self.results_DF_processed = self.results_DF_copy.copy()
        
        #removendo termos presentes na lista terms
        for term in terms_to_remove:
            if term in self.results_DF_processed[cat].values:
                self.results_DF_processed = self.results_DF_processed.reset_index().set_index(cat).drop(index = term).reset_index().set_index(['Filename', 'Counter'])
        
        #removing nan da array
        cat_array = np.delete(self.results_DF_processed[ cat ].values, np.where(self.results_DF_processed[ cat ].values.astype(str) == 'nan')[0])

        #contando as ocorrências
        cat_val_array = np.array(list(zip(*np.unique(cat_array, return_counts = True))), dtype=object)
        uniq, counts = np.unique(cat_array, return_counts = True)
        cat_val_array = cat_val_array[ np.argsort(counts) ][ - max_nbars : ][ :: - 1 ]

        #procurando se já existe o setup do plot
        plot_counter = self.get_plot_counter(f'plot_cat_barh{cat}')

        #replacing terms in Y-axis
        counter = -1
        for cat, val in cat_val_array:
            counter += 1
            try:
                cat_val_array[counter] = ( replace_terms_to_plot[cat] , val)
                print('  Replacing term to plot: ', cat, '->', replace_terms_to_plot[cat])
            except KeyError:    
                continue

        self.cat_barh(cat_val_array, font_scale = font_scale, x_label = x_label, y_label = y_label, 
                      log_scale = log_scale, plot_counter = plot_counter,
                      x_label_rotation = x_label_rotation)



    def plot_cat_cat_barv(self, cat_to_group = None, cat_to_count = None, vals_to_count = [], vals_to_ignore = [],
                        font_scale = None, x_label = None, y1_label = None, y2_label = None, max_nbars = 10, 
                        y1_lims = [], y2_lims = [], order_in_perc = False,
                        terms_to_remove = ['term1', 'term2'], x_label_rotation = 70, 
                        cats_terms_to_filter = { 'col1' : ['term1', 'term2'] },
                        replace_terms_to_plot = { 'term1' : 'term_to_replace'}, remove_duplicates = True):
        
        print('\n> function: plot_cat_cat_barh: ', cat_to_group, cat_to_count)
        
        #copiando a DF para processar
        columns_checked_in_DF = check_columns_in_DF( list(cats_terms_to_filter.keys()) , self.results_DF_copy )
        if columns_checked_in_DF is True:
            self.results_DF_processed = self.results_DF_copy[ [cat_to_group, cat_to_count] + list(cats_terms_to_filter.keys()) ].copy()
        else:
            self.results_DF_processed = self.results_DF_copy[ [cat_to_group, cat_to_count] ].copy()

        #checando se as cols estão no DF
        for col in [cat_to_group, cat_to_count]:
            if col not in self.results_DF_processed.columns:
                print(f'  Erro! A col {col} não está presente no DF.')
                print('  Cats presentes: ', self.results_DF_processed.columns)
                return
        
        #removendo termos presentes na lista terms
        for cat in [cat_to_group, cat_to_count]:
            for term in terms_to_remove:
                if term in self.results_DF_processed[cat].values:
                    self.results_DF_processed = self.results_DF_processed.reset_index().set_index(cat).drop(index = term).reset_index().set_index(['Filename', 'Counter'])

        #filtrando cats em colunas (quando filtra se determina um plot_title para mostrar o que foi filtrado)
        self.results_DF_processed, plot_title = filter_categories(self.results_DF_processed, 
                                                                columns_checked_in_DF, 
                                                                cats_terms_to_filter)

        #Eliminando os NaN
        self.results_DF_processed = self.results_DF_processed[ [cat_to_group, cat_to_count] ]
        self.results_DF_processed = remove_nan(self.results_DF_processed)

        #removendo as duplicatas (não se remove as duplicatas neste caso)
        if remove_duplicates is True:
            self.results_DF_processed = remove_duplicates(self.results_DF_processed, columns_to_consider = [cat_to_group, cat_to_count])
        
        #agrupando os termos
        groups_index_dic = self.results_DF_processed.groupby(by=cat_to_group).groups

        #extraindo os valores numéricos para cada grupo
        vals_cat_count = []
        vals_cat_perc = []
        for cat in groups_index_dic.keys():
            cat_array = self.results_DF_processed.loc[ groups_index_dic[ cat ] ][ cat_to_count ].values
            vat_count_dic = dict(list(zip(*np.unique(cat_array, return_counts = True))))
            for val_to_count in vals_to_count:
                try: 
                    val_count = vat_count_dic[ val_to_count ]
                    vals_cat_count.append( ( cat , val_count ) )
                    total_vals = sum([ vat_count_dic[key] for key in  vat_count_dic.keys() if key not in vals_to_ignore ])
                    vals_cat_perc.append( ( cat , val_count / total_vals ) )

                except KeyError:
                    continue

        cat_val_array = np.array(vals_cat_count, dtype=object)
        cat_perc_array = np.array(vals_cat_perc, dtype=object)

        #first sorting é com os maiores counts
        args_to_sort = np.argsort(cat_val_array[:, 1])
        cat_val_array = cat_val_array[ args_to_sort ][ - max_nbars : ][ :: - 1 ]
        cat_perc_array = cat_perc_array[ args_to_sort ][ - max_nbars : ][ :: - 1 ]

        #second sorting é com a percentagem
        if order_in_perc is True:
            args_to_sort = np.argsort(cat_perc_array[:, 1])
            cat_val_array = cat_val_array[ args_to_sort ][ :: - 1 ]
            cat_perc_array = cat_perc_array[ args_to_sort ][ :: - 1 ]

        #replacing terms in Y-axis
        counter = -1
        for cat, val in cat_val_array:
            counter += 1
            try:
                cat_val_array[counter] = ( replace_terms_to_plot[cat] , val)
                print('  Replacing term to plot: ', cat, '->', replace_terms_to_plot[cat])
            except KeyError:    
                continue

        #procurando se já existe o setup do plot
        plot_counter = self.get_plot_counter(f'plot_cat_cat_barv_{cat_to_group}_{cat_to_count}_{vals_to_count}_{vals_to_ignore}_{cats_terms_to_filter}')

        self.cat_barv(categories = cat_val_array[ : , 0], values_a = cat_val_array[ : , 1], values_b = cat_perc_array[ : , 1],
                      font_scale = font_scale, x_label = x_label, y1_label = y1_label, y2_label = y2_label, 
                      y1_lims = y1_lims, y2_lims = y2_lims, plot_title = plot_title,
                      plot_counter = plot_counter, x_label_rotation = x_label_rotation)



    def plot_cat_cat_heatmap(self, cats = ['None', 'None'], 
                             x_label = None, y_label = None,
                             cats_terms_to_filter = { 'col1' : ['term1', 'term2'] },
                             n_box_in_grid = 30, 
                             font_scale = 10,
                             x_label_rotation = 70,
                             terms_to_remove = ['term1', 'term2'],
                             plot_margins = {},
                             replace_terms_to_plot = { 'term1' : 'term_to_replace'}):
        
        print('\n> function: plot_cat_cat_heatmap: ', cats)

        #copiando a DF para processar
        columns_checked_in_DF = check_columns_in_DF( list(cats_terms_to_filter.keys()) , self.results_DF_copy )
        if columns_checked_in_DF is True:
            self.results_DF_processed = self.results_DF_copy[ cats + list(cats_terms_to_filter.keys()) ].copy()
        else:
            self.results_DF_processed = self.results_DF_copy[ cats ].copy()

        #checando se as categorias estão no DF
        for cat in cats:
            if cat not in self.results_DF_processed.columns:
                print('  Erro! A cat {cat} não está presente no DF.')
                print('  Cats presentes: ', self.results_DF_processed.columns)
                return

        #removendo termos presentes na lista terms
        for cat in cats:
            for term in terms_to_remove:
                if term in self.results_DF_processed[cat].values:
                    self.results_DF_processed = self.results_DF_processed.reset_index().set_index(cat).drop(index = term).reset_index().set_index(['Filename', 'Counter'])

        #filtrando cats em colunas (quando filtra se determina um plot_title para mostrar o que foi filtrado)
        self.results_DF_processed, plot_title = filter_categories(self.results_DF_processed, 
                                                                columns_checked_in_DF, 
                                                                cats_terms_to_filter)

        #Eliminando os NaN
        self.results_DF_processed = self.results_DF_processed[ cats ]
        self.results_DF_processed = remove_nan(self.results_DF_processed)

        #removendo as duplicatas
        self.results_DF_processed = remove_duplicates(self.results_DF_processed, columns_to_consider = cats)
        
        #contando as co-ocorrências
        cat_cat_list = [str(cat1) + '_' + str(cat2) for cat1, cat2 in zip(self.results_DF_processed[cats[0]].values, self.results_DF_processed[cats[1]].values)]
        uniq, counts = np.unique(cat_cat_list, return_counts = True)
        min_counts = np.sort(counts)[ - n_box_in_grid : ][0]
        
        #montando os labels dos eixos x e y
        cat_cat_val_array = []
        for cat_cat, counts in zip(uniq, counts):
            if counts >= min_counts:
                cat1 = re.search(r'[\w\s]+(?=_)', cat_cat).group()
                cat2 = re.search(r'(?<=_)[\w\s]+', cat_cat).group()
                cat_cat_val_array.append([cat1, cat2, counts])
        cat_cat_val_array = np.array(cat_cat_val_array, dtype = object)

        if len(cat_cat_val_array) >= 3:
            
            #procurando se já existe o setup do plot
            plot_counter = self.get_plot_counter(f'plot_cat_cat_heatmap_{cats[0]}_{cats[1]}_{str(cats_terms_to_filter)}')

            self.cat_cat_heatmap(cat_cat_val_array, font_scale = font_scale, x_label = x_label, y_label = y_label, 
                                 x_label_rotation = x_label_rotation, plot_title = plot_title,
                                 plot_counter = plot_counter, plot_margins = plot_margins,
                                 replace_terms_to_plot = replace_terms_to_plot)
        
        else:
            print('\nErro!\nNão foram encontradas três ou mais categorias para plotar.\n')



    def plot_cat_num_boxplot_stats_test(self, cat_col = 'None', num_col = 'None', 
                                   stat_func_to_extract = 'mean',
                                   x_label = None, y_label = None, y_lims = [0, 1], 
                                   n_cats = 10, min_num_vals_occur = 11, log_scale = False,
                                   cats_terms_to_filter = { 'col1' : ['term1', 'term2'] },
                                   x_label_rotation = 70,
                                   font_scale_boxplot = 10,
                                   font_scale_stats_test_grid = 10,
                                   box_values_font_scale = 10,
                                   stats_test_x_annotation_delta = 0, stats_test_y_annotation_delta = 0,
                                   boxplot_plot_margins = {'left':0.3},
                                   y_multiplier = None,
                                   stats_method = 'permutation', #anova #permutation
                                   remove_extreme_vals = None,
                                   terms_to_remove = ['term1', 'term2'],
                                   replace_terms_to_plot = { 'term1' : 'term_to_replace'},
                                   exclude_outliers_to_stats_test = False):
        
        print('\n> function: plot_cat_num_boxplot: ', cat_col, num_col)

        #1 plotando o boxplot

        #copiando a DF para processar
        columns_checked_in_DF = check_columns_in_DF( list(cats_terms_to_filter.keys()) , self.results_DF_copy )
        if columns_checked_in_DF is True:
            self.results_DF_processed = self.results_DF_copy[ [ cat_col, num_col] + list(cats_terms_to_filter.keys()) ].copy()
        else:
            self.results_DF_processed = self.results_DF_copy[ [ cat_col, num_col] ].copy()

        #checando se as cols estão no DF
        for col in [cat_col, num_col]:
            if col not in self.results_DF_processed.columns:
                print(f'  Erro! A col {col} não está presente no DF.')
                print('  Cats presentes: ', self.results_DF_processed.columns)
                return

        #removendo termos presentes na lista terms
        for term in terms_to_remove:
            if term in self.results_DF_processed[cat_col].values:
                self.results_DF_processed = self.results_DF_processed.reset_index().set_index(cat_col).drop(index = term).reset_index().set_index(['Filename', 'Counter'])

        #filtrando cats em colunas (quando filtra se determina um plot_title para mostrar o que foi filtrado)
        self.results_DF_processed, plot_title = filter_categories(self.results_DF_processed, 
                                                                columns_checked_in_DF, 
                                                                cats_terms_to_filter)

        #Eliminando os NaN
        self.results_DF_processed = self.results_DF_processed[ [ cat_col, num_col] ]
        self.results_DF_processed = remove_nan(self.results_DF_processed)

        #removendo as duplicatas
        self.results_DF_processed = remove_duplicates(self.results_DF_processed, columns_to_consider = [cat_col, num_col])

        #agrupando os termos
        groups_index_dic = self.results_DF_processed.groupby(by=cat_col).groups

        #extraindo os valores numéricos para cada grupo
        cat_val_len_list = []
        for cat in groups_index_dic.keys():
            vals = self.extract_vals_from_df_str(indexes = groups_index_dic[ cat ], 
                                               num_col = num_col,
                                               mode = stat_func_to_extract,
                                               interval_to_filter = y_lims,
                                               axis_multiplier = y_multiplier,
                                               remove_extreme_vals = remove_extreme_vals)

            vals = vals[~np.isnan(vals.astype(float))]
            cat_val_len_list.append( (cat, vals, len(vals)) )
            
            #if len(vals) >= min_num_vals_occur:
            #vals_cat_dic[ cat ] = vals
            #cat_len_list.append((cat, len(vals)))

        #filtrando somente cats que tenham len(vals) >= min_num_vals_occur
        cat_val_len_list = [ (cat, vals, n_vals) for cat, vals, n_vals in cat_val_len_list if n_vals >= min_num_vals_occur]
        #cat_vals_dic = dict([ (cat, vals) for cat, vals, length in cat_val_len_list ])

        #caso o número de categorias inserido seja maior que o número de grupos com valores numéricos
        print('  input n_cats: ', n_cats)
        n_cats = len(cat_val_len_list) if n_cats > len(cat_val_len_list) else n_cats
        print('  output n_cats: ', n_cats)

        #caso tenha mais de 1 categoria para plotar
        if n_cats <= 2:
            print('\nERRO!')
            print(f'O valor de "n_cats" ({n_cats}) é menor que 2.')
            print('Abortando a função.\n')
            return

        #sorting dic e coletando só as cats com mais valores
        cat_val_len_list = sorted( cat_val_len_list, key=lambda x: x[2])
        cat_val_len_list = sorted( cat_val_len_list[ -n_cats : ], key=lambda x: x[0])
        #cats_len_dic  = dict( cat_len_list)

        nums_grouped = []
        cats_grouped = []
        n_points_per_box = []
        for cat, vals, n_vals in cat_val_len_list:
            
            cats_grouped.append( cat )
            nums_grouped.append( vals )
            n_points_per_box.append(n_vals)

        if len(cats_grouped) < n_cats:
            print('\nERRO!')
            print(f'O valor de "n_cats" ({n_cats}) é menor que o número de categorias encontradas ({len(cats_grouped)}).')
            print('Ajustando "n_cats" para ', len(cats_grouped), '\n')
            n_cats = len(cats_grouped)

        #fazendo o box plot
        #procurando se já existe o setup do plot
        plot_counter = self.get_plot_counter(f'plot_cat_num_boxplot_{cat_col}_{num_col}_{str(cats_terms_to_filter)}')

        nums_grouped = self.box_plot(nums_grouped = nums_grouped, cats_grouped = cats_grouped, n_cats = n_cats, 
                                    n_points_per_box = n_points_per_box, log_scale = log_scale, y_lims = y_lims, 
                                    x_label = x_label, y_label = y_label, x_label_rotation = x_label_rotation, 
                                    font_scale_boxplot = font_scale_boxplot, plot_counter = plot_counter, 
                                    plot_title = plot_title, plot_margins = boxplot_plot_margins,
                                    replace_terms_to_plot = replace_terms_to_plot, exclude_outliers = exclude_outliers_to_stats_test)

        #fazendo o teste multicomponente
        plot_title = ''
        p_welch_f_value = round(permutation_test_welch_anova(nums_grouped,
                                                            n_resamples = 20000,
                                                            random_state = 42
                                                            ), 2)
        p_welch_f_value = str(p_welch_f_value) if p_welch_f_value > 0.01 else '< 0.01'
        plot_title = 'P. Welch F p-value: ' + p_welch_f_value

        stats_test_cat_list = []
        stats_test_val_list = []
        stats_test_mean_cat_val = {}
        for cat, vals in zip(cats_grouped, nums_grouped):
            stats_test_cat_list.extend( len(vals) * [cat] )
            stats_test_val_list.extend( list(vals) )
            stats_test_mean_cat_val[cat] = np.mean( vals )

        self.stats_results[f'boxplot_{stats_method}_{cat_col}_{num_col}_{str(cats_terms_to_filter)}'] = []

        if stats_method.lower() == 'anova':

            #fazendo o ANOVA para todos os cat - vals
            comp = mc.MultiComparison(stats_test_val_list, stats_test_cat_list)
            post_hoc_res = comp.tukeyhsd()
            stats_test_cat_cat_val_array = np.array(post_hoc_res._results_table.data[1:], dtype=object)
            stats_test_cat_cat_val_array = np.array(list(zip(stats_test_cat_cat_val_array[: , 0], stats_test_cat_cat_val_array[: , 1], stats_test_cat_cat_val_array[: , 3].astype(float) )), dtype = object)
            #stats_test_cat_cat_val_array = p_holm_bh_correction(stats_test_cat_cat_val_array)
            
            for cat1, cat2, p in stats_test_cat_cat_val_array:
                if p <= 0.05:
                    #salvando o anova em json
                    self.stats_results[f'boxplot_{stats_method}_{cat_col}_{num_col}_{str(cats_terms_to_filter)}'].append(
                        f'{cat1} - mean = { stats_test_mean_cat_val[cat1] / y_multiplier} : {cat2} - mean = { stats_test_mean_cat_val[cat2] / y_multiplier} : p-val = {p}')

        elif stats_method.lower() == 'permutation':

            #fazendo o permutation para todos os cat - vals
            cat_vals_dic = dict( [ (cat, vals) for cat, vals in zip(cats_grouped, nums_grouped) ] )
            
            stats_test_cat_cat_p_array = []
            for cat1, cat2 in itt.combinations(cat_vals_dic.keys(), 2):
                vals_group1 = cat_vals_dic[cat1]
                vals_group2 = cat_vals_dic[cat2]

                stats_test_cat_cat_p_array.append( (cat1 , cat2, permutation_test_AB(vals_group1, vals_group2,
                                                                                       statistic = diff_of_welch_t_means,
                                                                                       n_resamples = 20000,
                                                                                       alternative = 'two-sided',
                                                                                       random_state = 42)) )
                
            stats_test_cat_cat_p_array = np.array(stats_test_cat_cat_p_array, dtype=object)
            #stats_test_cat_cat_val_array = p_holm_bh_correction(stats_test_cat_cat_val_array)
            
            for cat1, cat2, p in stats_test_cat_cat_p_array:
                if p <= 0.05:
                    #salvando o anova em json
                    self.stats_results[f'boxplot_{stats_method}_{cat_col}_{num_col}_{str(cats_terms_to_filter)}'].append(
                        f'{cat1} - mean = { stats_test_mean_cat_val[cat1] / y_multiplier} : {cat2} - mean = { stats_test_mean_cat_val[cat2] / y_multiplier} : p-val = {p}')


        save_dic_to_json(self.diretorio + '/Outputs/analyses/dic_stats_results.json', self.stats_results)

        #fazendo o plot dos valores p
        if len(cats_grouped) >= 3:
            #plotando o stats_tets
            #procurando se já existe o setup do plot
            plot_counter = self.get_plot_counter(f'plot_cat_cat_heatmap_{stats_method}_{cat_col}_{num_col}_{str(cats_terms_to_filter)}')

            #stats_test_cat_cat_val_filtered_list = []
            #for cat1, cat2, p in stats_test_cat_cat_val_array:
            #    if cat1 in cats_len_dic.keys() and cat2 in cats_len_dic.keys():
            #        stats_test_cat_cat_val_filtered_list.append((cat1, cat2, p))
            
            #stats_test_cat_cat_p_array = np.array(stats_test_cat_cat_val_filtered_list, dtype = object)
            self.cat_cat_heatmap(stats_test_cat_cat_p_array, font_scale = font_scale_stats_test_grid, box_vals = 0.05, 
                                box_values_font_scale = box_values_font_scale, plot_counter = plot_counter, plot_title =plot_title, 
                                x_annotation_delta = stats_test_x_annotation_delta, y_annotation_delta = stats_test_y_annotation_delta, 
                                x_label_rotation = x_label_rotation, bar_label = 'p-val', replace_terms_to_plot = replace_terms_to_plot)
        else:
            print('\nErro!\nNão foram encontradas três ou mais categorias para plotar o stats_test gridplot.\n')



    def plot_cat_cat_num_heatmap(self, cats_col = [], num_col = 'None', 
                                stat_func_to_extract = 'mean',
                                stat_func_to_group = 'mean',
                                x_label = None, y_label = None, bar_label = None,
                                cats_terms_to_filter = { 'col1' : ['term1', 'term2'] },
                                n_box_in_grid = 10, 
                                min_num_vals_occur = 11,
                                font_scale_grid = 10,
                                box_values_font_scale = 10,
                                x_label_rotation = 70,
                                z_multiplier = None,
                                stats_method = 'anova', #permutation
                                z_lims = [],
                                x_annotation_delta = 0, y_annotation_delta = 0, 
                                remove_extreme_vals = None,
                                terms_to_remove = ['term1', 'term2'],
                                grid_plot_margins = None):
        
        print('\n> function: plot_cat_cat_num_heatmap: ', cats_col, num_col)

        #copiando a DF para processar
        columns_checked_in_DF = check_columns_in_DF( list(cats_terms_to_filter.keys()) , self.results_DF_copy )
        if columns_checked_in_DF is True:
            self.results_DF_processed = self.results_DF_copy[ [ cats_col[0], cats_col[1], num_col ] + list(cats_terms_to_filter.keys()) ].copy()
        else:
            self.results_DF_processed = self.results_DF_copy[ [ cats_col[0], cats_col[1], num_col ] ].copy()

        #checando se as categorias estão no DF
        for col in [ cats_col[0], cats_col[1], num_col]:
            if col not in self.results_DF_processed.columns:
                print(f'  Erro! A col {col} não está presente no DF.')
                print('  Cats presentes: ', self.results_DF_processed.columns)
                return

        #removendo termos presentes na lista terms
        for cat in cats_col:    
            for term in terms_to_remove:
                if term in self.results_DF_processed[cat].values:
                    self.results_DF_processed = self.results_DF_processed.reset_index().set_index(cat).drop(index = term).reset_index().set_index(['Filename', 'Counter'])

        #filtrando cats em colunas (quando filtra se determina um plot_title para mostrar o que foi filtrado)
        self.results_DF_processed, plot_title = filter_categories(self.results_DF_processed, 
                                                                columns_checked_in_DF, 
                                                                cats_terms_to_filter)

        #Eliminando os NaN
        self.results_DF_processed = self.results_DF_processed[ [ cats_col[0], cats_col[1], num_col ] ]
        self.results_DF_processed = remove_nan(self.results_DF_processed)

        #removendo as duplicatas
        self.results_DF_processed = remove_duplicates(self.results_DF_processed, columns_to_consider = [num_col] +  cats_col)

        #concatenando as cats para cada linha da DF
        cat_cat_list = [str(cat1) + '_' + str(cat2) for cat1, cat2 in zip(self.results_DF_processed[cats_col[0]].values, self.results_DF_processed[cats_col[1]].values)]
        self.results_DF_processed['concat_cats'] = cat_cat_list

        #agrupando os termos
        groups_index_dic = self.results_DF_processed.groupby(by='concat_cats').groups

        cat_cat_vals_dic = {}
        cat_cat_len_dic = {}
        cat_cat_len_list = []
        stats_test_cat_list = []
        stats_test_val_list = []
        for cat_cat in groups_index_dic.keys():
            vals = self.extract_vals_from_df_str(indexes = groups_index_dic[ cat_cat ], 
                                               num_col = num_col,
                                               mode = stat_func_to_extract,
                                               interval_to_filter = z_lims,
                                               axis_multiplier = z_multiplier,
                                               remove_extreme_vals = remove_extreme_vals)
            
            vals = vals[~np.isnan(vals.astype(float))]
            if len(vals) >= min_num_vals_occur:
                cat_cat_vals_dic[ cat_cat ] = vals
                cat_cat_len_dic[cat_cat] = len(vals)
                cat_cat_len_list.append( (cat_cat, len(vals)) )
                stats_test_cat_list.extend( len(vals) * [cat_cat] )
                stats_test_val_list.extend( list(vals) )
        
        num_vals_error = []
        cat_cat_mean_val_dic = {}
        cat_cat_stat_func_dic = {}
        for cat_cat in cat_cat_vals_dic.keys():

            if stat_func_to_group == 'mean':
                num_val = cat_cat_vals_dic[ cat_cat ].mean()
            elif stat_func_to_group == 'min':
                num_val = cat_cat_vals_dic[ cat_cat ].min()
            elif stat_func_to_group == 'max':
                num_val = cat_cat_vals_dic[ cat_cat ].max()
            elif stat_func_to_group == 'median':
                num_val = np.median(cat_cat_vals_dic[ cat_cat ])
            
            cat_cat_stat_func_dic[cat_cat] = num_val
            num_c = np.round(num_val, 1) if -10 < num_val < 10 else int(num_val)
            num_c = 0 if num_c == 0 else num_c #deixando 0 inteiro caso num_c seja float 0.0
            cat_cat_mean_val_dic[cat_cat] = num_c 
            num_vals_error.append((num_val, num_c))


        if len(cat_cat_mean_val_dic) >= 5:
            
            #fazendo o stats_test
            try: 
                self.stats_results[f'heatmap_{stats_method}_{cats_col[0]}_{cats_col[1]}_{num_col}_{str(cats_terms_to_filter)}']
            
            except KeyError:
                self.stats_results[f'heatmap_{stats_method}_{cats_col[0]}_{cats_col[1]}_{num_col}_{str(cats_terms_to_filter)}'] = []
                
                comp = mc.MultiComparison(stats_test_val_list, stats_test_cat_list)
                post_hoc_res = comp.tukeyhsd()
                stats_test_cat_cat_val_array = np.array(post_hoc_res._results_table.data[1:], dtype=object)
                stats_test_cat_cat_val_array = np.array(list(zip(stats_test_cat_cat_val_array[: , 0], stats_test_cat_cat_val_array[: , 1], stats_test_cat_cat_val_array[: , 3].astype(float) )), dtype = object)
                
                for cat_cat1, cat_cat2, p in stats_test_cat_cat_val_array:
                    if p <= 0.05:
                        #salvando o stats_test em json
                        self.stats_results[f'heatmap_{stats_method}_{cats_col[0]}_{cats_col[1]}_{num_col}_{str(cats_terms_to_filter)}'].append(
                            f'{cat_cat1} - {stat_func_to_group} = {cat_cat_stat_func_dic[cat_cat1] / z_multiplier} : {cat_cat2} - {stat_func_to_group} = {cat_cat_stat_func_dic[cat_cat2] / z_multiplier} : p-val = {p}')

                save_dic_to_json(self.diretorio + '/Outputs/analyses/dic_stats_results.json', self.stats_results)


            #caso o número de categorias inserido seja maior que o número de grupos com valores numéricos
            print('  input n_box_in_grid: ', n_box_in_grid)
            n_box_in_grid = len(cat_cat_vals_dic.keys()) if n_box_in_grid > len(cat_cat_vals_dic.keys()) else n_box_in_grid
            print('  output n_box_in_grid: ', n_box_in_grid)

            #sorting dic e coletando só as cats com mais valores
            
            cat_cat_len_list = sorted( cat_cat_len_list , key=lambda x: x[1])
            cat_cat_len_list = sorted( cat_cat_len_list[ - n_box_in_grid : ], key=lambda x: x[0])

            cat_cat_len_array = []
            cat_cat_men_val_array = []
            for cat_cat in cat_cat_mean_val_dic.keys():
                
                cat1 = re.search(r'[\w\s]+(?=_)', cat_cat).group()
                cat2 = re.search(r'(?<=_)[\w\s]+', cat_cat).group()

                if cat_cat in np.array(cat_cat_len_list, dtype = object)[ : , 0]:
                    #a ordem é (cat1, cat2, len(cat1_cat2), mean(cat1_cat2))
                    cat_cat_len_array.append( ( cat1, cat2, cat_cat_len_dic[cat_cat] ))
                    cat_cat_men_val_array.append( cat_cat_mean_val_dic[cat_cat] )

            #transformando em array
            cat_cat_len_array = np.array(cat_cat_len_array, dtype=object)

            #procurando se já existe o setup do plot
            try:
                plot_counter = int(self.plotsetup[f'plot_cat_cat_num_heatmap_{cats_col[0]}_{cats_col[1]}_{num_col}_{str(cats_terms_to_filter)}'])
            except KeyError:            
                self.last_fig_counter += 1
                plot_counter = self.last_fig_counter
                self.plotsetup[f'plot_cat_cat_num_heatmap_{cats_col[0]}_{cats_col[1]}_{num_col}_{str(cats_terms_to_filter)}'] = plot_counter

            if len(set(cat_cat_len_array[:, 2])) == 1:
                print('\nERRO!')
                print('Ajustar o parâmetro "z_multiplier" para colocar os valores numéricos > 1\n')
                print('(nums extracted : nums colected )')
                for i, j in num_vals_error:
                    print(i, ':', j)


            self.cat_cat_heatmap(cat_cat_len_array, font_scale = font_scale_grid, plot_title = plot_title,
                                x_label = x_label, y_label = y_label, bar_label = bar_label, x_label_rotation = x_label_rotation,
                                box_vals = cat_cat_men_val_array, x_annotation_delta = x_annotation_delta, y_annotation_delta = y_annotation_delta, 
                                box_values_font_scale = box_values_font_scale, plot_counter = plot_counter, plot_margins = grid_plot_margins)
        
        else:
            print('\nErro!\nNão foram encontradas 5 ou mais categorias para plotar.\n')



    def plot_scatter_graph(self, num_cols = [], stat_func_to_extract_x = 'mean', stat_func_to_extract_y = 'mean',
                           x_multiplier = None, y_multiplier = None, remove_extreme_vals_for_mean = None,
                           log_scale_x = False, log_scale_y = False,
                           min_num_vals_occur = 11,
                           stats_method = 'permutation', #anova #bootstrap #permutation
                           cats_terms_to_plot = ( '' , [''] ),
                           cats_terms_to_filter = { 'col1' : ['term1', 'term2'] },
                           x_lims = [], y_lims = [], x_label = None, y_label = None,
                           set_graph_region = None):
        

        print('\n> function: plot_scatter_graph: ', num_cols)
        print('> cats_terms_to_plot', cats_terms_to_plot)
        print('> cats_terms_to_filter', cats_terms_to_filter)

        #copiando a DF para processar
        columns_checked_in_DF = check_columns_in_DF( list(cats_terms_to_filter.keys()) , self.results_DF_copy )
        check_cats_to_plot = check_columns_in_DF( [ cats_terms_to_plot[0] ] , self.results_DF_copy )
        if columns_checked_in_DF is True and check_cats_to_plot is True:
            self.results_DF_processed = self.results_DF_copy[ num_cols + [ cats_terms_to_plot[0] ] + list(cats_terms_to_filter.keys()) ].copy()
        elif columns_checked_in_DF is True and check_cats_to_plot is False:
            self.results_DF_processed = self.results_DF_copy[ num_cols + list(cats_terms_to_filter.keys()) ].copy()
        elif columns_checked_in_DF is False and check_cats_to_plot is True:
            self.results_DF_processed = self.results_DF_copy[ num_cols + [ cats_terms_to_plot[0] ] ].copy()
        else:
            self.results_DF_processed = self.results_DF_copy[ num_cols ].copy()
        
        #filtrando cats em colunas (quando filtra se determina um plot_title para mostrar o que foi filtrado)
        self.results_DF_processed, plot_title = filter_categories(self.results_DF_processed, 
                                                                columns_checked_in_DF, 
                                                                cats_terms_to_filter)

        #Eliminando os NaN
        self.results_DF_processed = remove_nan(self.results_DF_processed)

        #removendo as duplicatas
        self.results_DF_processed = remove_duplicates(self.results_DF_processed, columns_to_consider = num_cols)

        for col, stat_func, multiplier, lims in zip(num_cols, (stat_func_to_extract_x, stat_func_to_extract_y), (x_multiplier, y_multiplier), (x_lims, y_lims)):
            #extraindo os valores numéricos
            self.results_DF_processed[col] = self.extract_vals_from_df_str(indexes = self.results_DF_processed.index, 
                                                                            num_col = col, 
                                                                            mode = stat_func, 
                                                                            axis_multiplier = multiplier,
                                                                            interval_to_filter = lims)

        #Eliminando os NaN devido aos x_lims e y_lims (caso os números estejam fora dos intervalos)
        self.results_DF_processed = remove_nan(self.results_DF_processed)
        
        ixy_all_array = np.array(list(zip(self.results_DF_processed.index.get_level_values(0), 
                                         self.results_DF_processed[num_cols[0]].values, 
                                         self.results_DF_processed[num_cols[1]].values)), dtype = object)


        #procurando se já existe o setup do plot
        plot_counter = self.get_plot_counter(f'plot_scatter_graph_{num_cols[0]}_{num_cols[1]}_{str(cats_terms_to_plot)}_{str(cats_terms_to_filter)}')

        self.scatter_plot(ixy_all_array = ixy_all_array, 
                          plot_counter = plot_counter, 
                          log_scale_x = log_scale_x, log_scale_y = log_scale_y, 
                          x_lims = x_lims, y_lims = y_lims, 
                          x_label = x_label, y_label = y_label,
                          set_graph_region = set_graph_region, 
                          plot_title = plot_title)


        #agrupando os termos        
        cats_len_dic = {}
        if check_cats_to_plot is True:

            cat_nums_dic = {}
            stats_test_stats_col_cat_val = {}
            stats_test_stats_col_cat_val[num_cols[0]] = {}
            stats_test_stats_col_cat_val[num_cols[1]] = {}
            stats_test_cat_vals_permutation_dic = {}
            stats_test_cat_vals_permutation_dic[num_cols[0]] = {}
            stats_test_cat_vals_permutation_dic[num_cols[1]] = {}         
            stats_test_results_dic = {}
            stats_test_results_dic[num_cols[0]] = {}
            stats_test_results_dic[num_cols[1]] = {}

            groups_index_dic = self.results_DF_processed.groupby(by=cats_terms_to_plot[0]).groups

            for cat in groups_index_dic:
                
                #coletando categorias que possuem mais que n vals
                if len(groups_index_dic[cat]) >= min_num_vals_occur:
                
                    #cats_to_plot = cats_terms_to_plot[1]
                    cats_len_dic[cat] = len(groups_index_dic[cat])

                    df_filtered = self.results_DF_processed.loc[groups_index_dic[cat]]
                    cat_nums_dic[cat] = np.array(list(zip(df_filtered.index.get_level_values(0),
                                                            df_filtered[num_cols[0]].values, 
                                                            df_filtered[num_cols[1]].values)), dtype = object)
                    
                    stats_test_cat_vals_permutation_dic[num_cols[0]][cat] = df_filtered[num_cols[0]].values
                    stats_test_cat_vals_permutation_dic[num_cols[1]][cat] = df_filtered[num_cols[1]].values


            #checando se há pelo menos duas categorias coletadas
            if len(cat_nums_dic.keys()) >= 2:
            
                #checando o tamanho das categorias
                cats_len_dic = dict( sorted( list(cats_len_dic.items()), key=lambda x: x[1]) )
                print(f'  Report for [{num_cols[0]} x {num_cols[1]}]')
                print(f'  Cat_col: {cats_terms_to_plot[0]}')
                print('  Cat -> len(vals) relation (10 top):')
                for cat, length in list(cats_len_dic.items())[ - 10 : ]:
                    print(f'  {cat} -> len {length}')

                #plotando cats selecionadas (se houver)
                cat_nums_dic_filtered = {}
                if type(cats_terms_to_plot[1]) == list and len(cats_terms_to_plot[1]) > 1:
                    for cat in cats_terms_to_plot[1]:
                        cat_nums_dic_filtered[cat] = cat_nums_dic[cat]
                
                    #procurando se já existe o setup do plot
                    plot_counter = self.get_plot_counter(f'plot_scatter_graph_{num_cols[0]}_{num_cols[1]}_{str(cats_terms_to_plot)}_{str(cats_terms_to_filter)}')

                    self.scatter_plot(ixy_all_array = ixy_all_array, cat_nums_dic = cat_nums_dic_filtered, 
                                      cats_len_dic = cats_len_dic, plot_counter = plot_counter,
                                      log_scale_x = log_scale_x, log_scale_y = log_scale_y, 
                                      x_lims = x_lims, y_lims = y_lims, 
                                      x_label = x_label, y_label = y_label,
                                      plot_title = plot_title)                        

                
                if stats_method.lower() == 'permutation':
                                
                    #fazendo o permutation para todos os cat - vals
                    stats_test_cat_cat_val_array = []
                    for cat1, cat2 in itt.combinations(stats_test_cat_vals_permutation_dic[num_cols[0]].keys(), 2):
                        vals_group1 = stats_test_cat_vals_permutation_dic[num_cols[0]][cat1]
                        vals_group2 = stats_test_cat_vals_permutation_dic[num_cols[0]][cat2]

                        vals_group1 = trim_array(vals_group1, remove_extreme_vals_for_mean)
                        vals_group2 = trim_array(vals_group2, remove_extreme_vals_for_mean)

                        stats_test_stats_col_cat_val[num_cols[0]][cat1] = np.mean(vals_group1)
                        stats_test_stats_col_cat_val[num_cols[0]][cat2] = np.mean(vals_group2)

                        stats_test_cat_cat_val_array.append( (cat1 , cat2, permutation_test_AB(vals_group1, vals_group2,
                                                                                                statistic = diff_of_welch_t_means,
                                                                                                n_resamples = 20000,
                                                                                                random_state = 42
                                                                                                )) )
                    
                    stats_test_cat_cat_val_array = np.array(stats_test_cat_cat_val_array, dtype=object)
                    #stats_test_cat_cat_val_array = p_holm_bh_correction(stats_test_cat_cat_val_array)
                    stats_test_results_dic[num_cols[0]]['results'] = stats_test_cat_cat_val_array


                    stats_test_cat_cat_val_array = []
                    for cat1, cat2 in itt.combinations(stats_test_cat_vals_permutation_dic[num_cols[1]].keys(), 2):
                        vals_group1 = stats_test_cat_vals_permutation_dic[num_cols[1]][cat1]
                        vals_group2 = stats_test_cat_vals_permutation_dic[num_cols[1]][cat2]

                        vals_group1 = trim_array(vals_group1, remove_extreme_vals_for_mean)
                        vals_group2 = trim_array(vals_group2, remove_extreme_vals_for_mean)
                        
                        stats_test_stats_col_cat_val[num_cols[1]][cat1] = np.mean(vals_group1)
                        stats_test_stats_col_cat_val[num_cols[1]][cat2] = np.mean(vals_group2)
                        
                        stats_test_cat_cat_val_array.append( (cat1 , cat2, permutation_test_AB(vals_group1, vals_group2,
                                                                                                statistic = diff_of_welch_t_means,
                                                                                                n_resamples = 20000,
                                                                                                random_state = 42
                                                                                                )) )

                    stats_test_cat_cat_val_array = np.array(stats_test_cat_cat_val_array, dtype=object)
                    #stats_test_cat_cat_val_array = p_holm_bh_correction(stats_test_cat_cat_val_array)
                    stats_test_results_dic[num_cols[1]]['results'] = stats_test_cat_cat_val_array

                cats_cols_to_plot = {}
                mean_cat_val = {}
                self.stats_results[f'scattergraph_{stats_method}_{num_cols[0]}_{num_cols[1]}_{cats_terms_to_plot[0]}_{str(cats_terms_to_filter)}'] = []
                for num_col in stats_test_results_dic.keys():
                    for cat1, cat2, p in stats_test_results_dic[num_col]['results']:
                        if p <= 0.05:

                            cats_cols_to_plot[(cat1, cat2)] = {}
                            cats_cols_to_plot[(cat1, cat2)][cat1] = cat_nums_dic[cat1]
                            cats_cols_to_plot[(cat1, cat2)][cat2] = cat_nums_dic[cat2]

                            axis = 'x' if num_col == num_cols[0] else 'y'
                            mean_cat_val[(cat1, cat2)] = {}
                            mean_cat_val[(cat1, cat2)][axis] = (stats_test_stats_col_cat_val[num_col][cat1], stats_test_stats_col_cat_val[num_col][cat2])
                            
                            #salvando o stats_test em json
                            multiplier = x_multiplier if num_col == num_cols[0] else y_multiplier
                            self.stats_results[f'scattergraph_{stats_method}_{num_cols[0]}_{num_cols[1]}_{cats_terms_to_plot[0]}_{str(cats_terms_to_filter)}'].append(
                                f'{num_col} > {cat1} - mean = { stats_test_stats_col_cat_val[num_col][cat1] / multiplier} : {cat2} - mean = { stats_test_stats_col_cat_val[num_col][cat2] / multiplier} : p-val = {p}')
                            
                save_dic_to_json(self.diretorio + '/Outputs/analyses/dic_stats_results.json', self.stats_results)

                #plotando cada par de cats com p < 0.05
                for cat1, cat2 in cats_cols_to_plot.keys():

                    #procurando se já existe o setup do plot
                    plot_counter = self.get_plot_counter(f'plot_scatter_graph_{num_cols[0]}_{num_cols[1]}_{cat1}_{cat2}_{num_col}_{str(cats_terms_to_filter)}')

                    self.scatter_plot(ixy_all_array = ixy_all_array, cat_nums_dic = cats_cols_to_plot[(cat1, cat2)], 
                                    cats_len_dic = cats_len_dic, axis_mean_vals = mean_cat_val[(cat1, cat2)],
                                    plot_counter = plot_counter, log_scale_x = log_scale_x, log_scale_y = log_scale_y, 
                                    x_lims = x_lims, y_lims = y_lims, x_label = x_label, y_label = y_label,
                                    plot_title = plot_title)



    def plot_axis_column_scatter_graph(self, num_col = None, cat_cols = [],
                                       stat_func_to_extract_y = 'mean', y_multiplier = 1,
                                       log_scale_y = False, x_label_rotation = 70,
                                       y_lims = [], y_label = None,
                                       font_scale_scatter_plot = 10, plotcircle_size = 1,
                                       y_bin_val = 10, offset_set = 0.1,
                                       g1_label = '', g2_label = '',
                                       x_labels_to_replace = {},
                                       cols_terms_to_remove = {'col1' : ['term1', 'term2']},
                                       cats_terms_to_filter = { 'col1' : ['term1', 'term2'] },
                                       representation_mode = 'random_spread',
                                       export_cats_len_json = False):

        print('\n> function: plot_axes_column_scatter_graph: ', num_col, cat_cols)

        #copiando a DF para processar
        columns_checked_in_DF = check_columns_in_DF( list(cats_terms_to_filter.keys()) , self.results_DF_copy )
        if columns_checked_in_DF is True:
            self.results_DF_processed = self.results_DF_copy[ [ num_col] + cat_cols + list(cats_terms_to_filter.keys()) ].copy()
        else:
            self.results_DF_processed = self.results_DF_copy[ [ num_col] + cat_cols].copy()

        #checando se as categorias estão no DF
        for col in [num_col] + cat_cols:
            if col not in self.results_DF_processed.columns:
                print(f'  Erro! A col {col} não está presente no DF.')
                print('  Cats presentes: ', self.results_DF_processed.columns)
                return
        
        #removendo termos presentes na lista terms
        for col in cols_terms_to_remove.keys():
            #caso a col não esteja no DF
            try:
                for term in cols_terms_to_remove[col]:
                    if term in self.results_DF_processed[col].values:
                        self.results_DF_processed = self.results_DF_processed.reset_index().set_index(col).drop(index = term).reset_index().set_index(['Filename', 'Counter'])
            except KeyError:
                pass

        #filtrando cats em colunas (quando filtra se determina um plot_title para mostrar o que foi filtrado)
        self.results_DF_processed, plot_title = filter_categories(self.results_DF_processed, 
                                                                columns_checked_in_DF, 
                                                                cats_terms_to_filter)

        col_cat_vals_len_dic = {}
        for col in cat_cols:

            #Eliminando os NaN
            temp_results_DF_processed = self.results_DF_processed[ [col, num_col] ].copy()
            temp_results_DF_processed = remove_nan(temp_results_DF_processed)

            #removendo as duplicatas
            temp_results_DF_processed = remove_duplicates(temp_results_DF_processed, columns_to_consider = [col, num_col])

            #extraindo os valores numéricos
            vals = self.extract_vals_from_df_str(indexes = temp_results_DF_processed.index, 
                                                num_col = num_col, 
                                                mode = stat_func_to_extract_y, 
                                                axis_multiplier = y_multiplier,
                                                interval_to_filter = y_lims)
            
            #agrupando os termos
            groups_index_dic = self.results_DF_processed.groupby(by=col).groups

            #extraindo os valores numéricos para cada grupo
            col_cat_vals_len_dic[col] = []
            for cat in groups_index_dic.keys():
                vals = self.extract_vals_from_df_str(indexes = groups_index_dic[ cat ], 
                                                num_col = num_col,
                                                mode = stat_func_to_extract_y,
                                                interval_to_filter = y_lims,
                                                axis_multiplier = y_multiplier)

                vals = vals[~np.isnan(vals.astype(float))]
                if len(vals) > 0:
                    col_cat_vals_len_dic[col].append( (cat, vals, len(vals)) )
                
                #if len(vals) >= min_num_vals_occur:
                #vals_cat_dic[ cat ] = vals
                #cat_len_list.append((cat, len(vals)))

            if export_cats_len_json is True:            
                cat_len_to_export = sorted([ (cat, n_vals) for cat, vals, n_vals in col_cat_vals_len_dic[col] ], key = lambda x: x[1])[::-1]
                self.cat_len_results[f'{num_col}_{col}_{cats_terms_to_filter}'] = dict(cat_len_to_export)
                save_dic_to_json(self.diretorio + f'/Outputs/analyses/dic_cat_len_results.json', self.cat_len_results, sort = False)


        #procurando se já existe o setup do plot
        plot_counter = self.get_plot_counter(f'plot_axis_column_scatter_graph_{num_col}_{cat_cols}_{cats_terms_to_filter}')

        self.axis_scatter_plots(col_cat_vals_len_dic,
                                log_scale = log_scale_y, y_lims = y_lims, 
                                y_label = y_label, x_label_rotation = x_label_rotation, 
                                y_bin_val = y_bin_val, offset_set = offset_set, plotcircle_size = plotcircle_size,
                                font_scale_axis_scatter = font_scale_scatter_plot, plot_counter = plot_counter, 
                                plot_title = plot_title, plot_plot_margins = {'left':0.3},
                                g1_label = g1_label, g2_label = g2_label,
                                x_labels_to_replace = x_labels_to_replace,
                                representation_mode = representation_mode)




    def cat_barv(self, categories = [], values_a = [], values_b = [],
                 font_scale = None, x_label = None, y1_label = None, y2_label = None,
                 y1_lims = [], y2_lims = [],
                 plot_counter = None, x_label_rotation = 70, plot_title = ''):

        # Set positions for each category on the x-axis
        x = np.arange(len(categories))
        width = 0.4

        n_colors = len(categories)
        palette = sns.color_palette(self.color_palette, n_colors)

        #adicionando o grid
        plt.figure(figsize=(20, 15))
        plot_grid = gridspec.GridSpec(1, 15) 
        plot_grid.update(wspace=0.2, hspace=0.2)
        
        #plotando a figura principal
        ax1 = plt.subplot(plot_grid[ : , : ])

        ax1.set_ylabel(y1_label, fontsize=font_scale, labelpad = 20)
        ax1.set_xlabel(x_label, fontsize=font_scale, labelpad = 20)

        ax1.set_title(plot_title + '\n', fontsize = font_scale)

        # Set category labels on the x-axis
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories, rotation=x_label_rotation, ha='right')

        ax1.tick_params(axis='x', bottom=True, labelsize=font_scale, width = 3, length = 10)
        ax1.tick_params(axis='y', left=True, labelsize=font_scale, width = 3, length = 10)

        ax1.spines['top'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        if y1_lims is not None:
            ax1.set_ylim([y1_lims[0], y1_lims[1]])

        # Create a second y-axis sharing the same x-axis
        ax2 = ax1.twinx()
        ax2.set_ylabel(y2_label, fontsize=font_scale, labelpad = 20)
        ax2.tick_params(axis='y', right=True, labelsize=font_scale, width = 3, length = 10)

        ax2.grid(which='major', visible='True', color='gray', linestyle='-', linewidth=1)

        
        # Plot the first dataset on the left y-axis
        rects1 = ax1.bar(x - width/2, values_a, width=width, label='Dataset A')

        i = 0
        for p in rects1:
            p.set_color(palette[i])
            i += 1

        # Plot the second dataset on the right y-axis
        rects2 = ax2.bar(x + width/2, values_b, width=width, label='Dataset B')

        i = 0
        for p in rects2:
            p.set_color(palette[ :: -1][i])
            i += 1

        if y2_lims is not None:
            ax2.set_ylim([y2_lims[0], y2_lims[1]])

        ax2.spines['top'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # Combine legends from both axes
        #lines1, labels1 = ax1.get_legend_handles_labels()
        #lines2, labels2 = ax2.get_legend_handles_labels()
        #ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        #salvando a figura
        filename = get_tag_name(plot_counter, prefix = '')
        plt.savefig(self.diretorio + f'/Outputs/analyses/{filename}.png', dpi=200, bbox_inches='tight', pad_inches=0.8)
        print(f'  salvando a figura {filename}...')
        save_dic_to_json(self.diretorio + '/Outputs/analyses/dic_plotsetup.json', self.plotsetup)



    def cat_barh(self, cat_val_array, font_scale = None, x_label = None, y_label = None, 
                plot_counter = None, log_scale = False, x_label_rotation = 70):

        n_colors = len(cat_val_array)
        palette = sns.color_palette(self.color_palette, n_colors)

        #adicionando o grid
        plt.figure(figsize=(20, 15))
        plot_grid = gridspec.GridSpec(1, 15) 
        plot_grid.update(wspace=0.2, hspace=0.2)
        
        #plotando a figura principal
        ax1 = plt.subplot(plot_grid[ : , : ])
        barlist = ax1.barh( cat_val_array[: , 0][ :: - 1] , cat_val_array[: , 1][ :: - 1].astype(float))
        
        i = 0
        for p in barlist:
            p.set_color(palette[i])
            i += 1

        i = -0.2
        max_val = cat_val_array[ : , 1].astype(float).max()
        for cat, val in cat_val_array[:: -1]:
            ax1.annotate(str(val), xy = (int(val) + (0.02 * max_val), i), ha='left', fontsize = font_scale)
            i+= 1

        ax1.set_ylabel(y_label, fontsize=font_scale, labelpad = 20)
        ax1.set_xlabel(x_label, fontsize=font_scale, labelpad = 20)
        ax1.xaxis.grid(True, linewidth = 1, c='gray', alpha=0.3)

        ax1.spines['top'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        ax1.tick_params(axis='x', bottom=True, labelsize=font_scale, labelrotation=x_label_rotation, width = 3, length = 10)
        ax1.tick_params(axis='y', left=True, labelsize=font_scale, width = 3, length = 10)

        if log_scale is True:
            ax1.set_xscale('log')

        #salvando a figura
        filename = get_tag_name(plot_counter, prefix = '')
        plt.savefig(self.diretorio + f'/Outputs/analyses/{filename}.png', dpi=200, bbox_inches='tight', pad_inches=0.5)
        print(f'  salvando a figura {filename}...')
        save_dic_to_json(self.diretorio + '/Outputs/analyses/dic_plotsetup.json', self.plotsetup)




    def cat_cat_heatmap(self, cat_cat_val_array, font_scale = None, box_vals = None, box_values_font_scale = None,
                        x_label = None, y_label = None, bar_label = None, x_label_rotation = 70, plot_title = '',
                        x_annotation_delta = 0, y_annotation_delta = 0, plot_counter = None, 
                        plot_margins = {}, replace_terms_to_plot = { 'term1' : 'term_to_replace'}):


        #registrando o max-min vals
        max_counts_val = cat_cat_val_array[: , 2].max()
        min_counts_val = cat_cat_val_array[: , 2].min()

        #montando os labels dos eixos x e y
        x_to_num = { v:k for k, v in enumerate(np.sort(np.unique(cat_cat_val_array[ : , 0]))) }
        y_to_num = { v:k for k, v in enumerate(np.sort(np.unique(cat_cat_val_array[ : , 1]))[ :: -1 ]) }
        x_tick_labels = list(x_to_num.keys())
        y_tick_labels = list(y_to_num.keys())

        #print('  x_tick_labels to plot:', x_tick_labels)
        #print('  y_tick_labels to plot:', y_tick_labels)

        #x_y_ratio = int( len(x_tick_labels) / len (y_tick_labels) )
        #x_y_ratio = x_y_ratio if x_y_ratio > 1 else 1
        #y_x_ratio = int( len(y_tick_labels) / len (x_tick_labels) )
        #y_x_ratio = y_x_ratio if y_x_ratio > 1 else 1

        #adicionando o grid
        plt.figure(figsize=(20, 15))
        plot_grid = gridspec.GridSpec(1, 15) 
        plot_grid.update(wspace=0.2, hspace=0.2, **plot_margins)
        
        #plotando a figura principal
        ax1 = plt.subplot(plot_grid[ : , : - 1 ])
        ax1.set_aspect("equal")

        #fazendo o setup do colormap
        n_colors = 256
        palette = sns.color_palette(self.color_palette, n_colors)
        color_min, color_max = [ cat_cat_val_array[: , 2].min() , cat_cat_val_array[: , 2].max() ] # Range of values that will be mapped to the palette, i.e. min and max possible correlation

        def value_to_color(val):
            try:
                val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range
                ind = int(val_position * (n_colors - 1)) # target index in the color palette
                return palette[ind]
            #caso só haja um valor para associar à cor
            except ZeroDivisionError:
                if val == 1:
                    return palette[-1]
                elif val == 0:
                    return palette[0]
        
        #mapeando as cores
        c = []
        vals = []
        for val in cat_cat_val_array[: , 2]:
            vals.append(val)
            c.append(value_to_color(val))

        #determinando o x, y e s
        x = []
        y = []
        cats_in_x = []
        cats_in_y = []
        for cat1, cat2, counts in cat_cat_val_array:
            
            #getting cats in x and y
            if cat1 not in cats_in_x:
                cats_in_x.append(cat1)

            if cat2 not in cats_in_y:
                cats_in_y.append(cat1)

            try:
                x_to_num[cat1]
                y_to_num[cat2]
                x.append(x_to_num[cat1])
                y.append(y_to_num[cat2])
        
            except KeyError:
                print(f'  Erro com os termos: {cat1} e {cat2}')
                print('  Considere eliminar esse termo usando o dicionário ~/Inputs/ngrams_to_remove.json')


        # Drawing the squares
        for xi, yi, ci in zip(x, y, c):
            # Create a rectangle that spans exactly one cell
            rect = patches.Rectangle(
                (xi - 0.5, yi - 0.5),   # lower-left corner
                1.0,                    # width
                1.0,                    # height
                facecolor=ci,    # or any color
                edgecolor="white",      
                linewidth=1
            )
            ax1.add_patch(rect)

        ax1.spines['top'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        ax1.spines['left'].set_position(('outward', 10))
        ax1.spines['bottom'].set_position(('outward', 10))

        ax1.xaxis.set_ticks_position('bottom')
        ax1.yaxis.set_ticks_position('left')

        ax1.tick_params(axis='x', which='both', bottom=True, labelsize= font_scale, length=6, width=3, color='black', direction='out')
        ax1.tick_params(axis='y', which='both', left=True, labelsize= font_scale, length = 6, width = 3, color='black', direction = 'out')

        ax1.set_title(plot_title + '\n', fontsize = font_scale)
        ax1.set_xticks([x_to_num[v] for v in x_tick_labels])
        ax1.set_yticks([y_to_num[v] for v in y_tick_labels])
                
        #replacing terms to plot
        counter = -1
        for label in x_tick_labels:
            counter += 1
            try:
                x_tick_labels[counter] = replace_terms_to_plot[label]
                print('  Replacing term to plot: ', label, '->', replace_terms_to_plot[label])
            except KeyError:
                pass
        
        counter = -1
        for label in y_tick_labels:
            counter += 1            
            try:
                y_tick_labels[counter] = replace_terms_to_plot[label]
                print('  Replacing term to plot: ', label, '->', replace_terms_to_plot[label])
            except KeyError:
                pass
                
        ha_aligment = 'right' if x_label_rotation != 90 else 'center'
        ax1.set_xticklabels(x_tick_labels, rotation=x_label_rotation, ha=ha_aligment)        
        ax1.set_yticklabels(y_tick_labels)
        
        #ax1.set_xticks( ax1.get_xticks() )
        #ax1.set_yticks( ax1.get_yticks() )

        ax1.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5]) 
        ax1.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])

        if x_label is not None:
            ax1.set_xlabel(x_label, labelpad = 20, fontdict={'fontsize': font_scale })
        if y_label is not None:
            ax1.set_ylabel(y_label, labelpad = 20, fontdict={'fontsize': font_scale })

        [ ax1.axhline(h_line + 0.5, linestyle='-', color='white') for h_line in list(y_to_num.values())[ : -1]] # horizontal lines
        [ ax1.axvline(v_line + 0.5, linestyle='-', color='white') for v_line in list(x_to_num.values())[ : -1]] # vertical lines
        #ax1.grid(which='minor', visible='True', color='white', linestyle='-', linewidth=1)
        ax1.set_facecolor('#ededed')

        #colocando os valores nas caixas
        if type(box_vals) == float and box_vals == 0.05:
            for i in range(len(vals)):
                if vals[i] < box_vals:
                    ax1.annotate('*', xy=(x[i] - x_annotation_delta * box_values_font_scale, y[i] - y_annotation_delta * box_values_font_scale),
                                 c=self.contrast_text_color_to_plot, ha='center', fontsize = box_values_font_scale, alpha = 0.7)

        elif box_vals is not None and len(box_vals) > 1:
            for i in range(len(box_vals)):
                ax1.annotate(str(box_vals[i]), xy=(x[i] - x_annotation_delta * box_values_font_scale, y[i] - y_annotation_delta * box_values_font_scale), 
                                 c='white', ha='center', fontsize = box_values_font_scale)
        
        #adicionando a barra lateral
        ax2 = plt.subplot(plot_grid[ : , - 1 :  ]) # Use the rightmost column of the plot

        bar_y=np.linspace(min_counts_val, max_counts_val, n_colors) # y coordinates for each of the n_colors bars

        bar_height = bar_y[1] - bar_y[0]
        ax2.barh(
            y=bar_y,
            width=2,
            left=0,
            height=bar_height,
            color=palette,
            linewidth=0
        )
        ax2.tick_params(axis='y', labelsize=font_scale)
        ax2.set_xlim(1, 2) # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
        ax2.grid(False) # Hide grid
        ax2.set_facecolor('white') # Make background white
        ax2.set_xticks([]) # Remove horizontal ticks
        ax2.set_yticks([max_counts_val, min_counts_val]) # Show vertical ticks for min, middle and max
        ax2.spines['top'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.yaxis.tick_right() # Show vertical ticks on the right

        if bar_label is not None:
            ax2.set_ylabel(bar_label, labelpad = 20, fontdict={'fontsize':font_scale})
            ax2.yaxis.set_label_position("right")
        
        #salvando a figura
        filename = get_tag_name(plot_counter, prefix = '')
        plt.savefig(self.diretorio + f'/Outputs/analyses/{filename}.png', dpi=200, bbox_inches='tight', pad_inches=0.5)
        print(f'  salvando a figura {filename}...')
        save_dic_to_json(self.diretorio + '/Outputs/analyses/dic_plotsetup.json', self.plotsetup)



    def box_plot(self, nums_grouped = None, cats_grouped = None, n_cats = None, n_points_per_box = None,
                 log_scale = False, y_lims = [], x_label = None, y_label = None, x_label_rotation = 70,
                 font_scale_boxplot = None, plot_counter = None, plot_margins = {}, plot_title = '',
                 replace_terms_to_plot = { 'term1' : 'term_to_replace'}, exclude_outliers = False):

        #plotando as figuras
        fig = plt.figure(figsize=(20, 15))
        plot_grid = gridspec.GridSpec(1, 15, figure = fig) 
        plot_grid.update(wspace=0.1, hspace=0.1, **plot_margins)
        
        #plotando a figura principal
        ax1 = fig.add_subplot(plot_grid[ : , : ])

        boxprops = dict(linestyle='-', linewidth=5, color='black')
        flierprops = dict(markersize=20, markeredgecolor='green')
        medianprops = dict(linestyle='-', linewidth=5, color='black')
        capprops = dict(linestyle='-', linewidth=5, color='black')
        whiskerprops = dict(linestyle='-', linewidth=5, color='black')
        
        box_positions = np.arange(1, len(nums_grouped)*2, 2).astype(int)
        bplot = ax1.boxplot(np.array(nums_grouped, dtype=object), 
                           positions = box_positions,
                           boxprops = boxprops, 
                           medianprops = medianprops, 
                           flierprops = flierprops, 
                           capprops = capprops, 
                           whiskerprops = whiskerprops,
                           vert=True,
                           patch_artist=True)

        n_colors = n_cats
        palette = sns.color_palette(self.color_palette, n_colors)

        # fill with colors
        for patch, color in zip(bplot['boxes'], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        #adicionando os valores em scatter points
        counter = 1
        for i in range(len(nums_grouped)):
            for num in nums_grouped[i]:
                if bplot['caps'][i * 2].get_ydata()[0] < num < bplot['caps'][i * 2  + 1].get_ydata()[0]:
                    ax1.scatter(x=np.random.uniform(low = counter - 0.2, high = counter + 0.2), y=num, s=50, c='gray', alpha=0.7)
                else:
                    ax1.scatter(x=counter, y=num, s=50, c='gray', alpha=0.7)
            counter += 2

        #getting the outliers
        nums_grouped_proc = []
        if exclude_outliers is True:
            outliers = [flier.get_ydata() for flier in bplot['fliers']]
            for i, values in enumerate(nums_grouped):
                nums_grouped_proc.append(values[~np.isin(values, outliers[i])])
            nums_grouped = nums_grouped_proc

        #histograms
        if log_scale is True:
            ax1.set_yscale('log')
        
        ax1.set_xlim([ -0.5 , (2 * n_cats) + 1.5])
        if y_lims is not None:
            ax1.set_ylim([y_lims[0], y_lims[1]])
        
        if x_label is not None:
            ax1.set_xlabel(x_label, fontdict={'fontsize':font_scale_boxplot})
        if y_label is not None:
            ax1.set_ylabel(y_label, labelpad = 20, fontdict={'fontsize': font_scale_boxplot})

        ax1.set_title(plot_title + '\n', fontsize=font_scale_boxplot)
        ax1.tick_params(axis='x', which='both', bottom=True, labelsize=font_scale_boxplot, width = 3, length = 10)
        ax1.tick_params(axis='y', which='both', left=True, labelsize=font_scale_boxplot, width = 3, length = 10)
        ha_aligment = 'right' if x_label_rotation != 90 else 'center'
        
        #replacing terms to plot on X-axis
        counter = -1
        cats_grouped_ = cats_grouped.copy()
        for label in cats_grouped_:
            counter += 1
            try:
                cats_grouped_[counter] = replace_terms_to_plot[label]
                print('  Replacing term to plot: ', label, '->', replace_terms_to_plot[label])
            except KeyError:
                continue

        ax1.set_xticklabels(cats_grouped_, rotation=x_label_rotation, ha=ha_aligment)
        ax1.yaxis.grid(True, linewidth = 2, c='gray', alpha=0.4)
        ax1.spines['top'].set_linewidth(2)
        ax1.spines['bottom'].set_linewidth(2)
        ax1.spines['left'].set_linewidth(2)
        ax1.spines['right'].set_linewidth(2)

        #histograms
        kde_positions = box_positions + 1
        for i, vals_group in enumerate(nums_grouped):

            hist_grid = np.linspace(vals_group.min(), vals_group.max(), 200)

            # Create a grid along the y-axis (covering the range of the data)
            if log_scale is True:
                data = np.log(vals_group.astype(float))
            else:
                data = vals_group.astype(float)

            #KDE
            kde = gaussian_kde(data)

            #KDE on the grid
            hist_grid = np.linspace(data.min(), data.max(), 200)
            density = (kde(hist_grid) / kde(hist_grid).max() ) * 0.7
            
            if log_scale is True:
                hist_grid = np.exp(hist_grid)

            # The base x-position for the KDE plot is given by kde_positions[i]
            base_x = kde_positions[i] - 0.4

            # Add the contour line along the right edge of the filled area
            ax1.plot(base_x + density, hist_grid, color='black', lw=2)
            ax1.fill_betweenx(hist_grid, base_x, base_x + density, facecolor=palette[i], alpha=0.7)

        #adicionando um segundo axes só para colocar os valores em x top
        ax2 = ax1.twiny()
        ax2.set_xlim([ -0.5 , (2 * n_cats) + 1.5])
        ax2.set_xticks(np.arange( 1, 2*(n_cats) + 1, 2))
        ax2.tick_params(axis='x', which='major', top=True, labelsize=font_scale_boxplot, width = 3, length = 10)
        ha_aligment = 'left' if x_label_rotation != 90 else 'center'
        ax2.set_xticklabels(n_points_per_box, rotation=x_label_rotation, ha=ha_aligment)

        #salvando a figura
        filename = get_tag_name(plot_counter, prefix = '')
        plt.savefig(self.diretorio + f'/Outputs/analyses/{filename}.png', dpi=200, bbox_inches='tight', pad_inches=0.5)
        print(f'  salvando a figura {filename}...')
        save_dic_to_json(self.diretorio + '/Outputs/analyses/dic_plotsetup.json', self.plotsetup)

        return nums_grouped



    def scatter_plot(self, ixy_all_array = None, cat_nums_dic = None, cats_len_dic = None, plot_counter = None, 
                     axis_mean_vals = None, log_scale_x = False, log_scale_y = False,
                     x_lims = [], y_lims = [], x_label = None, y_label = None, hist_bins_number = 20,
                     set_graph_region = None, plot_title = ''):
        

        fontsize = 40

        #plotando as figuras
        fig = plt.figure(figsize=(20, 15))
        plot_grid = gridspec.GridSpec(15, 18, figure = fig)
        plot_grid.update(wspace=0.2, hspace=0.2, left = 0.15, right = 0.95, top = 0.95, bottom = 0.2)
        
        #plotando a figura principal
        ax1 = fig.add_subplot(plot_grid[ 4 : , : 9 ])
        #plotando o histograma da direita
        ax2 = fig.add_subplot(plot_grid[ 4 : , 10 : 12 ])
        #plotando o histograma de cima
        ax3 = fig.add_subplot(plot_grid[ 1 : 3 , : 9 ])
        #plotando os nomes das cats
        ax4 = fig.add_subplot(plot_grid[ 4 : , 13 : ])

        if log_scale_x is True:
            ax1.set_xscale('log')
        if log_scale_y is True:
            ax1.set_yscale('log')

        if x_lims is not None:
            ax1.set_xlim([x_lims[0], x_lims[1]])        
        if y_lims is not None:
            ax1.set_ylim([y_lims[0], y_lims[1]])
        
        if x_label is not None:
            ax1.set_xlabel(x_label, labelpad = 20, fontsize = fontsize)
        if y_label is not None:
            ax1.set_ylabel(y_label, labelpad = 20, fontsize = fontsize)

        ax1.tick_params(axis='x', which='both', bottom=True, labelsize=fontsize, width = 3, length = 10)
        ax1.tick_params(axis='y', which='both', left=True, labelsize=fontsize, width = 3, length = 10)

        ax1.yaxis.grid(True, linewidth = 2, c='gray', alpha=0.2)
        ax1.xaxis.grid(True, linewidth = 2, c='gray', alpha=0.2)
        ax1.spines['top'].set_linewidth(2)
        ax1.spines['bottom'].set_linewidth(2)
        ax1.spines['left'].set_linewidth(2)
        ax1.spines['right'].set_linewidth(2)

        #histograma da direita
        if y_lims is not None:
            ax2.set_ylim([y_lims[0], y_lims[1]])        
        if log_scale_y is True:
            ax2.set_yscale('log')

        ax2.xaxis.set_tick_params(labelbottom=False)
        ax2.yaxis.set_tick_params(labelleft=False)
        ax2.set_xticks([])
        ax2.set_yticks([])

        ax2.spines['top'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_linewidth(2)
        ax2.spines['right'].set_visible(False)

        #histograma de cima
        ax3.set_title(plot_title + '\n', fontsize = fontsize)
        if x_lims is not None:
            ax3.set_xlim([x_lims[0], x_lims[1]])        
        if log_scale_x is True:
            ax3.set_xscale('log')

        ax3.xaxis.set_tick_params(labelbottom=False)
        ax3.yaxis.set_tick_params(labelleft=False)
        ax3.set_xticks([])
        ax3.set_yticks([])

        ax3.spines['top'].set_visible(False)
        ax3.spines['bottom'].set_linewidth(2)
        ax3.spines['left'].set_visible(False)
        ax3.spines['right'].set_visible(False)

        ax4.set_xlim([0, 10])
        ax4.set_ylim([0, 10]) 
        ax4.xaxis.set_tick_params(labelbottom=False)
        ax4.yaxis.set_tick_params(labelleft=False)
        ax4.set_xticks([])
        ax4.set_yticks([])

        ax4.spines['top'].set_visible(False)
        ax4.spines['bottom'].set_visible(False)
        ax4.spines['left'].set_visible(False)
        ax4.spines['right'].set_visible(False)

        #caso tudo seja plotado
        if cat_nums_dic is None:
            
            #scatter
            ax1.scatter(ixy_all_array[:, 1], ixy_all_array[:, 2], c='chocolate', s=100)

            #top histogram
            if log_scale_x is True:
                bins_x = np.logspace(math.log10(ixy_all_array[: , 1].min()), math.log10(ixy_all_array[: , 1].max()), hist_bins_number)
            else:
                bins_x = np.linspace(ixy_all_array[: , 1].min(), ixy_all_array[: , 1].max(), hist_bins_number)

            ax3.hist(ixy_all_array[: , 1], bins = bins_x, orientation='vertical', color='sandybrown', alpha = 0.7)

            #right histogram
            if log_scale_y is True:
                bins_y = np.logspace(math.log10(ixy_all_array[: , 2].min()), math.log10(ixy_all_array[: , 2].max()), hist_bins_number)
            else:
                bins_y = np.linspace(ixy_all_array[: , 2].min(), ixy_all_array[: , 2].max(), hist_bins_number)
            
            ax2.hist(ixy_all_array[: , 2], bins = bins_y, orientation='horizontal', color='sandybrown', alpha = 0.7)

        #caso seja dividido por categoria
        else:
            
            n_colors = len(cat_nums_dic.keys())
            palette = sns.color_palette(self.color_palette, n_colors)

            i = 0
            for cat in cat_nums_dic.keys():
                
                #scatter
                ax1.scatter( cat_nums_dic[cat][:, 1], cat_nums_dic[cat][:, 2], c=[palette[i]], s=100)

                #diminuindo a altura dos histogramas
                ax3.set_ylim([0, 1.5])
                ax2.set_xlim([0, 1.5])

                #top histogram
                if log_scale_x is True:
                    bins_x = np.logspace(math.log10(ixy_all_array[: , 1].min()), math.log10(ixy_all_array[: , 1].max()), hist_bins_number)
                else:
                    bins_x = np.linspace(ixy_all_array[: , 1].min(), ixy_all_array[: , 1].max(), hist_bins_number)

                #ax3.hist(cat_nums_dic[cat][:, 1], bins = bins_x, orientation = 'vertical', density = normalize_histograms, color=[palette[i]], alpha = 0.7)

                counts, edges = np.histogram(cat_nums_dic[cat][:, 1], bins = bins_x)                
                counts = counts / counts.max()

                #caso haja diferença estatística entre duas distribuição
                try: 
                    ax3.axvline(x=axis_mean_vals['x'][0], color='black', linestyle='-', linewidth=7, alpha=0.7)
                    ax3.axvline(x=axis_mean_vals['x'][1], color='black', linestyle='-', linewidth=7, alpha=0.7)
                except:
                    pass

                ax3.bar(edges[:-1], counts, width=np.diff(edges), color=[palette[i]], alpha = 0.6)


                #right histogram
                if log_scale_y is True:
                    bins_y = np.logspace(math.log10(ixy_all_array[: , 2].min()), math.log10(ixy_all_array[: , 2].max()), hist_bins_number)
                else:
                    bins_y = np.linspace(ixy_all_array[: , 2].min(), ixy_all_array[: , 2].max(), hist_bins_number)
                
                #ax2.hist(cat_nums_dic[cat][:, 2], bins = bins_y, orientation = 'horizontal', density = normalize_histograms, color=[palette[i]], alpha = 0.7)
                
                counts, edges = np.histogram(cat_nums_dic[cat][:, 2], bins = bins_y)
                counts = counts / counts.max()

                #caso haja diferença estatística entre duas distribuição
                try: 
                    ax2.axhline(y=axis_mean_vals['y'][0], color='black', linestyle='-', linewidth=7, alpha=0.7)
                    ax2.axhline(y=axis_mean_vals['y'][1], color='black', linestyle='-', linewidth=7, alpha=0.7)
                except:
                    pass

                ax2.barh(edges[:-1], counts, height=np.diff(edges), color=[palette[i]], alpha = 0.6)


                ax4.annotate(f'{cat} ({str(cats_len_dic[cat])})', xy=(0, 10 - i), c=palette[i], ha='left', fontsize = fontsize)
                
                i += 1

        #marcando e exportando indexes da região
        if set_graph_region is not None:
            for g in set_graph_region.keys():
                index_list = []
                x_f = []
                y_f = []
                for i, x, y in ixy_all_array:
                    cond1 = set_graph_region[g][0][0] <= x <= set_graph_region[g][0][1]
                    cond2 = set_graph_region[g][1][0] <= y <= set_graph_region[g][1][1]
                    if False not in (cond1, cond2):
                        x_f.append(x)
                        y_f.append(y)
                        index_list.append(i)

                #anotando os pontos
                ax1.scatter( x_f, y_f, edgecolors = 'black', facecolors='none', s=500)
                ax1.vlines( set_graph_region[g][0][0], ymin=set_graph_region[g][1][0], ymax=set_graph_region[g][1][1], colors='black', linewidth=2)
                ax1.vlines( set_graph_region[g][0][1], ymin=set_graph_region[g][1][0], ymax=set_graph_region[g][1][1], colors='black', linewidth=2)
                ax1.hlines( set_graph_region[g][1][0], xmin=set_graph_region[g][0][0], xmax=set_graph_region[g][0][1], colors='black', linewidth=2)
                ax1.hlines( set_graph_region[g][1][1], xmin=set_graph_region[g][0][0], xmax=set_graph_region[g][0][1], colors='black', linewidth=2)
                
                '''Não está sendo usada a anotação no gráfico
                delta_pos_x = x_lims[1] - x_lims[0]
                delta_pos_y = y_lims[1] - y_lims[0]
                print(set_graph_region[g][1][0], (0.2 * delta_pos_y))
                if set_graph_region[g][2] == 'top':
                    xy_pos = ( (set_graph_region[g][0][0] + set_graph_region[g][0][1]) / 2 , set_graph_region[g][1][1] + (0.1 * delta_pos_y) )
                    ha = 'center'
                elif set_graph_region[g][2] == 'bottom':
                    xy_pos = ( (set_graph_region[g][0][0] + set_graph_region[g][0][1]) / 2 , set_graph_region[g][1][0] - (0.1 * delta_pos_y) )
                    ha = 'center'
                elif set_graph_region[g][2] == 'right':
                    xy_pos = ( set_graph_region[g][0][1] + (0.1 * delta_pos_x), (set_graph_region[g][1][0] + set_graph_region[g][1][1]) / 2 )
                    ha = 'left'
                elif set_graph_region[g][2] == 'left':
                    xy_pos = ( set_graph_region[g][0][0] - (0.1 * delta_pos_x), (set_graph_region[g][1][0] + set_graph_region[g][1][1]) / 2 )
                    ha = 'right'
                
                ax1.annotate(g, xy=xy_pos, c='black', ha=ha, fontsize = 35)'''
                
                #checando o dicionário com as index_lists dos grupos
                if not os.path.exists(self.diretorio + '/Outputs/log/index_lists.json'):
                    index_lists = dict()
                else:
                    index_lists = load_dic_from_json(self.diretorio + '/Outputs/log/index_lists.json')
        
                index_lists[g] = index_list
                save_dic_to_json(self.diretorio + '/Outputs/log/index_lists.json', index_lists)

        #salvando a figura
        filename = get_tag_name(plot_counter, prefix = '')
        plt.savefig(self.diretorio + f'/Outputs/analyses/{filename}.png', dpi=200, bbox_inches='tight', pad_inches=0.5)
        print(f'  salvando a figura {filename}...')
        save_dic_to_json(self.diretorio + '/Outputs/analyses/dic_plotsetup.json', self.plotsetup)



    def axis_scatter_plots(self, col_cat_vals_len_dic, 
                            log_scale = False, y_lims = [], 
                            y_label = None, x_label_rotation = 70, 
                            font_scale_axis_scatter = 10, 
                            y_bin_val = 10, offset_set = 0.1, plotcircle_size = 1,
                            plot_counter = None, g1_label = '', g2_label = '',
                            plot_title = '', plot_plot_margins = {'left':0.05, 'right':0.05, 'top':0.05, 'bottom':0.05},
                            x_labels_to_replace = {'': ''},
                            representation_mode = 'random_spread'):

        g1_span = 4
        g2_span = 4

        #plotando as figuras
        fig = plt.figure(figsize=(35, 15))
        plot_grid = gridspec.GridSpec(40, len(col_cat_vals_len_dic) * (g1_span + 1) + len(col_cat_vals_len_dic) * (g2_span + 1), 
                                      figure = fig) 
        plot_grid.update(wspace=0.05, hspace=0.05, **plot_plot_margins)

        
        #gerando as tuplas para plot
        def gen_gridplot_intervals(g1_span: int = 1, g2_span: int = 2, vals_len: int = 10):
        
            start = 0
            intervals = []
            for _ in range(vals_len):
                end_g1 = g1_span + start
                end_g2 = end_g1 + g2_span + 1
                intervals.append( ( (start, end_g1) , (end_g1 + 1, end_g2) ) )
                start = end_g2 + 1
            
            return intervals

        plots = {}
        counter_columns = 0
        for (g1_i, g1_f), (g2_i, g2_f) in gen_gridplot_intervals(g1_span = g1_span, g2_span = g2_span, 
                                                                 vals_len = len(col_cat_vals_len_dic)):
            #plotando a figura principal
            plots[str(counter_columns) + 'b1'] = fig.add_subplot(plot_grid[ 1 : 3 , g1_i : g1_f ])
            plots[str(counter_columns) + 'b2'] = fig.add_subplot(plot_grid[ 1 : 3 , g2_i : g2_f ])
            plots[str(counter_columns) + 'g1'] = fig.add_subplot(plot_grid[ 4 : 40 , g1_i : g1_f ])
            plots[str(counter_columns) + 'g2'] = fig.add_subplot(plot_grid[ 4 : 40 , g2_i : g2_f ])
            counter_columns += 1


        counter_columns = 0
        for col in col_cat_vals_len_dic.keys():
            
            for var, graph_n, definition in [('cat', '1', g1_label), ('n_vals', '2', g2_label)]:
                
                #plotando os gráficos
                if var == 'cat':
                    cats = sorted(set([ cat for cat, vals, n_vals in col_cat_vals_len_dic[col] ]))
                    cat_i = dict([ (cat, i) for i, cat in enumerate(cats) ])
                    i_s = list(cat_i.values())
                elif var == 'n_vals':
                    i_s = sorted(set([ n_vals for cat, vals, n_vals in col_cat_vals_len_dic[col] ]))

                #associando uma cor a cada cat
                n_colors = 256
                palette = sns.color_palette(self.color_palette, n_colors=n_colors)
                min_val, max_val = [ min(i_s) , max(i_s) ]

                def value_to_color(val):
                    try:
                        val_position = float((val - min_val)) / (max_val - min_val) # position of value in the input range, relative to the length of the input range
                        ind = int(val_position * (n_colors - 1)) # target index in the color palette
                        return palette[ind]
                    #caso só haja um valor para associar à cor
                    except ZeroDivisionError:
                        if val == 1:
                            return palette[-1]
                        elif val == 0:
                            return palette[0]
            
                x = []
                y = []
                c = []
                for cat, vals, n_vals in col_cat_vals_len_dic[col]:
                    
                    #y
                    bin_vals = np.floor_divide(vals, y_bin_val) * y_bin_val #bin y values to facilitate visualization
                    y.extend( list(bin_vals) )

                    #c                    
                    if var == 'cat':
                        c.extend( [value_to_color(cat_i[cat])] * len(vals) )
                    elif var == 'n_vals':                    
                        c.extend( [value_to_color(n_vals)] * len(vals) )

                #x
                if representation_mode == 'random_spread':
                    max_offset = 0.4
                    x =  list(np.random.uniform(low = - max_offset, high = + max_offset, size = len(y)) )
                
                elif representation_mode == 'bin_swarm_spread':
                    # Prepare x-offsets: spread identical y-values evenly
                    unique_y = np.unique(y)
                    x = np.zeros_like(y, dtype=float)

                    for yi in unique_y:
                        
                        idx = np.where(y == yi)[0]
                        n = len(idx)
                        if n > 1:
                            # offsets: … −2·step, −step, 0, +step, +2·step, …
                            offsets = (np.arange(n) - (n-1)/2) * offset_set
                        else:
                            offsets = np.array([0])

                        x[idx] = 1 + offsets
                        
                plots[str(counter_columns) + f'g{graph_n}'].scatter(x = x, y = y, s=plotcircle_size,
                                                                    color = c, alpha=0.7)

                if y_lims is not None:
                    plots[str(counter_columns) + f'g{graph_n}'].set_ylim([y_lims[0], y_lims[1]])

                if log_scale is True:
                    plots[str(counter_columns) + f'g{graph_n}'].set_yscale('log')

                plots[str(counter_columns) + f'g{graph_n}'].yaxis.grid(True, linewidth = 2, c='gray', alpha=0.4)

                try:
                    x_label = x_labels_to_replace[col]
                except KeyError:
                    x_label = col
                
                ha_aligment = 'right' if x_label_rotation != 90 else 'center'
                plots[str(counter_columns) + f'g{graph_n}'].set_xlabel(x_label,
                                                                       fontsize = font_scale_axis_scatter,
                                                                       rotation = x_label_rotation,
                                                                       labelpad = 20,
                                                                       ha = ha_aligment)
                

                #plotando as barras
                bar_x=np.linspace(min_val, max_val, n_colors) # y coordinates for each of the n_colors bars

                bar_width = bar_x[1] - bar_x[0]
                plots[str(counter_columns) + f'b{graph_n}'].bar(x=bar_x,
                                                                height = 2,
                                                                bottom = 0,
                                                                width=bar_width,
                                                                color=palette,
                                                                linewidth=0)
                
                plots[str(counter_columns) + f'b{graph_n}'].tick_params(axis='x', labelsize=font_scale_axis_scatter)
                plots[str(counter_columns) + f'b{graph_n}'].set_ylim(0, 2) # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
                plots[str(counter_columns) + f'b{graph_n}'].grid(False) # Hide grid
                plots[str(counter_columns) + f'b{graph_n}'].set_facecolor('white') # Make background white
                plots[str(counter_columns) + f'b{graph_n}'].set_yticks([]) # Remove horizontal ticks
                plots[str(counter_columns) + f'b{graph_n}'].set_xticks([min_val, max_val]) # Show vertical ticks for min, middle and max
                plots[str(counter_columns) + f'b{graph_n}'].spines['top'].set_visible(False)
                plots[str(counter_columns) + f'b{graph_n}'].spines['bottom'].set_visible(False)
                plots[str(counter_columns) + f'b{graph_n}'].spines['left'].set_visible(False)
                plots[str(counter_columns) + f'b{graph_n}'].spines['right'].set_visible(False)
                plots[str(counter_columns) + f'b{graph_n}'].xaxis.tick_top() # Show vertical ticks on the right
                plots[str(counter_columns) + f'b{graph_n}'].set_xlabel(definition, labelpad = 20, fontdict={'fontsize':font_scale_axis_scatter})
                plots[str(counter_columns) + f'b{graph_n}'].xaxis.set_label_position("top")                

                #ajustando os contornos
                if counter_columns == 0 and graph_n == '1':
                    plots[str(counter_columns) + f'g{graph_n}'].spines['top'].set_visible(False)
                    plots[str(counter_columns) + f'g{graph_n}'].spines['bottom'].set_visible(False)
                    plots[str(counter_columns) + f'g{graph_n}'].spines['left'].set_visible(False)
                    plots[str(counter_columns) + f'g{graph_n}'].spines['right'].set_visible(False)
                    
                    plots[str(counter_columns) + f'g{graph_n}'].set_xticks([])
                    plots[str(counter_columns) + f'g{graph_n}'].tick_params(axis='y', which='both', left = True, 
                                                                            labelsize = font_scale_axis_scatter, 
                                                                            length = 1, width = 1, color='white', 
                                                                            direction = 'out')
                    
                    if y_label is not None:
                        plots[str(counter_columns) + f'g{graph_n}'].set_ylabel(y_label, labelpad = 20, 
                                                                               fontsize = font_scale_axis_scatter)
                            
                else:
                    plots[str(counter_columns) + f'g{graph_n}'].spines['top'].set_visible(False)
                    plots[str(counter_columns) + f'g{graph_n}'].spines['bottom'].set_visible(False)
                    plots[str(counter_columns) + f'g{graph_n}'].spines['left'].set_visible(False)
                    plots[str(counter_columns) + f'g{graph_n}'].spines['right'].set_visible(False)
                    plots[str(counter_columns) + f'g{graph_n}'].set_xticks([])
                    plots[str(counter_columns) + f'g{graph_n}'].tick_params(which='both',
                                                                            bottom=False,      # remove ticks on the bottom
                                                                            top=False,         # remove ticks on the top
                                                                            left=False,        # remove ticks on the left
                                                                            right=False,       # remove ticks on the right
                                                                            labelbottom=False,  # keep the labels if you want them
                                                                            labelleft=False)

            counter_columns +=1


        
        #salvando a figura
        filename = get_tag_name(plot_counter, prefix = '')
        plt.savefig(self.diretorio + f'/Outputs/analyses/{filename}.png', dpi=200, bbox_inches='tight', pad_inches=0.5)
        print(f'  salvando a figura {filename}...')
        save_dic_to_json(self.diretorio + '/Outputs/analyses/dic_plotsetup.json', self.plotsetup)



def check_columns_in_DF(columns_to_check, DF):
    
    #checando se todas as cats estão no DF
    all_columns_check = True
    
    for column in columns_to_check:
        if column in DF.columns:
            continue
        else:
            all_columns_check = False
            break

    return all_columns_check



def remove_duplicates(df, columns_to_consider = None):

    len_df_before = len(df.index)
    df = df.reset_index().drop_duplicates(subset=['Filename'] + columns_to_consider).set_index(['Filename', 'Counter'])
    print(f'  eliminando duplicatas: {len_df_before - len(df.index)} instâncias removidas.')
    
    return df


def remove_nan(df):

    #Eliminando os NaN
    len_df_before = len(df.index)
    df = df.dropna(axis = 0, how = 'any')
    print(f'  eliminando NaN: {len_df_before - len(df.index)} instâncias removidas.')

    return df



def filter_categories(df, columns_checked_in_DF : bool, cats_terms_to_filter : dict):
        
    #filtrando categorias
    cats_text = ''
    if columns_checked_in_DF is True:
        for cat in cats_terms_to_filter.keys():
            if type(cats_terms_to_filter[cat]) == list and len(cats_terms_to_filter[cat]) > 0:
                groups_index_dic = df.groupby(by=cat).groups
                index_group_list = []
                for group in groups_index_dic:
                    if group in cats_terms_to_filter[cat]:
                        index_group_list.extend( groups_index_dic[group] )
                        cats_text += group + ', '
                
                df = df.loc[ index_group_list ]
        
        cats_text = cats_text[ : -2]
        #print('\nDF pós filtro:')
        #print(self.results_DF_processed)

    return df, cats_text





    '''def cutting_outliers_in_DF_column(self, column = 'a', quantiles = [0, 1]):
        
        temp_series = self.results_DF_processed[column].dropna(axis=0, how='any')
        
        #definindo os limites
        q1, q2 = quantiles
        
        try:
            if type(q1) is str and q1.lower() == 'na':
                min_val , max_val = 0 , np.quantile( temp_series.values, q2)
            else:
                min_val , max_val  = np.quantile( temp_series.values, q1) , np.quantile( temp_series.values, q2)
            
            def cut_outliers(entry):
                if min_val <= entry <= max_val:
                    return entry
                else:
                    return None
                           
            final_Series = temp_series.map(cut_outliers)
            final_Series.dropna(how = 'any', inplace=True)
            
            return final_Series
        
        #caso a coluna numérica não tenha sido processada
        except TypeError:
            print('ERRO!')
            print('Os valores numéricos da coluna não estão corretos.')
            print('Provavelmente a função "process_columns_with_num_val" não foi usada.')
            return




    def multiply_column_by_factor(self, DF_columns=['a', 'b'], factor = 1):                        
    
        def multiply_f(entry):
            try:
                return entry * factor
            except TypeError:
                return entry
        
        for column in DF_columns:
            self.results_DF_processed[column] = self.results_DF_processed[column].apply(multiply_f)
        




    def process_columns_with_num_val(self, DF_columns=['a', 'b'], mode = 'mean', print_statics = False):
        
        
        
        #copiando os dados da DF raw
        results_DF_copy = self.results_DF.copy()
        
        #checando se a coluna está no DF de reultados        
        print('\nfunction: process_columns_with_num_val')
        for column in DF_columns:
            if column not in self.results_DF_processed.columns:
                print(f'Erro! A coluna "{column}" não existe na DF de resultados.')
                return
            else:                
                print(' Processing column: ', column)
                print('Number of vals in this column: ', len(self.results_DF_processed[column].dropna().values))
            
        def avg_intervals(entry):
            try:
                float(entry)
                return round(float(entry), 10)
            except ValueError:
                valores = re.findall(r'[\-\w\.]+', entry)
                avg_val = round( sum( [float(val) for val in valores] ) / len(valores) , 10)
                
                return avg_val
        
        def higher(entry):
            try:
                float(entry)
                return round(float(entry), 10)
            except ValueError:
                valores = re.findall(r'[\-\w\.]+', entry)
                higher_val = round( max( [float(val) for val in valores] ), 10 )
                return higher_val
            
        def lower(entry):
            try:
                float(entry)
                return round(float(entry), 10)
            except ValueError:
                valores = re.findall(r'[\-\w\.]+', entry)
                lower_val = round( min( [float(val) for val in valores] ), 10 )
                return lower_val
            
        #varrendo a DF
        if mode == 'mean':
            for column in DF_columns:
                self.results_DF_processed[column] = results_DF_copy[column].map(avg_intervals)

                #imprimir a estatística da coluna
                if print_statics is True:
                    
                    filtered_series = self.cutting_outliers_in_DF_column(column = column, quantiles = [0.05, 0.95])
                    mean_column = filtered_series.dropna().values.mean()
                    std_column = filtered_series.dropna().values.std()
                    min_column = filtered_series.dropna().values.min()
                    max_column = filtered_series.dropna().values.max()
                    median_column = np.median(filtered_series.dropna().values)
                    q1_column = np.quantile(filtered_series.dropna().values, 0.25)
                    q3_column = np.quantile(filtered_series.dropna().values, 0.75)
                    print('\nColumn statistic')
                    print('Column: ', column)
                    print('mean value: ', mean_column)
                    print('std value: ', std_column)
                    print('min value: ', min_column)
                    print('max value: ', max_column)
                    print('median value: ', median_column)
                    print('q1 value: ', q1_column)
                    print('q3 value: ', q3_column)
                
        elif mode == 'higher':
            for column in DF_columns:
                self.results_DF_processed[column] = results_DF_copy[column].map(higher)

                #imprimir a estatística da coluna
                if print_statics is True:                
                    
                    filtered_series = self.cutting_outliers_in_DF_column(column = column, quantiles = [0.05, 0.95])
                    mean_column = filtered_series.dropna().values.mean()
                    std_column = filtered_series.dropna().values.std()
                    min_column = filtered_series.dropna().values.min()
                    max_column = filtered_series.dropna().values.max()
                    median_column = np.median(filtered_series.dropna().values)
                    q1_column = np.quantile(filtered_series.dropna().values, 0.25)
                    q3_column = np.quantile(filtered_series.dropna().values, 0.75)
                    print('\nColumn statistic')
                    print('Column: ', column)
                    print('mean value: ', mean_column)
                    print('std value: ', std_column)
                    print('min value: ', min_column)
                    print('max value: ', max_column)
                    print('median value: ', median_column)
                    print('q1 value: ', q1_column)
                    print('q3 value: ', q3_column)
        
        elif mode == 'lower':
            for column in DF_columns:
                self.results_DF_processed[column] = results_DF_copy[column].map(lower)
                
                #imprimir a estatística da coluna
                if print_statics is True:                
                    
                    filtered_series = self.cutting_outliers_in_DF_column(column = column, quantiles = [0.05, 0.95])
                    mean_column = filtered_series.dropna().values.mean()
                    std_column = filtered_series.dropna().values.std()
                    min_column = filtered_series.dropna().values.min()
                    max_column = filtered_series.dropna().values.max()
                    median_column = np.median(filtered_series.dropna().values)
                    q1_column = np.quantile(filtered_series.dropna().values, 0.25)
                    q3_column = np.quantile(filtered_series.dropna().values, 0.75)
                    print('\nColumn statistic')
                    print('Column: ', column)
                    print('mean value: ', mean_column)
                    print('std value: ', std_column)
                    print('min value: ', min_column)
                    print('max value: ', max_column)
                    print('median value: ', median_column)
                    print('q1 value: ', q1_column)
                    print('q3 value: ', q3_column)



    def merge_columns_with_num_val(self, base_column = None, column_to_merge = None, invert_column_val = False):                

        #função para inverter os valores da coluna (exemplo C/O -> O/C)        
        def invert_column_vals(entry):            
            return 1/entry
        
        if invert_column_val is True:
            modified_df = self.results_DF_processed[[column_to_merge]].apply(invert_column_vals)
        else:
            modified_df = self.results_DF_processed[[column_to_merge]].copy()
        
        #substituir o nome da coluna
        modified_df.rename(columns={column_to_merge:base_column}, inplace=True)
        #combinar os valores das colunas        
        self.results_DF_processed = self.results_DF_processed.combine_first(modified_df)
        
        #eliminando a coluna que sofreu o merge
        self.results_DF_processed = self.results_DF_processed.drop(columns=[column_to_merge])



    def merge_columns_with_cat_val(self, base_column = None, column_to_merge = None):    

        s_base = self.results_DF_processed[[base_column]].copy()
        s_to_merge = self.results_DF_processed[[column_to_merge]].copy()
        
        #substituir o nome da coluna
        s_to_merge.rename(columns={column_to_merge:base_column}, inplace=True)
        
        #mudando a DF
        self.results_DF_processed[base_column] = s_base.combine_first(s_to_merge)



    def split_2grams_terms(self, DF_column = 'column1'):
        
        def separate_2gram_t1(entry):
            if len(re.findall(r'(\w+(?=\s)|(?<=\s)\w+)', entry)) == 2:
                term1, term2 = re.findall(r'(\w+(?=\s)|(?<=\s)\w+)', entry)
            return term1
        
        def separate_2gram_t2(entry):
            if len(re.findall(r'(\w+(?=\s)|(?<=\s)\w+)', entry)) == 2:
                term1, term2 = re.findall(r'(\w+(?=\s)|(?<=\s)\w+)', entry)
            return term2
        
        #checando se a coluna está no DF de reultados
        if DF_column not in self.results_DF_processed.columns:
            print(f'Erro! A coluna "{DF_column}" não existe na DF de resultados.')
            return
        
        temp_DF = self.results_DF_processed.copy()

        #limpando os NANs
        indexes_to_filter = temp_DF[DF_column].dropna().index
        temp_DF = temp_DF.loc[indexes_to_filter]
                
        #coletando todos os termos da coluna
        series_t1 = temp_DF[DF_column].str.lower().map(separate_2gram_t1)
        series_t2 = temp_DF[DF_column].str.lower().map(separate_2gram_t2)
        series_t1.name = f'{DF_column}_0'
        series_t2.name = f'{DF_column}_1'

        #testando se foi processado algum valor
        if len(series_t1.values) == 0:
            print(f'Erro! Não existem 2grams na coluna "{DF_column}".')
            return
                
        DF_2grams_splitted = pd.DataFrame([], index=series_t1.index)
        DF_2grams_splitted = DF_2grams_splitted.join(series_t1)
        DF_2grams_splitted = DF_2grams_splitted.join(series_t2)
        self.results_DF_processed = temp_DF.join(DF_2grams_splitted)



    def add_split_index_names_to_column(self, DF):

        

        #resetando o index
        DF.reset_index(inplace=True)

        #nome da coluna com os termos concatenados
        concat_terms_column_name = DF.columns.values[0][0]
        
        new_index_names = [ concat_terms_column_name ] + [multi_index[1] for multi_index in DF.columns.values[1:]]
        DF.columns = new_index_names
        
        def separate_2gram_t1(entry):
            if len(re.findall(r'([\w\s]+(?=\+)|(?<=\+)[\w\s]+)', entry)) == 2:
                term1, term2 = re.findall(r'([\w\s]+(?=\+)|(?<=\+)[\w\s]+)', entry)
            return term1
        
        def separate_2gram_t2(entry):
            if len(re.findall(r'([\w\s]+(?=\+)|(?<=\+)[\w\s]+)', entry)) == 2:
                term1, term2 = re.findall(r'([\w\s]+(?=\+)|(?<=\+)[\w\s]+)', entry)
            return term2

        #adicionando as novas colunas
        column1 = separate_2gram_t1(concat_terms_column_name)
        column2 = separate_2gram_t2(concat_terms_column_name)
        DF[column1] = DF[concat_terms_column_name].apply(separate_2gram_t1)
        DF[column2] = DF[concat_terms_column_name].apply(separate_2gram_t2)
        
        return DF



    def filter_groups_by_list(self, DF = None, grouped = None, filter_type = 'grouped_vals', group_list = []):

        
        
        if filter_type.lower() == 'grouped_vals' and type(group_list) == list:
            for group in grouped.groups:
                #mínima ocrrência por grupo
                if group not in group_list:
                    #print('group: ', group)
                    #print(grouped.get_group(group))
                    #print('len: ', len(grouped.get_group(group).values))
                    #print(f'group "{group}" excluido! ( < {min_occurrences})')
                    #time.sleep(3)
                    DF = DF.drop( grouped.get_group(group).index )
                else:
                    #print(f'group "{group}" coletado! (len = {len(grouped.get_group(group).values)})')
                    #time.sleep(3)
                    pass
        
        elif filter_type.lower() == 'grouped_filenames' and type(group_list) == list:
            #reseting index
            DF = DF.reset_index().set_index('doc')
            #filtrando os indexes
            DF = DF.loc[np.unique(DF.index.intersection(group_list).values)]
            #reseting index
            DF = DF.reset_index().set_index(['doc', 'counter'])

        return DF            



    def group_cat_columns_with_input_classes(self, DF_column = 'column1', groups_name = 'classes'):

        #
        
        
        
        print('\n\n> function: group_cat_columns_with_input_classes')
        
        group_dic = load_dic_from_json(self.diretorio + f'/Inputs/{groups_name}.json')

        def add_group(entry):
            #print(entry)
            #print(type(entry))
            #print('entry == nan', entry == 'nan')
            #print('entry == np.nan', entry == np.nan)
            #time.sleep(1)
            group_name_found = False
            for group_name in group_dic.keys():
                group_members = group_dic[group_name]
                for member in group_members:
                    try:
                        if member in entry:
                            group_name_found = True
                            group = group_name
                    except TypeError:
                        pass
            
            if group_name_found is True:                
                return group
            elif type(entry) == float:
                return np.nan
            else:
                return 'other'        
        
        temp_DF = self.results_DF_processed.copy()
        
        unique_vals = np.unique(temp_DF[DF_column].dropna().values)
        print('Unique groups encontrados para a coluna ', DF_column, ' - total : ', len(unique_vals))
        print(unique_vals)
        
        #criando a DF de grupo
        series_group = temp_DF[DF_column].str.lower().map(add_group)
        series_group.name = groups_name
        
        #concatenando com a DF de resultados
        self.results_DF_processed = temp_DF.join(series_group)




    def plot_pca_results(self, n_components = 2, DF_columns=['a', 'b'], loadings_label = ['','',''], axes_labels = ['X', 'Y'],
                         quantiles=['NA', 'NA'], x_min = None, x_max = None, y_min = None, y_max = None, plot_width=1000, plot_height=1000,
                         find_clusters = True, n_clusters = 3, loading_arrow_factor = 10, export_groups_to_csv = False, cluster_preffix='',
                         show_figure = False):
        
        print('\n\n> function: pca')
        print('Columns used: ', DF_columns)

        #checando se as colunas estão no DF de reultados
        for column in DF_columns:
            if column not in self.results_DF_processed.columns:
                print(f'Erro! A coluna "{column}" não existe na DF de resultados.')
                return
        #checando se as entradas de colunas são compatíveis
        if len(DF_columns) != len(loadings_label):
            print('Erro! O número de entradas do DF_columns deve ser igual ao do axes_labels')
            return

        #eliminando os outliers        
        DF_copy = pd.DataFrame([], index=self.results_DF_processed.index)
        for column in DF_columns:
            q1 , q2 = quantiles
            DF_copy = DF_copy.join( self.cutting_outliers_in_DF_column(column = column, quantiles = [q1, q2]) )
        
        DF_copy = DF_copy.dropna(axis=0, how='any')
        
        #limpando as duplicatas de index (doc)
        DF_copy = self.clear_duplicated_index(DF_copy, DF_indexes_names = ['doc', 'counter'])

        #scaling
        for column in DF_copy:
            DF_copy[column] = DF_copy[column].values - DF_copy[column].values.mean()
            DF_copy[column] = DF_copy[column].values / DF_copy[column].values.std()
        
        X = DF_copy.values
        
        #fazendo o PCA (a centralização da matriz é feita no pca)
        model = PCA(n_components=2)
        model.fit(X)
        reduced_X = model.fit_transform(X)
        var_ratio = model.explained_variance_ratio_
        
        #adicionando a variancia aos eixos
        axes_labels = axes_labels[0] + ' (' + str( round(var_ratio[0], 2) ) + ')' , axes_labels[1] + ' (' + str( round(var_ratio[1], 2) ) + ')'
        
        #convertendo para DF
        DF_pca = pd.DataFrame(reduced_X, columns=['PC1', 'PC2'], index=DF_copy.index)

        #plotando        
        self.pca_scatter_plot(DF = DF_pca, pca_model = model, loadings_label=loadings_label, 
                              axes_labels = axes_labels, input1 = 'PC1', input2 = 'PC2',
                              x_min = x_min, x_max = x_max, y_min = y_min, y_max = y_max,
                              plot_width=plot_width, plot_height=plot_height, 
                              find_clusters = find_clusters, n_clusters = n_clusters,
                              loading_arrow_factor = loading_arrow_factor,
                              export_groups_to_csv = export_groups_to_csv, 
                              cluster_preffix = cluster_preffix, show_figure = show_figure)

        

    def plot_cat_cat_stacked_barplots(self, DF_columns = ['0', '1'], axes_labels = ['a', 'b', '%'], 
                                      min_occurrences = 10, cat_to_filter_min_occurrences = '',
                                      size_factor = 0.5):
    
        print('\n\n> function: plot_cat_cat_stacked_barplots')
        print('Columns used: ', DF_columns)

        #checando se as colunas estão no DF de reultados
        for column in DF_columns:
            if column not in self.results_DF_processed.columns:
                print(f'Erro! A coluna "{column}" não existe na DF de resultados.')
                return
        
        
        np.seterr(invalid='ignore')
        
        #fazendo a cópia do DF
        DF_copy = self.results_DF_processed[[DF_columns[0], DF_columns[1]]].copy()
        filtered_DF = DF_copy.dropna(how='any')

        #limpando as duplicatas na coluna
        filtered_DF = self.clear_duplicates_in_columns(filtered_DF, column_names=DF_columns, DF_indexes_names = ['doc', 'counter'])

        #agrupando a DF (gera um objeto GROUPBY com multiindex)
        grouped = filtered_DF.groupby(cat_to_filter_min_occurrences)
        #filtrando por ocorrência
        filtered_DF = self.filter_groups_by_min_occurrences(DF = filtered_DF, grouped = grouped, min_occurrences = min_occurrences)

        #criando uma DF para colocar os resultados modificados
        modified_DF = pd.DataFrame([], columns=[ DF_columns[0], DF_columns[1], 'count', 'perc' ])

        #agrupando a DF com a primeira coluna (DF_columns[0])
        column_to_groupby = DF_columns[0]
        grouped = filtered_DF.groupby(column_to_groupby)
        counter = 0
        max_counts = 0
        for group in grouped.groups:
            if len(grouped.get_group(group).values) > max_counts:
                max_counts = len(grouped.get_group(group).values)
            #agrupando novamente
            grouped_inner = grouped.get_group(group).groupby(DF_columns[1])
            #coletando a serie agrupada
            series = grouped_inner.describe().loc[ : , ('Publication Year', 'count') ]
            #normalizando os valores
            series_perc = series / series.cumsum().max() * 100 
            for index in series.index:
                modified_DF.loc[counter] = group, index , series.loc[index], series_perc.loc[index]
                counter += 1

        #agrupando a DF com a segunda coluna (DF_columns[1])
        column_to_groupby = DF_columns[1]
        grouped = modified_DF.groupby(column_to_groupby)

        #ajustando os dados para plotar
        x_axis_labels = np.sort(np.unique(modified_DF[DF_columns[0]].values))
        categories = np.sort(np.unique(modified_DF[DF_columns[1]].values))

        #criando um dicionário para os dados
        data1 = {}
        data2 = {}
        data1['x_labels'] = x_axis_labels
        data2['x_labels'] = x_axis_labels

        for category in categories:
            group_DF = grouped.get_group(category)
            group_DF = group_DF.set_index(DF_columns[0])
            for index in x_axis_labels:
                if index not in group_DF.index.values:
                    group_DF.loc[index] = category, 0, 0 
            group_DF = group_DF.sort_index()
            
            data1[category] = group_DF['count'].values
            data2[category] = group_DF['perc'].values
        
        #cores        
        if len(categories) > 20:
            print('ERRO!')
            print('Aumentar o valor de min_occurence para que o número máximo de categorias seja 20.')
            return
        
        colors = Category20[20][ : len(categories) ]
        
        if len(axes_labels) != 3:
            print('ERRO!')
            print('A lista axes_labels precisa ter 3 valores')
            return
        
        x_label, y_label1, y_label2 = axes_labels
        
        #plotando o gráfico da esquerda
        p_top = figure(x_range=x_axis_labels,
                        toolbar_location=None, tools="",
                        plot_width=1500, plot_height=800,
                        title='min occurrences: ' + str(min_occurrences))

        p_top.vbar_stack(categories, x='x_labels', width=0.9, color=colors, source=data1)

        p_top.xaxis.axis_label = x_label
        p_top.xaxis.axis_label_text_font_size = str(int(size_factor) + 2) + 'pt'
        p_top.yaxis.axis_label = y_label1
        p_top.yaxis.axis_label_text_font_size = str(int(size_factor) + 2) + 'pt'
        p_top.y_range.start = 0
        p_top.x_range.range_padding = 0.1
        p_top.xgrid.grid_line_color = None
        p_top.axis.minor_tick_line_color = None
        p_top.outline_line_color = None
        p_top.title.text_font_size = str(int(size_factor) + 4) + 'pt'
        p_top.outline_line_color = "black"
        p_top.outline_line_width = 1
        p_top.axis.major_label_text_font_size = str(int(size_factor) + 2) + 'pt'
        p_top.axis.major_label_standoff = 10
        p_top.xaxis.major_label_orientation = np.pi/3
        
        p_legend = get_blank_figure(plot_width = 900, plot_height=2000, x_range=(0,700), y_range=(-600,50))
        
        for cat_i in range(len(categories)):
            
            p_legend.rect(x=30, y=-cat_i*20, width=50, height=20, color=colors[cat_i])
            leg_label = Label(x=65, y=-(cat_i*20)-6, text=categories[cat_i], text_font_size = '30pt', text_color = 'black')
            p_legend.add_layout(leg_label)

        #plotando o gráfico da direita
        p_bottom = figure(x_range=x_axis_labels,
                         toolbar_location=None, tools="",
                         plot_width=1500, plot_height=800,
                         title='min occurrences: ' + str(min_occurrences),
                         min_border=15)
        
        p_bottom.vbar_stack(categories, x='x_labels', width=0.9, color=colors, source=data2)

        p_bottom.xaxis.axis_label = x_label
        p_bottom.xaxis.axis_label_text_font_size = str(int(size_factor) + 2) + 'pt'
        p_bottom.yaxis.axis_label = y_label2
        p_bottom.yaxis.axis_label_text_font_size = str(int(size_factor) + 2) + 'pt'
        p_bottom.y_range.start = 0
        p_bottom.x_range.range_padding = 0.1
        p_bottom.xgrid.grid_line_color = None
        p_bottom.axis.minor_tick_line_color = None
        p_bottom.outline_line_color = None
        p_bottom.title.text_font_size = str(int(size_factor) + 4) + 'pt'
        p_bottom.outline_line_color = "black"
        p_bottom.outline_line_width = 1
        p_bottom.axis.major_label_text_font_size = str(int(size_factor) + 2) + 'pt'
        p_bottom.axis.major_label_standoff = 10
        p_bottom.xaxis.major_label_orientation = np.pi/3
        
        column1 = column(p_top, p_bottom)
        column2 = column(p_legend)
        
        layout = gridplot([[column1, column2]], merge_tools=False)
        
        #plotando
        
        print(f'Salvando a figura ~/Outputs/analyses/P{self.last_fig_filename_index}.png ...')
        export_png(layout, filename=self.diretorio + f'/Outputs/analyses/P{self.last_fig_filename_index}.png')
        
        #atualizando o número do plot_index
        self.last_fig_filename_index = ( ( len('0000') - len(str(int(self.last_fig_filename_index)+1)) ) * '0' ) + str(int(self.last_fig_filename_index) + 1)







    def plot_multicolumn_network_graph(self, DF_columns = ['0', '1', '2'], max_circle_size = 100, min_circle_size = 10, min_occurrences = 10,
                                       start_end_nodes_to_analyze = [], path_nodes_cutoff = None, min_edge_weight = 0,
                                       plot_graphs = False, graph_base_title = ''):
        
        
        

        print('\n\n> function: plot_multicolumn_network_graph')
        print('Columns used: ', DF_columns)

        #checando se as colunas estão no DF de reultados
        for column in DF_columns:
            if column not in self.results_DF_processed.columns:
                print(f'Erro! A coluna "{column}" não existe na DF de resultados.')
                return

        DF_column_name_join_list = []

        #copiando a DF
        DF_copy = self.results_DF_processed[DF_columns].copy().dropna(axis=0, how='any')

        #filtrando indexes que estão no arquivo /Inputs/ngrams_to_remove.json e /Inputs/ngrams_to_replace.json
        DF_copy = self.filter_indexes_by_category_inputs(DF = DF_copy,  DF_column_to_filter = DF_columns[0], DF_indexes_names = ['doc', 'counter'])
        DF_copy = self.filter_indexes_by_category_inputs(DF = DF_copy,  DF_column_to_filter = DF_columns[1], DF_indexes_names = ['doc', 'counter'])
        
        #eliminando duplicatas na coluna
        DF_copy = self.clear_duplicates_in_columns(DF_copy, column_names=DF_columns, DF_indexes_names = ['doc', 'counter'])

        #coletando o número de artigos usados
        n_articles = len(DF_copy.index.get_level_values(0))
        
        #varrendo as colunas para encontrar as conexões
        for column_i in range(len(DF_columns)-1):
            
            name1 = DF_columns[column_i]
            name2 = DF_columns[column_i+1]
                        
            #fazendo o merge das duas colunas para grouping
            DF_column_name_join = name1 + '_' + name2
            DF_column_name_join_list.append(DF_column_name_join)
            DF_copy[DF_column_name_join] = pd.Series(DF_copy[name1].values + '_' + DF_copy[name2].values, index=DF_copy.index)
    
        #definindo uma tag para cada coluna coletada no DataFrame
        column_group_tag_list = []

        #varrendo as colunas para definir as tags de cada entrada
        for column in DF_columns:
            #definindo uma tag para cada intem (node) de coluna coletada no DataFrame
            tags = list( zip( list(np.unique(DF_copy[column].values)) , [column] * len(np.unique(DF_copy[column].values)) ) )
            column_group_tag_list.extend(tags)
        
        #definindo um dicionário com o tag de cada termo usado (node)
        nodes_categories = dict(column_group_tag_list)

        #criando o DF para concatenar
        concat_network_relat = {}
        concat_network_relat['graph'] = pd.DataFrame([])
        #agrupando os resultados
        for join_name in DF_column_name_join_list:

            #agrupando a DF (gera um objeto GROUPBY com multiindex)
            grouped = DF_copy.groupby(join_name)
            DF_copy = self.filter_groups_by_min_occurrences(DF = DF_copy, grouped = grouped, min_occurrences = min_occurrences)

            #agrupando para fazer a contagem de aparição das correlações
            grouped_DF = DF_copy.groupby(join_name).describe()
            concat_network_relat['graph'] = pd.concat([ concat_network_relat['graph'] , grouped_DF.loc[ : , ( DF_columns[0] , 'count' ) ]])            
        
        #resetando o index e renomeando a coluna
        concat_network_relat['graph'] = concat_network_relat['graph'].reset_index()
        concat_network_relat['graph'].rename(columns={0:'weight'}, inplace=True)
        
        self.network_analysis_graph_plot(concat_network_relat, nodes_categories = nodes_categories,
                                         max_circle_size = max_circle_size, min_circle_size = min_circle_size,                                         
                                         start_end_nodes_to_analyze=start_end_nodes_to_analyze, 
                                         path_nodes_cutoff = path_nodes_cutoff, 
                                         min_edge_weight = min_edge_weight,
                                         plot_graphs = plot_graphs, graph_base_title = graph_base_title,
                                         n_articles = n_articles)



    def plot_2column_network_chord(self, DF_columns = ['0', '1'], min_occurrences = 10):
        
        


        print('\n\n> function: plot_2column_network_chord')    
        print('Columns used: ', DF_columns)

        #checando se as colunas estão no DF de reultados
        for column in DF_columns:
            if column not in self.results_DF_processed.columns:
                print(f'Erro! A coluna "{column}" não existe na DF de resultados.')
                return

        #copiando a DF
        DF_copy = self.results_DF_processed[DF_columns].copy().dropna(axis=0, how='any')

        #filtrando indexes que estão no arquivo /Inputs/ngrams_to_remove.json e /Inputs/ngrams_to_replace.json
        filtered_DF = self.filter_indexes_by_category_inputs(DF = DF_copy,  DF_column_to_filter = DF_columns[0], DF_indexes_names = ['doc', 'counter'])
        filtered_DF = self.filter_indexes_by_category_inputs(DF = filtered_DF,  DF_column_to_filter = DF_columns[1], DF_indexes_names = ['doc', 'counter'])
        
        #limpando as duplicatas na coluna
        filtered_DF = self.clear_duplicates_in_columns(filtered_DF, column_names=DF_columns, DF_indexes_names = ['doc', 'counter'])

        #fazendo o merge das duas colunas para grouping
        DF_column_name_join = DF_columns[0] + '_' + DF_columns[1]
        filtered_DF[DF_column_name_join] = pd.Series(filtered_DF[DF_columns[0]].values + '_' + filtered_DF[DF_columns[1]].values, index=filtered_DF.index)

        #agrupando a DF (gera um objeto GROUPBY com multiindex)
        grouped = filtered_DF.groupby(DF_column_name_join)
        filtered_DF = self.filter_groups_by_min_occurrences(DF = filtered_DF, grouped = grouped, min_occurrences = min_occurrences)

        #agrupando para fazer a contagem de aparição das correlações
        grouped_DF = filtered_DF.groupby(DF_column_name_join).describe().loc[ : , ( DF_columns[0] , 'count' ) ]

        #plotando
        self.chord_plot(grouped_DF)



    def plot_cat_cat_gridplot_bins(self, DF_columns = ['0', '1'],
                                   min_occurrences = 10, size_factor = 0.4, colobar_nticks = 20,
                                   palette = 'Viridis256',
                                   background_fill_color = 'blue',
                                   plot_width=1000, plot_height=1000):

        
        
        print('\n\n> function: plot_cat_cat_gridplot_bins')    
        print('DF_columns: ', DF_columns)
        
        #checando se as colunas estão no DF de reultados
        for column in DF_columns:
            if column not in self.results_DF_processed.columns:
                print(f'Erro! A coluna "{column}" não existe na DF de resultados.')
                return
        
        #copiando a DF        
        DF_copy = self.results_DF_processed[[DF_columns[0], DF_columns[1]]].copy().dropna(axis=0, how='any')

        #filtrando indexes que estão no arquivo /Inputs/ngrams_to_remove.json e /Inputs/ngrams_to_replace.json
        DF_copy = self.filter_indexes_by_category_inputs(DF = DF_copy,  DF_column_to_filter = DF_columns[0], DF_indexes_names = ['doc', 'counter'])
        DF_copy = self.filter_indexes_by_category_inputs(DF = DF_copy,  DF_column_to_filter = DF_columns[1], DF_indexes_names = ['doc', 'counter'])

        #eliminando duplicatas na coluna
        DF_copy = self.clear_duplicates_in_columns(DF_copy, column_names=DF_columns, DF_indexes_names = ['doc', 'counter'])

        #fazendo o merge das duas colunas para grouping
        DF_column_name_join = DF_columns[0] + '+' + DF_columns[1]
        DF_copy[DF_column_name_join] = pd.Series(DF_copy[DF_columns[0]].values + '+' + DF_copy[DF_columns[1]].values, index=DF_copy.index)
        
        #agrupando a DF (gera um objeto GROUPBY com multiindex)
        grouped = DF_copy.groupby(DF_column_name_join)
        #limpando os grupos que possuem poucos valores (menores que min_values_for_column) e que estão no ~/Inputs
        DF_copy = self.filter_groups_by_min_occurrences(DF = DF_copy, grouped = grouped, min_occurrences = min_occurrences)

        #agrupando a DF (gera um objeto GROUPBY com multiindex)
        DF_grouped = DF_copy.groupby(DF_column_name_join).describe()        

        DF_grouped = self.add_split_index_names_to_column(DF_grouped)

        
        #coletando apenas a coluna com o número de ocorrências
        DF_grouped = DF_grouped[[DF_columns[0], DF_columns[1], 'count']].copy()
        
        #exportando alguns valores genéricos em csv
        #DF_grouped.drop_duplicates(subset=DF_columns[0]).to_csv('/home/amaurijp/Desktop/species.csv')

        #plotando
        self.cat_cat_gridplot(DF = DF_grouped,
                              DF_x_column = DF_columns[0], DF_y_column = DF_columns[1], DF_val_column = 'count',
                              min_occurrences = min_occurrences,
                              size_factor=size_factor, 
                              colobar_nticks = colobar_nticks,
                              palette = palette,
                              background_fill_color = background_fill_color,
                              plot_width=plot_width, plot_height=plot_width)



    def plot_cat_loc_gridplot_bins(self, DF_column = '', axes_labels = ['a', 'b'], min_occurrences = 10, size_factor = 0.4, 
                                   colobar_nticks = 20,
                                   palette = 'Viridis256',
                                   background_fill_color = 'blue',
                                   plot_width=1000, plot_height=1000):

        
        

        print('\n\n> function: plot_cat_loc_gridplot_bins')
        print('DF_column: ', DF_column)
        
        #checando se as colunas estão no DF de reultados
        if DF_column not in self.results_DF_processed.columns:
            print(f'Erro! A coluna "{DF_column}" não existe na DF de resultados.')
            return
            
        def cluster_location(line_vals):
            return [val for val in line_vals if val is not np.nan]
        
        #abrindo o DF com as locations
        temp_series = self.doc_location_DF.apply(cluster_location, axis=1)
        temp_DF1 = pd.DataFrame([])
        temp_DF1['doc'] = temp_series.index.values
        temp_DF1['location'] = temp_series.values

        #fazendo o merge com as locations
        temp_DF2 = pd.merge(self.results_DF_processed.reset_index(), temp_DF1, on='doc')

        #redefinindo o index
        temp_DF2 = temp_DF2[[DF_column, 'location', 'doc']].copy()
        
        #unstacking o location
        loc_unstacked = []
        loc_multipler = []
        for loc_list in temp_DF2['location'].values:
            loc_unstacked.extend(loc_list)
            loc_multipler.append(len(loc_list))
        
        #criando a lista associada com os valores da coluna com a categoria
        cat_unstacked = []
        doc_unstacked = []
        for i in range(len(temp_DF2[DF_column].values)):
            cat_unstacked.extend( [temp_DF2[DF_column].values[i]] * loc_multipler[i] )
            doc_unstacked.extend( [temp_DF2['doc'].values[i]] * loc_multipler[i] )
        
        #definindo DF final
        DF_copy = pd.DataFrame(list(zip(cat_unstacked , loc_unstacked)), columns=[DF_column, 'location'])        
        DF_copy['doc'] = doc_unstacked
        DF_copy['counter'] = list(range(len(doc_unstacked)))
        DF_copy = DF_copy.set_index(['doc', 'counter'])

        #eliminando os NANs
        DF_copy = DF_copy.dropna(how='any')

        #eliminando duplicatas na coluna da categoria
        DF_copy = self.clear_duplicates_in_columns(DF_copy, column_names=[DF_column, 'location'], DF_indexes_names = ['doc', 'counter'])
        
        #fazendo o merge das duas colunas para grouping
        DF_column_name_join = DF_column + '+location'

        DF_copy[DF_column_name_join] = pd.Series(DF_copy[DF_column].values + '+' + DF_copy['location'].values, index=DF_copy.index)

        #agrupando a DF (gera um objeto GROUPBY com multiindex)
        grouped = DF_copy.groupby(DF_column_name_join)        
        #limpando os grupos que possuem poucos valores (menores que min_values_for_column) e que estão no ~/Inputs
        DF_copy = self.filter_groups_by_min_occurrences(DF = DF_copy, grouped = grouped, min_occurrences = min_occurrences)

        #agrupando a DF (gera um objeto GROUPBY com multiindex)
        DF_grouped = DF_copy.groupby(DF_column_name_join).describe()        
        DF_grouped = self.add_split_index_names_to_column(DF_grouped)
        #coletando apenas a coluna com o número de ocorrências
        DF_grouped = DF_grouped[[DF_column, 'location', 'count']].copy()

        #plotando
        self.cat_cat_gridplot(DF = DF_grouped,
                              DF_x_column = DF_column, DF_y_column = 'location', DF_val_column = 'count',
                              min_occurrences = min_occurrences,
                              size_factor = size_factor,
                              colobar_nticks = colobar_nticks,
                              palette = palette,
                              background_fill_color = background_fill_color,
                              plot_width=plot_width, plot_height=plot_height)


    
    def plot_group_num_boxplot_correlation(self, DF_column_with_cat_vals = 'a', 
                                           DF_column_with_num_vals = 'b', 
                                           categories_to_get = None,
                                           column_cat_to_filter = None,
                                           num_values_to_filter = [None, None],
                                           axes_labels=['', ''],
                                           y_quantiles = [0, 1], 
                                           min_values_for_column = 10,
                                           size_factor_boxplot = 1,
                                           size_factor_anova_grid = 1,
                                           grouplabel_x_offset = 0,
                                           colobar_nticks = 5,
                                           palette = 'Viridis256',
                                           background_fill_color='blue',
                                           box_plot_plot_width=1000, box_plot_plot_height=1000,
                                           grid_plot_width=1000, grid_plot_height=1000,
                                           export_groups_to_csv = False):
        

        print('\n\n> function: plot_group_num_boxplot_correlation')

        #checando se as colunas estão no DF de reultados
        for column in (DF_column_with_cat_vals, DF_column_with_num_vals):
            if column not in self.results_DF_processed.columns:
                print(f'Erro! A coluna "{column}" não existe na DF de resultados.')
                return

        #caso haja categorias para filtrar
        if type(categories_to_get) == list and len(categories_to_get) > 0:
            #agrupando
            grouped = self.results_DF_processed.groupby(DF_column_with_cat_vals)
            #DF para concatenar
            concat_DF = pd.DataFrame([], columns=self.results_DF_processed.columns, index=[[],[]])
            concat_DF.index.names = self.results_DF_processed.index.names
            for cat in categories_to_get:
                concat_DF = pd.concat([concat_DF, grouped.get_group(cat)])

            #sorting the index            
            self.results_DF_processed = concat_DF.sort_index()

        #caso não haja uma coluna específica para filtrar, usa-se a do plot
        if column_cat_to_filter is None:
            column_cat_to_filter = DF_column_with_num_vals

        #caso haja valores númericos para filtrar
        if len(num_values_to_filter) > 0 and (num_values_to_filter[0] is not None or num_values_to_filter[1] is not None):

            min_val = num_values_to_filter[0]
            max_val = num_values_to_filter[1]
                
            #filtrando a coluna com valores
            if min_val is not None and max_val is None:
                self.results_DF_processed = self.results_DF_processed[ (self.results_DF_processed[column_cat_to_filter] >= min_val ) ]
            elif min_val is None and max_val is not None:
                self.results_DF_processed = self.results_DF_processed[ (self.results_DF_processed[column_cat_to_filter] <= max_val) ]
            elif min_val is not None and max_val is not None:
                self.results_DF_processed = self.results_DF_processed[ (self.results_DF_processed[column_cat_to_filter] >= min_val ) & (self.results_DF_processed[column_cat_to_filter] <= max_val) ]

        if len(self.results_DF_processed.index) == 0:
            print('ERRO!')
            print('DF com nenhum valor após filtragem dos valores numéricos.')
            return   

        #eliminando os outliers
        input1 , input2 = DF_column_with_cat_vals , DF_column_with_num_vals
        DF_copy = pd.DataFrame([], index=self.results_DF_processed.index)
        DF_copy = DF_copy.join(self.results_DF_processed[input1])
        y_q1 , y_q2 = y_quantiles
        DF_copy = DF_copy.join( self.cutting_outliers_in_DF_column(column = input2, quantiles = [y_q1, y_q2]) )
        DF_copy = DF_copy[[DF_column_with_cat_vals, DF_column_with_num_vals]].dropna(axis=0, how='any')       

        #limpando as duplicatas na coluna
        DF_copy = self.clear_duplicates_in_columns(DF_copy, column_names=[DF_column_with_cat_vals, DF_column_with_num_vals], DF_indexes_names = ['doc', 'counter'])

        #agrupando a DF (gera um objeto GROUPBY com multiindex)
        grouped = DF_copy.groupby(DF_column_with_cat_vals)
        #limpando os grupos que possuem poucos valores (menores que min_values_for_column) e que estão no ~/Inputs
        DF_copy = self.filter_groups_by_min_occurrences(DF = DF_copy, grouped = grouped, min_occurrences = min_values_for_column) 

        #analise das categorias que ficaram
        if type(categories_to_get) == list and len(np.unique(DF_copy[DF_column_with_cat_vals].values)) != len(categories_to_get):
            print('\nERRO!')
            print('Algumas das categorias inseridas não estão presentes após filtragem da DF.')
            print('Categorias únicas para a coluna: ', DF_column_with_cat_vals, ' após a filtragem:')
            print(np.unique(DF_copy[DF_column_with_cat_vals].values))
            print('Modificar os valores de ', categories_to_get)
            return
        
        #agrupando a serie
        grouped_filtered = DF_copy[input2].groupby(DF_copy[input1])
        
        #valores mínimo e máximo do input numérico
        ymin , ymax = DF_copy[input2].values.min() , DF_copy[input2].values.max()

        #plotando
        self.group_num_boxplot(grouped_DF = grouped_filtered, 
                               input1 = input1, 
                               input2 = input2, 
                               axes_label1 = axes_labels[0],
                               axes_label2 = axes_labels[1],
                               ymin = ymin, ymax = ymax,
                               min_occurrences = min_values_for_column,
                               size_factor = size_factor_boxplot,
                               grouplabel_x_offset = grouplabel_x_offset,
                               colobar_nticks = colobar_nticks,
                               plot_width=box_plot_plot_width, plot_height=box_plot_plot_height,
                               export_groups_to_csv = export_groups_to_csv)
        
        
        #plotando o anova entre os grupo        
        stacked_data = pd.DataFrame([])
        #varrendo os grupos
        for group_name in grouped_filtered.groups:
            temp_df = grouped_filtered.get_group(group_name).reset_index()
            series = temp_df[input2]
            temp_df['values'] = series
            temp_df['group'] = [group_name] * len(temp_df['values'].index)
            stacked_data = pd.concat([stacked_data, temp_df[['values', 'group']].copy()])

        #fazendo o TUKEY HONESTLY SIGNIFICANT DIFFERENCE
        comp = mc.MultiComparison(stacked_data['values'], stacked_data['group'])
        post_hoc_res = comp.tukeyhsd()        
        post_hoc_df = pd.DataFrame(data=post_hoc_res._results_table.data[1:], columns=post_hoc_res._results_table.data[0])

        self.cat_cat_gridplot(DF = post_hoc_df, DF_x_column = 'group1', DF_y_column = 'group2', DF_val_column = 'p-adj', 
                              min_occurrences = 0, size_factor=size_factor_anova_grid, color_mapper_range = [0, 1],
                              colobar_nticks = colobar_nticks, palette = palette, background_fill_color = background_fill_color,
                              plot_width=grid_plot_width, plot_height=grid_plot_height)



    def plot_num_num_correlation(self, DF_columns_with_num_vals = ['a', 'b'], axes_labels = ['d', 'e'], 
                                 x_quantiles = [0, 1], y_quantiles = [0, 1],
                                 x_min = None, x_max = None, y_min = None, y_max = None,
                                 hex_size = 10, plot_width=1000, plot_height=1000, mode='scatter', 
                                 regression = False, find_clusters = False, n_clusters = 3,
                                 export_groups_to_csv = False, cluster_preffix = '',
                                 show_figure = False):
            
        

        print('\n\n> function: plot_num_num_correlation')

        #checando se as colunas estão no DF de reultados        
        for column in DF_columns_with_num_vals:
            if column not in self.results_DF_processed.columns:
                print(f'Erro! A coluna "{column}" não existe na DF de resultados.')
                return
        
        #eliminando os outliers
        input1 , input2 = DF_columns_with_num_vals
        DF_copy = pd.DataFrame([], index=self.results_DF_processed.index)
        x_q1 , x_q2 = x_quantiles
        DF_copy = DF_copy.join( self.cutting_outliers_in_DF_column(column = input1, quantiles = [x_q1, x_q2]) )
        y_q1 , y_q2 = y_quantiles
        DF_copy = DF_copy.join( self.cutting_outliers_in_DF_column(column = input2, quantiles = [y_q1, y_q2]) )
        
        #limpando os valores NANs
        DF_copy = DF_copy[DF_columns_with_num_vals].dropna(axis=0, how='any')

        #limpando as duplicatas de index (doc)
        DF_copy = self.clear_duplicated_index(DF_copy, DF_indexes_names = ['doc', 'counter'])

        #plotando
        if mode.lower() == 'hexbin':
            self.num_num_hexbinplot(DF = DF_copy, axes_labels = axes_labels, input1 = input1, input2 = input2, 
                                     x_min = x_min, x_max = x_max, y_min = y_min, y_max = y_max, 
                                     hex_size = hex_size, plot_width=plot_width, plot_height=plot_height)
        elif mode.lower() == 'scatter':
            self.num_num_scatterplot(DF = DF_copy, axes_labels = axes_labels, input1 = input1, input2 = input2, 
                                     x_min = x_min, x_max = x_max, y_min = y_min, y_max = y_max, 
                                     plot_width=plot_width, plot_height=plot_height,
                                     regression = regression, find_clusters = find_clusters, n_clusters=n_clusters,
                                     export_groups_to_csv = export_groups_to_csv, cluster_preffix = cluster_preffix,
                                     show_figure = show_figure)



    def plot_group_num_num_correlation(self, DF_columns_with_num_vals = ['a', 'b'], axes_labels = ['d', 'e'], 
                                       x_quantiles = [0, 1], y_quantiles = [0, 1],
                                       DF_column_with_cat_vals_to_group = None,
                                       categories_to_get = None,
                                       x_min = None, x_max = None, y_min = None, y_max = None,
                                       min_values_for_column = 10, mode='scatter',
                                       hex_size = 10, plot_width=1000, plot_height=1000,
                                       regression = False, export_groups_to_csv = False,
                                       cluster_preffix=''):
            
        

        print('\n\n> function: plot_group_num_num_correlation')

        #checando se as colunas estão no DF de reultados        
        for column in DF_columns_with_num_vals:
            if column not in self.results_DF_processed.columns:
                print(f'Erro! A coluna "{column}" não existe na DF de resultados.')
                return            
        if DF_column_with_cat_vals_to_group not in self.results_DF_processed.columns:
            print(f'Erro! A coluna "{DF_column_with_cat_vals_to_group}" não existe na DF de resultados.')
            return

        DF_copy = pd.DataFrame([], index=self.results_DF_processed.index)

        #adicionando a coluna com o parametro categorico a agrupar
        cat_to_group = DF_column_with_cat_vals_to_group
        DF_copy = DF_copy.join(self.results_DF_processed[cat_to_group])
        
        try:
            #eliminando os outliers
            input1 , input2 = DF_columns_with_num_vals
            x_q1 , x_q2 = x_quantiles
            DF_copy = DF_copy.join( self.cutting_outliers_in_DF_column(column = input1, quantiles = [x_q1, x_q2]) )
            y_q1 , y_q2 = y_quantiles
            DF_copy = DF_copy.join( self.cutting_outliers_in_DF_column(column = input2, quantiles = [y_q1, y_q2]) )
            
            #limpando os valores NANs
            DF_copy = DF_copy[DF_columns_with_num_vals + [DF_column_with_cat_vals_to_group]].dropna(axis=0, how='any')
        
        #caso a coluna numérica não tenha sido processada
        except TypeError:
            print('ERRO!')
            print('Os valores numéricos da coluna não estão corretos.')
            print('Provavelmente a função "process_columns_with_num_val" não foi usada.')
            return
            
        #filtrando indexes que estão no arquivo /Inputs/ngrams_to_remove.json e /Inputs/ngrams_to_replace.json
        DF_copy = self.filter_indexes_by_category_inputs(DF = DF_copy,  DF_column_to_filter = DF_column_with_cat_vals_to_group, 
                                                         DF_indexes_names = ['doc', 'counter'])

        #eliminando duplicatas na coluna da categoria
        DF_copy = self.clear_duplicates_in_columns(DF_copy, column_names=[DF_column_with_cat_vals_to_group], DF_indexes_names = ['doc', 'counter'])

        #agrupando a DF (gera um objeto GROUPBY com multiindex)
        grouped = DF_copy[[input1, input2, cat_to_group]].groupby(cat_to_group)
        #limpando os grupos que possuem poucos valores (menores que min_values_for_column) e que estão no ~/Inputs
        DF_copy = self.filter_groups_by_min_occurrences(DF = DF_copy, grouped = grouped, min_occurrences = min_values_for_column)

        if len(DF_copy.index) == 0:
            print('ERRO!')
            print('Não há entradas no DF filtrado.')
            print('Diminuir o valor de "min_values_for_column".')
            return

        #coletando somente os grupos de interesse
        if categories_to_get is not None:
            #agrupando a DF (gera um objeto GROUPBY com multiindex)
            grouped = DF_copy[[input1, input2, cat_to_group]].groupby(cat_to_group)
            DF_copy = self.filter_groups_by_list(DF = DF_copy, grouped = grouped, filter_type = 'grouped_vals', group_list = categories_to_get)
            if len (DF_copy.index) == 0:
                print('ERRO!')
                print('Nenhuma categoria foi encontrada com as entradas:')
                print(categories_to_get)
                return

        #agrupando a DF (gera um objeto GROUPBY com multiindex)
        grouped_filtered = DF_copy[[input1, input2, cat_to_group]].groupby(cat_to_group)
        print('Número de grupos encontrados: ', len(grouped_filtered))
        
        if mode.lower() == 'hexbin':
            if len(grouped_filtered) > len(self.palette_bins):
                print('ERRO! O número de grupos é maior que o número de paletas de cores para plotagem do group_num_num_hexbin.')
                print('Número de grupos: ', len(grouped_filtered))
                print('Número de paletas: ', len(self.palette_bins))
            else:
                #plotando
                self.group_num_num_hexbinplot(DF = DF_copy, grouped = grouped_filtered, axes_labels = axes_labels, input1 = input1, input2 = input2, 
                                              x_min = x_min, x_max = x_max, y_min = y_min, y_max = y_max, 
                                              hex_size = hex_size, plot_width=plot_width, plot_height=plot_height)
        elif mode.lower() == 'scatter':
                self.group_num_num_scatterplot(DF = DF_copy, grouped = grouped_filtered, axes_labels = axes_labels, input1 = input1, input2 = input2, 
                                               x_min = x_min, x_max = x_max, y_min = y_min, y_max = y_max, 
                                               hex_size = hex_size, plot_width=plot_width, plot_height=plot_height, 
                                               regression = regression, export_groups_to_csv = export_groups_to_csv,
                                               cluster_preffix = cluster_preffix)



    def plot_grouped_cat_barplot_from_fileindexes(self, plot_name = None, DF_column_with_cat_vals_to_plot = None, maximum_values_to_plot = 10,
                                                  cluster_preffix = '', graph_title = ''):
        
        
        try:
            #carregando o INDEX LIST em ~/Inputs
            index_list_dic = load_dic_from_json(self.diretorio + '/Inputs/Index_lists.json')
            index_lists = index_list_dic[plot_name]
        except KeyError:
            print('ERRO!')
            print('As variáveis para "plot_name" e "group_name" não foram encontradas em "Index_lists.json"')
            
        if DF_column_with_cat_vals_to_plot not in self.results_DF_processed.columns:
            print(f'Erro! A coluna "{DF_column_with_cat_vals_to_plot}" não existe na DF de resultados.')
            return
        
        #copiando a DF                
        DF_copy = self.results_DF_processed[[DF_column_with_cat_vals_to_plot]].copy().dropna(axis=0, how='any')

        #filtrando indexes que estão no arquivo /Inputs/ngrams_to_remove.json e /Inputs/ngrams_to_replace.json
        DF_copy = self.filter_indexes_by_category_inputs(DF = DF_copy,  DF_column_to_filter = DF_column_with_cat_vals_to_plot, DF_indexes_names = ['doc', 'counter'])

        #eliminando duplicatas na coluna
        DF_copy = self.clear_duplicates_in_columns(DF_copy, column_names=[DF_column_with_cat_vals_to_plot], DF_indexes_names = ['doc', 'counter'])
                 
        #varrendo os grupos para encontrar todas as categorias
        unique_cats = []
        cat_vals_dic={}
        for key in index_lists.keys():
            
            #coletando os filenames
            filenames = index_lists[key]
            
            #filtrando os grupos no DF
            DF_filtered = self.filter_groups_by_list(DF = DF_copy, filter_type = 'grouped_filenames', group_list = filenames)
            
            #guardando os valores para cada grupo
            cats_counts = np.unique(DF_filtered[DF_column_with_cat_vals_to_plot].values, return_counts=True)
            cat_vals_dic[key] = dict( list( zip(cats_counts[0], cats_counts[1])))

            #identificando as cats unicas e fazendo os counts das categorias para cada grupo            
            unique_cats.extend([cat for cat in cats_counts[0] if cat not in unique_cats])                        
            unique_cats.sort()

        
        #padronizando as categorias para todos os grupos e colocando os counts
        concat_cat_vals_dic = {}
        for key in index_lists.keys():
            
            #lista para colocar os valores
            concat_cat_vals_dic[key] = []
            #varrendo as categorias
            for cat in unique_cats:
                if cat in cat_vals_dic[key].keys():
                    concat_cat_vals_dic[key].append( cat_vals_dic[key][cat] )
                else:
                    concat_cat_vals_dic[key].append(0)
                
            #fazendo a soma dos valores para todas as categorias
            try:
                all_vals_array = all_vals_array + np.array(concat_cat_vals_dic[key])
            except NameError:
                all_vals_array = np.array(concat_cat_vals_dic[key]                )

            #modificando para um dicionário
            concat_cat_vals_dic[key] = dict( list( zip( unique_cats, concat_cat_vals_dic[key] )))
            
        #concatenando as categorias com os valores somados
        all_cats_vals_list = list( zip( unique_cats, all_vals_array ))
        
        #estabelecendo as cores para os barplots
        colors = {}
        counter = 0
        for key in index_lists.keys():
            colors[key] = self.palette_colors1[counter]
            counter += 1

        #coletando somente as categorias que contem os maiores valores                
        cat_vals_to_plot = sorted( all_cats_vals_list, key = lambda t: t[1] , reverse = True)[ : maximum_values_to_plot]
        #organizando por ordem alfabética
        cat_vals_to_plot = np.array( sorted(cat_vals_to_plot, key = lambda t: t[0]) )
        
        bar_plots = {}
        for key in index_lists.keys():
        
            #categorias a serem plotadas
            cats_to_plot = cat_vals_to_plot[:, 0]
            
            #valores da soma a serem plotados
            vals_to_plot = cat_vals_to_plot[:, 1].astype('float32') / cat_vals_to_plot[:, 1].astype('float32').max()
            
            #valores a serem plotados para o grupo
            group_vals = np.array( [ concat_cat_vals_dic[key][cat] for cat in cats_to_plot ] )
            
            group_results_array = group_vals.astype('float32') / group_vals.astype('float32').max()
        
            #plotando o gráfico
            bar_plots[key] = figure(x_range=cats_to_plot,
                                    toolbar_location=None, tools="",
                                    plot_width=1500, plot_height=1400,
                                    title = cluster_preffix + key + ': Norm. ' + graph_title,
                                    min_border = 20)
            
            #adicionado as barras para o grupo
            bar_plots[key].vbar(cats_to_plot, 
                                top=group_results_array, 
                                color = colors[key],
                                fill_alpha = 0.6,
                                width=0.9)

            #adicionando as barras para a somatória normalizada de todos os grupos
            bar_plots[key].vbar(cats_to_plot, 
                                top=vals_to_plot, 
                                fill_alpha=0.0, 
                                line_color = 'black',
                                line_width = 10,
                                line_dash="dashed",
                                width=0.9)
 
            bar_plots[key].xaxis.axis_label = graph_title
            bar_plots[key].xaxis.axis_label_text_font_size = '50pt'
            bar_plots[key].y_range.start = 0
            bar_plots[key].yaxis.visible = False
            bar_plots[key].x_range.range_padding = 0.1
            bar_plots[key].ygrid.grid_line_color = None
            bar_plots[key].xgrid.grid_line_color = None
            bar_plots[key].axis.minor_tick_line_color = None
            bar_plots[key].outline_line_color = None
            bar_plots[key].title.text_font_size = '50pt'
            bar_plots[key].outline_line_color = "black"
            bar_plots[key].outline_line_width = 1
            bar_plots[key].axis.major_label_text_font_size = '50pt'
            bar_plots[key].axis.major_label_standoff = 10
            bar_plots[key].xaxis.major_label_orientation = np.pi/2
            bar_plots[key].outline_line_color = 'white'            
        
        #organizando o gridplot
        grid_list=[]
        for key in index_lists.keys():
            grid_list.append([bar_plots[key]])

        layout = gridplot(grid_list)
        
        print(f'Salvando a figura ~/Outputs/analyses/P{self.last_fig_filename_index}.png ...')
        export_png(layout, filename=self.diretorio + f'/Outputs/analyses/P{self.last_fig_filename_index}.png')        
        
        #atualizando o número do plot_index
        self.last_fig_filename_index = ( ( len('0000') - len(str(int(self.last_fig_filename_index)+1)) ) * '0' ) + str(int(self.last_fig_filename_index) + 1)
        


    def plot_grouped_filtered_cat_boxplot(self, column_to_group = None, 
                                 category_to_analyze = None,
                                 columns_with_num_vals_to_filter = [None],
                                 min_max_vals_to_filter = [[None,None]],
                                 ymin_ymax_list = [[None,None]], 
                                 x_axis_label = None, 
                                 y_axes_labels=[], 
                                 export_groups_to_csv=False,
                                 show_figure = False):
        
        
        if column_to_group is not None:            
            
            #agrupando e tratando os dados
            DF_grouped = self.results_DF_processed.groupby(column_to_group)
            print('\nFiltering column: ', column_to_group, ' only for category: ', category_to_analyze)
            
            #filtrando a DF agrupada por categoria
            DF_cat_filtered = DF_grouped.get_group(category_to_analyze)

            #varrendo as colunas
            columns = columns_with_num_vals_to_filter
            for i in range(len(columns)):
                
                min_val = min_max_vals_to_filter[i][0]
                max_val = min_max_vals_to_filter[i][1]
                
                #filtrando a coluna com valores
                if min_val is not None and max_val is None and columns[i] is not None:
                    DF_cat_filtered = DF_cat_filtered[ (DF_cat_filtered[columns[i]] >= min_val ) ]
                elif min_val is None and max_val is not None and columns[i] is not None:
                    DF_cat_filtered = DF_cat_filtered[ (DF_cat_filtered[columns[i]] <= max_val) ]
                elif min_val is not None and max_val is not None and columns[i] is not None:
                    DF_cat_filtered = DF_cat_filtered[ (DF_cat_filtered[columns[i]] >= min_val ) & (DF_cat_filtered[columns[i]] <= max_val) ]

            if len(DF_cat_filtered.index) == 0:
                print('ERRO!')
                print('DF com nenhum valor após filtragem dos valores numéricos.')
                return
            
            #coletando a DF com as estatística descritivas do category_to_analyze após filtragem
            DF_described = DF_cat_filtered.groupby(column_to_group).describe().loc[category_to_analyze]

            #colunas a plotar            
            columns_to_plot = DF_described.index.levels[0].values

            if len(columns_to_plot) != len(y_axes_labels) or len(columns_to_plot) != len(ymin_ymax_list):
                print('ERRO!')
                print('Ou número de labels para os axes (y axes labels) é diferente do número de colunas selecionado.')
                print('Ou número de valores de ymin e ymax é diferente do número de colunas selecionado.')
                print('DF Columns: ', columns_to_plot)
                return        

            #palettes
            #paleta de cores
            colors1 = plasma(len(columns_to_plot))
            colors2 = turbo(len(columns_to_plot))
            
            #dic de plots
            plots = {}
            
            #varrendo os indexes
            
            for i in range(len(columns_to_plot)):
                try:
                    #encontrando os valores de percentiles para cada grupo
                    mean = DF_described.loc[columns_to_plot[i], 'mean']
                    q1 = DF_described.loc[columns_to_plot[i], '25%']
                    q2 = DF_described.loc[columns_to_plot[i], '50%']
                    q3 = DF_described.loc[columns_to_plot[i], '75%']
                    
                    #maximo e mínimo
                    iqr = q3 - q1
                    upper = q3 + 1.5*iqr
                    lower = q1 - 1.5*iqr
                    
                    #setup do grafico
                    plot_width = 200
                    plot_height = 1000
                    cats_number = 1
                    size_factor = 20
                    cat = [category_to_analyze]
                    ymin = ymin_ymax_list[i][0]
                    ymax = ymin_ymax_list[i][1]              
                    
                    if ymin is None:
                        ymin = lower
                    if ymax is None:
                        ymax = upper
                    
                    plots[i] = set_bokeh_main_boxplot(ymin, ymax, cat, x_axis_label, y_axes_labels[i], None, plot_width, plot_height, cats_number, size_factor, 10)
                    
                    # stems
                    plots[i].segment(cat, upper, cat, q3, line_width=3, line_color="black")
                    plots[i].segment(cat, lower, cat, q1, line_width=3, line_color="black")            
                    
                    # boxes
                    plots[i].vbar(cat, 0.7, q2, q3, line_width=2, fill_color=colors1[i], line_color="white")
                    plots[i].vbar(cat, 0.7, q1, q2, line_width=2, fill_color=colors2[i], line_color="white")
                    
                    # whiskers (almost-0 height rects simpler than segments)
                    plots[i].rect(cat, lower, width=0.3, height=0.1, line_width=3, line_color="black")
                    plots[i].rect(cat, upper, width=0.3, height=0.1, line_width=3, line_color="black")
                    
                    print('Plotting column: ', columns_to_plot[i])
                
                #caso não haja nenhum valor na coluna
                except ValueError:
                    continue
                
                        #layout
            plot_list = []
            for k in plots.keys():
                plot_list.append(plots[k])

            layout = row(plot_list)

            if show_figure is True:        
                show(layout)
            
            else:            
                print(f'Salvando a figura ~/Outputs/analyses/P{self.last_fig_filename_index}.png ...')
                export_png(layout, filename=self.diretorio + f'/Outputs/analyses/P{self.last_fig_filename_index}.png')
    
                #salvando os grupos
                if export_groups_to_csv is True:

                    
                    
                    
                    DF_labeled = DF_cat_filtered.reset_index()
                    DF_labeled['groups'] = 1
                    DF_labeled = DF_labeled.reset_index()[['doc','groups']]
                    DF_labeled = DF_labeled.drop_duplicates()                    

                    index_lists_dic = load_dic_from_json(self.diretorio + '/Inputs/Index_lists.json')
                    index_lists_dic['P' + self.last_fig_filename_index] = {}
                    labels_grouped_DF = DF_labeled.groupby('groups')
                    for group in labels_grouped_DF.groups:
                        index_lists_dic['P' + self.last_fig_filename_index][group] = list(labels_grouped_DF.get_group(group)['doc'].values)
                    save_dic_to_json(self.diretorio + '/Inputs/Index_lists.json', index_lists_dic)
                    print('Salvando a index_list dos grupos em ~/Inputs/Index_lists.json ...')
                
                #atualizando o número do plot_index
                self.last_fig_filename_index = ( ( len('0000') - len(str(int(self.last_fig_filename_index)+1)) ) * '0' ) + str(int(self.last_fig_filename_index) + 1)

        else:
            print('ERRO!')
            print('Inserir um termo para agrupar.')



    def analyze_1column_network_digraph(self, DF_column = '', max_circle_size = 100, min_circle_size = 10, min_occurrences = 0, graph_base_title = '',
                                        start_end_nodes_to_analyze = None, path_nodes_cutoff = None, min_edge_weight = 0, 
                                        plot_graphs = False, print_unique_vals_in_DF = False):

        
        print('\n\n> function: plot_1column_network_graph')
        print('Column used: ', DF_column)
        
        if DF_column not in self.results_DF_processed.columns:
            print(f'Erro! A coluna "{DF_column}" não existe na DF de resultados.')
            return
        
        #copiando a DF
        DF_copy = self.results_DF_processed[DF_column].copy().dropna(axis=0, how='any')

        #filtrando indexes que estão no arquivo /Inputs/ngrams_to_remove.json e /Inputs/ngrams_to_replace.json
        filtered_series = self.filter_indexes_by_category_inputs(DF = DF_copy,  DF_column_to_filter = DF_column, DF_indexes_names = ['doc', 'counter'])

        if print_unique_vals_in_DF is True:
            print('\nUnique vals in network:')
            print(np.unique(filtered_series.values))

        #eliminando duplicatas na coluna (somente será usado para o graph e não para o digraph)
        filtered_series_without_duplicates = self.clear_duplicates_in_columns(filtered_series, column_names=[DF_column], DF_indexes_names = ['doc', 'counter'])

        #agrupando a DF (gera um objeto GROUPBY com multiindex)
        grouped = filtered_series_without_duplicates.groupby(DF_column)
        filtered_series_without_duplicates = self.filter_groups_by_min_occurrences(DF = filtered_series_without_duplicates, grouped = grouped, min_occurrences = min_occurrences)

        #coletando o número de artigos usados
        n_articles = len(np.unique(filtered_series.index.get_level_values(0)))

        #criando uma df para colocar as relações da rede
        concat_network_relat = {}
        concat_network_relat['graph'] = pd.Series([], name='weight', dtype=object)
        concat_network_relat['digraph'] = pd.Series([], name='weight', dtype=object)
        
        #varrendos as entradas para cada artigo para montar o digraph
        for file in filtered_series_without_duplicates.index.levels[0]:
            
            try:
                column_vals = filtered_series_without_duplicates.loc[ (file, ) , ].values
            except KeyError:
                continue
            
            #reshaping
            column_vals = column_vals.reshape(-1)
            
            #criando a DF para o digraph                    
            if len(column_vals) > 1:
                for i in range(len(column_vals)-1):
                    if column_vals[i] != column_vals[i+1]:
                        index = column_vals[i] + '_' + column_vals[i+1]
                        try:
                            concat_network_relat['digraph'].loc[index] = concat_network_relat['digraph'].loc[index] + 1
                        except KeyError:
                            concat_network_relat['digraph'].loc[index] = 1

        
        #varrendos as entradas para cada artigo para montar o graph
        for file in filtered_series_without_duplicates.index.levels[0]:

            try:
                column_vals = filtered_series_without_duplicates.loc[ (file, ) , ].values
            except KeyError:
                continue            

            #reshaping
            column_vals = column_vals.reshape(-1)

            #criando a DF para o graph                
            vals = list(set([val for val in column_vals]))
            
            if len(vals) > 1:
                for comb in itt.combinations(vals, 2):                    
                    sorted_comb = tuple(sorted(comb))

                    index = sorted_comb[0] + '_' + sorted_comb[1]
                    try:
                        concat_network_relat['graph'].loc[index] = concat_network_relat['graph'].loc[index] + 1
                    except KeyError:
                        concat_network_relat['graph'].loc[index] = 1


        #ajustando e normalizando os weights
        concat_network_relat['graph'].index.name = 'index'
        concat_network_relat['graph'] = concat_network_relat['graph'].reset_index()
        concat_network_relat['graph']['weight'] = concat_network_relat['graph']['weight'].values / concat_network_relat['graph']['weight'].values.max()        
        
        concat_network_relat['digraph'].index.name = 'index'
        concat_network_relat['digraph'] = concat_network_relat['digraph'].reset_index()
        concat_network_relat['digraph']['weight'] = concat_network_relat['digraph']['weight'].values / concat_network_relat['digraph']['weight'].values.max()

        #definindo um dic para report
        self.network_report_dic = {}

        self.network_analysis_graph_plot(concat_network_relat, max_circle_size = max_circle_size, min_circle_size = min_circle_size,
                                         start_end_nodes_to_analyze=start_end_nodes_to_analyze, 
                                         path_nodes_cutoff = path_nodes_cutoff, min_edge_weight = min_edge_weight, 
                                         plot_graphs = plot_graphs, graph_base_title = graph_base_title,
                                         n_articles = n_articles)    
                
        #passando para array para plotar ocorrências de termos
        self.network_report_dic['cat_occurrences'] = dict(list( zip( *np.unique(filtered_series_without_duplicates.values.reshape(-1), return_counts=True) ) ))
        
        return self.network_report_dic



    #funções de plotagem            
    def network_analysis_graph_plot(self, concat_network_relat, nodes_categories = None, 
                                    max_circle_size = 100, min_circle_size = 20, graph_base_title = '',
                                    start_end_nodes_to_analyze=[], path_nodes_cutoff = None, min_edge_weight = 0, 
                                    plot_graphs = False, n_articles = None):

        self.network_report_dic['group_name'] = graph_base_title
        
        
        def get_first_column_names(term):
            
            split_term = re.findall(r'[\w\s]+(?=_)', term)
            return split_term[0]

        def get_second_column_names(term):
            
            split_term = re.findall(r'(?<=_)[\w\s]+', term)
            return split_term[0]
        
        #analisando os graphs
        for graph_type in ('graph', 'digraph'): #a análise do graph está desativada
            try:
                print(f'\nAnalisando o {graph_type}...')
                concat_network_relat[graph_type]
                
                #mudando o dtype para a coluna weight para plotar
                edges_weights_DF = pd.DataFrame([])
                edges_weights_DF['weight'] = concat_network_relat[graph_type]['weight'].values.astype('float32')
        
                #coletando os sources e targets
                edges_weights_DF['source'] = concat_network_relat[graph_type]['index'].apply(get_first_column_names)
                edges_weights_DF['target'] = concat_network_relat[graph_type]['index'].apply(get_second_column_names)

                if graph_type == 'graph':        
                    #obtendo a rede no networkx
                    G = nx.from_pandas_edgelist(edges_weights_DF, source='source', target='target', edge_attr='weight', create_using=nx.Graph)
                elif graph_type == 'digraph':
                    #obtendo a rede no networkx
                    G = nx.from_pandas_edgelist(edges_weights_DF, source='source', target='target', edge_attr='weight', create_using=nx.DiGraph)
        
                print('* Network analysis *')
                print('Graph name: ', graph_base_title)
                print('N nodes: ', nx.number_of_nodes(G))
                print('N edges: ', nx.number_of_edges(G))
                print('Density:', round(nx.density(G), 4))
                
                self.network_report_dic['n_nodes'] = nx.number_of_nodes(G)
                self.network_report_dic['n_edges'] = nx.number_of_edges(G)
                self.network_report_dic['density'] = round(nx.density(G), 4)
        
                #analisando os neighbours do node central
                if graph_type == 'graph':

                    #centralidade
                    degree_cent = nx.degree_centrality(G)
                    degree_centrality_sorted = sorted(degree_cent.items(), reverse = True, key = lambda t: t[1])

                    node_higher_degree_cent = degree_centrality_sorted[0][0]
                    #avaliando propriedades de neighborhoods para o node de maior in_degree centrality
                    print('Node with higher degree centrality: ', node_higher_degree_cent)
                    
                    self.network_report_dic['degree_nodes'] = dict(sorted(G.degree(), key = lambda t: t[1] , reverse=True))

                    #organizando as edges pelo weight
                    edges_weight_list = []
                    for start, end in G.edges():
                        edges_weight_list.append( [start + ' ' +  end , G[start][end]['weight'] ] )
                    
                    self.network_report_dic['graph_edges'] = dict(sorted( edges_weight_list, reverse = True, key = lambda t: t[1]))
                                        
                    #plotando os grafos
                    if plot_graphs is True:
                        #determinando os vizinhos para plotar
                        neighbor_G = get_neighbor_graph_from_graph(node_name = node_higher_degree_cent, graph = G)                    
                        self.network_weighted_plot(neighbor_G, 30, 60, net_layout = 'random', directed_edges = False, 
                                                   plot_title = str(graph_type) + ' - ' + str(graph_base_title) + ' - node: ' + str(node_higher_degree_cent) + ' (neighbors) [ articles = ' + str(n_articles) + ']')

                elif graph_type == 'digraph':                    

                    #centralidade
                    degree_cent = nx.degree_centrality(G)
                    degree_centrality_sorted = sorted(degree_cent.items(), reverse = True, key = lambda t: t[1])

                    node_higher_degree_cent = degree_centrality_sorted[0][0]
                    #avaliando propriedades de neighborhoods para o node de maior in_degree centrality
                    print('Node with higher degree centrality: ', node_higher_degree_cent)
                    
                    #organizando os nodes pelo degree
                    in_degrees = sorted(G.in_degree(), key = lambda t: t[1] , reverse=True)
                    out_degrees = sorted(G.out_degree(), key = lambda t: t[1] , reverse=True)

                    self.network_report_dic['in_degree_nodes'] = dict( in_degrees )
                    self.network_report_dic['out_degree_nodes'] = dict( out_degrees )                    

                    #organizando as edges pelo weight
                    edges_weight_list = []
                    for start, end in G.edges():
                        edges_weight_list.append( [start + ' - ' +  end , G[start][end]['weight'] ] )
                    
                    self.network_report_dic['digraph_edges'] = dict(sorted( edges_weight_list, reverse = True, key = lambda t: t[1]))
                    
                #plotando a rede completa
                if plot_graphs is True:
                    self.network_weighted_plot(G, min_circle_size, max_circle_size,
                                               net_layout = 'random',
                                               nodes_categories = nodes_categories,
                                               directed_edges = False,
                                               plot_title = str(graph_type) + ' - ' + str(graph_base_title) + ' [ articles = ' + str(n_articles) + ']')

                #análise de paths
                if len(start_end_nodes_to_analyze) == 1 and graph_type == 'digraph':
        
                    #encontrando o heaviest path a partir de um start node
                    heaviest_path, heaviest_weight = find_heaviest_path_from_graph(G, start=start_end_nodes_to_analyze[0], min_edge_weight=min_edge_weight, 
                                                                                   network_type = 'directed', network_name = 'Full network (directed)')

                    self.network_report_dic['path'] = {} 
                    self.network_report_dic['path']['heaviest_weight'] = heaviest_weight
                    self.network_report_dic['path']['heaviest_path'] = heaviest_path
                    self.network_report_dic['path']['start'] = heaviest_path[0]
                    self.network_report_dic['path']['end'] = heaviest_path[-1]

                #análise de paths
                elif len(start_end_nodes_to_analyze) == 2 and graph_type == 'digraph':
                    start_node = start_end_nodes_to_analyze[0]
                    end_node = start_end_nodes_to_analyze[1]

                    #encontrando o heaviest path a partir de um start node
                    heaviest_path, heaviest_weight = find_heaviest_path_from_graph(G, start=start_node, end=end_node, min_edge_weight=min_edge_weight, 
                                                                                   network_type = 'directed', network_name = 'Full network (directed)')
                    #o metodo do networkX não está sendo usado
                    #simple_paths = nx.all_simple_edge_paths(G, start_node, end_node, cutoff=path_nodes_cutoff)
                    #heaviest_path, heaviest_weight = find_heaviest_path_from_edges(simple_paths, G)
                    self.network_report_dic['path'] = {} 
                    self.network_report_dic['path']['heaviest_weight'] = heaviest_weight
                    self.network_report_dic['path']['heaviest_path'] = heaviest_path                    
                    self.network_report_dic['path']['start'] = heaviest_path[0]
                    self.network_report_dic['path']['end'] = heaviest_path[-1]

            except KeyError:
                print(f'ERRO! Não há {graph_type} para analisar')
                pass



    def network_weighted_plot(self, graph, min_circle_size, max_circle_size, net_layout = 'random', nodes_categories = None, directed_edges = False, plot_title = ''):
                
        
        #definindo o graph
        
        G = graph
        
        #calculando o tamanho dos círculos
        degrees = dict(nx.degree(G))

        max_degree_val = np.array(list(degrees.values())).max()
        min_degree_val = np.array(list(degrees.values())).min()

        #dicionário para o tamanho do node (círculo)
        node_size = {}

        #caso haja diferentes degrees
        same_degree_val = False
        if len(set(degrees.values())) > 1:
            for node_name in degrees.keys():
                node_size[node_name] = ( ( (degrees[node_name] - min_degree_val) / (max_degree_val - min_degree_val) ) * (max_circle_size - min_circle_size) ) + min_circle_size
        #caso seja um clique
        else:
            same_degree_val = True
            for node_name in degrees.keys():
                node_size[node_name] = max_circle_size

        #calculando a espessura das edges
        edges_weights = nx.get_edge_attributes(G,'weight')

        #setando os maiores valores de linha
        max_edge_line_width = 10
        min_edge_line_width = 2
        
        #encontrando os maiores valores de weights
        max_edge_val = max(list(edges_weights.values()))
        min_edge_val = min(list(edges_weights.values()))
        
        for edge_name in edges_weights.keys():
            edges_weights[edge_name] = round( (((edges_weights[edge_name] - min_edge_val) / (max_edge_val - min_edge_val) ) * (max_edge_line_width - min_edge_line_width)), 3) + min_edge_line_width

        #o número de cores é equivalente ao número de degrees diferentes no grafo
        n_degrees = len(list(set(degrees.values())))
        palette_nodes = turbo(n_degrees)

        #atribuindo um cores para os nodes
        nodes_color = {}        
        
        #dicionário do tipo: degree_val : color_number
        degree_color_ident_dic = dict( zip( sorted(set(degrees.values())) , range(len(set(degrees.values())))) )            
        
        #definindo a cor em um dic do tipo node : color
        for node in G.nodes():
            nodes_color[node] = palette_nodes[ degree_color_ident_dic[degrees[node]] ]
        
        #fazendo o color map dos edges weights
        #encontrando os valores únicos de weight
        unique_weights_list = list(set(edges_weights.values()))
        unique_weights_list.sort()
        #coletando uma paleta com o número de cores correspondente ao número de edge weights
        palette_colors = turbo(len(unique_weights_list))
        unique_weight_colors_dic = dict(zip(unique_weights_list, palette_colors))
        
        #dicionário com as cores
        edge_colors = {}
        for edge_name in edges_weights:
            weight = edges_weights[edge_name]            
            edge_colors[edge_name] = unique_weight_colors_dic[weight]

        #função de plotagem        
        if net_layout.lower() == 'random':
            nodes_position_dic = nx.drawing.layout.random_layout(G)
        elif net_layout.lower() == 'shell':
            nodes_position_dic = nx.drawing.layout.shell_layout(G)
        elif net_layout.lower() == 'spectral':
            nodes_position_dic = nx.drawing.layout.spectral_layout(G)
        else:
            print('ERRO! Insira um layout válido para plotagem.')
            return
            
        x_pos, y_pos = list(zip(*[ [item[1][0], item[1][1]] for item in nodes_position_dic.items() ]))
        xmax = max(x_pos)
        xmin = min(x_pos)
        ymax = max(y_pos)
        ymin = min(y_pos)

        #plotando a rede
        p = figure(toolbar_location=None, 
                   title=plot_title,                   
                   match_aspect=False,
                   x_range=Range1d(xmin-((xmax-xmin)*0.2), xmax+((xmax-xmin)*0.4)), y_range=Range1d(ymin-((ymax-ymin)*0.2), ymax+((ymax-ymin)*0.3)),
                   background_fill_color='white', 
                   plot_width=1500, plot_height=1500,
                   min_border=40)

        p.title.text_font_size = '30pt'
        p.axis.visible = False
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
                
        #adicionando as edges
        for edge_name in edges_weights.keys():
            source = edge_name[0]
            target = edge_name[1]
            #print('source_name: ', source, nodes_position_dic[source], '; target: ', target, nodes_position_dic[target])
            if directed_edges is True:
                p.add_layout(Arrow(end=VeeHead(size=40), line_color="gray", line_width=edges_weights[edge_name],
                                   x_start=nodes_position_dic[source][0], y_start=nodes_position_dic[source][1], 
                                   x_end=nodes_position_dic[target][0], y_end=nodes_position_dic[target][1]))
            else:
                p.line([nodes_position_dic[source][0], nodes_position_dic[target][0]], [nodes_position_dic[source][1], nodes_position_dic[target][1]], 
                       line_width=edges_weights[edge_name], color='gray', alpha=0.4)
                
            
        #adicionando o círculo (node)
        for node_name in nodes_position_dic.keys():
            
            #caso não seja clique
            if same_degree_val is False:
                node_color = nodes_color[node_name]
            else:
                node_color = 'gray'
            
            p.circle(nodes_position_dic[node_name][0], nodes_position_dic[node_name][1], 
                     size=node_size[node_name], 
                     color=node_color, 
                     line_color='black',
                     line_width=2,
                     alpha=1)

        #adicionando os textos
        text_list = []
        x_pos_list = []
        y_pos_list = []
        for node in nodes_position_dic.keys():
            text_list.append(node)
            x_pos_list.append(nodes_position_dic[node][0] + 0.005)
            y_pos_list.append(nodes_position_dic[node][1] + 0.005)        
        xytext_dic = dict(x=x_pos_list, y=y_pos_list, text=text_list)
        fontsize_scale_factor = 0.5
        offset_scale_factor=0.3
        for i in range(len(text_list)):
            label = xytext_dic['text'][i]
            x_pos = xytext_dic['x'][i]
            y_pos = xytext_dic['y'][i]
            
            #caso não seja clique
            if same_degree_val is False:
                node_text_color = nodes_color[label]
            else:
                node_text_color = 'black'
            
            citation = Label(x=x_pos, y=y_pos, angle=0, text=label,
                             x_offset=node_size[label]*offset_scale_factor, y_offset=node_size[label]*offset_scale_factor,
                             text_font_size = str(int(node_size[label]*fontsize_scale_factor)) + 'pt', 
                             text_font='helvetica',
                             text_font_style='bold',
                             text_color=node_text_color,
                             background_fill_alpha=1.0)
            p.add_layout(citation)

        #caso o grafo não seja clique
        if same_degree_val is False:
            color = LinearColorMapper(palette = 'Turbo256',
                                      low = min_degree_val,
                                      high = max_degree_val)
        
            color_bar = ColorBar(color_mapper = color,
                                 label_standoff = 14,
                                 orientation='horizontal',
                                 location = (0,0),
                                 major_label_text_font_size='30pt',
                                 bar_line_color='black',
                                 bar_line_width=2,
                                 title = '',
                                 title_text_font_size='30pt')
              
            # Defining the position of the color bar
            p.add_layout(color_bar, 'above')

        print(f'Salvando a figura ~/Outputs/analyses/P{self.last_fig_filename_index}.png ...')
        export_png(p, filename=self.diretorio + f'/Outputs/analyses/P{self.last_fig_filename_index}.png')
        
        #atualizando o número do plot_index
        self.last_fig_filename_index = ( ( len('0000') - len(str(int(self.last_fig_filename_index)+1)) ) * '0' ) + str(int(self.last_fig_filename_index) + 1)



    def chord_plot(self, grouped_DF):
        
        def get_first_column_names(term):            
            
            split_term = re.findall(r'[\w\s]+(?=_)', term)
            return split_term[0]

        def get_second_column_names(term):            
            
            split_term = re.findall(r'(?<=_)[\w\s]+', term)
            return split_term[0] 
        
        #resetando o index        
        grouped_DF = grouped_DF.reset_index()
        grouped_DF.columns = grouped_DF.columns.get_level_values(1)
        
        #coletando em uma DF somente os valores conexão (weights)
        weights_DF = grouped_DF.copy()
        
        #mudando o dtype para a coluna weight para plotar
        weights_DF['count'] = weights_DF['count'].values.astype('int32')
        
        #associando um valor inteiro para cada label para usar o hollowviews
        sources_names_series = grouped_DF[''].apply(get_first_column_names)
        taget_names_series = grouped_DF[''].apply(get_second_column_names)
        all_names = list(np.unique(np.hstack([sources_names_series.values, taget_names_series.values])))

        #função para substiuição        
        names_dic = {}
        for i in range(len(all_names)):
            names_dic[all_names[i]] = i

        def replace_name_by_numbers(name):
            return names_dic[name]
        
        weights_DF['source'] = sources_names_series.apply(replace_name_by_numbers)
        weights_DF['target'] = taget_names_series.apply(replace_name_by_numbers)

        #adicionando os nodes labels a um DF
        nodes_labels_df = pd.DataFrame([])
        nodes_labels_df['index'] = names_dic.values()
        nodes_labels_df['name'] = names_dic.keys()
        
        #plotting with holoviews        
        hv.extension('bokeh')
        hv.output(size=500)
        #chord = hv.Chord(concat_network_relat)
        
        nodes = hv.Dataset(nodes_labels_df, 'index')
        chord = hv.Chord((weights_DF[['source', 'target', 'count']].copy(), nodes)).select(value=(5, None))
        chord.opts(opts.Chord(cmap='Category20', edge_cmap='Category20', edge_color=dim('source').str(), 
                              labels='name', node_color=dim('index').str()))
        chord.opts(label_text_font_size='23pt')
        
        print(f'Salvando a figura ~/Outputs/analyses/P{self.last_fig_filename_index}.png ...')
        hv.save(chord, self.diretorio + f'/Outputs/analyses/P{self.last_fig_filename_index}.png', backend='bokeh')
        
        #atualizando o número do plot_index
        self.last_fig_filename_index = ( ( len('0000') - len(str(int(self.last_fig_filename_index)+1)) ) * '0' ) + str(int(self.last_fig_filename_index) + 1)



    def group_num_boxplot(self, grouped_DF = None, input1 = '', input2 = '', 
                          axes_label1 = '', axes_label2 = '', ymin = None, ymax = None, 
                          min_occurrences = 0, size_factor = 1, grouplabel_x_offset = 0, 
                          colobar_nticks = 20, plot_width=800, plot_height=1200,
                          export_groups_to_csv = False):
        
        
        print('input1: ', input1)
        print('input2: ', input2)

        cats = grouped_DF.describe().index.values
        cats_counter = grouped_DF.describe()['count'].values
        cats_number = len(cats)
        max_cat_allowed = 50
        if cats_number > max_cat_allowed:
            print(f'ERRO! Reduzir para {max_cat_allowed} o número de categorias para o group_num_plot.')
            print('Número atual: ', cats_number)
            return
        
        #encontrando os valores de percentiles para cada grupo
        qmin = grouped_DF.quantile(q=0.00)
        q1 = grouped_DF.quantile(q=0.25)
        q2 = grouped_DF.quantile(q=0.5)
        q3 = grouped_DF.quantile(q=0.75)
        qmax = grouped_DF.quantile(q=1.00)
        iqr = q3 - q1
        upper = q3 + 1.5*iqr
        lower = q1 - 1.5*iqr
        
        # find the outliers for each category
        def outliers(group):
            cat = group.name
            result = group[ ( group > upper[cat] ) | ( group < lower[cat] ) ]
            return result
                    
        out = grouped_DF.apply(outliers).dropna(axis=0, how='any')
                
        # prepare outlier data for plotting, we need coordinates for every outlier.
        if not out.empty:
            outx = []
            outy = []
            for keys in out.index:
                outx.append(keys[0])
                outy.append(out.loc[keys[0]].loc[keys[1]].loc[keys[2]])
        
        #colocando os números em frente às categorias para exportar os grupos
        if export_groups_to_csv is True:
            cats_with_group_numbers = []
            cat_counter = 1
            for cat in cats:
                cats_with_group_numbers.append(cat + ' (' + str(cat_counter) + ')')  
                cat_counter += 1
            cats = cats_with_group_numbers
        
        p = set_bokeh_main_boxplot(ymin, ymax, cats, axes_label1, axes_label2, min_occurrences, plot_width, plot_height, cats_number, size_factor, 200)

        # if no outliers, shrink lengths of stems to be no longer than the minimums or maximums
        upper_values = [min([x,y]) for (x,y) in zip(list(qmax.values), upper.values)]
        lower_values = [max([x,y]) for (x,y) in zip(list(qmin.values), lower.values)]
                
        # stems
        p.segment(cats, upper_values, cats, q3.values, line_width=3, line_color="black")
        p.segment(cats, lower_values, cats, q1.values, line_width=3, line_color="black")
        
        #palettes
        #paleta de cores
        colors1 = plasma(len(q1.values))
        colors2 = turbo(len(q1.values))
        
        # boxes
        p.vbar(cats, 0.7, q2.values, q3.values, line_width=2, fill_color=colors1, line_color="white")
        p.vbar(cats, 0.7, q1.values, q2.values, line_width=2, fill_color=colors2, line_color="white")
        
        # whiskers (almost-0 height rects simpler than segments)
        p.rect(cats, lower_values, width=0.3, height=0.1, line_width=3, line_color="black")
        p.rect(cats, upper_values, width=0.3, height=0.1, line_width=3, line_color="black")
        
        # outliers
        if not out.empty:
            p.circle(outx, outy, size=11, color="red", fill_alpha=0.6)
            
        #adicionando o count sobre as barras
        citation_yval = max(outy) + (max(outy)*0.1)
        for i in range(len(cats_counter)):
            citation = Label(x=i+cats_number*size_factor*0.025, x_offset=grouplabel_x_offset, 
                             y=citation_yval, angle=np.pi/2, text=str(int(cats_counter[i])),
                             text_font_size = str(int(cats_number*size_factor) + 0) + 'pt', background_fill_alpha=1.0)
            p.add_layout(citation)
            
        #salvando os grupos
        if export_groups_to_csv is True:
            
            
            
            index_lists_dic = load_dic_from_json(self.diretorio + '/Inputs/Index_lists.json')
            index_lists_dic['P' + self.last_fig_filename_index] = {}
            
            group_counter = 0
            for group in grouped_DF.groups:

                index_lists_dic['P' + self.last_fig_filename_index][group_counter] = list(grouped_DF.get_group(group).index.get_level_values(0).values)
                group_counter += 1

            save_dic_to_json(self.diretorio + '/Inputs/Index_lists.json', index_lists_dic)
            print('Salvando a index_list dos grupos em ~/Inputs/Index_lists.json ...')

        print(f'Salvando a figura ~/Outputs/analyses/P{self.last_fig_filename_index}.png ...')
        export_png(p, filename=self.diretorio + f'/Outputs/analyses/P{self.last_fig_filename_index}.png')
        
        #atualizando o número do plot_index
        self.last_fig_filename_index = ( ( len('0000') - len(str(int(self.last_fig_filename_index)+1)) ) * '0' ) + str(int(self.last_fig_filename_index) + 1)
        


    def cat_cat_gridplot(self, DF = None, DF_x_column = '', DF_y_column = '', DF_val_column = '', 
                         min_occurrences = 0, size_factor = 0.4, colobar_nticks = 20, palette = 'Viridis256', 
                         background_fill_color = 'blue', color_mapper_range = None, plot_width=1000, plot_height=1000):


        #proc_vals = len(DF.index)
        #print('articles: ', proc_vals)
        
        #determinando os valores mínimos e máximos de ocorrências
        lowest_val , highest_val = DF[DF_val_column].values.min(), DF[DF_val_column].values.max()

        #valores de x e y                
        x_vals = sorted(list(np.unique(DF[DF_x_column].values)))
        y_vals = sorted(list(np.unique(DF[DF_y_column].values)))[::-1]
        max_axis_vals = max(len(x_vals),len(y_vals))

        p = figure(toolbar_location=None,
                   title='.', #'articles = ' + str(proc_vals),
                   background_fill_color = background_fill_color,
                   min_border = 200,
                   x_range=x_vals, y_range=y_vals,
                   x_axis_location="above", 
                   plot_width=plot_width, plot_height=plot_height)
        
        #definições gerais
        p.title.text_font_size = str(int(max_axis_vals*size_factor) + 2) + 'pt'
        p.outline_line_color = "black"
        p.outline_line_width = 1
        p.grid.grid_line_color = 'gray'
        p.grid.grid_line_width = 1
        p.grid.grid_line_alpha = 0.5
        p.axis.axis_line_color = None
        p.axis.major_tick_line_color = None
        p.axis.major_label_text_font_size = str(int(max_axis_vals*size_factor) + 0) + 'pt'
        p.axis.major_label_standoff = 0
        p.xaxis.major_label_orientation = np.pi/3

        if color_mapper_range is not None and len(color_mapper_range) == 2:
            bottom_map_val = color_mapper_range[0]
            upper_map_val = color_mapper_range[1]
        else:
            bottom_map_val = lowest_val
            upper_map_val = highest_val

        # color mapping
        mapper = LinearColorMapper(palette=palette,
                                   low=bottom_map_val,
                                   high=upper_map_val)
        
        p.rect(x=DF_x_column, y=DF_y_column, width=1, height=1,
               source=DF,
               fill_color={'field': DF_val_column, 'transform': mapper},
               line_color=None)
        
        #colorbar
        color_bar = ColorBar(color_mapper=mapper,
                             ticker=BasicTicker(desired_num_ticks=colobar_nticks),
                             orientation = 'horizontal',
                             label_standoff=6, border_line_color=None, location=(0, 0))
        color_bar.title_text_font_size = str(int(max_axis_vals*size_factor) + 0) + 'pt'
        color_bar.major_label_text_font_size = str(int(max_axis_vals*size_factor) + 0) + 'pt'
        color_bar.bar_line_color = 'black'
        color_bar.bar_line_width = 0
        color_bar.major_tick_line_alpha = 0
        color_bar.major_tick_out = int(max_axis_vals*size_factor*0.8)
        color_bar.title = f'occurrences (min = {min_occurrences})'
        color_bar.title_text_font_size = str(int(max_axis_vals*size_factor) + 0) + 'pt'
        
        p.add_layout(color_bar, 'above')
        
        print(f'Salvando a figura ~/Outputs/analyses/P{self.last_fig_filename_index}.png ...')
        export_png(p, filename=self.diretorio + f'/Outputs/analyses/P{self.last_fig_filename_index}.png', width=plot_width, height=plot_height)
        
        #atualizando o número do plot_index
        self.last_fig_filename_index = ( ( len('0000') - len(str(int(self.last_fig_filename_index)+1)) ) * '0' ) + str(int(self.last_fig_filename_index) + 1)


    
    def pca_scatter_plot(self, DF = None, pca_model = None, loadings_label = ['', ''], 
                         axes_labels = ['', ''], input1 = '', input2 = '',                         
                         x_min = None, x_max = None, y_min = None, y_max = None,
                         plot_width=500, plot_height=500, 
                         find_clusters = False, n_clusters = 3,
                         loading_arrow_factor = 10,
                         export_groups_to_csv = False,
                         cluster_preffix = '',
                         show_figure = False):
        
        
        print('input1: ', input1)
        print('input2: ', input2)
        proc_vals = len(DF.index)
        print('articles: ', proc_vals)        
        
        DF, xmin, xmax, ymin, ymax = filter_DF_by_min_max_vals(DF, input1=input1, input2=input2, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

        x = DF[input1].values
        y = DF[input2].values
        
        print('filtrando...')
        print('xmin filtered = ', x.min())
        print('xmax filtered = ', x.max())
        print('ymin filtered = ', y.min())
        print('ymax filtered = ', y.max())    
        
        #encontrando clusters
        if find_clusters is True:
            xy_vals_labeled, c_labels, c_centers = clusterize(x, y, n_clusters)
            DF_labeled = DF.copy()
            DF_labeled['groups'] = c_labels + 1
            DF_labeled = DF_labeled.reset_index()[['doc','groups']]
            DF_labeled = DF_labeled.drop_duplicates()
            
        x_delta_limit = xmax*0.1
        y_delta_limit = ymax*0.1
        x_label , y_label = axes_labels

        #plotando
        if show_figure is True:
            tooltips = [("(x,y)", "($x, $y)"),
                        ("doc", "@doc")]
            p = set_bokeh_main_numplot(proc_vals, xmin, xmax, ymin, ymax, x_delta_limit, y_delta_limit, x_label, y_label, plot_width, plot_height,
                                    tooltips = tooltips, toolbar_location = 'right')
        else:
            p = set_bokeh_main_numplot(proc_vals, xmin, xmax, ymin, ymax, x_delta_limit, y_delta_limit, x_label, y_label, plot_width, plot_height)
        
        ph = set_bokeh_horizontal_hist(p)
        pv = set_bokeh_vertical_hist(p)

        #adicionando as linhas dos eixos
        p.line(x=[xmin,xmax], y=[0,0], line_alpha=1, line_color='gray')
        p.line(x=[0,0], y=[ymin,ymax], line_alpha=1, line_color='gray')
        
        #plotando os scores
        if find_clusters is True:
            
            color_list = [self.palette_colors1[ c_label ] for c_label in c_labels]
            
            #adicionando os pontos
            source = ColumnDataSource(data=dict(x = DF[input1], y = DF[input2].values, doc = DF.index.levels[0].values, color=color_list))
            p.circle('x', 'y', color='color', source=source, size=25, alpha=0.4)

            #adicionando os centroides
            for i in range(len(c_centers)):
                p.circle(c_centers[i][0], c_centers[i][1], line_color='black', line_width=3, fill_color=self.palette_colors1[i] , size=40, alpha=0.8)
            
            #adicionando os textos
            cluster_labels = [cluster_preffix + str(i) for i in range(1, n_clusters+1)]
            source = ColumnDataSource(dict(x=c_centers[:,0] + (c_centers[:, 0].max() * 0.09), y=c_centers[:,1] + (c_centers[:, 1].max() * 0.09), text=cluster_labels))
            glyph = Text(x="x", y="y", text="text", angle=0, text_color="black", text_font_size='35pt')
            p.add_glyph(source, glyph)

        else:
            p.circle(x, y, color="blue", size=25, alpha=0.3)

        #plotando os vetores de correlação        
        loadings = pca_model.components_.T * np.sqrt(pca_model.explained_variance_)
        for loading in loadings:
            p.add_layout(Arrow(end=VeeHead(size=35), line_color="black", line_width=3,
                               x_start=0, y_start=0, x_end=loading[0]*loading_arrow_factor, y_end=loading[1]*loading_arrow_factor))

        #adicionando os label dos loadings
        source = ColumnDataSource(dict(x=(loadings[:,0] * loading_arrow_factor) + (loadings[:,0] * loading_arrow_factor * 0.03), 
                                       y=(loadings[:,1] * loading_arrow_factor) + (loadings[:,1] * loading_arrow_factor * 0.03), 
                                       text=loadings_label))
        glyph = Text(x="x", y="y", text="text", angle=0, text_color="black", text_font_size='32pt')
        p.add_glyph(source, glyph)
    
        #plotando o histograma de cima
        hhist, hedges = np.histogram(x, bins=20, range=(xmin,xmax))        
        hmax = max(hhist)*1.1
        ph.x_range = p.x_range
        ph.y_range = Range1d( 0 - (0.1*hmax), hmax)
        ph.yaxis.ticker = (0, int(hmax))
        ph.yaxis.bounds = (0, int(hmax))
        ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hhist, color="white", line_color="red", line_width = 2)

        #plotando o histograma do lado
        vhist, vedges = np.histogram(y, bins=20, range=(ymin, ymax))
        vmax = max(vhist)*1.1
        pv.x_range = Range1d( 0 - (0.1*vmax), vmax)
        pv.y_range = p.y_range
        pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vhist, color="white", line_color="red", line_width = 2)
        pv.xaxis.ticker = (0, int(vmax))
        pv.xaxis.bounds = (0, int(vmax))        

        #fazendo o layout            
        layout = gridplot([[p, pv], [ph, None]], merge_tools=False)
        
        if show_figure is True:        
            show(layout)
        
        else:                        
            print(f'Salvando a figura ~/Outputs/analyses/P{self.last_fig_filename_index}.png ...')
            export_png(layout, filename=self.diretorio + f'/Outputs/analyses/P{self.last_fig_filename_index}.png')
            
            #salvando os grupos
            if find_clusters is True and export_groups_to_csv is True:
                
                
                index_lists_dic = load_dic_from_json(self.diretorio + '/Inputs/Index_lists.json')
                index_lists_dic['P' + self.last_fig_filename_index] = {}
                labels_grouped_DF = DF_labeled.groupby('groups')
                for group in labels_grouped_DF.groups:
                    index_lists_dic['P' + self.last_fig_filename_index][group] = list(labels_grouped_DF.get_group(group)['doc'].values)
                save_dic_to_json(self.diretorio + '/Inputs/Index_lists.json', index_lists_dic)
                print('Salvando a index_list dos grupos em ~/Inputs/Index_lists.json ...')
            
            #atualizando o número do plot_index
            self.last_fig_filename_index = ( ( len('0000') - len(str(int(self.last_fig_filename_index)+1)) ) * '0' ) + str(int(self.last_fig_filename_index) + 1)



    def num_num_scatterplot(self, DF = None, axes_labels = ['', ''], input1 = '', input2 = '', 
                            x_min = None, x_max = None, y_min = None, y_max = None,
                            plot_width=1000, plot_height=1000, regression=False, 
                            find_clusters = True, n_clusters = 3, export_groups_to_csv = False, cluster_preffix = '',
                            show_figure = False):

        
        print('input1: ', input1)
        print('input2: ', input2)
        proc_vals = len(DF.index)
        print('articles: ', proc_vals)        

        DF, xmin, xmax, ymin, ymax = filter_DF_by_min_max_vals(DF, input1=input1, input2=input2, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

        x = DF[input1].values
        y = DF[input2].values

        print('filtrando...')
        print('xmin filtered = ', x.min())
        print('xmax filtered = ', x.max())
        print('ymin filtered = ', y.min())
        print('ymax filtered = ', y.max())
        
        #encontrando clusters
        if find_clusters is True:
            xy_vals_labeled, c_labels, c_centers = clusterize(x, y, n_clusters)
            DF_labeled = DF.copy()
            DF_labeled['groups'] = c_labels + 1
            DF_labeled = DF_labeled.reset_index()[['doc','groups']]        
            DF_labeled = DF_labeled.drop_duplicates()
            
        x_delta_limit = xmax*0.1
        y_delta_limit = ymax*0.1
        x_label , y_label = axes_labels
        
        #plotando
        if show_figure is True:
            tooltips = [("(x,y)", "($x, $y)"),
                        ("doc", "@doc")]
            p = set_bokeh_main_numplot(proc_vals, xmin, xmax, ymin, ymax, x_delta_limit, y_delta_limit, x_label, y_label, plot_width, plot_height,
                                    tooltips = tooltips, toolbar_location = 'right')
        else:
            p = set_bokeh_main_numplot(proc_vals, xmin, xmax, ymin, ymax, x_delta_limit, y_delta_limit, x_label, y_label, plot_width, plot_height)
        
        ph = set_bokeh_horizontal_hist(p)
        pv = set_bokeh_vertical_hist(p)

        #plotando
        if find_clusters is True:

            color_list = [self.palette_colors1[ c_label ] for c_label in c_labels]            
            
            #adicionando os pontos
            source = ColumnDataSource(data=dict(x = DF[input1].values, y = DF[input2].values, doc = DF.index.get_level_values(1).values, color=color_list))
            p.circle('x', 'y', color='color', source=source, size=25, alpha=0.4)
                        
            #adicionando os centroides
            for i in range(len(c_centers)):
                p.circle(c_centers[i][0], c_centers[i][1], line_color='black', line_width=3, fill_color=self.palette_colors1[i] , size=40, alpha=0.8)
            
            #adicionando os textos
            cluster_labels = [cluster_preffix + str(i) for i in range(1, n_clusters+1)]
            source = ColumnDataSource(dict(x=c_centers[:,0] + (c_centers[:, 0].max() * 0.02), y=c_centers[:,1] + (c_centers[:, 1].max() * 0.02), text=cluster_labels))
            glyph = Text(x="x", y="y", text="text", angle=0, text_color="black", text_font_size='45pt')
            p.add_glyph(source, glyph)                    

        else:
            p.circle(x, y, color="blue", size=25, alpha=0.3)
        
        if regression is True:
            #fazendo a regressão linear
            b0, b1, r_sq = lin_regression(x,y)
            p.line([xmin, xmax],[b0 + ymin*b1, b0 + ymax*b1], line_width=15, color='red', line_alpha=0.6, line_dash='solid')

        #plotando o histograma de cima
        hhist, hedges = np.histogram(x, bins=10, range=(xmin,xmax))        
        hmax = max(hhist)*1.1
        ph.x_range = p.x_range
        ph.y_range = Range1d( 0 - (0.1*hmax), hmax)
        ph.yaxis.ticker = (0, int(hmax))
        ph.yaxis.bounds = (0, int(hmax))
        ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hhist, color="white", line_color="red", line_width=3)

        #plotando o histograma do lado
        vhist, vedges = np.histogram(y, bins=10, range=(ymin, ymax))
        vmax = max(vhist)*1.1
        pv.x_range = Range1d( 0 - (0.1*vmax), vmax)
        pv.y_range = p.y_range
        pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vhist, color="white", line_color="red", line_width=3)
        pv.xaxis.ticker = (0, int(vmax))
        pv.xaxis.bounds = (0, int(vmax))

        layout = gridplot([[p, pv], [ph, None]], merge_tools=False)
        
        if show_figure is True:
            show(layout)
        
        else:                        
            print(f'Salvando a figura ~/Outputs/analyses/P{self.last_fig_filename_index}.png ...')
            export_png(layout, filename=self.diretorio + f'/Outputs/analyses/P{self.last_fig_filename_index}.png')
            
            #salvando os grupos
            if find_clusters is True and export_groups_to_csv is True:
                
                
                index_lists_dic = load_dic_from_json(self.diretorio + '/Inputs/Index_lists.json')
                index_lists_dic['P' + self.last_fig_filename_index] = {}
                labels_grouped_DF = DF_labeled.groupby('groups')
                for group in labels_grouped_DF.groups:
                    index_lists_dic['P' + self.last_fig_filename_index][group] = list(labels_grouped_DF.get_group(group)['doc'].values)
                save_dic_to_json(self.diretorio + '/Inputs/Index_lists.json', index_lists_dic)
                print('Salvando a index_list dos grupos em ~/Inputs/Index_lists.json ...')
            
            #atualizando o número do plot_index
            self.last_fig_filename_index = ( ( len('0000') - len(str(int(self.last_fig_filename_index)+1)) ) * '0' ) + str(int(self.last_fig_filename_index) + 1)

    

    def num_num_hexbinplot(self, DF = None, axes_labels = ['', ''], input1 = '', input2 = '', 
                           x_min = None, x_max = None, y_min = None, y_max = None, hex_size = 10,
                           plot_width=1000, plot_height=1000):

        
        
        print('input1: ', input1)
        print('input2: ', input2)
        proc_vals = len(DF.index)
        print('articles: ', proc_vals)
        
        #coletando os resultados de interesse
        x = DF[input1].values
        y = DF[input2].values

        print('xmin found (DF) = ', DF[input1].values.min())
        print('xmax found (DF) = ', DF[input1].values.max())
        print('ymin found (DF) = ', DF[input2].values.min())
        print('ymax found (DF) = ', DF[input2].values.max())
        
        #xmin
        if x_min is None:
            xmin = DF[input1].values.min()
        else:
            xmin = x_min
        #xmax
        if x_max is None:
            xmax = DF[input1].values.max()
        else:
            xmax = x_max        
        #ymin
        if y_min is None:
            ymin = DF[input2].values.min()
        else:
            ymin = y_min
        #ymax
        if y_max is None:
            ymax = DF[input2].values.max()
        else:
            ymax = y_max            
        
        x_delta_limit, y_delta_limit = xmax*0.1, ymax*0.1
        x_label , y_label = axes_labels

        #gerando os plots
        p = set_bokeh_main_numplot(proc_vals, xmin, xmax, ymin, ymax, x_delta_limit, y_delta_limit, x_label, y_label, plot_width, plot_height)
        ph = set_bokeh_horizontal_hist(p)
        pv = set_bokeh_vertical_hist(p)

        #plotando
        r, bins = p.hexbin(x, y, size=hex_size, palette=self.palette_bins[0])
        p.circle(x, y, color="gray", size=6, alpha=0)

        #adicionando o colormap
        mapper = LinearColorMapper(palette=self.palette_bins[0], low=bins['counts'].values.min(), high=bins['counts'].values.max())
        color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="7px",
                             ticker=BasicTicker(desired_num_ticks=12),
                             orientation = 'horizontal',
                             label_standoff=12, border_line_color=None, location=(0, 0))
        color_bar.major_label_text_font_size = '22pt'
        color_bar.bar_line_color = 'black'
        color_bar.bar_line_width = 2
        color_bar.major_tick_line_alpha = 0
        color_bar.major_tick_out = 5
        color_bar.title = 'Occurrences'
        color_bar.title_text_font_size = '19pt'
        
        p.add_layout(color_bar, 'above')    
    
        #plotando o histograma de cima
        hhist, hedges = np.histogram(x, bins=20, range=(xmin,xmax))        
        hmax = max(hhist)*1.1
        ph.x_range = p.x_range
        ph.y_range = Range1d( 0 - (0.1*hmax), hmax)
        ph.yaxis.ticker = (0, int(hmax))
        ph.yaxis.bounds = (0, int(hmax))
        ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hhist, color="white", line_color="red")

        #plotando o histograma do lado
        vhist, vedges = np.histogram(y, bins=20, range=(ymin, ymax))
        vmax = max(vhist)*1.1
        pv.x_range = Range1d( 0 - (0.1*vmax), vmax)
        pv.y_range = p.y_range
        pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vhist, color="white", line_color="red")
        pv.xaxis.ticker = (0, int(vmax))
        pv.xaxis.bounds = (0, int(vmax))        
        
        layout = gridplot([[p, pv], [ph, None]], merge_tools=False)
        
        print(f'Salvando a figura ~/Outputs/analyses/P{self.last_fig_filename_index}.png ...')
        export_png(layout, filename=self.diretorio + f'/Outputs/analyses/P{self.last_fig_filename_index}.png')
        
        #atualizando o número do plot_index
        self.last_fig_filename_index = ( ( len('0000') - len(str(int(self.last_fig_filename_index)+1)) ) * '0' ) + str(int(self.last_fig_filename_index) + 1)


    
    def group_num_num_scatterplot(self, DF = None, grouped = None, axes_labels = ['', ''], input1 = '', input2 = '', 
                                  x_min = None, x_max = None, y_min = None, y_max = None,
                                  hex_size = 10, plot_width=1000, plot_height=1000, 
                                  regression = False, export_groups_to_csv = False,
                                  cluster_preffix = ''):        
        

        print('input1: ', input1)
        print('input2: ', input2)
        proc_vals = len(DF.index)
        print('articles: ', proc_vals)

        print('xmin found (aut) = ', DF[input1].values.min())
        print('xmax found (aut) = ', DF[input1].values.max())
        print('ymin found (aut) = ', DF[input2].values.min())
        print('ymax found (aut) = ', DF[input2].values.max())        
        
        #xmin
        if x_min is None:
            xmin = DF[input1].values.min()
        else:
            xmin = x_min
        #xmax
        if x_max is None:
            xmax = DF[input1].values.max()
        else:
            xmax = x_max        
        #ymin
        if y_min is None:
            ymin = DF[input2].values.min()
        else:
            ymin = y_min
        #ymax
        if y_max is None:
            ymax = DF[input2].values.max()
        else:
            ymax = y_max
        
        print('xmin inserted = ', xmin)
        print('xmax inserted = ', xmax)
        print('ymin inserted = ', ymin)
        print('ymax inserted = ', ymax)
                
        x_delta_limit, y_delta_limit = xmax*0.1, ymax*0.1
        x_label , y_label = axes_labels
        n_groups = len(grouped.groups)
        print('grupos de categoria encontrados: ', n_groups)
        if n_groups > len(self.palette_colors1):
            print('ERRO!')
            print('O número de grupos é maior que o número de cores na paleta para plotar.')
            return
        
        #gerando os plots
        p = set_bokeh_main_numplot(proc_vals, xmin, xmax, ymin, ymax, x_delta_limit, y_delta_limit, x_label, y_label, plot_width, plot_height)
        ph = set_bokeh_horizontal_hist(p)
        pv = set_bokeh_vertical_hist(p)        
        above = get_blank_figure(main_plot = p, plot_height=700, x_range=(0, 200), y_range=(-(800), 100))

        #criando a DF para exportar
        if export_groups_to_csv is True:
            DF_labeled = pd.DataFrame([], index=[[],[]])

        #plotando
        counter = 0
        hmax = 0
        vmax = 0
        for group in grouped.groups:
            grouped_DF = grouped.get_group(group)

            #coletando os resultados de interesse
            x = grouped_DF[input1].values
            y = grouped_DF[input2].values
        
            p.circle(x, y, color=self.palette_colors1[counter], size=17, alpha=0.3, line_color='black', line_width=1)

            if regression is True:
                #fazendo a regressão linear
                b0, b1, r_sq = lin_regression(x,y)
                p.line([xmin, xmax],[b0 + ymin*b1, b0 + ymax*b1], line_width=15, color=self.palette_colors1[counter], line_alpha=0.6, line_dash='solid')

            #adicionando as legendas
            n_vals = len(x)
            above.circle(x=20, y=-counter*70, size=40, alpha=0.6, color=self.palette_colors1[counter], line_color='black', line_width=3)
            leg_label = Label(x=28, y=-(counter*70)-34, text=cluster_preffix+str(counter+1)+': '+group+f' ({n_vals})', text_font_size = '30pt', text_color = 'black')
            above.add_layout(leg_label)            
        
            #plotando o histograma de cima
            hhist, hedges = np.histogram(x, bins=10, range=(xmin,xmax))        
            hmax = 1.1
            ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hhist/hhist.max(), color=self.palette_colors1[counter], alpha=0.6)
            
            #plotando o histograma do lado
            vhist, vedges = np.histogram(y, bins=10, range=(ymin, ymax))    
            vmax = 1.1
            pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vhist/vhist.max(), color=self.palette_colors1[counter], alpha=0.6)        

            #ajustando a DF para exportar os grupos
            if export_groups_to_csv is True:
                temp_DF = grouped_DF.copy()
                temp_DF['groups'] = [counter + 1] * n_vals
                temp_DF = temp_DF.reset_index()[['doc', 'groups']]
                temp_DF = temp_DF.drop_duplicates()  ###################################################### checar
                DF_labeled = pd.concat( [ DF_labeled, temp_DF], axis=0)
            
            counter += 1

        #ajustando a escala dos histogramas após a plotagem
        ph.x_range = p.x_range
        ph.y_range = Range1d( 0 - (0.1*hmax), hmax)
        ph.yaxis.visible = False
        ph.yaxis.bounds = (0, int(hmax))
        pv.x_range = Range1d( 0 - (0.1*vmax), vmax)
        pv.y_range = p.y_range
        pv.xaxis.visible = False
        pv.xaxis.bounds = (0, int(vmax))
        
        layout = gridplot([[above, None], [p, pv], [ph, None]], merge_tools=False)
        
        print(f'Salvando a figura ~/Outputs/analyses/P{self.last_fig_filename_index}.png ...')
        export_png(layout, filename=self.diretorio + f'/Outputs/analyses/P{self.last_fig_filename_index}.png')
        
        #salvando os grupos
        if export_groups_to_csv is True:
            
            
            index_lists_dic = load_dic_from_json(self.diretorio + '/Inputs/Index_lists.json')
            index_lists_dic['P' + self.last_fig_filename_index] = {}
            labels_grouped_DF = DF_labeled.groupby('groups')
            for group in labels_grouped_DF.groups:
                index_lists_dic['P' + self.last_fig_filename_index][group] = list(labels_grouped_DF.get_group(group)['doc'].values)
            save_dic_to_json(self.diretorio + '/Inputs/Index_lists.json', index_lists_dic)
            print('Salvando a index_list dos grupos em ~/Inputs/Index_lists.json ...')
        
        #atualizando o número do plot_index
        self.last_fig_filename_index = ( ( len('0000') - len(str(int(self.last_fig_filename_index)+1)) ) * '0' ) + str(int(self.last_fig_filename_index) + 1)



    def group_num_num_hexbinplot(self, DF = None, grouped = None, axes_labels = ['', ''], input1 = '', input2 = '', 
                                 x_min = None, x_max = None, y_min = None, y_max = None,
                                 hex_size = 10, plot_width=1000, plot_height=1000):

        
        
        
        
        
        
        print('input1: ', input1)
        print('input2: ', input2)
        proc_vals = len(DF.index)
        print('articles: ', proc_vals)

        print('xmin found (aut) = ', DF[input1].values.min())
        print('xmax found (aut) = ', DF[input1].values.max())
        print('ymin found (aut) = ', DF[input2].values.min())
        print('ymax found (aut) = ', DF[input2].values.max())        
        
        #xmin
        if x_min is None:
            xmin = DF[input1].values.min()
        else:
            xmin = x_min
        #xmax
        if x_max is None:
            xmax = DF[input1].values.max()
        else:
            xmax = x_max        
        #ymin
        if y_min is None:
            ymin = DF[input2].values.min()
        else:
            ymin = y_min
        #ymax
        if y_max is None:
            ymax = DF[input2].values.max()
        else:
            ymax = y_max
        
        print('xmin inserted = ', xmin)
        print('xmax inserted = ', xmax)
        print('ymin inserted = ', ymin)
        print('ymax inserted = ', ymax)
                
        x_delta_limit, y_delta_limit = xmax*0.1, ymax*0.1
        x_label , y_label = axes_labels

        #gerando os plots
        p = set_bokeh_main_numplot(proc_vals, xmin, xmax, ymin, ymax, x_delta_limit, y_delta_limit, x_label, y_label, plot_width, plot_height)
        ph = set_bokeh_horizontal_hist(p)
        pv = set_bokeh_vertical_hist(p)
        c_bars = set_bokeh_cbar(p, self.palette_bins)

        #plotando
        counter = 0
        hmax = 0
        vmax = 0
        for group in grouped.groups:
            grouped_vals = grouped.get_group(group)
            #coletando os resultados de interesse
            x = grouped_vals[input1].values
            y = grouped_vals[input2].values
            n_vals = len(grouped_vals[input2].values)
        
            r, bins = p.hexbin(x, y, size=hex_size, palette=self.palette_bins[counter])
            p.circle(x, y, color="gray", size=6, alpha=0)
    
            #adicionando o colormap
            mapper = LinearColorMapper(palette=self.palette_bins[counter], low=bins['counts'].values.min(), high=bins['counts'].values.max())
            color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="7px",
                                 ticker=BasicTicker(desired_num_ticks=12),
                                 orientation = 'horizontal',
                                 label_standoff=12, border_line_color=None, location=(0, 0))
            color_bar.major_label_text_font_size = '22pt'
            color_bar.bar_line_color = 'black'
            color_bar.bar_line_width = 2
            color_bar.major_tick_line_alpha = 0
            color_bar.major_tick_out = 5
            color_bar.title = 'Occurrences'
            color_bar.title_text_font_size = '19pt'
            
            p.add_layout(color_bar, 'above')
        
            #plotando o histograma de cima
            hhist, hedges = np.histogram(x, bins=20, range=(xmin,xmax))        
            if max(hhist)*1.1 > hmax:
                hmax = max(hhist)*1.1
            ph.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hhist, color=self.palette_bins[counter][-1], alpha=0.5)
            
            #plotando o histograma do lado
            vhist, vedges = np.histogram(y, bins=20, range=(ymin, ymax))
            if max(vhist)*1.1 > vmax:
                vmax = max(vhist)*1.1
            pv.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vhist, color=self.palette_bins[counter][-1], alpha=0.5)

            #adicionando a legenda no gráfico acima
            for color_i in range(len(self.palette_bins[counter])):
                rect_color = c_bars.rect(10+(color_i*10), 0-(counter*10), 10, 5, line_color = 'white', fill_color=self.palette_bins[counter][color_i])            
                c_bars.add_layout(rect_color)

            y_base_pos = -7.5
            leg_label1 = Label(x=0, y=y_base_pos-(counter*10), text='0',
                              text_font_size = '16pt', text_color = 'gray', background_fill_alpha=0.8)            
            leg_label2 = Label(x=7+(len(self.palette_bins[0])*10), y=y_base_pos-(counter*10), text=str(n_vals),
                              text_font_size = '16pt', text_color = 'gray', background_fill_alpha=0.8)
            leg_label3 = Label(x=30+(len(self.palette_bins[0])*10), y=y_base_pos-(counter*10), text=re.sub(r'_', ' ', group),
                               text_font_size = '16pt', text_color = self.palette_bins[counter][-1], background_fill_alpha=0.8)
            c_bars.add_layout(leg_label1)
            c_bars.add_layout(leg_label2)
            c_bars.add_layout(leg_label3)
            
            counter += 1
            
        
        #ajustando a escala dos histogramas após a plotagem
        ph.x_range = p.x_range
        ph.y_range = Range1d( 0 - (0.1*hmax), hmax)
        ph.yaxis.ticker = (0, int(hmax))
        ph.yaxis.bounds = (0, int(hmax))
        pv.x_range = Range1d( 0 - (0.1*vmax), vmax)
        pv.y_range = p.y_range
        pv.xaxis.ticker = (0, int(vmax))
        pv.xaxis.bounds = (0, int(vmax))
        
        layout = gridplot([[c_bars,None], [p, pv], [ph, None]], merge_tools=False)
        
        print(f'Salvando a figura ~/Outputs/analyses/P{self.last_fig_filename_index}.png ...')
        export_png(layout, filename=self.diretorio + f'/Outputs/analyses/P{self.last_fig_filename_index}.png')
        
        #atualizando o número do plot_index
        self.last_fig_filename_index = ( ( len('0000') - len(str(int(self.last_fig_filename_index)+1)) ) * '0' ) + str(int(self.last_fig_filename_index) + 1)



def lin_regression(x, y):
    
    x = x.reshape(-1, 1)
    
    model = LinearRegression()        
    model.fit(x, y)
    
    
    b0 = model.intercept_
    b1 = model.coef_
    r_sq = model.score(x, y)

    return b0, b1, r_sq



def clusterize(x, y, n_clusters):

    X = np.dstack((x,y)).reshape(x.shape[0], 2)
    
    k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    k_means.fit(X)
    k_means_cluster_centers = k_means.cluster_centers_
    k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)

    xy_vals = list(zip(x,y))
    xy_vals_labeled = np.array( [ [xy_vals[i], k_means_labels[i]] for i in range(len(xy_vals)) ], dtype=object )
    
    return xy_vals_labeled, k_means_labels, k_means_cluster_centers

    
def filter_DF_by_min_max_vals(DF, input1=None, input2=None, x_min=None, x_max=None, y_min=None, y_max=None):
    
    
    print('xmin found (DF) = ', DF[input1].values.min())
    print('xmax found (DF) = ', DF[input1].values.max())
    print('ymin found (DF) = ', DF[input2].values.min())
    print('ymax found (DF) = ', DF[input2].values.max())
    
    #xmin
    if x_min is None:
        xmin = DF[input1].values.min()
    else:
        xmin = x_min
    #xmax
    if x_max is None:
        xmax = DF[input1].values.max()
    else:
        xmax = x_max        
    #ymin
    if y_min is None:
        ymin = DF[input2].values.min()
    else:
        ymin = y_min
    #ymax
    if y_max is None:
        ymax = DF[input2].values.max()
    else:
        ymax = y_max
                
    print('xmin inserted = ', xmin)
    print('xmax inserted = ', xmax)
    print('ymin inserted = ', ymin)
    print('ymax inserted = ', ymax)

    def filtering_max_min(entry, min_val = 0, max_val = 10):
        
        
        
        if min_val <= entry <= max_val:
            return entry    
        else:
            return np.nan

    #coletando os resultados de interesse entre os limites
    DF = DF[ (xmin <= DF[input1]) & (DF[input1] <= xmax) ]
    DF = DF[ (ymin <= DF[input2]) & (DF[input2] <= ymax) ]

    return DF, xmin, xmax, ymin, ymax



def set_bokeh_main_boxplot(ymin, ymax, cats, axes_label1, axes_label2, min_occurrences, plot_width, plot_height, cats_number, size_factor, min_border):    
    
    
    if min_occurrences is None:
        g_title = ''
        title_text_font_size = '0pt'
        
    else:
        g_title = 'min occurences: ' + str(min_occurrences)
        title_text_font_size = '25pt'
        
    #plotando o boxplot
    p = figure(toolbar_location=None,
               match_aspect=False,
               title = g_title,
               x_range=cats,                  
               background_fill_color='white', 
               plot_width=plot_width, plot_height=plot_height,
               min_border=min_border)

    #general settings
    p.grid.visible = True
    font = 'helvetica'
    p.title.text_font_size = title_text_font_size
    p.outline_line_color = "white"
    p.outline_line_width = 0
    p.axis.axis_line_width = 2
    p.axis.axis_line_color = 'red'
    p.axis.major_tick_line_color = 'red'
    p.axis.major_tick_line_width = 3
    p.axis.minor_tick_line_color = 'red'
    p.axis.minor_tick_line_width = 2
    p.axis.minor_tick_in = -2
    p.axis.minor_tick_out = 6
    
    #x
    p.xaxis.axis_label = axes_label1
    p.xaxis.axis_label_text_font = font
    p.xaxis.axis_label_text_color = 'black'
    p.xaxis.axis_label_text_font_size = str(int(cats_number*size_factor) + 0) + 'pt'
    p.xaxis.axis_label_standoff = 10
    p.xaxis.major_label_text_color = "black"
    p.xaxis.major_label_orientation = np.pi/3
    p.xaxis.major_label_text_font_size = str(int(cats_number*size_factor) + 0) + 'pt'
    p.xaxis.major_label_standoff = 10
    
    #y
    p.yaxis.axis_label = axes_label2
    p.yaxis.axis_label_text_font = font
    p.yaxis.axis_label_text_color = 'black'
    p.yaxis.axis_label_text_font_size = str(int(cats_number*size_factor) + 0) + 'pt'
    p.yaxis.axis_label_standoff = 10
    p.yaxis.major_label_text_color = "black"
    p.yaxis.major_label_orientation = "horizontal"
    p.yaxis.major_label_text_font_size = str(int(cats_number*size_factor) + 0) + 'pt'
    p.yaxis.major_label_standoff = 10
    p.yaxis.bounds = (ymin, ymax)
    p.ygrid.grid_line_color = 'gray'
    p.ygrid.grid_line_alpha = 0.1
    p.ygrid.minor_grid_line_color = 'gray'
    p.ygrid.minor_grid_line_alpha = 0.1    
    p.y_range = Range1d(ymin, ymax+(ymax*0.4))
    p.yaxis.ticker = [round(val, 0) for val in np.arange(ymin, ymax + (ymax*0.05), (ymax-ymin+1)/6) ]
    
    return p



def set_bokeh_main_numplot(proc_vals, xmin, xmax, ymin, ymax, x_delta_limit, y_delta_limit, x_label, y_label, plot_width, plot_height,
                            tooltips = None, toolbar_location=None):

    
    

    #plotando o scatter plot
    p = figure(tooltips = tooltips,
               toolbar_location=toolbar_location,
               title='articles = ' + str(proc_vals),
               match_aspect=False,
               background_fill_color='white', 
               x_range=(xmin - x_delta_limit, xmax + x_delta_limit), y_range=(ymin - y_delta_limit, ymax + y_delta_limit),
               plot_width=plot_width, plot_height=plot_height,
               min_border=50)

    #general settings
    p.grid.visible = False
    font = 'helvetica'
    p.title.text_font_size = '30pt'
    p.outline_line_color = "white"
    p.outline_line_width = 0
    p.axis.axis_line_width = 2
    p.axis.axis_line_color = 'red'
    p.axis.major_tick_line_color = 'red'
    p.axis.major_tick_line_width = 3
    p.axis.minor_tick_line_color = 'red'
    p.axis.minor_tick_line_width = 2
    p.axis.minor_tick_in = -2
    p.axis.minor_tick_out = 6
    
    #x
    p.xaxis.axis_label = x_label
    p.xaxis.axis_label_text_font = font
    p.xaxis.axis_label_text_color = 'black'
    p.xaxis.axis_label_text_font_size = '40pt'
    p.xaxis.axis_label_standoff = 10
    p.xaxis.major_label_text_color = "black"
    p.xaxis.major_label_orientation = "horizontal"
    p.xaxis.major_label_text_font_size = '40pt'
    p.xaxis.major_label_standoff = 10
    p.xaxis.bounds = (xmin, xmax)
    p.xgrid.grid_line_color = 'gray'
    p.xgrid.grid_line_alpha = 0.1
    p.xgrid.minor_grid_line_color = 'gray'
    p.xgrid.minor_grid_line_alpha = 0.1
    p.xaxis.ticker = [int(val) for val in np.arange(xmin, xmax + (xmax*0.05), (xmax-xmin+1)/5) ]
    
    #y
    p.yaxis.axis_label = y_label
    p.yaxis.axis_label_text_font = font
    p.yaxis.axis_label_text_color = 'black'
    p.yaxis.axis_label_text_font_size = '40pt'
    p.yaxis.axis_label_standoff = 10
    p.yaxis.major_label_text_color = "black"
    p.yaxis.major_label_orientation = "vertical"
    p.yaxis.major_label_text_font_size = '40pt'
    p.yaxis.major_label_standoff = 10
    p.yaxis.bounds = (ymin, ymax)
    p.ygrid.grid_line_color = 'gray'
    p.ygrid.grid_line_alpha = 0.1
    p.ygrid.band_fill_color = "gray"
    p.ygrid.band_fill_alpha = 0.1
    p.ygrid.minor_grid_line_color = 'gray'
    p.ygrid.minor_grid_line_alpha = 0.1
    p.yaxis.ticker = [int(val) for val in np.arange(ymin, ymax + (ymax*0.05), (ymax-ymin+1)/5) ]
    
    return p



def set_bokeh_horizontal_hist(main_plot):
    
    
    
    #plotando o histograma de cima
    ph = figure(toolbar_location=None, plot_width=main_plot.plot_width, plot_height=200,
                min_border=50)
    ph.outline_line_color = "white"
    ph.outline_line_width = 0
    ph.grid.visible = False
    ph.yaxis.major_label_orientation = 'horizontal'
    ph.background_fill_color = "white"        
    ph.xaxis.visible = False
    ph.yaxis.major_label_text_color = "black"
    ph.yaxis.major_label_text_font_size = '40pt'
    ph.yaxis.axis_label_text_font_size = '40pt'
    ph.yaxis.axis_label = 'freq'
    
    return ph



def set_bokeh_vertical_hist(main_plot):

    
    
    
    #plotando o histograma de lado
    pv = figure(toolbar_location=None, plot_width=200, plot_height=main_plot.plot_height,
                min_border=50)
    pv.outline_line_color = "white"
    pv.outline_line_width = 0
    pv.grid.visible = False
    pv.xaxis.major_label_orientation = np.pi/3
    pv.background_fill_color = "white"        
    pv.yaxis.visible = False
    pv.xaxis.major_label_text_color = "black"
    pv.xaxis.major_label_text_font_size = '40pt'
    pv.xaxis.axis_label_text_font_size = '40pt'
    pv.xaxis.axis_label = 'freq'
    
    return pv



def set_bokeh_cbar(main_plot, palette_bins, plot_height=200):

    
    
    #color bars        
    c_bars = figure(toolbar_location=None, plot_width=main_plot.plot_width, plot_height=plot_height,
                    x_range=(0, (len(palette_bins[0])*10) + 60), min_border=50)
    c_bars.outline_line_color = "white"
    c_bars.outline_line_width = 0
    c_bars.grid.visible = False
    c_bars.background_fill_color = "white"
    c_bars.yaxis.visible = False
    c_bars.xaxis.visible = False
    
    return c_bars



def get_blank_figure(main_plot=None, plot_width = None, plot_height=None, x_range=(-5, 5), y_range=(0, 10)):
    
    if plot_width is None:
        plot_width_ = main_plot.plot_width
    else:
        plot_width_ = plot_width
        
    if plot_height is None:
        plot_height_ = main_plot.plot_height
    else:
        plot_height_ = plot_height
    
    fig = figure(toolbar_location=None, plot_width=plot_width_, plot_height=plot_height_,
                 x_range=x_range, y_range=y_range, min_border=50)
    fig.outline_line_color = "white"
    fig.outline_line_width = 0
    fig.grid.visible = False
    fig.background_fill_color = "white"
    fig.yaxis.visible = False
    fig.xaxis.visible = False
    
    return fig



def find_heaviest_path_from_edges(path_generator, graph):
    
    counter = 0
    heaviest_weight = 0
    heaviest_path = None
    
    for path in path_generator:
        weight = 0
        for start, end in path:            
            weight += graph[start][end]['weight']
        if weight >= heaviest_weight:
            heaviest_weight = weight
            nodes = [edge[0] for edge in path]
            nodes.append(path[-1][1])
            heaviest_path = nodes
        counter += 1
        if counter % 1000000 == 0:
            print('looking for path number ', counter)            
            print('current heaviest path: ', heaviest_path)
            print('current weight: ', heaviest_weight)
    
    return heaviest_path, heaviest_weight
    


def find_heaviest_path_from_graph(G, start = None, end=None, min_edge_weight=None, network_type='directed', network_name = ''):


    def search_out_nodes(G, in_node, end_node=None, path = [], max_weight=0, heaviest_path=[], cum_path_weights=[], min_edge_weight = None, close_path = False):
    
        #andamento
        if round(random.random(), 7) == 0.0000001:
            print('> heaviest path: ', heaviest_path)
            print('> max weight: ', max_weight)
        
        #print('begin - search_out_nodes cum path weight: ', sum(cum_path_weights), ' - max weight: ', max_weight)
        
        if network_type.lower() == 'directed':
            node_list = list(G.successors(in_node))
        elif network_type.lower() == 'undirected':
            node_list = list(G.neighbors(in_node))

        #encerrando a procura
        if close_path is True:
            #print('\nreturning close_path')
            return cum_path_weights, path, max_weight, heaviest_path, close_path
            
        #varrendo os out_nodes
        for out_node in node_list:
            
            #check close path
            close_path = False
            
            if out_node not in path:
                                
                path_weight = G[in_node][out_node]['weight']
                #print('in: ', in_node, ' ; out: ', out_node, ' ; path weight: ', path_weight)
                
                #checks
                get_edge = False
                
                #caso o path esteja completo                
                if out_node == end_node:
                    get_edge = True
                    close_path = True                                    
                #caso haje um path weight mínimo
                elif min_edge_weight is not None and path_weight >= min_edge_weight:
                    get_edge = True                    
                #ignorando essa edge pois está abaixo do valor mínimo
                elif min_edge_weight is not None and path_weight < min_edge_weight:
                    continue
                #não há valor mínimo inserido
                else:
                    get_edge = True
                
                #coletando a edge
                if get_edge is True:
                    
                    #coletando informações do caminho
                    path = path + [out_node]
                    cum_path_weights.append(path_weight)
        
                    #print('get_edge...')
                    #print('current path: ', path)
                    #print('cum path weight: ', cum_path_weights, ' ; sum: ', sum(cum_path_weights), ' ; max weight: ', max_weight)
                    
                    #recurssão                                        
                    new_cum_path_weights, new_path, max_weight, heaviest_path, close_path = search_out_nodes(G, 
                                                                                                             out_node, 
                                                                                                             end_node = end_node, 
                                                                                                             path=path, 
                                                                                                             max_weight=max_weight, 
                                                                                                             heaviest_path=heaviest_path, 
                                                                                                             cum_path_weights=cum_path_weights,
                                                                                                             close_path = close_path)
                        

                    update_heaviest_weight = False
                    if end_node is None and sum(new_cum_path_weights) > max_weight:
                        update_heaviest_weight = True
                    elif end_node is not None and close_path is True and sum(new_cum_path_weights) > max_weight:
                        update_heaviest_weight = True

                    #atualizando o heaviest weight
                    if update_heaviest_weight is True:
                        max_weight = sum(new_cum_path_weights)
                        heaviest_path = new_path
                        #print('> heaviest path atualizado: ', heaviest_path, ' ; max weight: ', max_weight)
                        #print('> current path: ', path)
                        #print('> cum path weight: ', new_cum_path_weights, ' ; sum: ', sum(new_cum_path_weights), )
                        
                    path = new_path[ : -1]
                    cum_path_weights = new_cum_path_weights[ : -1]
                    #print('\ndroping last elements..')
                    #print('path: ', path)
                    #print('cum_path_weights: ', cum_path_weights, ' ; sum: ', sum(cum_path_weights), ' ; max_weight: ', max_weight, '\n')                    
    
            #time.sleep(1)
    
        #print('\nNenhum out_node encontrado! (out of loop)')
        #print('last path: ', path)
        #print('cum path weight: ', cum_path_weights, ' ; sum: ', sum(cum_path_weights),  ' ; max_weight: ', max_weight)
        return cum_path_weights, path, max_weight, heaviest_path, close_path


    #encontra o heaviest weight a partir de um node específico    
    if start is not None:
        print('\nFinding heaviest path...')
        print('Network: ', network_name)
        print('Start node: ', start)
        _, _, heaviest_path_weight, heaviest_path, close_path = search_out_nodes(G, start, end_node = end, 
                                                                                 min_edge_weight = min_edge_weight, 
                                                                                 path = [start], 
                                                                                 max_weight=0, 
                                                                                 heaviest_path=[], 
                                                                                 cum_path_weights=[])    
        print('Heaviest path: ', heaviest_path)
        print('Heaviest path weight: ', heaviest_path)    
        
        return heaviest_path, heaviest_path_weight
    
    else:
        print('ERRO!')
        print('Inserir um node para o atributo "start" do path.')
        return



class groups_results(object):
    
    def __init__(self, *results_dic):
        
        
        
        self.diretorio = os.getcwd()
        
        self.groups_dic = {}
        self.groups_name_list = []

        #resultados que serão usados        
        self.results_types = ['cat_occurrences', 
                              'degree_nodes', 
                              'in_degree_nodes', 
                              'out_degree_nodes', 
                              'graph_edges', 
                              'digraph_edges', 
                              'unique_cat_occurrences',
                              'unique_degree_nodes', 
                              'unique_in_degree_nodes', 
                              'unique_out_degree_nodes', 
                              'unique_graph_edges', 
                              'unique_digraph_edges']

        #dicionário para encontrar os valores únicos de categorias para todos os grupos
        all_cats = {}
        for result_type in self.results_types:
            all_cats[result_type] = []

        #encontrandos as categorias únicas para todos os grupos
        for dic in results_dic:
            for result_type in self.results_types:
                if result_type[ : len('unique')] != 'unique':
                    all_cats[result_type] = all_cats[result_type] + [val for val in dic[result_type].keys() if val not in all_cats[result_type]]

        #varrendo os diferentes grupos para padronizar as categorias dos resultados
        unique_vals = {}
        self.group_number = 0
        for dic in results_dic:   
            
            group_name = dic['group_name']
            self.groups_dic[group_name] = {}
            unique_vals[group_name] = {}
            self.groups_name_list.append(group_name)
            self.group_number += 1
                                        
            #padronizando os valores das categorias para todos os resultados
            for result_type in self.results_types:
                
                #definindo uma key para valores que será trabalhados como únicos
                if result_type[ : len('unique') ] == 'unique':
                    unique_vals[group_name][result_type] = dic[result_type[ len('unique_') : ]]
                    
                #resultados de outros tipos
                else:
                    try:
                        self.groups_dic[group_name][result_type] 
                    except KeyError:
                        self.groups_dic[group_name][result_type] = []
                    
                    for cat in all_cats[result_type]:
                        try:
                            self.groups_dic[group_name][result_type].append([ cat, dic[result_type][cat] ])
                        except KeyError:
                            self.groups_dic[group_name][result_type].append([ cat, 0 ])
                    self.groups_dic[group_name][result_type] = sorted(self.groups_dic[group_name][result_type], key = lambda t: t[0])                    

        #filtrando o dicionário para ficar somente com os resultados únicos
        for group_name in self.groups_name_list:
            for result_type in [ result for result in self.results_types if result[ : len('unique')] == 'unique' ]:
                temp_result = list( zip( unique_vals[group_name][result_type].keys() , unique_vals[group_name][result_type].values() ) )                
                cat_in_other_groups_to_be_excluded = []
                for group_name_inner in self.groups_name_list:
                    if group_name != group_name_inner:
                        cat_in_other_groups_to_be_excluded.extend([ cat for cat in unique_vals[group_name_inner][result_type].keys() ])
                
                self.groups_dic[group_name][result_type] = [ [cat, val] for cat, val in temp_result if cat not in cat_in_other_groups_to_be_excluded ]
                self.groups_dic[group_name][result_type] = sorted(self.groups_dic[group_name][result_type] , key = lambda t: t[0])

        #econtrando o ultimo numerador dos arquivos de figuras
        filenames = get_filenames_from_folder(folder = self.diretorio + '/Outputs/analyses', file_type='png')
        try:
            if len(filenames) > 0:
                fig_filenames = [file for file in filenames if 'P' in file]
                last_counter = int(fig_filenames[-1][ len('P') :  ]) + 1
                #atualizando o número do plot_index
                self.last_fig_filename_index = ( ( len('0000') - len(str(last_counter)) ) * '0' ) + str(last_counter)
        except TypeError:
            self.last_fig_filename_index = '0001'
            
        #color
        self.palette_colors1 = ['#4EACC5', '#FF9C34', '#4E9A06', '#BE2929', '#9944EE', '#4C6663', '#0339B7', '#A87940', '#090909',
                                "#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]



    def bar_plot(self, maximum_values_to_plot = 20):
        
        bar_plots = {}
        group_names = []
        
                
        #dando um título para cada resultado
        graph_title = {}        
        graph_title['cat_occurrences'] = 'processes'
        graph_title['degree_nodes'] = 'nodes degree (graph)'
        graph_title['in_degree_nodes'] = 'nodes degree (in)'
        graph_title['out_degree_nodes'] = 'nodes degree (out)'
        graph_title['graph_edges'] = 'edge weight (graph)'
        graph_title['digraph_edges'] = 'edge weight (digraph)'
        graph_title['unique_cat_occurrences'] = 'unique processes'
        graph_title['unique_degree_nodes'] = 'unique nodes degree (graph)'
        graph_title['unique_in_degree_nodes'] = 'unique nodes degree (in)'
        graph_title['unique_out_degree_nodes'] = 'unique nodes degree (out)'
        graph_title['unique_graph_edges'] = 'unique edge weight (graph)'
        graph_title['unique_digraph_edges'] = 'unique edge weight (digraph)'

        #concatenando os resultados para encontrar os maiores valores
        concat_dic={}
        concat_dic['cat'] = {}
        concat_dic['vals'] = {}
        concat_dic['cat_vals'] = {}

        for group_name in self.groups_name_list:
            for result_type in [ result for result in self.results_types if result[ : len('unique')] != 'unique' ]:
                try:                    
                    concat_dic['vals'][result_type] = concat_dic['vals'][result_type] + np.array(self.groups_dic[group_name][result_type])[:, 1].astype('float32')
                    concat_dic['cat_vals'][result_type] = list( zip( concat_dic['cat'][result_type], concat_dic['vals'][result_type] ) )
                except KeyError:
                    concat_dic['vals'][result_type] = np.array(self.groups_dic[group_name][result_type])[:, 1].astype('float32')
                    concat_dic['cat'][result_type] = np.array(self.groups_dic[group_name][result_type])[:, 0]

        #montando os group name para os group sets e criando os dic para o plot
        for group_name in self.groups_name_list:
            group_names.extend([group_name] * len(self.results_types))
            #dic para as plotagens
            bar_plots[group_name] = {}

        #os resultados que serão plotados
        bar_plot_sets = list( zip( group_names , self.results_types * self.group_number, self.palette_colors1[ : len(self.results_types) ] * self.group_number ) )
                
        #plotando

        for group_name, result_type, color in bar_plot_sets:

            if result_type[ : len('unique') ] == 'unique':
                #encontrando o array para os cat, vals
                results_array = np.array( self.groups_dic[group_name][result_type] )

                if len(results_array) == 0:
                    bar_plots[group_name][result_type] = figure()
                else:
                    group_results_array = results_array[:, 1].astype('float32') / results_array[:, 1].astype('float32').max()
                    
                    cats_to_plot = results_array[:, 0]
    
                    #plotando o gráfico
                    bar_plots[group_name][result_type] = figure(x_range=cats_to_plot,
                                                                toolbar_location=None, tools="",
                                                                plot_width=1000, plot_height=2000,
                                                                title = group_name + ': Norm. ' + graph_title[result_type],
                                                                min_border = 20)
                
                    #adicionado as barras para o grupo
                    bar_plots[group_name][result_type].vbar(cats_to_plot, 
                                                            top=group_results_array, 
                                                            color = color,
                                                            width=0.9)
            
            else:
                #coletando somente as categorias que contem os maiores valores                           
                cat_vals_to_plot = sorted( concat_dic['cat_vals'][result_type], key = lambda t: t[1] , reverse = True)[ : maximum_values_to_plot]                
                #organizando por ordem alfabética
                cat_vals_to_plot = sorted(cat_vals_to_plot, key = lambda t: t[0])

                all_results_array = np.array([ val for cat, val in cat_vals_to_plot], dtype='float32') / np.array([ val for cat, val in cat_vals_to_plot], dtype='float32').max()
                
                #coletando todas as categorias para plotagem
                cats_to_plot = [ cat_val[0] for cat_val in cat_vals_to_plot ]
                
                #encontrando o array para os cat, vals
                group_results_array = np.array([ val for cat, val in self.groups_dic[group_name][result_type] if cat in cats_to_plot])
                
                group_results_array = group_results_array.astype('float32') / group_results_array.astype('float32').max()

                #plotando o gráfico
                bar_plots[group_name][result_type] = figure(x_range=cats_to_plot,
                                                            toolbar_location=None, tools="",
                                                            plot_width=1500, plot_height=1400,
                                                            title = group_name + ': Norm. ' + graph_title[result_type],
                                                            min_border = 20)        
            
                #adicionado as barras para o grupo
                bar_plots[group_name][result_type].vbar(cats_to_plot, 
                                                        top=group_results_array, 
                                                        color = color,
                                                        fill_alpha = 0.6,
                                                        width=0.9)
                
                #adicionando as barras para a somatória normalizada de todos os grupos
                bar_plots[group_name][result_type].vbar(cats_to_plot, 
                                                        top=all_results_array, 
                                                        fill_alpha=0.0, 
                                                        line_color = 'black',
                                                        line_width = 10,
                                                        line_dash="dashed",
                                                        width=0.9)
            
            bar_plots[group_name][result_type].xaxis.axis_label = graph_title[result_type]
            bar_plots[group_name][result_type].xaxis.axis_label_text_font_size = '50pt'
            bar_plots[group_name][result_type].y_range.start = 0
            bar_plots[group_name][result_type].yaxis.visible = False
            bar_plots[group_name][result_type].x_range.range_padding = 0.1
            bar_plots[group_name][result_type].ygrid.grid_line_color = None
            bar_plots[group_name][result_type].xgrid.grid_line_color = None
            bar_plots[group_name][result_type].axis.minor_tick_line_color = None
            bar_plots[group_name][result_type].outline_line_color = None
            bar_plots[group_name][result_type].title.text_font_size = '50pt'
            bar_plots[group_name][result_type].outline_line_color = "black"
            bar_plots[group_name][result_type].outline_line_width = 1
            bar_plots[group_name][result_type].axis.major_label_text_font_size = '50pt'
            bar_plots[group_name][result_type].axis.major_label_standoff = 10
            bar_plots[group_name][result_type].xaxis.major_label_orientation = np.pi/2
            bar_plots[group_name][result_type].outline_line_color = 'white'

        #organizando o gridplot
        grid_list=[]
        for group_name in self.groups_name_list:
            grid_list.append([bar_plots[group_name]['cat_occurrences'],
                              #bar_plots[group_name]['degree_nodes'],                               
                              #bar_plots[group_name]['in_degree_nodes'], 
                              #bar_plots[group_name]['out_degree_nodes'],
                              #bar_plots[group_name]['graph_edges'],                        
                              bar_plots[group_name]['digraph_edges']])
                              #bar_plots[group_name]['unique_cat_occurrences'])
                              #bar_plots[group_name]['unique_degree_nodes'],                               
                              #bar_plots[group_name]['unique_in_degree_nodes'], 
                              #bar_plots[group_name]['unique_out_degree_nodes'],
                              #bar_plots[group_name]['unique_graph_edges'],                      
                              #bar_plots[group_name]['unique_digraph_edges']])
            
        layout = gridplot(grid_list)
        
        
        print(f'Salvando a figura ~/Outputs/analyses/P{self.last_fig_filename_index}.png ...')
        export_png(layout, filename=self.diretorio + f'/Outputs/analyses/P{self.last_fig_filename_index}.png')        
        
        #atualizando o número do plot_index
        self.last_fig_filename_index = ( ( len('0000') - len(str(int(self.last_fig_filename_index)+1)) ) * '0' ) + str(int(self.last_fig_filename_index) + 1)
        

def get_neighbor_graph_from_graph(node_name = '', graph = None):
    
    
    
    G = nx.Graph()    
    for neighbor in list(graph[node_name].keys()):
        G.add_edge(node_name, neighbor, weight=graph[node_name][neighbor]['weight'])
        
    return G



def get_neighbor_graph_from_digraph(node_name = '', graph = None, mode= 'in'):
    
    
    
    if mode.lower() == 'in':
        in_out = 1
    elif mode.lower() == 'out':
        in_out = 0
            
    in_edges = [edge for edge in graph.edges() if edge[in_out] == node_name]
    G = nx.Graph()
    for source, target in in_edges:
        G.add_edge(source, target, weight=graph[source][target]['weight'])
        
    return G



def get_subgraph(nodes_names = [], graph = None):
    
            
    edges = [edge for edge in graph.edges() if ((edge[0] in nodes_names) and (edge[1] in nodes_names))]
    G = nx.Graph()
    for source, target in edges:
        G.add_edge(source, target, weight=graph[source][target]['weight'])
        
    return G'''