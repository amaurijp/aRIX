#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
diretorio = os.getcwd()
sys.path.append(diretorio + '/Modules')


#instanciando o DataFrame consolidado
from RESULTS import results
results = results(DF_input_name = 'consolidated_DF', diretorio=diretorio)
#results.remove_terms_in_DF(cats = ['materials_composition'])
results.replace_terms_in_DF(cats = {'materials_composition':['', ]})

results.plot_cat_barh(cat = 'materials_composition', font_scale = 20, x_label = 'Counts', y_label = 'Composition',
                      max_nbars = 15, plot_margins = {'left':0.3, 'right':0.9, 'top':0.9, 'bottom':0.2})

results.plot_cat_cat_heatmap(cats = ['materials_composition', 'morphology'],
                             x_label = 'Composition', y_label = 'Morphology',
                             font_scale = 20,
                             mark_scale = 50,
                             n_box_size = 100, 
                             terms_to_remove = ['In'],
                             grid_plot_margins = {'left':0.3, 'right':0.9, 'top':0.9, 'bottom':0.4})

results.plot_cat_num_boxplot_anova(cat_col = 'materials_composition', num_col = 'ecoli', num_mode = 'min', 
                                   y_lims = None,
                                   x_label = 'Elements', y_label = 'Conc', 
                                   y_multiplier = 1, 
                                   n_cats = 9, 
                                   log_scale = True,
                                   remove_extreme_vals = 0.07,
                                   font_scale_boxplot = 40, 
                                   font_scale_anova_grid = 20,
                                   mark_scale_anova_grid = 50,
                                   box_values_font_scale = 30,
                                   anova_x_annotation_delta = 0, anova_y_annotation_delta = 0.01,
                                   terms_to_remove = ['In'],
                                   boxplot_plot_margins = {'left':0.3, 'right':0.9, 'top':0.9, 'bottom':0.3},
                                   grid_plot_margins = {'left':0.3, 'right':0.8, 'top':0.9, 'bottom':0.4})

results.plot_cat_cat_num_heatmap(cats_col = ['materials_composition', 'morphology'], num_col = 'ecoli', 
                                 num_mode = 'min', 
                                 x_label = 'x', y_label = 'y', bar_label = 'ecc',
                                 n_box_size = 100, font_scale_grid = 20, 
                                 mark_scale_grid = 130, box_values_font_scale = 25,
                                 z_multiplier = 100, z_lims = None, remove_extreme_vals = 0.05, 
                                 x_annotation_delta = 0, y_annotation_delta = 0.01,
                                 terms_to_remove = ['In'],
                                 grid_plot_margins = {'left':0.3, 'right':0.9, 'top':0.9, 'bottom':0.5})

results.plot_scatter_graph(num_cols = ['ecoli', 'size'], cat_col = None, cats_to_plot = 'all', #['silver'], 
                           num_mode = 'min', x_multiplier = 10, y_multiplier = 1000000, remove_extreme_vals = None,
                           log_scale = False, x_lims = [-0.1, 2.1], y_lims = [-0.01, 0.1], x_label = 'AA', y_label = 'BB',
                           terms_to_remove = [],
                           set_graph_region = {'R1':[[0.01, 0.4],[0.001, 0.02], 'right'],
                                               'R2':[[0.5, 1.0],[0.04, 0.06], 'bottom']})

'''
#Fig1
#instanciando o DataFrame consolidado
from RESULTS import results
results = results(results_DF_input_name = '_CONSOLIDATED_DF', diretorio=diretorio)
results.plot_cat_loc_gridplot_bins(DF_column = 'raw_materials', 
                                   min_occurrences = 15, size_factor = 0.65, colobar_nticks = 10, palette = 'Turbo256',
                                   background_fill_color = 'white', plot_width=2000, plot_height=2000)

#Fig2
#instanciando o DataFrame consolidado
from RESULTS import results
results = results(results_DF_input_name = '_CONSOLIDATED_DF', diretorio=diretorio)
results.split_2grams_terms(DF_column = 'raw_materials') #splitting nGrams
results.plot_cat_cat_gridplot_bins(DF_columns = ['raw_materials_0', 'raw_materials_1'],
                                   min_occurrences = 10, size_factor = 0.8, colobar_nticks = 10, palette = 'Turbo256',
                                   background_fill_color = 'white', plot_width=2000, plot_height=2000)

#Fig3
#instanciando o DataFrame consolidado
from RESULTS import results
results = results(results_DF_input_name = '_CONSOLIDATED_DF', diretorio=diretorio)
results.plot_cat_cat_stacked_barplots(DF_columns = ['Publication Year', 'raw_materials'],
                                      axes_labels = ['publication year', 'number of publications', '%-publications'],
                                      min_occurrences = 80,
                                      cat_to_filter_min_occurrences = 'raw_materials',
                                      size_factor = 30)

#Fig4
#instanciando o DataFrame consolidado
from RESULTS import results
results = results(results_DF_input_name = '_CONSOLIDATED_DF', diretorio=diretorio)
results.group_cat_columns_with_input_classes(DF_column = 'applications', groups_name = 'classes_applications') #grouping (o groups_name está no arquivo JSON no folder "~/Inputs")
results.plot_2column_network_chord(DF_columns = ['raw_materials', 'classes_applications'], min_occurrences = 10)


#numerical values plots
#Fig5
#Fig6
#use "NA" nos quantiles caso queira começar com o valor zero no intervalo (ex: x_quantiles = ['NA', 0.98], y_quantiles = ['NA', 0.98])
#instanciando o DataFrame consolidado
from RESULTS import results
results = results(results_DF_input_name = '_CONSOLIDATED_DF', diretorio=diretorio)
results.process_columns_with_num_val(DF_columns=['temperature_carbonization'], mode='avg')
results.plot_group_num_boxplot_correlation(DF_column_with_cat_vals = 'raw_materials',
                                           DF_column_with_num_vals = 'temperature_carbonization',
                                           axes_labels=['biomass precursors', 'avg carb temperature (°C)'],
                                           y_quantiles = [0.02, 0.98],
                                           min_values_for_column = 20,
                                           colobar_nticks = 10,
                                           palette = 'Viridis256',
                                           background_fill_color = 'blue',
                                           size_factor_boxplot = 1.1,
                                           grouplabel_x_offset = 10,
                                           box_plot_plot_width=2000, box_plot_plot_height=1000,
                                           size_factor_anova_grid = 0.7,
                                           grid_plot_width=1000, grid_plot_height=1000)

#Fig7
#Fig8
#use "NA" nos quantiles caso queira começar com o valor zero no intervalo (ex: x_quantiles = ['NA', 0.98], y_quantiles = ['NA', 0.98])
#instanciando o DataFrame consolidado
from RESULTS import results
results = results(results_DF_input_name = '_CONSOLIDATED_DF', diretorio=diretorio)
results.set_DF_by_category(category_name = 'pyrolysis', DF_column_with_cat_vals_to_group = 'synthesis_method', 
                           DF_columns_with_num_vals = ['temperature_carbonization', 'raw_materials'])
results.process_columns_with_num_val(DF_columns=['temperature_carbonization'], mode='avg')
results.plot_group_num_boxplot_correlation(DF_column_with_cat_vals = 'raw_materials',
                                           DF_column_with_num_vals = 'temperature_carbonization',
                                           axes_labels=['biomass precursors', 'avg pyrol temperature (°C)'],
                                           y_quantiles = [0.02, 0.98],
                                           min_values_for_column = 12,
                                           colobar_nticks = 10,
                                           palette = 'Viridis256',
                                           background_fill_color = 'blue',
                                           size_factor_boxplot = 1.1,
                                           grouplabel_x_offset = 9,
                                           box_plot_plot_width=2000, box_plot_plot_height=1000,
                                           size_factor_anova_grid = 0.7,
                                           grid_plot_width=1000, grid_plot_height=1000)

#Fig9
#Fig10
#use "NA" nos quantiles caso queira começar com o valor zero no intervalo (ex: x_quantiles = ['NA', 0.98], y_quantiles = ['NA', 0.98])
#instanciando o DataFrame consolidado
from RESULTS import results
results = results(results_DF_input_name = '_CONSOLIDATED_DF', diretorio=diretorio)
results.set_DF_by_category(category_name = 'pyrolysis', DF_column_with_cat_vals_to_group = 'synthesis_method', 
                           DF_columns_with_num_vals = ['time_carbonization', 'raw_materials'])
results.process_columns_with_num_val(DF_columns=['time_carbonization'], mode='avg')
results.plot_group_num_boxplot_correlation(DF_column_with_cat_vals = 'raw_materials',
                                           DF_column_with_num_vals = 'time_carbonization',
                                           axes_labels=['biomass precursors', 'avg pyrol time (min)'],
                                           y_quantiles = [0.02, 0.98],
                                           min_values_for_column = 10,
                                           colobar_nticks = 10,
                                           palette = 'Viridis256',
                                           background_fill_color = 'blue',
                                           size_factor_boxplot = 3.0,
                                           grouplabel_x_offset = -20,
                                           box_plot_plot_width=2000, box_plot_plot_height=1400,
                                           size_factor_anova_grid = 2.0,
                                           grid_plot_width=1000, grid_plot_height=1000)

#Fig11
#use "NA" nos quantiles caso queira começar com o valor zero no intervalo (ex: x_quantiles = ['NA', 0.98], y_quantiles = ['NA', 0.98])
#instanciando o DataFrame consolidado
from RESULTS import results
results = results(results_DF_input_name = '_CONSOLIDATED_DF', diretorio=diretorio)
results.set_DF_by_category(category_name = 'pyrolysis', DF_column_with_cat_vals_to_group = 'synthesis_method', 
                           DF_columns_with_num_vals = ['temperature_carbonization', 'time_carbonization', 'raw_materials'])
results.process_columns_with_num_val(DF_columns=['temperature_carbonization'], mode='avg')
results.process_columns_with_num_val(DF_columns=['time_carbonization'], mode='avg')
results.plot_group_num_num_correlation(DF_columns_with_num_vals = ['temperature_carbonization', 'time_carbonization'],
                                       axes_labels = ['avg pyrol temperature (°C)', 'avg pyrol time (min)'], 
                                       x_quantiles = [0.05, 0.95], y_quantiles = [0.05, 0.95],
                                       DF_column_with_cat_vals_to_group = 'raw_materials',
                                       categories_to_get = None,
                                       x_min = None, x_max = None, y_min = None, y_max = 1300,
                                       min_values_for_column = 15, mode='scatter', regression=False,
                                       hex_size = 20, plot_width=1200, plot_height=1200,
                                       cluster_preffix='', export_groups_to_csv=False)

#Fig12
#use "NA" nos quantiles caso queira começar com o valor zero no intervalo (ex: x_quantiles = ['NA', 0.98], y_quantiles = ['NA', 0.98])
#instanciando o DataFrame consolidado
from RESULTS import results
results = results(results_DF_input_name = '_CONSOLIDATED_DF', diretorio=diretorio)
results.process_columns_with_num_val(DF_columns=['temperature_carbonization'], mode='avg')
results.process_columns_with_num_val(DF_columns=['time_carbonization'], mode='avg')
results.plot_group_num_num_correlation(DF_columns_with_num_vals = ['temperature_carbonization', 'time_carbonization'], 
                                       axes_labels = ['avg carb temperature (°C)', 'avg carb time (min)'], 
                                       x_quantiles = [0.05, 0.95], y_quantiles = [0.05, 0.95],
                                       DF_column_with_cat_vals_to_group = 'synthesis_method',
                                       categories_to_get = ['pyrolysis', 'hydrothermal', 'gasification', 'torrefaction'],
                                       x_min = None, x_max = None, y_min = None, y_max = 1300,
                                       min_values_for_column = 20, mode='scatter', regression=True,
                                       hex_size = 20, plot_width=1200, plot_height=1200,
                                       cluster_preffix='', export_groups_to_csv=False)

#Fig13
#use "NA" nos quantiles caso queira começar com o valor zero no intervalo (ex: x_quantiles = ['NA', 0.98], y_quantiles = ['NA', 0.98])
#instanciando o DataFrame consolidado
from RESULTS import results
results = results(results_DF_input_name = '_CONSOLIDATED_DF', diretorio=diretorio)
results.set_DF_by_category(category_name = 'pyrolysis', DF_column_with_cat_vals_to_group = 'synthesis_method', 
                           DF_columns_with_num_vals = ['temperature_carbonization', 'H#C', 'C#H'])
results.process_columns_with_num_val(DF_columns=['temperature_carbonization'], mode='avg')
results.process_columns_with_num_val(DF_columns=['C#H'], mode='higher')
results.process_columns_with_num_val(DF_columns=['H#C'], mode='lower')
#results.merge_columns_with_num_val(base_column = 'C#H', column_to_merge = 'H#C', invert_column_val = True)
results.multiply_column_by_factor(DF_columns=['H#C'], factor = 1000)
results.plot_num_num_correlation(DF_columns_with_num_vals = ['temperature_carbonization', 'H#C'], 
                                 axes_labels = ['avg pyrol temperature (°C)', 'l H/C (x 0.001)'], 
                                 x_quantiles = [0.05, 0.95], y_quantiles = [0.05, 0.95],
                                 x_min = 150, x_max = 900, y_min = None, y_max = None,
                                 hex_size = 20, plot_width=1200, plot_height=1200,
                                 mode='scatter', regression = True, find_clusters = True, n_clusters = 2,
                                 cluster_preffix='H/C ', export_groups_to_csv=False)

#Fig14
#use "NA" nos quantiles caso queira começar com o valor zero no intervalo (ex: x_quantiles = ['NA', 0.98], y_quantiles = ['NA', 0.98])
#instanciando o DataFrame consolidado
from RESULTS import results
results = results(results_DF_input_name = '_CONSOLIDATED_DF', diretorio=diretorio)
results.set_DF_by_category(category_name = 'pyrolysis', DF_column_with_cat_vals_to_group = 'synthesis_method', 
                           DF_columns_with_num_vals = ['temperature_carbonization', 'O#C', 'C#O'])
results.process_columns_with_num_val(DF_columns=['temperature_carbonization'], mode='avg')
results.process_columns_with_num_val(DF_columns=['C#O'], mode='higher')
results.process_columns_with_num_val(DF_columns=['O#C'], mode='lower')
#results.merge_columns_with_num_val(base_column = 'C#O', column_to_merge = 'O#C', invert_column_val = True)
results.multiply_column_by_factor(DF_columns=['O#C'], factor = 1000)
results.plot_num_num_correlation(DF_columns_with_num_vals = ['temperature_carbonization', 'O#C'], 
                                 axes_labels = ['avg pyrol temperature (°C)', 'l O/C (x 0.001)'], 
                                 x_quantiles = [0.05, 0.95], y_quantiles = [0.05, 0.95],
                                 x_min = 150, x_max = 900, y_min = None, y_max = None,
                                 hex_size = 20, plot_width=1200, plot_height=1200,
                                 mode='scatter', regression = True, find_clusters = True, n_clusters = 2,
                                 cluster_preffix='', export_groups_to_csv=True)

#Fig15
#use "NA" nos quantiles caso queira começar com o valor zero no intervalo (ex: x_quantiles = ['NA', 0.98], y_quantiles = ['NA', 0.98])
#instanciando o DataFrame consolidado
from RESULTS import results
results = results(results_DF_input_name = '_CONSOLIDATED_DF', diretorio=diretorio)
results.set_DF_by_category(category_name = 'pyrolysis', DF_column_with_cat_vals_to_group = 'synthesis_method', 
                           DF_columns_with_num_vals = ['temperature_carbonization', 'N#C', 'C#N'])
results.process_columns_with_num_val(DF_columns=['temperature_carbonization'], mode='avg')
results.process_columns_with_num_val(DF_columns=['C#N'], mode='lower')
results.process_columns_with_num_val(DF_columns=['N#C'], mode='higher')
#results.merge_columns_with_num_val(base_column = 'N#C', column_to_merge = 'C#N', invert_column_val = True)
results.multiply_column_by_factor(DF_columns=['C#N'], factor = 1)
results.plot_num_num_correlation(DF_columns_with_num_vals = ['temperature_carbonization', 'C#N'],
                                 axes_labels = ['avg pyrol temperature (°C)', 'l C/N'],
                                 x_quantiles = [0.05, 0.95], y_quantiles = [0.05, 0.95],
                                 x_min = 150, x_max = 900, y_min = None, y_max = None,
                                 hex_size = 20, plot_width=1200, plot_height=1200,
                                 mode='scatter', regression = True, find_clusters = True, n_clusters = 2,
                                 cluster_preffix='', export_groups_to_csv=True,
                                 show_figure = True)

#Fig16
#use "NA" nos quantiles caso queira começar com o valor zero no intervalo (ex: x_quantiles = ['NA', 0.98], y_quantiles = ['NA', 0.98])
#instanciando o DataFrame consolidado
from RESULTS import results
results = results(results_DF_input_name = '_CONSOLIDATED_DF', diretorio=diretorio)
results.set_DF_by_category(category_name = 'pyrolysis', DF_column_with_cat_vals_to_group = 'synthesis_method', 
                           DF_columns_with_num_vals = ['temperature_carbonization','concentration_hhv'])
results.process_columns_with_num_val(DF_columns=['temperature_carbonization'], mode='avg')
results.process_columns_with_num_val(DF_columns=['concentration_hhv'], mode='higher')
results.multiply_column_by_factor(DF_columns=['concentration_hhv'], factor = 0.001)
results.plot_num_num_correlation(DF_columns_with_num_vals = ['temperature_carbonization', 'concentration_hhv'], 
                                 axes_labels = ['avg pyrol temperature (°C)', 'h HHV (MJ kg-1)'],
                                 x_quantiles = [0.05, 0.95], y_quantiles = [0.05, 0.95],
                                 x_min = 150, x_max = 900, y_min = None, y_max = 80,
                                 hex_size = 20, plot_width=1200, plot_height=1200,
                                 mode='scatter', regression = True, find_clusters = True, n_clusters = 2, 
                                 cluster_preffix='', export_groups_to_csv=True)

#Fig17
#use "NA" nos quantiles caso queira começar com o valor zero no intervalo (ex: x_quantiles = ['NA', 0.98], y_quantiles = ['NA', 0.98])
#instanciando o DataFrame consolidado
from RESULTS import results
results = results(results_DF_input_name = '_CONSOLIDATED_DF', diretorio=diretorio)
results.set_DF_by_category(category_name = 'pyrolysis', DF_column_with_cat_vals_to_group = 'synthesis_method', 
                           DF_columns_with_num_vals = ['temperature_carbonization', 'C'])
results.process_columns_with_num_val(DF_columns=['temperature_carbonization'], mode='avg')
results.process_columns_with_num_val(DF_columns=['C'], mode='higher')
results.plot_num_num_correlation(DF_columns_with_num_vals = ['temperature_carbonization', 'C'],
                                 axes_labels = ['avg pyrol temperature (°C)', 'h C (%)'],
                                 x_quantiles = [0.05, 0.95], y_quantiles = [0.05, 0.95],
                                 x_min = 150, x_max = 900, y_min = None, y_max = None,
                                 hex_size = 20, plot_width=1200, plot_height=1200,
                                 mode='scatter', regression = True, find_clusters = False, n_clusters = 0, 
                                 cluster_preffix='', export_groups_to_csv=False)

#Fig18
#use "NA" nos quantiles caso queira começar com o valor zero no intervalo (ex: x_quantiles = ['NA', 0.98], y_quantiles = ['NA', 0.98])
#instanciando o DataFrame consolidado
from RESULTS import results
results = results(results_DF_input_name = '_CONSOLIDATED_DF', diretorio=diretorio)
results.set_DF_by_category(category_name = 'pyrolysis', DF_column_with_cat_vals_to_group = 'synthesis_method', 
                           DF_columns_with_num_vals = ['temperature_carbonization', 'N'])
results.process_columns_with_num_val(DF_columns=['temperature_carbonization'], mode='avg')
results.process_columns_with_num_val(DF_columns=['N'], mode='higher')
results.plot_num_num_correlation(DF_columns_with_num_vals = ['temperature_carbonization', 'N'],
                                 axes_labels = ['avg pyrol temperature (°C)', 'h N (%)'],
                                 x_quantiles = [0.05, 0.95], y_quantiles = [0.05, 0.95],
                                 x_min = 150, x_max = 900, y_min = None, y_max = None,
                                 hex_size = 20, plot_width=1200, plot_height=1200,
                                 mode='scatter', regression = True, find_clusters = False, n_clusters = 0, 
                                 cluster_preffix='', export_groups_to_csv=False)

#Fig19
#use "NA" nos quantiles caso queira começar com o valor zero no intervalo (ex: x_quantiles = ['NA', 0.98], y_quantiles = ['NA', 0.98])
#instanciando o DataFrame consolidado
from RESULTS import results
results = results(results_DF_input_name = '_CONSOLIDATED_DF', diretorio=diretorio)
results.set_DF_by_category(category_name = 'pyrolysis', DF_column_with_cat_vals_to_group = 'synthesis_method', 
                           DF_columns_with_num_vals = ['temperature_carbonization', 'P'])
results.process_columns_with_num_val(DF_columns=['temperature_carbonization'], mode='avg')
results.process_columns_with_num_val(DF_columns=['P'], mode='higher')
results.plot_num_num_correlation(DF_columns_with_num_vals = ['temperature_carbonization', 'P'],
                                 axes_labels = ['avg pyrol temperature (°C)', 'h P (%)'],
                                 x_quantiles = [0.05, 0.95], y_quantiles = [0.05, 0.95],
                                 x_min = 150, x_max = 900, y_min = None, y_max = None,
                                 hex_size = 20, plot_width=1200, plot_height=1200,
                                 mode='scatter', regression = True, find_clusters = False, n_clusters = 0, 
                                 cluster_preffix='', export_groups_to_csv=False)

#Fig20
#use "NA" nos quantiles caso queira começar com o valor zero no intervalo (ex: x_quantiles = ['NA', 0.98], y_quantiles = ['NA', 0.98])
#instanciando o DataFrame consolidado
from RESULTS import results
results = results(results_DF_input_name = '_CONSOLIDATED_DF', diretorio=diretorio)
results.set_DF_by_category(category_name = 'pyrolysis', DF_column_with_cat_vals_to_group = 'synthesis_method', 
                           DF_columns_with_num_vals = ['temperature_carbonization','surface_area'])
results.process_columns_with_num_val(DF_columns=['temperature_carbonization'], mode='avg')
results.process_columns_with_num_val(DF_columns=['surface_area'], mode='higher')
results.plot_num_num_correlation(DF_columns_with_num_vals = ['temperature_carbonization', 'surface_area'], 
                                 axes_labels = ['avg pyrol temperature (°C)', 'h surface area (m2 g-1)'],
                                 x_quantiles = [0.02, 0.98], y_quantiles = [0.02, 0.98],
                                 x_min = 150, x_max = 900, y_min = None, y_max = 1500,
                                 hex_size = 20, plot_width=1200, plot_height=1200,
                                 mode='scatter', regression = True, find_clusters = True, n_clusters = 4, 
                                 cluster_preffix='SA ', export_groups_to_csv=False)

#Fig21
#use "NA" nos quantiles caso queira começar com o valor zero no intervalo (ex: x_quantiles = ['NA', 0.98], y_quantiles = ['NA', 0.98])
#instanciando o DataFrame consolidado
from RESULTS import results
results = results(results_DF_input_name = '_CONSOLIDATED_DF', diretorio=diretorio)
results.set_DF_by_category(category_name = 'pyrolysis', DF_column_with_cat_vals_to_group = 'synthesis_method', 
                           DF_columns_with_num_vals = ['surface_area','distance_particlesize'])
results.process_columns_with_num_val(DF_columns=['surface_area'], mode='higher')
results.process_columns_with_num_val(DF_columns=['distance_particlesize'], mode='higher')
results.multiply_column_by_factor(DF_columns=['distance_particlesize'], factor = 1000000)
results.plot_num_num_correlation(DF_columns_with_num_vals = ['distance_particlesize', 'surface_area'],
                                 axes_labels = ['h particle size (µm)', 'h surface area (m2 g-1)'],
                                 x_quantiles = [0.05, 0.95], y_quantiles = [0.05, 0.95],
                                 x_min = None, x_max = 2000, y_min = None, y_max = 1500,
                                 hex_size = 20, plot_width=1200, plot_height=1200,
                                 mode='scatter', regression = True, find_clusters = False, n_clusters = 0,
                                 cluster_preffix='', export_groups_to_csv=False)

#Fig22
#use "NA" nos quantiles caso queira começar com o valor zero no intervalo (ex: x_quantiles = ['NA', 0.98], y_quantiles = ['NA', 0.98])
#instanciando o DataFrame consolidado
from RESULTS import results
results = results(results_DF_input_name = '_CONSOLIDATED_DF', diretorio=diretorio)
results.set_DF_by_category(category_name = 'pyrolysis', DF_column_with_cat_vals_to_group = 'synthesis_method', 
                           DF_columns_with_num_vals = ['surface_area', 'concentration_mass_mass'])
results.process_columns_with_num_val(DF_columns=['concentration_mass_mass'], mode='higher')
results.process_columns_with_num_val(DF_columns=['surface_area'], mode='higher')
results.multiply_column_by_factor(DF_columns=['concentration_mass_mass'], factor = 1000)
results.plot_num_num_correlation(DF_columns_with_num_vals = ['concentration_mass_mass', 'surface_area'], 
                                 axes_labels = ['h adsorption cap (g kg-1)', 'h surface area (m2 g-1)'],
                                 x_quantiles = [0.05, 0.95], y_quantiles = [0.05, 0.95],
                                 x_min = None, x_max = None, y_min = None, y_max = 1500,
                                 hex_size = 20, plot_width=1200, plot_height=1200,
                                 mode='scatter', regression = True, find_clusters = False, n_clusters = 4, 
                                 cluster_preffix='', export_groups_to_csv=True)

#Fig23
#use "NA" nos quantiles caso queira começar com o valor zero no intervalo (ex: x_quantiles = ['NA', 0.98], y_quantiles = ['NA', 0.98])
#instanciando o DataFrame consolidado
from RESULTS import results
results = results(results_DF_input_name = '_CONSOLIDATED_DF', diretorio=diretorio)
results.set_DF_by_category(category_name = 'pyrolysis', DF_column_with_cat_vals_to_group = 'synthesis_method', 
                           DF_columns_with_num_vals = ['temperature_carbonization', 'time_carbonization', 'surface_area'])
results.process_columns_with_num_val(DF_columns=['temperature_carbonization'], mode='avg')
results.process_columns_with_num_val(DF_columns=['time_carbonization'], mode='avg')
results.process_columns_with_num_val(DF_columns=['surface_area'], mode='higher')
results.plot_pca_results(DF_columns=['temperature_carbonization', 'time_carbonization', 'surface_area'], 
                         loadings_label=['temperature', 'time', 'surface area'],
                         axes_labels=['PC1', 'PC2',],
                         quantiles = [0.05, 0.95],
                         x_min = -3, x_max = 5, y_min = -1.5, y_max = 4.5,
                         plot_width=1000, plot_height=1000,
                         find_clusters = True, n_clusters = 3, cluster_preffix='',
                         loading_arrow_factor = 4.0, export_groups_to_csv=True,
                         show_figure = False)

#Fig24
from RESULTS import results
results = results(results_DF_input_name = '_CONSOLIDATED_DF', diretorio=diretorio)
results.plot_grouped_cat_barplot_from_fileindexes(plot_name = 'P0013', DF_column_with_cat_vals_to_plot = 'raw_materials', maximum_values_to_plot = 15,
                                                  cluster_preffix = 'H/C ', graph_title = 'Biomass precursors')


#Fig25
#Fig26
#Fig27
#plotando os grafos de grupos
#instanciando o DataFrame consolidado
from RESULTS import results
results = results(results_DF_input_name = 'process_P0013_G1_FULL_DF')
graph_results_1 = results.analyze_1column_network_digraph(DF_column = 'processes', max_circle_size = 90, min_circle_size = 50, min_occurrences = 5, graph_base_title = 'H/C 1',
                                                    start_end_nodes_to_analyze = [], path_nodes_cutoff = 11, min_edge_weight = 0.5,
                                                    plot_graphs = False, print_unique_vals_in_DF = False)

#Fig28
#Fig29
#Fig30
#instanciando o DataFrame consolidado
from RESULTS import results
results = results(results_DF_input_name = 'process_P0013_G2_FULL_DF')
graph_results_2 = results.analyze_1column_network_digraph(DF_column = 'processes', max_circle_size = 90, min_circle_size = 50, min_occurrences = 5, graph_base_title = 'H/C 2',
                                                    start_end_nodes_to_analyze = [], path_nodes_cutoff = 11, min_edge_weight = 0.5,
                                                    plot_graphs = False, print_unique_vals_in_DF = False)

#Fig31
from RESULTS import groups_results
grop_results = groups_results(graph_results_1, graph_results_2)
grop_results.bar_plot(maximum_values_to_plot = 20)


#instanciando o DataFrame consolidado
from RESULTS import results
results = results(results_DF_input_name = 'process_P0014_G1_FULL_DF')
graph_results_3 = results.analyze_1column_network_digraph(DF_column = 'processes', max_circle_size = 60, min_circle_size = 30, min_occurrences = 5, graph_base_title = 'O/C 1',
                                                    start_end_nodes_to_analyze = [], path_nodes_cutoff = 11, min_edge_weight = 0.4,
                                                    plot_graphs = True, print_unique_vals_in_DF = False)

#instanciando o DataFrame consolidado
from RESULTS import results
results = results(results_DF_input_name = 'process_P0014_G2_FULL_DF')
graph_results_4 = results.analyze_1column_network_digraph(DF_column = 'processes', max_circle_size = 60, min_circle_size = 30, min_occurrences = 5, graph_base_title = 'O/C 2',
                                                    start_end_nodes_to_analyze = [], path_nodes_cutoff = 11, min_edge_weight = 0.4,
                                                    plot_graphs = True, print_unique_vals_in_DF = False)

#Fig32
from RESULTS import groups_results
grop_results = groups_results(graph_results_3, graph_results_4)
grop_results.bar_plot(maximum_values_to_plot = 20)


#Fig33
from RESULTS import results
results = results(results_DF_input_name = '_CONSOLIDATED_DF', diretorio=diretorio)
results.plot_grouped_cat_barplot_from_fileindexes(plot_name = 'P0020', DF_column_with_cat_vals_to_plot = 'raw_materials', maximum_values_to_plot = 15,
                                                  cluster_preffix = 'SA ', graph_title = 'Biomass precursors')

#plotando os grafos de grupos
#instanciando o DataFrame consolidado
from RESULTS import results
results = results(results_DF_input_name = 'process_P0020_G1_FULL_DF')
graph_results_5 = results.analyze_1column_network_digraph(DF_column = 'processes', max_circle_size = 60, min_circle_size = 30, min_occurrences = 10, graph_base_title = 'SA 1',
                                                    start_end_nodes_to_analyze = ['drying', 'storage'], path_nodes_cutoff = 11, min_edge_weight = 0.5,
                                                    plot_graphs = False, print_unique_vals_in_DF = False)

#plotando os grafos de grupos
#instanciando o DataFrame consolidado
from RESULTS import results
results = results(results_DF_input_name = 'process_P0020_G2_FULL_DF')
graph_results_6 = results.analyze_1column_network_digraph(DF_column = 'processes', max_circle_size = 60, min_circle_size = 30, min_occurrences = 10, graph_base_title = 'SA 2',
                                                    start_end_nodes_to_analyze = ['drying', 'storage'], path_nodes_cutoff = 11, min_edge_weight = 0.5,
                                                    plot_graphs = False, print_unique_vals_in_DF = False)

#instanciando o DataFrame consolidado
from RESULTS import results
results = results(results_DF_input_name = 'process_P0020_G3_FULL_DF')
graph_results_7 = results.analyze_1column_network_digraph(DF_column = 'processes', max_circle_size = 60, min_circle_size = 30, min_occurrences = 10, graph_base_title = 'SA 3',
                                                    start_end_nodes_to_analyze = ['drying', 'storage'], path_nodes_cutoff = 11, min_edge_weight = 0.5,
                                                    plot_graphs = False, print_unique_vals_in_DF = False)

#instanciando o DataFrame consolidado
from RESULTS import results
results = results(results_DF_input_name = 'process_P0020_G4_FULL_DF')
graph_results_8 = results.analyze_1column_network_digraph(DF_column = 'processes', max_circle_size = 60, min_circle_size = 30, min_occurrences = 10, graph_base_title = 'SA 4',
                                                    start_end_nodes_to_analyze = ['drying', 'storage'], path_nodes_cutoff = 11, min_edge_weight = 0.5,
                                                    plot_graphs = False, print_unique_vals_in_DF = False)

#Fig34
from RESULTS import groups_results
grop_results = groups_results(graph_results_5, graph_results_6, graph_results_7, graph_results_8)
grop_results.bar_plot(maximum_values_to_plot = 15)



#Fig35
#use "NA" nos quantiles caso queira começar com o valor zero no intervalo (ex: x_quantiles = ['NA', 0.98], y_quantiles = ['NA', 0.98])
#instanciando o DataFrame consolidado
from RESULTS import results
results = results(results_DF_input_name = '_CONSOLIDATED_DF', diretorio=diretorio)
results.process_columns_with_num_val(DF_columns=['surface_area'], mode='higher')
results.process_columns_with_num_val(DF_columns=['O#C'], mode='lower')
results.multiply_column_by_factor(DF_columns=['O#C'], factor = 1000)
results.plot_group_num_num_correlation(DF_columns_with_num_vals = ['O#C', 'surface_area'], 
                                       axes_labels = ['l O/C (x 0.001)', 'h surface area (m2 g-1)'], 
                                       x_quantiles = [0.05, 0.95], y_quantiles = [0.05, 0.95],
                                       DF_column_with_cat_vals_to_group = 'synthesis_method',
                                       categories_to_get = None,
                                       x_min = None, x_max = None, y_min = None, y_max = 1300,
                                       min_values_for_column = 23, mode='scatter', regression=True,
                                       hex_size = 20, plot_width=1200, plot_height=1200,
                                       cluster_preffix='', export_groups_to_csv=False)

#Fig36
#use "NA" nos quantiles caso queira começar com o valor zero no intervalo (ex: x_quantiles = ['NA', 0.98], y_quantiles = ['NA', 0.98])
#instanciando o DataFrame consolidado
from RESULTS import results
results = results(results_DF_input_name = '_CONSOLIDATED_DF', diretorio=diretorio)
results.process_columns_with_num_val(DF_columns=['surface_area'], mode='higher')
results.process_columns_with_num_val(DF_columns=['temperature_carbonization'], mode='avg')
results.multiply_column_by_factor(DF_columns=['H#C'], factor = 1000)
results.plot_group_num_num_correlation(DF_columns_with_num_vals = ['temperature_carbonization', 'surface_area'], 
                                       axes_labels = ['avg carb temperature (°C)', 'h surface area (m2 g-1)'], 
                                       x_quantiles = [0.05, 0.95], y_quantiles = [0.05, 0.95],
                                       DF_column_with_cat_vals_to_group = 'synthesis_method',
                                       categories_to_get = None,
                                       x_min = None, x_max = None, y_min = None, y_max = 1300,
                                       min_values_for_column = 23, mode='scatter', regression=False,
                                       hex_size = 20, plot_width=1200, plot_height=1200,
                                       cluster_preffix='', export_groups_to_csv=False)


#Fig37
from RESULTS import results
results = results(results_DF_input_name = '_CONSOLIDATED_DF', diretorio=diretorio)
results.plot_grouped_cat_barplot_from_fileindexes(plot_name = 'P0014', DF_column_with_cat_vals_to_plot = 'raw_materials', maximum_values_to_plot = 15,
                                                  cluster_preffix = 'O/C ', graph_title = 'Biomass precursors')


#Fig38
from RESULTS import results
results = results(results_DF_input_name = '_CONSOLIDATED_DF', diretorio=diretorio)
results.plot_grouped_cat_barplot_from_fileindexes(plot_name = 'P0038', DF_column_with_cat_vals_to_plot = 'raw_materials', maximum_values_to_plot = 15,
                                                  cluster_preffix = 'PCA ', graph_title = 'Biomass precursors')



#Fig39
#use "NA" nos quantiles caso queira começar com o valor zero no intervalo (ex: x_quantiles = ['NA', 0.98], y_quantiles = ['NA', 0.98])
#instanciando o DataFrame consolidado
from RESULTS import results
results = results(results_DF_input_name = '_CONSOLIDATED_DF', diretorio=diretorio)
results.set_DF_by_category(category_name = 'pyrolysis', DF_column_with_cat_vals_to_group = 'synthesis_method', 
                           DF_columns_with_num_vals = ['H#C', 'O#C', 'surface_area'])
results.process_columns_with_num_val(DF_columns=['H#C'], mode='lower')
results.process_columns_with_num_val(DF_columns=['O#C'], mode='lower')
results.process_columns_with_num_val(DF_columns=['surface_area'], mode='higher')
results.multiply_column_by_factor(DF_columns=['H#C'], factor = 1000)
results.multiply_column_by_factor(DF_columns=['O#C'], factor = 1000)
results.plot_pca_results(DF_columns=['H#C', 'O#C', 'surface_area'], 
                         loadings_label=['l H/C', 'l O/C', 'h surface area'],
                         axes_labels=['PC1', 'PC2'],
                         quantiles = [0.05, 0.95],
                         x_min = -3, x_max = 5, y_min = -1.5, y_max = 4.5,
                         plot_width=1000, plot_height=1000,
                         find_clusters = False, n_clusters = 3, cluster_preffix='',
                         loading_arrow_factor = 4.0, export_groups_to_csv=False,
                         show_figure = False)


#Fig40
#use "NA" nos quantiles caso queira começar com o valor zero no intervalo (ex: x_quantiles = ['NA', 0.98], y_quantiles = ['NA', 0.98])
#instanciando o DataFrame consolidado
from RESULTS import results
results = results(results_DF_input_name = '_CONSOLIDATED_DF', diretorio=diretorio)
results.process_columns_with_num_val(DF_columns=['surface_area'], mode='higher')
results.process_columns_with_num_val(DF_columns=['O#C'], mode='lower')
results.multiply_column_by_factor(DF_columns=['O#C'], factor = 1000)
results.plot_group_num_num_correlation(DF_columns_with_num_vals = ['O#C', 'surface_area'], 
                                       axes_labels = ['l O/C (x 0.001)', 'h surface area (m2 g-1)'], 
                                       x_quantiles = [0.05, 0.95], y_quantiles = [0.05, 0.95],
                                       DF_column_with_cat_vals_to_group = 'synthesis_method',
                                       categories_to_get = None,
                                       x_min = None, x_max = None, y_min = None, y_max = 1300,
                                       min_values_for_column = 23, mode='scatter', regression=True,
                                       hex_size = 20, plot_width=1200, plot_height=1200,
                                       cluster_preffix='', export_groups_to_csv=False)

#Fig41
#use "NA" nos quantiles caso queira começar com o valor zero no intervalo (ex: x_quantiles = ['NA', 0.98], y_quantiles = ['NA', 0.98])
#instanciando o DataFrame consolidado
from RESULTS import results
results = results(results_DF_input_name = '_CONSOLIDATED_DF', diretorio=diretorio)
results.process_columns_with_num_val(DF_columns=['distance_particlesize'], mode='higher')
results.process_columns_with_num_val(DF_columns=['surface_area'], mode='higher')
results.multiply_column_by_factor(DF_columns=['distance_particlesize'], factor = 1000000)
results.plot_group_num_num_correlation(DF_columns_with_num_vals = ['distance_particlesize', 'surface_area'], 
                                       axes_labels = ['h particle size (micron)', 'h surface area (m2 g-1)'], 
                                       x_quantiles = [0.05, 0.95], y_quantiles = [0.05, 0.95],
                                       DF_column_with_cat_vals_to_group = 'raw_materials',
                                       categories_to_get = ['wheat straw', 'rice husk', 'rice straw', 'pine wood', 'peanut shell', 'corn stover', 'wood chip'],
                                       x_min = None, x_max = 5000, y_min = None, y_max = 1300,
                                       min_values_for_column = 1, mode='scatter', regression=False,
                                       hex_size = 20, plot_width=1200, plot_height=1200,
                                       cluster_preffix='', export_groups_to_csv=False)

#Fig42
#Fig43
#use "NA" nos quantiles caso queira começar com o valor zero no intervalo (ex: x_quantiles = ['NA', 0.98], y_quantiles = ['NA', 0.98])
#instanciando o DataFrame consolidado
from RESULTS import results
results = results(results_DF_input_name = '_CONSOLIDATED_DF', diretorio=diretorio)
results.set_DF_by_category(category_name = 'pyrolysis', DF_column_with_cat_vals_to_group = 'synthesis_method', 
                           DF_columns_with_num_vals = ['surface_area', 'raw_materials', 'H#C'])
results.process_columns_with_num_val(DF_columns=['surface_area'], mode='higher')
results.plot_group_num_boxplot_correlation(DF_column_with_cat_vals = 'raw_materials',
                                           DF_column_with_num_vals = 'surface_area',
                                           categories_to_get = ['corn stover', 'peanut shell', 'pine wood', 'rice straw', 'rice husk', 'sugarcane bagasse', 'wheat straw', 'wood chip'],
                                           num_values_to_filter = [100, 1500],
                                           axes_labels=['biomass precursors', 'h surface area (m2 g-1)'],
                                           y_quantiles = [0.02, 0.98],
                                           min_values_for_column = 10,
                                           colobar_nticks = 10,
                                           palette = 'Viridis256',
                                           background_fill_color = 'blue',
                                           size_factor_boxplot = 4.5,
                                           grouplabel_x_offset = -35,
                                           box_plot_plot_width=1700, box_plot_plot_height=1700,
                                           size_factor_anova_grid = 5.1,
                                           grid_plot_width=1500, grid_plot_height=1500,
                                           export_groups_to_csv = False)

#Fig44
#Fig45
#use "NA" nos quantiles caso queira começar com o valor zero no intervalo (ex: x_quantiles = ['NA', 0.98], y_quantiles = ['NA', 0.98])
#instanciando o DataFrame consolidado
from RESULTS import results
results = results(results_DF_input_name = '_CONSOLIDATED_DF', diretorio=diretorio)
results.set_DF_by_category(category_name = 'pyrolysis', DF_column_with_cat_vals_to_group = 'synthesis_method', 
                           DF_columns_with_num_vals = ['surface_area', 'raw_materials', 'H#C'])
results.process_columns_with_num_val(DF_columns=['H#C'], mode='lower')
results.multiply_column_by_factor(DF_columns=['H#C'], factor = 1000)
results.plot_group_num_boxplot_correlation(DF_column_with_cat_vals = 'raw_materials',
                                           DF_column_with_num_vals = 'H#C',
                                           categories_to_get = ['corn stover', 'peanut shell', 'pine wood', 'rice straw', 'rice husk', 'sugarcane bagasse', 'wheat straw', 'wood chip'],
                                           num_values_to_filter = [0, None],
                                           axes_labels=['biomass precursors', 'l H/C (x 0.001)'],
                                           y_quantiles = [0.02, 0.98],
                                           min_values_for_column = 2,
                                           colobar_nticks = 10,
                                           palette = 'Viridis256',
                                           background_fill_color = 'blue',
                                           size_factor_boxplot = 4.5,
                                           grouplabel_x_offset = -35,
                                           box_plot_plot_width=1700, box_plot_plot_height=1700,
                                           size_factor_anova_grid = 5.1,
                                           grid_plot_width=1500, grid_plot_height=1500,
                                           export_groups_to_csv = False)

#Fig46
for cat in ('wheat straw', 'rice straw', 'peanut shell'):
    from RESULTS import results
    results = results(results_DF_input_name = '_CONSOLIDATED_DF', diretorio=diretorio)
    results.set_DF_by_category(category_name = 'pyrolysis', DF_column_with_cat_vals_to_group = 'synthesis_method', 
                               DF_columns_with_num_vals = ['raw_materials', 'temperature_carbonization', 'surface_area', 'H#C'])
    results.process_columns_with_num_val(DF_columns=['temperature_carbonization'], mode='avg')
    results.process_columns_with_num_val(DF_columns=['surface_area'], mode='higher')
    results.process_columns_with_num_val(DF_columns=['H#C'], mode='lower')
    results.multiply_column_by_factor(DF_columns=['H#C'], factor = 1000)
    results.plot_grouped_filtered_cat_boxplot(column_to_group = 'raw_materials', 
                                              category_to_analyze = cat,
                                              columns_with_num_vals_to_filter = ['surface_area'],
                                              min_max_vals_to_filter = [[0, 1000]],
                                              ymin_ymax_list = [[0, 1000], [0, 1000], [0, 1500]], 
                                              x_axis_label = '', 
                                              y_axes_labels=['avg pyrol temperature (°C)', 'h surface area (m2 g-1)', 'l H/C (x 0.001)'], 
                                              export_groups_to_csv=False,
                                              show_figure = False)
'''