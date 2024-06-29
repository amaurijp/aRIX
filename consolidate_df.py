#!/usr/bin/env python3
# -*- coding: utf-8 -*-

    
def process():

    import sys
    import os
    program_folder = os.getcwd()
    sys.path.append(program_folder + '/Modules')
    
    from DFs import DataFrames    
    from FUNCTIONS import extract_inputs_from_csv
    inputs_to_consolidate = extract_inputs_from_csv(csv_filename = 'DFs_to_consolidate', diretorio = program_folder, mode = 'consolidate_df')
    
    for i in inputs_to_consolidate.keys():
        
        DF = DataFrames(diretorio = program_folder)
        DF.set_settings_to_concatenate(dic_inputs = inputs_to_consolidate[i])
        DF.consolidate_DF()


#executando a função
process()
