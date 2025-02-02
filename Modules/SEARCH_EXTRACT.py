#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import regex as re
from FUNCTIONS import extract_inputs_from_csv
from DFs import DataFrames
from SCHENG import search_engine
from FUNCTIONS import load_dic_from_json

def main():
      
    #Função principal   
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-s', '--search_input_index', default = 0, help = 'Introduzir o index com os inputs de SEARCH/EXTRACT (manipular arquivo em ~/Settings/SE_inputs.csv).', type = int)
    parser.add_argument('-r', '--revise_search_cond', default = 'no', help ='Decidir se será feita a checagem de todas as condições de busca.', type = str)
    parser.add_argument('-d', '--diretorio', default = 'None', help ='Introduzir o Master Folder do programa.', type = str)
    
    args = parser.parse_args()

    process(args.search_input_index, args.revise_search_cond, args.diretorio)


    
def process(search_input_index, revise_search_cond, diretorio):

    print('\n(function: search_extract)')

    SE_inputs = extract_inputs_from_csv(csv_filename = 'SE_inputs', diretorio = diretorio, mode = 'search_extract')
    search_input_index = int(search_input_index)
    print()
    print('> SE_input line index: ', search_input_index)
    
    #definindo as variáveis
    print('> filename: ', SE_inputs[search_input_index]['filename'])
    print('> parameter to extract: ', SE_inputs[search_input_index]['parameter_to_extract'])
    print('> scan_sent_by_sent: ', SE_inputs[search_input_index]['scan_sent_by_sent'])
    print('> index_list_name: ', SE_inputs[search_input_index]['index_list_name'])
    for key in SE_inputs[search_input_index]['search_inputs'].keys():
        print('>', key, ': ' , SE_inputs[search_input_index]['search_inputs'][key])

    #inserir a combinação de procura no arquivo /Settings/SE_inputs.csv
    if SE_inputs[search_input_index]['search_status'].lower() != 'finished':
        print('\n> Go to search engine...')
        se = search_engine(diretorio = diretorio)
        se.set_search_conditions(SE_inputs = SE_inputs[search_input_index], revise_search_conditions = revise_search_cond)
        se.search_with_combined_models()


    if SE_inputs[search_input_index]['export_status'].lower() != 'finished':
        
        print('\n> Go to extract engine...')
        DF = DataFrames(diretorio = diretorio)
        DF.set_settings_for_se(SE_inputs = SE_inputs[search_input_index])
        #use o regex para pegar parâmetros numéricos dentro das sentenças
        DF.get_data()



###############################################################################################
#executando a função
main()
