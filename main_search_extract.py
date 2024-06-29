#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
local_dir = os.getcwd()
sys.path.append(local_dir + '/Modules')

from FUNCTIONS import extract_inputs_from_csv


def main():
    
    #Função principal   
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-r', '--revise_search_conditions', default = 'yes', help ='Decidir se será feita a checagem de todas as condições de busca (yes or no; sim ou nao).', type = str)
    parser.add_argument('-s', '--se_index', default = 0, help ='Introduzir o index da procura (ver em "SE_inputs.csv") caso a busca seja manual', type = int)
    
    args = parser.parse_args()

    process(args.revise_search_conditions, args.se_index)


def process(revise_search_conditions, se_index):
    
    SE_inputs = extract_inputs_from_csv(csv_filename = 'SE_inputs', diretorio = local_dir)
    
    #for key in SE_inputs:
    #    print('> ', key, ': ', SE_inputs)

    #caso todos as entradas serão rodadas automaticamente
    if revise_search_conditions.lower()[0] == 'n':
        for i in SE_inputs.keys():
            #caso a entrada introduzida já tenha sido feita
            if SE_inputs[i]['search_status'].lower() == 'finished' and SE_inputs[i]['export_status'].lower() == 'finished':
                print(f'\nA rotina de busca e extração para o "se_index" ({i}) já está completa.\n')
            
            elif SE_inputs[i]['search_status'].lower() != 'finished' or SE_inputs[i]['export_status'].lower() != 'finished':
                os.system(f'python {local_dir}/Modules/SEARCH_EXTRACT.py --search_input_index={i} --diretorio={local_dir}')
                print(f'python {local_dir}/Modules/SEARCH_EXTRACT.py --search_input_index={i} --revise_search_cond={revise_search_conditions.lower()} --diretorio={local_dir}')
    
    if revise_search_conditions.lower()[0] == 'y':
        #caso a entrada introduzida já tenha sido feita
        if SE_inputs[se_index]['search_status'].lower() == 'finished' and SE_inputs[se_index]['export_status'].lower() == 'finished':
            print(f'\nA rotina de busca e extração para o "se_index" introduzido ({se_index}) já está completa.')
            print('Para dúvidas, digite "python main_search_extract.py -h" para ajuda.\n')

        #caso uma entrada específica tenha sido introduzida
        elif SE_inputs[se_index]['search_status'].lower() != 'finished' or SE_inputs[se_index]['export_status'].lower() != 'finished':
            os.system(f'python {local_dir}/Modules/SEARCH_EXTRACT.py --search_input_index={se_index} --revise_search_cond={revise_search_conditions.lower()} --diretorio={local_dir}')



if __name__ == '__main__':
    main()