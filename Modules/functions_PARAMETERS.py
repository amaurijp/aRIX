#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import regex as re # type: ignore
import pandas as pd # type: ignore
import json
import os

'''Lembre de adicionar novas unidades à função "get_physical_units_to_replace" caso necessário'''

#------------------------------
def convert_ner_dic_to_nGrams_df(diretorio):
    
    print('\n( Function: convert_ner_dic_to_nGrams_df )')

    #criando a pasta /Outputs/ngrams/semantic/
    if not os.path.exists(diretorio + f'/Outputs/ngrams/semantic/'):
        os.makedirs(diretorio + f'/Outputs/ngrams/semantic/')

    #abrindo o dicionário com os ner rules
    with open(diretorio + '/Inputs/ner_rules.json', 'r') as ner_json_file:
        ner_dic = json.load(ner_json_file)
        ner_json_file.close()
    
    #varrendo as entidades ner
    for ner_entity in ner_dic.keys():

        #os termos associados a essa entidade serão o index da df
        ner_terms = ner_dic[ner_entity]['terms']
        nxgram_ner_terms_DF = pd.DataFrame(columns=['Sem_App_Counter'], index=ner_terms)
        nxgram_ner_terms_DF.index.name = 'index'

        #setando as condições inicias da DF
        for term in nxgram_ner_terms_DF.index:
            nxgram_ner_terms_DF.loc[term] = 0
        nxgram_ner_terms_DF.to_csv(diretorio + f'/Outputs/ngrams/semantic/nxgram_{ner_entity}.csv')
        print(f'Exportando a DF ~/Outputs/ngrams/semantic/nxgram_{ner_entity}.csv')



#------------------------------
def extract_textual_num_parameter_from_json_str(text: str):
    
    #textual vals
    textual_params = re.findall(r'(?<=[\{\,][\s\n\t]*\")[0-9A-Za-z\s\.\;\:\-\+\/]+(?=\")', text)
    text_with_textual_params = ''
    for val in textual_params:
        text_with_textual_params += val + ', '


    #nums_vals
    num_params = re.findall(r'(?<=\:[\s\n\t\[]*)[\sA-Za-z0-9\,\.\%\"]+(?=[\s\n\t\]]*)', text)
    text_with_num_params = ''
    for val in num_params:
        text_with_num_params += val + ', '

    return text_with_textual_params, text_with_num_params



#------------------------------
def get_physical_units():

    dic = {}
        
    dic['areaagro'] = ['ha', 'Ha', 'HA']
    dic['areametric'] = ['nm2', 'um2', 'mm2', 'cm2', 'dm2', 'm2', 'km2', 'Km2']
    dic['distance'] = ['nm', 'um', 'mm', 'cm', 'dm', 'm' , 'km' , 'Km']
    dic['energy'] = ['mJ', 'J', 'kJ', 'KJ', 'MJ', 'mcal', 'cal', 'kcal', 'Kcal', 'Mcal', 'mCal', 'Cal', 'kCal', 'KCal', 'MCal']
    dic['electricpotential'] = ['mV', 'V', 'kV', 'KV']
    dic['force'] = ['dyn', 'kdyn', 'Kdyn', 'Mdyn', 'dyne', 'kdyne', 'Kdyne', 'Mdyne', 'nN', 'uN', 'mN', 'cN', 'dN', 'N', 'kN']
    dic['log10'] = ['log', 'logs']
    dic['molarity'] = ['umol', 'µmol', 'mmol', 'cmol', 'mol']
    dic['potency'] = ['watts', 'W']
    dic['percentage'] = ['%']
    dic['weight_percentage'] = ['wtperc', 'wtvperc' , 'volperc']
    dic['pressure'] = ['Pa', 'kPa', 'KPa', 'MPa', 'Bar', 'bar', 'kBar', 'kbar', 'KBar', 'Kbar', 'MBar', 'Mbar']
    dic['temperature'] = ['C', '°C', 'K']
    dic['time'] = ['s', 'sec', 'secs', 'Sec', 'Secs', 'min', 'mins', 'Min', 'Mins', 'h', 'hour', 'hours', 'Hour', 'Hours']
    dic['viscosity'] = ['cP']
    dic['volume'] = ['mm3', 'dm3', 'cm3', 'cc', 'm3', 'ul', 'uL', 'ml', 'mL',  'l', 'L']
    dic['weight'] = ['ng', 'ug', 'mg', 'g', 'kg', 'Kg', 't','ton', 'tonne', 'TON']

    return dic



#------------------------------
def get_physical_units_cats():

    dic_cats = {}

    #coletando todas as unidades físicas
    PU_dic = get_physical_units()

    for key in PU_dic.keys():
        for unit in PU_dic[key]:
            dic_cats[unit] = key
    
    return dic_cats



#------------------------------
def get_conversion_other_physical_units_to_SI():

    dic = {}

    dic['energy'] = {}
    dic['energy']['units'] = ['cal', 'Cal']
    dic['energy']['factor'] = 4.184
    dic['energy']['operation'] = 'multiply'
    dic['energy']['SI_unit'] = 'J'

    dic['force'] = {}
    dic['force']['units'] = ['dyn', 'dyne']
    dic['force']['factor'] = 1e-5
    dic['force']['operation'] = 'multiply'
    dic['force']['SI_unit'] = 'N'

    dic['temperature'] = {}
    dic['temperature']['units'] = ['K']
    dic['temperature']['factor'] = -273
    dic['temperature']['operation'] = 'add'
    dic['temperature']['SI_unit'] = 'C'
    
    dic['time_sec'] = {}
    dic['time_sec']['units'] = ['s', 'sec', 'secs', 'Sec', 'Secs']
    dic['time_sec']['factor'] = 1/60
    dic['time_sec']['operation'] = 'multiply'
    dic['time_sec']['SI_unit'] = 'min'


    dic['time_hour'] = {}
    dic['time_hour']['units'] = ['h', 'hour', 'hours', 'Hour', 'Hours']
    dic['time_hour']['factor'] = 60
    dic['time_hour']['operation'] = 'multiply'
    dic['time_hour']['SI_unit'] = 'min'

    dic['volume'] = {}
    dic['volume']['units'] = ['m3']
    dic['volume']['factor'] = 1000
    dic['volume']['operation'] = 'multiply'
    dic['volume']['SI_unit'] = 'l'

    dic['weight'] = {}
    dic['weight']['units'] = ['t', 'ton', 'tonne', 'TON']
    dic['weight']['factor'] = 1e6
    dic['weight']['operation'] = 'multiply'
    dic['weight']['SI_unit'] = 'g'

    return dic



#------------------------------
def get_physical_units_SI_normalized():

    dic = {}
    
    #unidades normalizadas
    dic['areaagro'] = 'ha'
    dic['areametric'] = 'm2'
    dic['distance'] = 'm'
    dic['energy'] = 'J'
    dic['electricpotential'] = 'V'
    dic['force'] = 'N'
    dic['log10'] = 'log'
    dic['molarity'] = 'mol'
    dic['potency'] = 'W'
    dic['percentage'] = '%'
    dic['weight_percentage'] = 'wtperc'
    dic['weightvol_percentage'] = 'wtvperc'
    dic['volvol_percentage'] = 'volperc'
    dic['pressure'] = 'Pa'
    dic['temperature'] = 'C'
    dic['time'] = 'min'
    dic['viscosity'] = 'cP'
    dic['volume'] = 'l'
    dic['weight'] = 'g'
    
    return dic



#------------------------------
def get_physical_units_SI_unnormalized():

    dic = {}
    
    #outras formas de unidades padronizadas que podem aparecer além das apresentadas na função "get_physical_units_SI_normalized()"
    dic['areaagro'] = ['ha', 'Ha', 'HA']
    dic['areametric'] = ['m2']
    dic['distance'] = ['m']
    dic['energy'] = ['J', 'Joule', 'joule']
    dic['electricpotential'] = ['V', 'volts', 'Volts']
    dic['force'] = ['N']
    dic['log10'] = ['log', 'logs']
    dic['molarity'] = ['mol']
    dic['potency'] = ['W', 'Watts', 'watts']
    dic['percentage'] = ['%']
    dic['weight_percentage'] = ['wtperc']
    dic['weightvol_percentage'] = ['wtvperc']
    dic['volvol_percentage'] = ['volperc']
    dic['pressure'] = ['Pa']
    dic['temperature'] = ['C', '°C']
    dic['time'] = ['min', 'mins', 'Min', 'Mins']
    dic['viscosity'] = ['cP']
    dic['volume'] = ['l', 'L']
    dic['weight'] = ['g']
    
    return dic



#------------------------------
def get_factor_conversion():

    dic = {}
        
    dic['n'] = 1e-9
    dic['u'] = 1e-6
    dic['m'] = 1e-3
    dic['c'] = 1e-2
    dic['d'] = 1e-1
    dic['k'] = 1e3
    dic['K'] = 1e3
    dic['M'] = 1e6
    
    return dic



#------------------------------
def get_physical_units_combined(first_parameter = '', second_parameter = None, get_inverse = False):
        
    #motando as combinações de unidades físicas
    PU_units_combined = {}
    PU_units_combined['separated'] = []
    PU_units_combined['joint'] = []

    #coletando todas as unidades físicas
    PU_dic = get_physical_units()

    #varrendo as unidades físicas da classe primária
    for unit1 in PU_dic[first_parameter]:        
        
        #varrendo todas as classe de unidades físicas
        for key in PU_dic.keys():

            #caso se queira obter uma unidade combinada específica. EX: mg L
            if (second_parameter is not None) and (key != second_parameter):
                continue
            
            #caso se queria obter todas as combinações de unidades
            elif (second_parameter == key) or (second_parameter is None):
                #fazendo a combinação com todos os parâmetros
                if get_inverse is False:
                    #obtendo as unidades como str
                    PU_units_combined['joint'].extend( [ ( unit1 + ' ' +  unit2 ) for unit2 in PU_dic[key] ] )
                    #obtendo as unidades como tuplas
                    PU_units_combined['separated'].extend( [ ( unit1, unit2 ) for unit2 in PU_dic[key] ] )
                #encontrando as unidades inversas de todas os outros parâmetros e combinando o parâmetro introduzido
                elif get_inverse is True:
                    #obtendo as unidades como str
                    PU_units_combined['joint'].extend( [ ( unit1 + ' ' +  get_physical_unit_inverse(unit2) ) for unit2 in PU_dic[key] ] )
                    #obtendo as unidades como tuplas
                    PU_units_combined['separated'].extend( [ ( unit1, get_physical_unit_inverse(unit2) ) for unit2 in PU_dic[key] ] )
    
    return PU_units_combined


#------------------------------
def get_physical_all_units_combined():

    PU_all_units_combined = {}
    PU_all_units_combined['separated'] = []
    PU_all_units_combined['joint'] = []

    PU_units = get_physical_units()

    for cat in PU_units:
        
        PU_units_combined = get_physical_units_combined(first_parameter = cat)
        
        PU_all_units_combined['separated'].extend( PU_units_combined['separated'] )
        PU_all_units_combined['joint'].extend( PU_units_combined['joint'] )

    return PU_all_units_combined


#------------------------------
#essa função é usada na extração dos dados numéricos das sentenças
def get_physical_units_converted_to_SI(PUs):

    #dicionário para trabalhar com as unidades de entrada (raw)
    units = {}
    units['factor_list'] = []
    units['factor_operation'] = []
    units['raw_unit'] = []
    units['SI_unit'] = []
    
    #obtendo as categorias para as unidades físicas
    PU_units_cats = get_physical_units_cats()

    #obtendo as unidades físicas standard (SI)
    PU_units_SI_normalized = get_physical_units_SI_normalized()
    PU_units_SI_unnormalized = get_physical_units_SI_unnormalized()

    #obtendo as outras unidades
    other_units = get_conversion_other_physical_units_to_SI()

    #obtendo os fatores de conversão
    PU_factor_conversion = get_factor_conversion()


    for PU in PUs:

        found_pu = False
        units['raw_unit'].append(PU)

        #caso seja inverso
        if '-' in PU:
            PU = get_physical_unit_inverse(PU)
        
        cat = PU_units_cats[PU]
        units['SI_unit'].append( PU_units_SI_normalized[ cat ] )

        #tentando unidades SI sem fator
        for SI_unit in PU_units_SI_unnormalized[cat]:
            if SI_unit == PU:

                units['factor_list'].append( 1 )
                units['factor_operation'].append('multiply')
                found_pu = True
        
        
        if found_pu is False:
            try:
                #tentando outras unidades físicas (não SI) sem fator    
                for unit in other_units[cat]['units']:
                    if unit == PU:

                        units['factor_list'].append( other_units[cat]['factor'] )
                        units['factor_operation'].append( other_units[cat]['operation'] )
                        found_pu = True
            except KeyError:
                pass

        
        if found_pu is False:
            #tentando as unidades SI com fator
            for SI_unit in PU_units_SI_unnormalized[cat]:
                for factor_letter in PU_factor_conversion.keys():

                    if re.search(r'{factor_letter}{SI_unit}'.format(factor_letter = factor_letter, SI_unit = SI_unit), PU):
                        
                        factor = PU_factor_conversion[factor_letter]
                        
                        #caso tenha expoente ex: km2 -> 1000 ** 2
                        if PU[-1] in '23':
                            factor = factor ** int(PU[-1])

                        units['factor_list'].append( factor )
                        units['factor_operation'].append('multiply')
                        found_pu = True

        
        if found_pu is False:
            #tentando outras unidades físicas (não SI) com fator        
            for unit in other_units[cat]['units']:
                for factor_letter in PU_factor_conversion.keys():
            
                    if re.search(r'{factor_letter}{unit}'.format(factor_letter = factor_letter, unit = unit), PU):
                        
                        factor = PU_factor_conversion[factor_letter]
                        
                        #caso tenha expoente ex: km2 -> 1000 ** 2
                        if PU[-1] in '23':
                            factor = factor ** int(PU[-1])

                        units['factor_list'].append( factor * other_units[cat]['factor'] )
                        units['factor_operation'].append( other_units[cat]['operation'] )
                        found_pu = True

            
    #caso todas as unidades tenham sido identificadas
    if len(units['factor_list']) == len(units['raw_unit']) == len(units['SI_unit']):
        
        #fator de conversão
        conv_factor_to_multiply = 1
        conv_factor_to_add = 0
        #lista para guardar as unidades no SI
        SI_units_list = []
        
        #varrendo as unidades encontradas (as três listas do dic tem o mesmo length)
        for i in range(len(units['raw_unit'])):
            
            #caso a PU encontrada seja inversa
            if '-' in units['raw_unit'][i]:
                #invertendo a PU
                inverse_PU = get_physical_unit_inverse( units['SI_unit'][i] )
                if inverse_PU not in SI_units_list:            
                    #invertendo o factor de conversão
                    conv_factor_to_multiply = round( conv_factor_to_multiply * ( 1 / units['factor_list'][i] ), 9)
                    SI_units_list.append( inverse_PU )                
                    #print('Conversão de unidade: ', units['raw_unit'][i] , ' > ' , inverse_PU, '( fator: ' , ( 1 / units['factor_list'][i] ) , ' ; multiply )' )
            else:                
                #não precisa inverter a PU
                direct_PU = units['SI_unit'][i]
                if direct_PU not in SI_units_list:
                    #caso a conversão seja por somatório
                    if units['factor_operation'][i] == 'add':
                        #caso a conversão seja por multiplicação
                        conv_factor_to_add = conv_factor_to_add + units['factor_list'][i]
                        SI_units_list.append( direct_PU )                                            
                    elif units['factor_operation'][i] == 'multiply':
                        #caso a conversão seja por multiplicação
                        conv_factor_to_multiply = round( conv_factor_to_multiply * units['factor_list'][i], 9)
                        SI_units_list.append( direct_PU )
                    #print('Conversão de unidade: ', units['raw_unit'][i] , ' > ' , direct_PU, '( fator: ' , units['factor_list'][i], ' ; ', units['factor_operation'][i], ' )' )
        
        #montando as PUs no SI
        SI_units = ''
        for i in range(len(SI_units_list)):
            if i == len(SI_units_list) - 1:
                SI_units += SI_units_list[i]
            else:
                SI_units += SI_units_list[i] + ' '
                
        #print('Converted PUs: ', SI_units)
            
        #time.sleep(5)
        return conv_factor_to_multiply , conv_factor_to_add , SI_units 
    
    #caso nenhuma PU tenha sido identificada ou só parcialmente identificadas
    else:
        print('Erro de extração das PUs: unidade não identificada.')
        return None, None, None



#------------------------------
def get_physical_unit_exponent(unit):
    
    if unit[-1] not in ('23456789'):
        return unit , '1'
        
    else:
        return unit[ : -1 ] , unit[-1]


#------------------------------
def get_physical_unit_inverse(unit):
    
    if '-' in unit:

        if unit[-2:] == '-1':
            return unit[:-2]
        
        else:
            return re.sub(r'\-(?=[23456789])', '', unit)


    else:
        if unit[-1] not in '23':
            return unit + '-1'
            
        else:
            return unit[ : -1 ] + '-' + unit[-1]


#------------------------------
def find_textual_num_params(parameter: str):

    if parameter == 'biological_models_and_percentage_killing':        
        return 'species toxicological models', 'percentage'



#------------------------------
def list_numerical_parameter():

    #ATENÇÃO: ao adicionar uma nova unidade numérica, o primeiro nome deve estar presenta na lista da função "get_physical_units"    
    #ou se for uma unidade composta, deve ser descrita na função "regex_patt_from_parameter" na lista de base_parameters

    base_list = ['concentration_mass_mass',
                 'concentration_mass_vol',
                 'concentration_molar',
                 'electricpotential',
                 'elementcontent',
                 'distance',
                 'microbe_log_reduction',
                 'microbe_mic_inhibition',
                 'microbe_percentage_killing',
                 'nanomaterial_concentration',
                 'nanomaterial_size',
                 'nanomaterial_surface_area',
                 'nanomaterial_zeta_potential',
                 'percentage',
                 'surface_tension',
                 'temperature',
                 'toxicity_ec50',
                 'toxicity_lc50',
                 'toxicity_ld50',
                 'toxicity_ic50',
                 'time',
                 'viscosity_cp',
                 'volume',
                 'weight']

    mod_list1 = [item +'_inc' for item in base_list]
    mod_list2 = [item +'_dec' for item in base_list]
    
    return base_list + mod_list1 + mod_list2



#------------------------------
def list_textual_parameter(diretorio = None):

    from FUNCTIONS import get_filenames_from_folder

    parameter_list = []
    filenames = get_filenames_from_folder(diretorio + '/Outputs/ngrams/semantic', file_type = 'csv')
    for filename in filenames:
        parameter_list.append( re.search(r'(?<=n[12x]gram_).+', filename).group() )

    return parameter_list



#------------------------------
def list_textual_num_parameter():

    base_list = ['biological_models_and_percentage_killing']
    
    return base_list



#------------------------------
def process_input_parameter(parameter: str, diretorio: str = None):

    textual_parameter = None
    num_parameter = None

    if parameter in list_textual_parameter(diretorio):
        textual_parameter = parameter
    
    elif parameter in list_numerical_parameter():
        num_parameter = parameter

    elif parameter in list_textual_num_parameter():
        textual_parameter, num_parameter = find_textual_num_params(parameter)

    return textual_parameter, num_parameter



#------------------------------
def regex_patt_from_parameter(parameter):

    print('Encontrando padrão regex para o parâmetro: ', parameter)

    pattern_dic = {}
    #esse termo é para indicar se foi encontrado algum parâmetro
    found_parameter = False

    #checando se o paramêtro será "inc" ou "dec"
    parameter_suffix = None
    if parameter[ -4 : ] == '_inc':
        parameter = parameter[ : -4 ]
        parameter_suffix = '_inc'
    elif parameter[ -4 : ] == '_dec':
        parameter = parameter[ : -4 ]
        parameter_suffix = '_dec'

    #dicionário com as unidades físicas        
    PU_unit_dic = get_physical_units()
    all_PU_unit = [unit for cat in list(PU_unit_dic.values()) for unit in cat]
    
    #determinando as unidades físicas de interesse
    if parameter.lower() == 'concentration_mass_mass':
        
        pattern_dic['first_parameter'] = 'weight'
        pattern_dic['second_parameter'] = 'weight'
        
        #lista de unidades a não serem encontradas
        PU_units_to_find = get_physical_units_combined(first_parameter = pattern_dic['first_parameter'], second_parameter = pattern_dic['second_parameter'], get_inverse = True)
        
        #determinando o número mínimo e máximo de caracteres numéricos        
        n_min_len , n_max_len = 1 , 5
        ndec_min_len, ndec_max_len = 0 , 3
        found_parameter = True
        parameter_type = 'combined'
    
    elif parameter.lower() == 'concentration_mass_vol':
        
        pattern_dic['first_parameter'] = 'weight'
        pattern_dic['second_parameter'] = 'volume'
        
        #lista de unidades a não serem encontradas
        PU_units_to_find = get_physical_units_combined(first_parameter = pattern_dic['first_parameter'], second_parameter = pattern_dic['second_parameter'], get_inverse = True)
        
        #determinando o número mínimo e máximo de caracteres numéricos        
        n_min_len , n_max_len = 1 , 5
        ndec_min_len, ndec_max_len = 0 , 3
        found_parameter = True
        parameter_type = 'combined'
    
    elif parameter.lower() == 'concentration_molar':

        pattern_dic['first_parameter'] = 'molarity'
        pattern_dic['second_parameter'] = 'volume'
        
        #lista de unidades a não serem encontradas
        PU_units_to_find = get_physical_units_combined(first_parameter = pattern_dic['first_parameter'], second_parameter = pattern_dic['second_parameter'], get_inverse = True)

        #determinando o número mínimo e máximo de caracteres numéricos                
        n_min_len , n_max_len = 1 , 5
        ndec_min_len, ndec_max_len = 0 , 3
        found_parameter = True
        parameter_type = 'combined'

    elif parameter.lower() == 'distance':
        
        pattern_dic['first_parameter'] = 'distance'
        pattern_dic['second_parameter'] = None

        #lista de unidades a serem encontradas
        PU_units_to_find = PU_unit_dic[pattern_dic['first_parameter']]
        
        #determinando o número mínimo e máximo de caracteres numéricos
        n_min_len , n_max_len = 1 , 5
        ndec_min_len, ndec_max_len = 0 , 3
        found_parameter = True
        parameter_type = 'single'
        
    elif parameter.lower() == 'microbe_log_reduction':
        
        pattern_dic['first_parameter'] = 'log10'
        pattern_dic['second_parameter'] = None

        #lista de unidades a serem encontradas
        PU_units_to_find = PU_unit_dic[pattern_dic['first_parameter']]
        
        #determinando o número mínimo e máximo de caracteres numéricos        
        n_min_len , n_max_len = 1 , 5
        ndec_min_len, ndec_max_len = 0 , 3
        found_parameter = True
        parameter_type = 'single'

    elif parameter.lower() == 'microbe_mic_inhibition':
        
        pattern_dic['first_parameter'] = 'weight'
        pattern_dic['second_parameter'] = 'volume'
        
        #lista de unidades a não serem encontradas
        PU_units_to_find = get_physical_units_combined(first_parameter = pattern_dic['first_parameter'], second_parameter = pattern_dic['second_parameter'], get_inverse = True)
        
        #determinando o número mínimo e máximo de caracteres numéricos        
        n_min_len , n_max_len = 1 , 5
        ndec_min_len, ndec_max_len = 0 , 3
        found_parameter = True
        parameter_type = 'combined'

    elif parameter.lower() == 'microbe_percentage_killing':
        
        pattern_dic['first_parameter'] = 'percentage'
        pattern_dic['second_parameter'] = None

        #lista de unidades a serem encontradas        
        PU_units_to_find = PU_unit_dic[pattern_dic['first_parameter']]

        #determinando o número mínimo e máximo de caracteres numéricos        
        n_min_len , n_max_len = 1 , 5
        ndec_min_len, ndec_max_len = 0 , 3
        found_parameter = True
        parameter_type = 'single'

    elif parameter.lower() == 'nanomaterial_concentration':
        
        pattern_dic['first_parameter'] = 'weight'
        pattern_dic['second_parameter'] = 'volume'
        
        #lista de unidades a não serem encontradas
        PU_units_to_find = get_physical_units_combined(first_parameter = pattern_dic['first_parameter'], second_parameter = pattern_dic['second_parameter'], get_inverse = True)
        
        #determinando o número mínimo e máximo de caracteres numéricos        
        n_min_len , n_max_len = 1 , 5
        ndec_min_len, ndec_max_len = 0 , 3
        found_parameter = True
        parameter_type = 'combined'

    elif parameter.lower() == 'nanomaterial_size':
        
        pattern_dic['first_parameter'] = 'distance'
        pattern_dic['second_parameter'] = None

        #lista de unidades a serem encontradas
        PU_units_to_find = PU_unit_dic[pattern_dic['first_parameter']]
        
        #determinando o número mínimo e máximo de caracteres numéricos
        n_min_len , n_max_len = 1 , 5
        ndec_min_len, ndec_max_len = 0 , 3
        found_parameter = True
        parameter_type = 'single'

    elif parameter.lower() == 'nanomaterial_surface_area':

        pattern_dic['first_parameter'] = 'areametric'
        pattern_dic['second_parameter'] = 'weight'        
        
        #lista de unidades a não serem encontradas
        PU_units_to_find = get_physical_units_combined(first_parameter = pattern_dic['first_parameter'], second_parameter = pattern_dic['second_parameter'], get_inverse = True)

        #determinando o número mínimo e máximo de caracteres numéricos        
        n_min_len , n_max_len = 1 , 5
        ndec_min_len, ndec_max_len = 0 , 3
        found_parameter = True
        parameter_type = 'combined'

    elif parameter.lower() == 'nanomaterial_zeta_potential':
        
        pattern_dic['first_parameter'] = 'electricpotential'
        pattern_dic['second_parameter'] = None

        #lista de unidades a serem encontradas
        PU_units_to_find = PU_unit_dic[pattern_dic['first_parameter']]
        
        #determinando o número mínimo e máximo de caracteres numéricos        
        n_min_len , n_max_len = 0 , 3
        ndec_min_len, ndec_max_len = 0 , 3
        found_parameter = True
        parameter_type = 'single'

    elif parameter.lower() == 'percentage':
        
        pattern_dic['first_parameter'] = 'percentage'
        pattern_dic['second_parameter'] = None

        #lista de unidades a serem encontradas        
        PU_units_to_find = PU_unit_dic[pattern_dic['first_parameter']]

        #determinando o número mínimo e máximo de caracteres numéricos        
        n_min_len , n_max_len = 1 , 5
        ndec_min_len, ndec_max_len = 0 , 3
        found_parameter = True
        parameter_type = 'single'

    elif parameter.lower() == 'surface_tension':

        pattern_dic['first_parameter'] = 'force'
        pattern_dic['second_parameter'] = 'distance'
        
        #lista de unidades a não serem encontradas
        PU_units_to_find = get_physical_units_combined(first_parameter = pattern_dic['first_parameter'], second_parameter = pattern_dic['second_parameter'], get_inverse = True)

        #determinando o número mínimo e máximo de caracteres numéricos        
        n_min_len , n_max_len = 1 , 5
        ndec_min_len, ndec_max_len = 0 , 3
        found_parameter = True
        parameter_type = 'combined'

    elif parameter[ : ].lower() == 'temperature':
        
        pattern_dic['first_parameter'] = 'temperature'
        pattern_dic['second_parameter'] = None

        #lista de unidades a serem encontradas
        PU_units_to_find = PU_unit_dic[pattern_dic['first_parameter']]
        
        #determinando o número mínimo e máximo de caracteres numéricos
        n_min_len , n_max_len = 1 , 5
        ndec_min_len, ndec_max_len = 0 , 3
        found_parameter = True
        parameter_type = 'single'
    
    elif parameter.lower() == 'time':
        
        pattern_dic['first_parameter'] = 'time'
        pattern_dic['second_parameter'] = None

        #lista de unidades a serem encontradas
        PU_units_to_find = PU_unit_dic[pattern_dic['first_parameter']]
        
        #determinando o número mínimo e máximo de caracteres numéricos        
        n_min_len , n_max_len = 1 , 5 
        ndec_min_len, ndec_max_len = 0 , 3
        found_parameter = True
        parameter_type = 'single'

    elif parameter.lower() == 'toxicity_ec50':
        
        pattern_dic['first_parameter'] = 'weight'
        pattern_dic['second_parameter'] = 'volume'
        
        #lista de unidades a não serem encontradas
        PU_units_to_find = get_physical_units_combined(first_parameter = pattern_dic['first_parameter'], second_parameter = pattern_dic['second_parameter'], get_inverse = True)
        
        #determinando o número mínimo e máximo de caracteres numéricos        
        n_min_len , n_max_len = 1 , 5
        ndec_min_len, ndec_max_len = 0 , 3
        found_parameter = True
        parameter_type = 'combined'

    elif parameter.lower() == 'toxicity_lc50':
        
        pattern_dic['first_parameter'] = 'weight'
        pattern_dic['second_parameter'] = 'volume'
        
        #lista de unidades a não serem encontradas
        PU_units_to_find = get_physical_units_combined(first_parameter = pattern_dic['first_parameter'], second_parameter = pattern_dic['second_parameter'], get_inverse = True)
        
        #determinando o número mínimo e máximo de caracteres numéricos        
        n_min_len , n_max_len = 1 , 5
        ndec_min_len, ndec_max_len = 0 , 3
        found_parameter = True
        parameter_type = 'combined'

    elif parameter.lower() == 'toxicity_ld50':
        
        pattern_dic['first_parameter'] = 'weight'
        pattern_dic['second_parameter'] = 'weight'
        
        #lista de unidades a não serem encontradas
        PU_units_to_find = get_physical_units_combined(first_parameter = pattern_dic['first_parameter'], second_parameter = pattern_dic['second_parameter'], get_inverse = True)
        
        #determinando o número mínimo e máximo de caracteres numéricos        
        n_min_len , n_max_len = 1 , 5
        ndec_min_len, ndec_max_len = 0 , 3
        found_parameter = True
        parameter_type = 'combined'

    elif parameter.lower() == 'toxicity_ic50':
        
        pattern_dic['first_parameter'] = 'weight'
        pattern_dic['second_parameter'] = 'volume'
        
        #lista de unidades a não serem encontradas
        PU_units_to_find = get_physical_units_combined(first_parameter = pattern_dic['first_parameter'], second_parameter = pattern_dic['second_parameter'], get_inverse = True)
        
        #determinando o número mínimo e máximo de caracteres numéricos        
        n_min_len , n_max_len = 1 , 5
        ndec_min_len, ndec_max_len = 0 , 3
        found_parameter = True
        parameter_type = 'combined'

    elif parameter.lower() == 'viscosity_cp':
        
        pattern_dic['first_parameter'] = 'viscosity'
        pattern_dic['second_parameter'] = None

        #lista de unidades a serem encontradas
        PU_units_to_find = PU_unit_dic[pattern_dic['first_parameter']]
        
        #determinando o número mínimo e máximo de caracteres numéricos        
        n_min_len , n_max_len = 1 , 5 
        ndec_min_len, ndec_max_len = 0 , 3
        found_parameter = True
        parameter_type = 'single'

    elif parameter[ : ].lower() == 'volume':
        
        pattern_dic['first_parameter'] = 'volume'
        pattern_dic['second_parameter'] = None

        #lista de unidades a serem encontradas
        PU_units_to_find = PU_unit_dic[pattern_dic['first_parameter']]
        
        #determinando o número mínimo e máximo de caracteres numéricos
        n_min_len , n_max_len = 1 , 5
        ndec_min_len, ndec_max_len = 0 , 3
        found_parameter = True
        parameter_type = 'single'

    elif parameter[ : ].lower() == 'weight':
        
        pattern_dic['first_parameter'] = 'weight'
        pattern_dic['second_parameter'] = None

        #lista de unidades a serem encontradas
        PU_units_to_find = PU_unit_dic[pattern_dic['first_parameter']]
        
        #determinando o número mínimo e máximo de caracteres numéricos
        n_min_len , n_max_len = 1 , 5
        ndec_min_len, ndec_max_len = 0 , 4
        found_parameter = True
        parameter_type = 'single'

    else:
        print(f'Erro! O parâmetro introduzido ({parameter.lower()}) não foi encontrado')
        print('Ver abaixo os parâmetros definidos:')
        for parameter_set in list_numerical_parameter():
            print(parameter_set)
    
    
    #caso o parâmetro tenha sido encontrado
    if found_parameter is True:
        
        print('Padrão regex encontrado para o parâmetro: ', parameter)

        #definindo a lista de PUs
        if parameter_type == 'single':
            list_PUs_to_find = PU_units_to_find

        elif parameter_type == 'combined':
            list_PUs_to_find = PU_units_to_find['joint']

        #lista de unidades físicas a serem encontradas
        pattern_dic['PUs'] = list_PUs_to_find

        #montando os padrões que devem ser encontrados
        text_PUs_to_find = ''
        for i in range(len(list_PUs_to_find)):            
            if i == len(list_PUs_to_find) - 1:
                text_PUs_to_find += list_PUs_to_find[i]
            else:
                text_PUs_to_find += list_PUs_to_find[i] + '|'


        #montando o regex pattern  ################# lembrar do < e >

        inter_chars = '[\s\,\;\(\[\)\]]'
        spacer = '{inter_chars}*(and|or|to|from)?{inter_chars}+'.format(inter_chars = inter_chars)
        range_separator = '\-'
        uncert_separator = '\+\/\-|±'
        back_spacer = '[\s\,\;\(\[\)\]\.]'

        num_pattern = '\-?\s*[0-9]{n_min_len},{n_max_len}\.?[0-9]{ndec_min_len},{ndec_max_len}?'.format(n_min_len = '{' + str(n_min_len), 
                                                                                                        n_max_len = str(n_max_len) + '}', 
                                                                                                        ndec_min_len = '{' + str(ndec_min_len),
                                                                                                        ndec_max_len = str(ndec_max_len) + '}')

        initial_pattern = '(({spacer})+({num_pattern})\s*({PUs_to_find})?\s*({range_separator})?)?'.format(spacer = spacer, 
                                                                                                        num_pattern = num_pattern, 
                                                                                                        range_separator = range_separator,
                                                                                                        PUs_to_find = text_PUs_to_find)
        
        mid_pattern = '({spacer})+({num_pattern})\s*(({uncert_separator})\s*{num_pattern})?\s*({PUs_to_find}){back_spacer}'.format(spacer = spacer, 
                                                                                                                                 num_pattern = num_pattern, 
                                                                                                                                 uncert_separator = uncert_separator,
                                                                                                                                 PUs_to_find = text_PUs_to_find, 
                                                                                                                                 back_spacer = back_spacer)

        if parameter_suffix == '_inc':
            pre_pattern_to_complete = 'increas|rais|ris|rais|enhanc|grow'
            pre_pattern = '({pre_pattern_to_complete})(e|ing|ed)\s*({inter_chars}|by)?\s*'.format(pre_pattern_to_complete = pre_pattern_to_complete,
                                                                                                  inter_chars = inter_chars)
        elif parameter_suffix == '_dec':
            pre_pattern_to_complete = 'decreas|reduc|lower|diminish|reduc'
            pre_pattern = '({pre_pattern_to_complete})(e|ing|ed)\s*({inter_chars}|by)?\s*'.format(pre_pattern_to_complete = pre_pattern_to_complete,
                                                                                                  inter_chars = inter_chars)
        else:
            pre_pattern = ''


        #caso seja um parâmetro do tipo single, não se quer encontrar outros PUs em frente a ele
        if parameter_type == 'single':

            #montando os padrões que não devem ser encontrados
            list_PUs_not_to_find = all_PU_unit + [ get_physical_unit_inverse(unit) for unit in all_PU_unit ]
            
            PUs_not_to_find = ''
            for i in range(len(list_PUs_not_to_find)):
                if i == len(list_PUs_not_to_find) - 1:
                    PUs_not_to_find += list_PUs_not_to_find[i]
                else:
                    PUs_not_to_find += list_PUs_not_to_find[i] + '|'
                
            last_pattern = '(?!({PUs_not_to_find}){back_spacer})'.format(PUs_not_to_find = PUs_not_to_find, back_spacer = back_spacer)

        elif parameter_type == 'combined':
            last_pattern = '()'


        #gerando a pattern para encontrar
        pattern_dic['PU_to_find_regex'] = r'{pre_pattern}{initial_pattern}{initial_pattern}{initial_pattern}{initial_pattern}{initial_pattern}{mid_pattern}{last_pattern}'.format(pre_pattern = pre_pattern,
                                                                                                                                                                                  initial_pattern = initial_pattern, 
                                                                                                                                                                                  mid_pattern = mid_pattern,
                                                                                                                                                                                  last_pattern = last_pattern)

    #print('PU_to_find_regex: ', pattern_dic['PU_to_find_regex'])    
    
    return pattern_dic