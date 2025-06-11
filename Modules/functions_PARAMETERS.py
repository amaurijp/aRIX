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
def get_llm_personality_to_check(input_parameter):

    text = None
    regex_to_match_llm_output = None
    depend_params_in_consol_DF = None #checa se os parâmetros inseridos estão no dataframe consolidado
    check_params_already_filtered = None #checa se os parâmetros já foram checados no json (~/Outputs/log) com os resultados da checagem
    params_to_be_filtered_along = None #caso checado o atual parâmetro, esses outros também serão checados

    
    if input_parameter.lower() == 'biofilm_killing_perc':

        def biofilm_killing_perc():
            
            t = "You are a data miner checker specializing in nanotechnology/nanotoxicology. " \
                "Your primary responsibility is to verify whether the previously extracted parameter, 'percentage of biofilm killing,' including numerical values (and any corresponding units, such as '%'), accurately matches the input text. " \
                "You must confirm that the extracted percentage of biofilm killing indeed applies to a 'biological species' or (if applicable) a 'nanomaterial core composition' that was also previously extracted, " \
                "ensuring the input text explicitly associates that percentage with the particular biological species (for example, 'Escherichia coli', 'Staphylococcus aureus') or " \
                "nanomaterial core composition (for example, 'silver', 'carbon', 'zinc oxide', 'silica', 'titania', etc.). " \
                "Keep in mind possible biological species abbreviations (for example, 'E. coli', 'S. aureus'). " \
                "Consider only the nanomaterial's core composition (for example, the core composition of 'functionalized carbon nanotubes' is 'C'; the core composition of 'PEG-modified silver NPs' is 'Ag'). " \
                "You must also verify that the stated percentage of biofilm killing (for example, '10 %', '25 %', or approximate descriptors such as 'around 20 %') indeed corresponds to the biological species or nanomaterial core composition mentioned in the input text, " \
                "and that any units of measurement ('%', etc.) match what appears in the input text. " \
                "After reviewing, you must output the numerical value defining the percentage of biofilm killing (or range) (for example, '10 %', or '25-50 %'). " \
                "If more than one value was extracted (for example, '10 %, 20 %, and 30 %'), " \
                "you must output only the numerical value that correlates with the previously extracted biological species or nanomaterial core composition (for example, '10 %', '20 %', or '30 %'). " \
                "Key points to consider: " \
                "You do not need to provide any additional commentary beyond outputting the percentage of biofilm killing (for example, '20 %'), regardless of whether the extraction is correct or if errors exist."
            
            return t
        
        text = biofilm_killing_perc()

        depend_params_in_consol_DF = [ 'nanomaterial_composition', 'biological_species' ]
        check_params_already_filtered = ['nanomaterial_composition', 'biological_species' ]
        params_to_be_filtered_along = []
        regex_to_match_llm_output = r'\s*[\-\+/]*\s*[0-9]+'


    elif input_parameter.lower() == 'biological_species':
        
        def biological_species():

            t = "You are a data miner checker specializing in nanotoxicology. " \
                "Your primary responsibility is to verify whether the extracted 'biological species' and the nanomaterial description (including 'core composition' " \
                "and, if applicable, 'morphology') accurately match the text. " \
                "Specifically, you must verify that the biological species (examples: 'Escherichia coli', 'Staphylococcus aureus', 'Daphnia magna') or " \
                "its abbreviation (examples: 'E. coli', 'S. aureus') was indeed tested or exposed to a nanomaterial in the text. " \
                "Besides, you must confirm that the nanomaterial's core composition (examples: 'silver' = 'Ag', 'carbon' = 'C', 'zinc oxide' = 'ZnO', 'titanium oxide' = 'TiO2'), " \
                "along with any morphology (examples: 'nanoparticle' = 'NPs', 'nanotube' = 'NTs', 'nanorod'), is correctly identified from the text. " \
                "Finally, you must confirm that the species-nanomaterial relationship is correct, ensuring that the text actually indicates that species was exposed to or tested with that specific nanomaterial. " \
                "Key points to consider: " \
                "- Ignore additional functionalizations or coatings (examples: 'PEG-modified', 'functionalized') when identifying the core composition; focus on the elemental or compound core (examples: 'Ag', 'ZnO', 'TiO2'). " \
                "- In case the nanomaterial morphology is not provided, look for the relation between the species and the nanomaterial composition only. " \
                "- Pay attention to synonyms (examples: 'zebrafish' = 'Danio rerio'). " \
                "- If multiple species or multiple nanomaterials are mentioned, verify that each pair (species + nanomaterial) is accurately extracted and actually tested or exposed in the text. " \
                "- Approve the extracted information if both the species and nanomaterial (composition + morphology) align with the text, and the text indicates a testing or exposure relationship. " \
                "- You do not need to provide any extra commentary beyond confirming whether the extraction is accurate or not and explaining any errors if they exist."
            
            return t
        
        text = biological_species()

        depend_params_in_consol_DF = [ 'nanomaterial_composition' ]
        check_params_already_filtered = ['nanomaterial_composition']
        params_to_be_filtered_along = []
        regex_to_match_llm_output = r'(?=\')[Yy]es(?=\')'


    elif input_parameter.lower() == 'microbe_killing_log':

        def microbe_killing_log():

            t = "You are a data miner checker specializing in nanotechnology/nanotoxicology. " \
                "Your primary responsibility is to verify whether the previously extracted parameter, 'reduction of the microbial population in log units' (including numerical values and any corresponding 'log' notation), accurately matches the input text. " \
                "You must confirm that the extracted log reduction indeed applies to a 'biological species' or (if applicable) a 'nanomaterial core composition' that was also previously extracted, " \
                "ensuring the input text explicitly associates that log reduction with the particular biological species (for example, 'Escherichia coli', 'Staphylococcus aureus') or " \
                "nanomaterial core composition (for example, 'silver', 'carbon', 'zinc oxide', 'silica', 'titania', etc.). " \
                "Keep in mind possible biological species abbreviations (for example, 'E. coli', 'S. aureus'). " \
                "Consider only the nanomaterial's core composition (for example, the core composition of 'functionalized carbon nanotubes' is 'C'; the core composition of 'PEG-modified silver NPs' is 'Ag'). " \
                "You must also verify that the stated log reduction (for example, '2 log', '4.5 log', or approximate descriptors such as 'around 3 log') indeed corresponds to the biological species or nanomaterial core composition mentioned in the input text, " \
                "and that the log notation ('log units') matches what appears in the input text. " \
                "After reviewing, you must output the numerical value defining the reduction in microbial population (or range) along with its associated 'log' notation (for example, '2 log' or '4-5 log'). " \
                "If more than one value was extracted (for example, '2 log, 3 log, and 4 log'), " \
                "you must output only the numerical value that correlates with the previously extracted biological species or nanomaterial core composition (for example, '2 log', '3 log', or '4 log'). " \
                "Key points to consider: " \
                "- You do not need to provide any additional commentary beyond outputting the log reduction and its notation (for example, '2.3 log'), regardless of whether the extraction is correct or if errors exist."
        
            return t

        text = microbe_killing_log

        depend_params_in_consol_DF = [ 'nanomaterial_composition', 'biological_species' ]
        check_params_already_filtered = ['nanomaterial_composition', 'biological_species' ]
        params_to_be_filtered_along = []
        regex_to_match_llm_output = r'\s*[\-\+/]*\s*[0-9]+'


    elif input_parameter.lower() == 'microbe_killing_mbc':

        def microbe_killing_mbc():
            t = "You are a data miner checker specializing in nanotechnology/nanotoxicology. " \
                "Your primary responsibility is to verify whether the previously extracted parameter, 'minimum bactericidal concentration' (MBC), including numerical values and any corresponding units, accurately matches the input text. " \
                "You must confirm that the extracted MBC indeed applies to a 'biological species' or (if applicable) a 'nanomaterial core composition' that was also previously extracted, " \
                "ensuring the input text explicitly associates that MBC with the particular biological species (for example, 'Escherichia coli', 'Staphylococcus aureus') or " \
                "nanomaterial core composition (for example, 'silver', 'carbon', 'zinc oxide', 'silica', 'titania', etc.). " \
                "Keep in mind possible biological species abbreviations (for example, 'E. coli', 'S. aureus'). " \
                "Consider only the nanomaterial's core composition (for example, the core composition of 'functionalized carbon nanotubes' is 'C'; the core composition of 'PEG-modified silver NPs' is 'Ag'). " \
                "You must also verify that the stated MBC (for example, '10 µg/mL', '25 µg mL-1', or approximate descriptors such as 'around 20 µg mL-1') indeed corresponds to the biological species or nanomaterial core composition mentioned in the input text, " \
                "and that any units of measurement ('µg mL-1', 'g L-1', 'g/mL', 'g/L') match what appears in the input text. " \
                "After reviewing, you must output the numerical value defining the minimum bactericidal concentration (or range) along with its associated units (for example, '10 µg mL-1' or '25-50 µg mL-1'). " \
                "If more than one value was extracted (for example, '10 µg mL-1, 20 µg mL-1, and 30 µg mL-1'), " \
                "you must output only the numerical value that correlates with the previously extracted biological species or nanomaterial core composition (for example, '10 µg mL-1', '20 µg mL-1', or '30 µg mL-1'). " \
                "Key points to consider: " \
                "- You do not need to provide any additional commentary beyond outputting the MBC and its units (for example, '20 µg mL-1'), regardless of whether the extraction is correct or if errors exist."
        
            return t
        
        text = microbe_killing_mbc()
        
        depend_params_in_consol_DF = [ 'nanomaterial_composition', 'biological_species' ]
        check_params_already_filtered = ['nanomaterial_composition', 'biological_species' ]
        params_to_be_filtered_along = []
        regex_to_match_llm_output = r'\s*[\-\+/]*\s*[0-9]+'


    elif input_parameter.lower() == 'microbe_killing_mic':

        def microbe_killing_mic():
            t = "You are a data miner checker specializing in nanotechnology/nanotoxicology. " \
                "Your primary responsibility is to verify whether the previously extracted parameter, 'minimum inhibitory concentration' (MIC), including numerical values and any corresponding units, accurately matches the input text. " \
                "You must confirm that the extracted MIC indeed applies to a 'biological species' or (if applicable) a 'nanomaterial core composition' that was also previously extracted, " \
                "ensuring the input text explicitly associates that MIC with the particular biological species (for example, 'Escherichia coli', 'Staphylococcus aureus') or " \
                "nanomaterial core composition (for example, 'silver', 'carbon', 'zinc oxide', 'silica', 'titania', etc.). " \
                "Keep in mind possible biological species abbreviations (for example, 'E. coli', 'S. aureus'). " \
                "Consider only the nanomaterial's core composition (for example, the core composition of 'functionalized carbon nanotubes' is 'C'; the core composition of 'PEG-modified silver NPs' is 'Ag'). " \
                "You must also verify that the stated MIC (for example, '10 µg/mL', '25 µg mL-1', or approximate descriptors such as 'around 20 µg mL-1') indeed corresponds to the biological species or nanomaterial core composition mentioned in the input text, " \
                "and that any units of measurement ('µg mL-1', 'g L-1', 'g/mL', 'g/L') match what appears in the input text. " \
                "After reviewing, you must output the numerical value defining the minimum inhibitory concentration (or range) along with its associated units (for example, '10 µg mL-1' or '25-50 µg mL-1'). " \
                "If more than one value was extracted (for example, '10 µg mL-1, 20 µg mL-1, and 30 µg mL-1'), " \
                "you must output only the numerical value that correlates with the previously extracted biological species or nanomaterial core composition (for example, '10 µg mL-1', '20 µg mL-1', or '30 µg mL-1'). " \
                "Key points to consider: " \
                "- You do not need to provide any additional commentary beyond outputting the MIC and its units (for example, '20 µg mL-1'), regardless of whether the extraction is correct or if errors exist."
            
            return t
        
        text = microbe_killing_mic()
        
        depend_params_in_consol_DF = [ 'nanomaterial_composition', 'biological_species' ]
        check_params_already_filtered = ['nanomaterial_composition', 'biological_species' ]
        params_to_be_filtered_along = []
        regex_to_match_llm_output = r'\s*[\-\+/]*\s*[0-9]+'


    elif input_parameter.lower() == 'nanomaterial_morphology':
        
        def nanomaterial_morphology():

            t = "You are a data miner checker specializing in nanotechnology. " \
                "Your primary responsibility is to determine whether nanomaterial descriptions extracted from text are accurate. " \
                "Specifically, you must verify that the 'core composition' (examples: 'silver', 'carbon', 'zinc oxide', 'silica', 'titania', etc.) " \
                "and the 'morphology' (examples: 'nanoparticle' or 'NPs', 'nanotube' or 'NTs', 'nanosheet', 'nanorod', etc.) were correctly extracted from the input text. " \
                "The core composition and morphology together are related to a single nanomaterial (examples: 'silver nanoparticle' or 'AgNPs', " \
                "'carbon nanotube' or 'CNTs', 'carbon dot' or 'Cdots', 'titania nanotube' or 'TiO2 nanotubes', etc.). " \
                "Consider only the core composition of the nanomaterial (examples: the core composition of 'functionalized carbon nanotubes' is 'C' " \
                "and the morphology is 'nanotube'; the core composition of 'PEG-modified silver NPs' is 'Ag' and the morphology is 'nanoparticle'). " \
                "Consider that core chemical composition can be defined with abbreviations (examples: 'silver' = 'Ag', 'carbon' = 'C', 'zinc oxide' = 'ZnO', 'titanium oxide' = 'TiO2'). " \
                "Also, consider that the morphology of a nanomaterial can be defined with abbreviations (examples: 'nanoparticles' = 'NPs', 'nanotubes' = 'NTs'). " \
                "Do not distinguish singular and plural extracted values (example: 'nanotubes' is equivalent to 'nanotube')."
            
            return t
        
        text = nanomaterial_morphology()
    
        depend_params_in_consol_DF = [ 'nanomaterial_composition' ]
        check_params_already_filtered = []
        params_to_be_filtered_along = ['nanomaterial_composition']
        regex_to_match_llm_output = r'(?=\')[Yy]es(?=\')'


    elif input_parameter.lower() == 'nanomaterial_size':

        def nanomaterial_size():
            
            t = "You are a data miner checker specializing in nanotechnology. " \
                "Your primary responsibility is to verify whether the previously extracted 'nanomaterial size' (including numerical values and units of measurement) accurately matches the input text. " \
                "*Rules* " \
                "- You must confirm that the extracted size information indeed applies to a 'nanomaterial core composition' or (if applicable) a 'nanomaterial morphology' that was also previously extracted, " \
                "ensuring the input text explicitly associates that size with the particular nanomaterial core composition (for example, 'silver', 'carbon', 'zinc oxide', 'silica', 'titania', etc.) or " \
                "the particular nanomaterial morphology (examples: 'nanoparticle' = 'NPs', 'nanotube' = 'NTs', 'nanorod'). " \
                "- Consider only the nanomaterial's core composition (for example, the core composition of 'functionalized carbon nanotubes' is 'C'; the core composition of 'PEG-modified silver NPs' is 'Ag'). " \
                "- You must also check that the stated size (for example, '10 nm', '50-100 nm', or approximate descriptors such as 'around 20 nm') indeed corresponds to the nanomaterial core composition or morphology mentioned in the input text, " \
                "and that any units of measurement (nm, µm, etc.) match what appears in the input text. " \
                "- After checking, you must output the numerical value defining the nanomaterial size (or size range) along with its associated units of measurement (for example, '10 nm' or '50-100 nm'). " \
                "- If more than one value was extracted (for example, '10 nm, 20 nm, and 30 nm' or '10, 20, and 30 nm'), " \
                "you must output only the numerical value and units of measurement that correlate with the previously extracted nanomaterial core composition or morphology (for example, just '10 nm' or '20 nm' or '30 nm'). " \
                "*Key points to consider* " \
                "- You do not need to provide any additional commentary beyond outputting the size and the unit of measurement (for example, '1.3 µm'), regardless of whether the extraction is correct or if errors exist. " \
                "- Do not distinguish singular and plural extracted values (example: 'nanotubes' is equivalent to 'nanotube'). " \
                "*EXAMPLES* " \
                "EXAMPLE 1: " \
                "text: 'Transmission-electron microscopy revealed silver nanoparticles (Ag NPs) with diameters of 25 ± 5 nm dispersed in the polymer matrix.' " \
                "candidate_size: '25 ± 5 nm' " \
                "candidate_core: 'silver' " \
                "candidate_morphology: 'nanoparticle' " \
                "your output:  '25 ± 5 nm' " \
                "EXAMPLE 2: " \
                "text: 'The nanorods (ZnO) exhibited lengths of 200 nm and widths near 40 nm, whereas carbon nanotubes (CNTs) averaged 15 nm in diameter.' " \
                "candidate_size: '40 nm' " \
                "candidate_core: 'ZnO' " \
                "candidate_morphology: 'nanorod' " \
                "your output: 'None' # 40 nm belongs to CNT width, not ZnO nanorods " \
                "-- end examples --"
            
            return t

        text = nanomaterial_size()

        depend_params_in_consol_DF = [ 'nanomaterial_composition', 'morphology' ]
        check_params_already_filtered = ['nanomaterial_composition', 'morphology' ]
        params_to_be_filtered_along = []
        regex_to_match_llm_output = r'\s*[\-\+/]*\s*[0-9]+'


    elif input_parameter.lower() == 'nanomaterial_surface_area':

        def nanomaterial_surface_area():
        
            t = "You are a data miner checker specializing in nanotechnology. " \
                "Your primary responsibility is to verify whether the previously extracted 'nanomaterial surface area' (including numerical values and units of measurement) accurately matches the input text. " \
                "You must confirm that the extracted surface area information indeed applies to a 'nanomaterial core composition' or (if applicable) a 'nanomaterial morphology' that was also previously extracted, " \
                "ensuring the input text explicitly associates that surface area with the particular nanomaterial core composition (for example, 'silver', 'carbon', 'zinc oxide', 'silica', 'titania', etc.) or " \
                "the particular nanomaterial morphology (examples: 'nanoparticle' = 'NPs', 'nanotube' = 'NTs', 'nanorod'). " \
                "Consider only the nanomaterial's core composition (for example, the core composition of 'functionalized carbon nanotubes' is 'C'; the core composition of 'PEG-modified silver NPs' is 'Ag'). " \
                "You must also check that the stated surface area (for example, '10 m2 g-1', '50-100 m2 g-1', or approximate descriptors such as 'around 20 m2 g-1') indeed corresponds to the nanomaterial core composition or morphology mentioned in the input text, " \
                "and that any units of measurement (m2/mg, m2/g, m2 mg-1, m2 g-1, etc.) match what appears in the input text. " \
                "After checking, you must output the numerical value defining the nanomaterial surface area (or range) along with its associated units of measurement (for example, '10 m2 g-1' or '50-100 m2 g-1'). " \
                "If more than one value was extracted (for example, '10 m2 g-1, 20 m2 g-1, and 30 m2 g-1'), " \
                "you must output only the numerical value and units of measurement that correlate with the previously extracted nanomaterial core composition or morphology (for example, just '10 m2 g-1' or '20 m2 g-1' or '30 m2 g-1'). " \
                "Key points to consider: " \
                "- You do not need to provide any additional commentary beyond outputting the surface area and its unit of measurement (for example, '1.3 m2 g-1'), regardless of whether the extraction is correct or if errors exist. " \
                "- Do not distinguish singular and plural extracted values (example: 'nanotubes' is equivalent to 'nanotube'). "

            return t
        
        text = nanomaterial_surface_area()

        depend_params_in_consol_DF = [ 'nanomaterial_composition', 'morphology' ]
        check_params_already_filtered = ['nanomaterial_composition', 'morphology' ]
        params_to_be_filtered_along = []
        regex_to_match_llm_output = r'\s*[\-\+/]*\s*[0-9]+'


    elif input_parameter.lower() == 'nanomaterial_zeta_potential':

        def nanomaterial_zeta_potential():

            t = "You are a data miner checker specializing in nanotechnology. " \
                "Your primary responsibility is to verify whether the previously extracted 'nanomaterial zeta potential' (including numerical values and units of measurement) accurately matches the input text. " \
                "You must confirm that the extracted zeta potential indeed applies to a 'nanomaterial core composition' or (if applicable) a 'nanomaterial morphology' that was also previously extracted, " \
                "ensuring the input text explicitly associates that zeta potential with the particular nanomaterial core composition (for example, 'silver', 'carbon', 'zinc oxide', 'silica', 'titania', etc.) or " \
                "the particular nanomaterial morphology (examples: 'nanoparticle' = 'NPs', 'nanotube' = 'NTs', 'nanorod'). " \
                "Consider only the nanomaterial's core composition (for example, the core composition of 'functionalized carbon nanotubes' is 'C'; the core composition of 'PEG-modified silver NPs' is 'Ag'). " \
                "You must also check that the stated zeta potential (for example, '-20 mV', '10-30 mV', or approximate descriptors such as 'around -15 mV') indeed corresponds to the nanomaterial core composition or morphology mentioned in the input text, " \
                "and that any units of measurement (µV, mV, etc.) match what appears in the input text. " \
                "After checking, you must output the numerical value defining the nanomaterial zeta potential (or range) along with its associated units of measurement (for example, '-20 mV' or '10-30 mV'). " \
                "If more than one value was extracted (for example, '10 mV, 20 mV, and 30 mV'), " \
                "you must output only the numerical value and units of measurement that correlate with the previously extracted nanomaterial core composition or morphology (for example, just '10 mV' or '20 mV' or '30 mV'). " \
                "Key points to consider: " \
                "- You do not need to provide any additional commentary beyond outputting the zeta potential and its unit of measurement (for example, '-1.3 mV'), regardless of whether the extraction is correct or if errors exist. " \
                "- Do not distinguish singular and plural extracted values (example: 'nanotubes' is equivalent to 'nanotube')."

            return t

        text = nanomaterial_zeta_potential()

        depend_params_in_consol_DF = [ 'nanomaterial_composition', 'morphology' ]
        check_params_already_filtered = ['nanomaterial_composition', 'morphology' ]
        params_to_be_filtered_along = []
        regex_to_match_llm_output = r'\s*[\-\+/]*\s*[0-9]+'


    elif input_parameter.lower() == 'toxicity_lc50':

        def toxicity_lc50():
            t = "You are a data miner checker specializing in nanotechnology/nanotoxicology. " \
                "Your primary responsibility is to verify whether the previously extracted parameter, 'Lethal Concentration 50%' (LC50), including numerical values and any corresponding units, accurately matches the input text. " \
                "You must confirm that the extracted LC50 indeed applies to a 'biological species' or (if applicable) a 'nanomaterial core composition' that was also previously extracted, " \
                "ensuring the input text explicitly associates that LC50 with the particular biological species (for example, 'Daphnia magna', 'Danio rerio') or " \
                "nanomaterial core composition (for example, 'silver', 'carbon', 'zinc oxide', 'silica', 'titania', etc.). " \
                "Keep in mind possible biological species abbreviations (for example, 'D. magna', 'D. rerio'). " \
                "Consider only the nanomaterial's core composition (for example, the core composition of 'functionalized carbon nanotubes' is 'C'; the core composition of 'PEG-modified silver NPs' is 'Ag'). " \
                "You must also verify that the stated LC50 (for example, '10 µg/mL', '25 µg mL-1', or approximate descriptors such as 'around 20 µg mL-1') indeed corresponds to the biological species or nanomaterial core composition mentioned in the input text, " \
                "and that any units of measurement (for example, 'µg mL-1', 'g L-1', 'g/mL', 'g/L') match what appears in the input text. " \
                "After reviewing, you must output the numerical value defining the LC50 (or range) along with its associated units (for example, '10 µg mL-1' or '25-50 µg mL-1'). " \
                "If more than one value was extracted (for example, '10 µg mL-1, 20 µg mL-1, and 30 µg mL-1'), " \
                "you must output only the numerical value that correlates with the previously extracted biological species or nanomaterial core composition (for example, '10 µg mL-1', '20 µg mL-1', or '30 µg mL-1'). " \
                "Key points to consider: " \
                "You do not need to provide any additional commentary beyond outputting the LC50 and its units (for example, '20 µg mL-1'), regardless of whether the extraction is correct or if errors exist."
        
            return t

        text = toxicity_lc50()

        depend_params_in_consol_DF = [ 'nanomaterial_composition', 'biological_species' ]
        check_params_already_filtered = ['nanomaterial_composition', 'biological_species' ]
        params_to_be_filtered_along = []
        regex_to_match_llm_output = r'\s*[\-\+/]*\s*[0-9]+'


    elif input_parameter.lower() == 'toxicity_yes_no':

        def toxicity_yes_no():
            t = "You are a nanotoxicology evidence validator. " \
                "TASK: " \
                "Given a text input and a candidate conclusion ('yes' or 'no' to the question 'Is your previous answer about the toxicity assessment correct?'), decide whether that conclusion is supported by the text input. " \
                "DEFINITIONS: " \
                "- 'Toxic' = the passage reports at least one statistically or descriptively significant adverse biological outcome attributed to the nanomaterial in a living organism, cell culture, or in vitro substitute. " \
                "Valid endpoints include (but are not limited to): mortality, abnormal hatching, developmental delay, growth inhibition, oxidative-stress markers, genotoxicity, enzyme dysregulation, bioaccumulation, behavioural change, " \
                "reproduction impairment, morphological pathology. " \
                "- 'Not toxic' = the passage explicitly states the nanomaterial produced no significant adverse effect in the study (e.g. 'no significant toxicity', 'did not affect growth', 'no mortality observed'). " \
                "DECISION RULES: " \
                "1. If multiple endpoints are reported, a single valid adverse outcome is enough to label the study 'toxic'. " \
                "2. Ignore dose magnitude, exposure time, or mitigation comments ('may recover after five days')—only the presence or absence of adverse outcome matters for this classification. " \
                "4. Output format: " \
                "• output 'yes' or 'no' to the question 'Is your previous answer about the toxicity assessment correct?' if you conclude that the previous asnwer is correct based on the current inputs. " \
                "• output 'no' if you conclude that the previous asnwer is wrong based on the current inputs or if you conclude that there is insufficient or conflicting evidence. " \
                "Return nothing else — no explanations, no JSON keys. " \
                "EXAMPLE 1 - confirmation of the previous answer " \
                "> input text: (beginning of the input text) From the sentence 'Silver nanoparticles caused >30 % mortality and severe oxidative stress in Danio rerio embryos', " \
                "it was evaluated if there is (or not) a toxicity effect for the endpoint 'mortality'. " \
                "Consider the question: " \
                "Is there any evidence in the input sentence of a possible toxic effect for the endpoint 'mortality' considering the species 'Danio rerio'? " \
                "Your previous answer for this question was: 'yes'. " \
                "Is your previous answer about the toxicity assessment correct? (end of the input text) " \
                "> candidate output: 'yes' - a correct answer was previously given, there is indeed a 'toxic' effect described " \
                "> your output: 'yes' " \
                "EXAMPLE 2 - confirmation of the previous answer " \
                "> input text: (beginning of the input text) From the sentence 'Administration of a suspension of carbon nanotubes did not affect the growth of the C. elegans population studied', " \
                "it was evaluated if there is (or not) a toxicity effect for the endpoint 'developmental alterations'. " \
                "Consider the question: " \
                "Is there any evidence in the input sentence of a possible toxic effect for the endpoint 'developmental alterations' considering the species 'Caenorhabditis elegans'? " \
                "Your previous answer for this question was: 'no'. " \
                "Is your previous answer about the toxicity assessment correct? (end of the input text) " \
                "> candidate output: 'yes' - a correct answer was previously given - there is 'no toxic' effect described " \
                "> your output: 'yes' " \
                "EXAMPLE 3 - identifying a wrong answer previously given " \
                "> input text: (beginning of the input text) From the sentence 'Exposure to the nanomaterial significantly increased ROS levels in D. magna, indicating oxidative stress.', " \
                "it was evaluated if there is (or not) a toxicity effect for the endpoint 'oxidative stress'. " \
                "Consider the question: " \
                "Is there any evidence in the input sentence of a possible toxic effect for the endpoint 'oxidative stress' considering the species 'Daphnia magna'? " \
                "Your previous answer for this question was: 'no'. " \
                "Is your previous answer about the toxicity assessment correct? (end of the input text) " \
                "> candidate output: 'no' - there is a wrong assessment about the toxicity. The input text describes a toxic behavior of an agent but your previous answer was " \
                "that there is no toxic behavior. " \
                "> your output: 'No' " \
                "-- end examples --"
        
            return t

        text = toxicity_yes_no()

        depend_params_in_consol_DF = [ 'nanomaterial_composition', 'biological_species', 'toxicity_endpoints' ]
        check_params_already_filtered = [ 'nanomaterial_composition', 'biological_species' ]
        params_to_be_filtered_along = [ 'toxicity_endpoints' ]
        regex_to_match_llm_output = r'(?<=\')[Yy]es(?=\')'


    return depend_params_in_consol_DF, check_params_already_filtered, params_to_be_filtered_along, text, regex_to_match_llm_output



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
    dic['weight'] = ['ng', 'ug', 'mg', 'g', 'kg', 'Kg', 't', 'ton', 'tonne', 'TON']

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
def list_numerical_parameter():

    #ATENÇÃO: ao adicionar uma nova unidade numérica, o primeiro nome deve estar presenta na lista da função "get_physical_units"    
    #ou se for uma unidade composta, deve ser descrita na função "regex_patt_from_parameter" na lista de base_parameters

    base_list = ['biofilm_killing_perc',
                 'concentration_mass_mass',
                 'concentration_mass_vol',
                 'concentration_molar',
                 'electricpotential',
                 'elementcontent',
                 'distance',
                 'microbe_killing_log',
                 'microbe_killing_mic',
                 'microbe_killing_mbc',
                 'microbe_killing_perc',
                 'nanomaterial_concentration',
                 'nanomaterial_size',
                 'nanomaterial_surface_area',
                 'nanomaterial_zeta_potential',
                 'percentage',
                 'surface_tension',
                 'temperature',
                 'time',
                 'toxic_ec50',
                 'toxicity_lc50',
                 'toxic_ld50',
                 'toxic_ic50',
                 'viscosity_cp',
                 'volume',
                 'weight']
    
    return base_list



#------------------------------
def list_textual_parameter_for_ngrams_search(diretorio = None):

    from FUNCTIONS import get_filenames_from_folder

    parameter_list = []
    filenames = get_filenames_from_folder(diretorio + '/Outputs/ngrams/semantic', file_type = 'csv')
    for filename in filenames:
        parameter_list.append( re.search(r'(?<=n[12x]gram_).+', filename).group() )

    return parameter_list



#------------------------------
def list_textual_parameter_for_llm_search():
    
    parameter_list = ['2d_materials',
                      'metallic_materials',
                      'oxide_materials',
                      'qdots_materials',

                      'toxicity_a_thaliana_all',
                      'toxicity_a_thaliana_bioaccumulation',
                      'toxicity_a_thaliana_development',
                      'toxicity_a_thaliana_enzyme',                  
                      'toxicity_a_thaliana_genotox',
                      'toxicity_a_thaliana_germination',
                      'toxicity_a_thaliana_morphology',
                      'toxicity_a_thaliana_oxi_stress',
                      'toxicity_a_thaliana_photosynthesis',
                      'toxicity_a_thaliana_seedling_phototropic',
                      
                      'toxicity_c_elegans_all',
                      'toxicity_c_elegans_behavior',
                      'toxicity_c_elegans_bioaccumulation',
                      'toxicity_c_elegans_development',
                      'toxicity_c_elegans_enzyme',
                      'toxicity_c_elegans_genotox',
                      'toxicity_c_elegans_mortality',
                      'toxicity_c_elegans_morphology',
                      'toxicity_c_elegans_oxi_stress',
                      'toxicity_c_elegans_reproduction',
                      
                      'toxicity_d_magna_all',
                      'toxicity_d_magna_behavior',
                      'toxicity_d_magna_bioaccumulation',
                      'toxicity_d_magna_development',
                      'toxicity_d_magna_enzyme',
                      'toxicity_d_magna_genotox',
                      'toxicity_d_magna_mortality',
                      'toxicity_d_magna_morphology',
                      'toxicity_d_magna_oxi_stress',
                      'toxicity_d_magna_reproduction',

                      'toxicity_d_rerio_all',
                      'toxicity_d_rerio_behavior',
                      'toxicity_d_rerio_bioaccumulation',
                      'toxicity_d_rerio_development',
                      'toxicity_d_rerio_enzyme',
                      'toxicity_d_rerio_genotox',
                      'toxicity_d_rerio_mortality',
                      'toxicity_d_rerio_morphology',
                      'toxicity_d_rerio_oxi_stress',
                      'toxicity_d_rerio_reproduction'
                      ]

    return parameter_list



#------------------------------
def list_textual_parameter_for_llm_check():
    
    parameter_list = ['biofilm_killing_perc',
                      'microbe_killing_log',
                      'microbe_killing_mbc',
                      'microbe_killing_mic',
                      'nanomaterial_morphology',
                      'nanomaterial_size',
                      'nanomaterial_surface_area',
                      'nanomaterial_zeta_potential',
                      'biological_species',
                      'toxicity_lc50',
                      'toxicity_yes_no'
                      ]

    return parameter_list



'''#------------------------------
def process_input_parameter(parameter: str, diretorio: str = None):

    textual_parameter = None
    num_parameter = None

    if parameter in list_textual_parameter(diretorio):
        textual_parameter = parameter
    
    elif parameter in list_numerical_parameter():
        num_parameter = parameter

    elif parameter in list_textual_num_parameter():
        textual_parameter, num_parameter = find_textual_num_params(parameter)

    return textual_parameter, num_parameter'''



#------------------------------
def regex_patt_from_parameter(parameter):

    print('Encontrando padrão regex para o parâmetro: ', parameter)

    pattern_dic = {}
    #esse termo é para indicar se foi encontrado algum parâmetro
    found_parameter = False

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
        
    elif parameter.lower() == 'microbe_killing_log':
        
        pattern_dic['first_parameter'] = 'log10'
        pattern_dic['second_parameter'] = None

        #lista de unidades a serem encontradas
        PU_units_to_find = PU_unit_dic[pattern_dic['first_parameter']]
        
        #determinando o número mínimo e máximo de caracteres numéricos        
        n_min_len , n_max_len = 1 , 5
        ndec_min_len, ndec_max_len = 0 , 3
        found_parameter = True
        parameter_type = 'single'

    elif parameter.lower() == 'microbe_killing_mic':
        
        pattern_dic['first_parameter'] = 'weight'
        pattern_dic['second_parameter'] = 'volume'
        
        #lista de unidades a não serem encontradas
        PU_units_to_find = get_physical_units_combined(first_parameter = pattern_dic['first_parameter'], second_parameter = pattern_dic['second_parameter'], get_inverse = True)
        
        #determinando o número mínimo e máximo de caracteres numéricos        
        n_min_len , n_max_len = 1 , 5
        ndec_min_len, ndec_max_len = 0 , 3
        found_parameter = True
        parameter_type = 'combined'

    elif parameter.lower() == 'microbe_killing_mbc':
        
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

    elif parameter.lower() == 'biofilms_killing_perc':
        
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

    elif parameter.lower() == 'size':
        
        pattern_dic['first_parameter'] = 'distance'
        pattern_dic['second_parameter'] = None

        #lista de unidades a serem encontradas
        PU_units_to_find = PU_unit_dic[pattern_dic['first_parameter']]
        
        #determinando o número mínimo e máximo de caracteres numéricos
        n_min_len , n_max_len = 1 , 5
        ndec_min_len, ndec_max_len = 0 , 3
        found_parameter = True
        parameter_type = 'single'

    elif parameter.lower() == 'surface_area':

        pattern_dic['first_parameter'] = 'areametric'
        pattern_dic['second_parameter'] = 'weight'        
        
        #lista de unidades a não serem encontradas
        PU_units_to_find = get_physical_units_combined(first_parameter = pattern_dic['first_parameter'], second_parameter = pattern_dic['second_parameter'], get_inverse = True)

        #determinando o número mínimo e máximo de caracteres numéricos        
        n_min_len , n_max_len = 1 , 5
        ndec_min_len, ndec_max_len = 0 , 3
        found_parameter = True
        parameter_type = 'combined'

    elif parameter.lower() == 'zeta_potential':
        
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

    elif parameter.lower() == 'toxic_ec50':
        
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

    elif parameter.lower() == 'toxic_ld50':
        
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


        #montando o regex pattern
        range_separator = '\s*\-\s*|\s*\+\s*\/\s*\-\s*|\s*±\s*|\s*to\s*'

        num_pattern = '\-?[0-9]{n_min_len},{n_max_len}\.?[0-9]{ndec_min_len},{ndec_max_len}?'.format(n_min_len = '{' + str(n_min_len), 
                                                                                                        n_max_len = str(n_max_len) + '}', 
                                                                                                        ndec_min_len = '{' + str(ndec_min_len),
                                                                                                        ndec_max_len = str(ndec_max_len) + '}')
        
        initial_pattern = '({num_pattern})\s*({PUs_to_find})?\s*((?<!\s*\,\s*)({range_separator})\s*({num_pattern}))?\s*({PUs_to_find})'.format(num_pattern = num_pattern, 
                                                                                                                                            range_separator = range_separator,
                                                                                                                                            PUs_to_find = text_PUs_to_find)


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
                
            last_pattern = '(?!\s*({PUs_not_to_find}))'.format(PUs_not_to_find = PUs_not_to_find)

        elif parameter_type == 'combined':
            last_pattern = '()'


        #gerando a pattern para encontrar
        pattern_dic['PU_to_find_regex'] = r'{initial_pattern}{last_pattern}'.format(initial_pattern = initial_pattern,
                                                                                    last_pattern = last_pattern)

    #print('PU_to_find_regex: ', pattern_dic['PU_to_find_regex'])    
    
    return pattern_dic