#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class DataFrames(object):
    
    def __init__(self, mode = 'select_parameters_manual', consolidated_DataFrame_name = 'ConsolDF' , save_to_GoogleDrive = False, diretorio = None):
        
        print('\n=============================\nDATA FRAMES...\n=============================\n')
        print('( Class: DataFrames )')
        

        import pandas as pd
        from GoogleDriveAPI import gDrive

        self.diretorio = diretorio
        self.class_name = 'DataFrames'        
        self.mode = mode.lower()
        self.consolidated_DataFrame_name = consolidated_DataFrame_name
        self.save_to_GDrive = save_to_GoogleDrive
        if self.save_to_GDrive == True:
            self.drive = gDrive()

        #testando os erros de inputs
        from FUNCTIONS import error_incompatible_strings_input        
        self.abort_class = error_incompatible_strings_input('mode', mode, ('collect_parameters_automatic', 'collect_parameters_manual', 'select_sentences'), class_name = self.class_name)
                
        print(f'Mode chose: {mode}')            


    def set_settings(self, 
                     input_DF_name = 'frags_extracted', 
                     search_mode = '',
                     output_DF_name = 'Paramaters', 
                     parameters = ['temperature_pyrolysis'], 
                     hold_samples = False, 
                     hold_sample_number = False,
                     ngram_for_textual_search = 2,
                     min_ngram_appearence = 5,
                     lower_sentence_in_textual_search = True,
                     numbers_extraction_mode = 'first',
                     filter_unique_results = False,
                     match_all_sent_findings = True,
                     get_avg_num_results = False):
        
        print('( Function: set_files )')
        print('Setting Files...')
        
        self.input_DF_name = input_DF_name
        self.search_mode = search_mode
        self.output_DF_name = output_DF_name
        self.input_parameters_list = parameters
        self.hold_samples = hold_samples
        self.hold_sample_number = hold_sample_number
        self.min_ngram_appearence = min_ngram_appearence
        self.lower_sentence_in_textual_search = lower_sentence_in_textual_search
        self.ngram_for_textual_search = ngram_for_textual_search
        self.numbers_extraction_mode = numbers_extraction_mode
        self.filter_unique_results = filter_unique_results
        self.match_all_sent_findings = match_all_sent_findings
        self.get_avg_num_results = get_avg_num_results


    
    def get_data(self, max_token_in_sent = 100):

        print('( Function: get_data )')
        
        #checando erros de instancia????o/inputs
        from FUNCTIONS import error_print_abort_class
        if self.abort_class is True:        
            error_print_abort_class(self.class_name)
            return
        from FUNCTIONS import create_PDFFILE_list
        from FUNCTIONS import error_incompatible_strings_input
        from FUNCTIONS import load_dic_from_json
        from FUNCTIONS import create_DF_columns_names_with_index
        from functions_PARAMETERS import list_numerical_parameter
        from functions_PARAMETERS import list_textual_parameter
        import regex as re
        import time
        import pandas as pd
        import webbrowser as wb
        import os

        #checando se existe um Data Frame de fragmentos
        print(f'Procurando... {self.diretorio}/Outputs/Extracted/' + f'{self.input_DF_name}_extracted_mode_{self.search_mode}.csv')
        if os.path.exists(self.diretorio + '/Outputs/Extracted/' + f'{self.input_DF_name}_extracted_mode_{self.search_mode}.csv'):
            self.extracted_fragsDF = pd.read_csv(self.diretorio + '/Outputs/Extracted/' + f'{self.input_DF_name}_extracted_mode_{self.search_mode}.csv', index_col=[0,1], dtype=object)
            print(f'Carregando o DataFrame de INPUT com os frags/sents extraidos (~/Outputs/Extracted/{self.input_DF_name}_extracted_mode_{self.search_mode}.csv)')
            #se multiplica por 2 o valor de self.input_parameters_list pois uma coluna ?? dos indexes das senten??as
            if len(self.extracted_fragsDF.columns) / 2 == len(self.input_parameters_list):
                
                #dic para salvar as SI PUs padronizadas para cada features (ou par??metros)
                if not os.path.exists(self.diretorio + '/Outputs/DataFrames/SI_PUs.json'):
                    self.SI_PUs_dic_to_record = {}
                else:
                    self.SI_PUs_dic_to_record = load_dic_from_json(self.diretorio + '/Outputs/DataFrames/SI_PUs.json')
                
                #definindo os inputs de par??metros
                for filename in self.extracted_fragsDF.index.levels[0]:
                    
                    print('----------------------------------------------------------------------------------------------------------------------------------------------------')
                    self.filename = filename
                    
                    #montando os nomes das colunas do DF Tempor??rio (isso pois os nomes das colunas do DF que ser?? exportado sair?? das fun????o de extra????o de dados)
                    Temp_DF_columns_name = create_DF_columns_names_with_index(self.input_parameters_list, index_columns = [], index_name = '_index')
                                        
                    #checando se j?? existe um Data Frame para esses par??metros
                    if os.path.exists(self.diretorio + f'/Outputs/DataFrames/{self.output_DF_name}_mode_{self.numbers_extraction_mode}_match_{self.match_all_sent_findings}.csv'):
                        self.output_DF = pd.read_csv(self.diretorio + f'/Outputs/DataFrames/{self.output_DF_name}_mode_{self.numbers_extraction_mode}_match_{self.match_all_sent_findings}.csv', index_col=[0,1], dtype=object)
                        self.output_DF.index.names = ['Filename', 'Counter']
                        self.create_output_DF = False
                        #print(f'Carregando o DataFrame de OUTPUT (~/Outputs/Extracted/{self.output_DF_name}.csv)')
                        try:
                            last_pdf_processed = self.output_DF.index.levels[0][-1]
                        except IndexError:
                            last_pdf_processed = 'PDF00000'
                    else:
                        print(f'Output DF {self.output_DF_name}_mode_{self.numbers_extraction_mode}_match_{self.match_all_sent_findings}.csv n??o encontrado.')
                        print(f'Criando o output_DF data frame: {self.output_DF_name}_mode_{self.numbers_extraction_mode}_match_{self.match_all_sent_findings}.csv')
                        #caso tenha que ser gerada a output_DF
                        self.create_output_DF = True
                        last_pdf_processed = 'PDF00000'
                    
                    pdf_extracted_list = create_PDFFILE_list(last_pdf_processed)

                    #abrindo o search-extract report
                    if os.path.exists(self.diretorio + f'/Outputs/LOG/counter_SE_report_{self.input_DF_name}_extracted_mode_{self.search_mode}.json'):
                        #carregando o dicion??rio
                        self.search_report_dic = load_dic_from_json(self.diretorio + f'/Outputs/LOG/counter_SE_report_{self.input_DF_name}_extracted_mode_{self.search_mode}.json')
                        if self.search_report_dic['search']['searching_status'] != 'finished':
                            print(f'Erro! O processo de extra????o para o search_input {self.input_DF_name} ainda n??o terminou.' )
                            return                        
                        
                        elif last_pdf_processed == 'PDF00000':
                            self.search_report_dic['export'] = {}
                            self.search_report_dic['export']['last_PDF_processed'] = None
                            self.search_report_dic['export']['total_finds'] = 0
                            self.search_report_dic['export']['pdf_finds'] = 0
                            
                    else:
                        print('Erro! LOG counter_SE_report n??o encontrado em ~/outputs/LOG' )
                        print(f'Erro! O processo de extra????o para o search_input {self.input_DF_name} n??o foi feito.' )
                        return

                    
                    if self.filename not in pdf_extracted_list:
                        print('# Processando ', self.filename)                
                        
                        #dicion??rio para coletar os par??metros num??ricos extra??dos
                        self.parameters_extracted = {}
                        self.parameters_extracted['filename'] = self.filename                    
                                            
                        #varrendo as colunas do input_DF (coluna = feature)
                        for parameter_N in range(len(self.extracted_fragsDF.columns)):
                            
                            #nome da coluna do input DF
                            input_DF_column_name = self.extracted_fragsDF.columns[parameter_N]
                            #nome do par??metro de entrada (os par??metros de fato sair??o das senten??as)
                            input_parameter = Temp_DF_columns_name[parameter_N]
                            
                            #ignora-se as colunas dos indexes dos par??metros
                            if input_DF_column_name[ -len('_index') : ] != '_index':                                

                                print('\nParameter column: ( ', input_parameter, ' )') 
                                print('Fragments extracted from: ', self.filename)                                

                                #criando uma key de coluna, para o index de senten??a coletada e para confirma????o dos dados extra??dos
                                self.parameters_extracted['column' + str(parameter_N)] = {}                                                                                            
                                self.parameters_extracted['column' + str(parameter_N)]['selected_sent_index'] = None
                                self.parameters_extracted['column' + str(parameter_N)]['extracted_paramaters_from_column'] = False

                                #varrendo as linhas com as senten??as para esse PDF (input_DF)
                                for i in self.extracted_fragsDF.loc[ (self.filename , ) , ].index:                                    
                                    
                                    #checando se o n??mero de tokens ?? maior que o desejado                                    
                                    sent_len = len( str(self.extracted_fragsDF.loc[ (self.filename , i ) , input_DF_column_name ]).split() )

                                    #senten??a
                                    sent = self.extracted_fragsDF.loc[ (self.filename , i ) , input_DF_column_name ]                                        
                                    #index de senten??a
                                    sent_index = self.extracted_fragsDF.loc[ (self.filename , i ) , input_DF_column_name + '_index' ]

                                    #coletando a senten??a e o sent_index                                        
                                    self.parameters_extracted['column' + str(parameter_N)][i] = {}                                        
                                    self.parameters_extracted['column' + str(parameter_N)][i]['sent'] = sent
                                    self.parameters_extracted['column' + str(parameter_N)][i]['sent_index'] = sent_index                                        
                                                                            
                                    #Mostrando as senten??as a serem processadas para esse PDF
                                    print(f'\nIndex {i} (sent_index {sent_index}):', sent, '\n')
                                    
                                    #s?? entramos na sequ??ncia abaixo para coletar os par??metros
                                    if self.mode != 'select_sentences':                                            

                                        #essa key definida aqui coleta somente os valores extra??dos (serve para compara????o/matching)
                                        self.parameters_extracted['column' + str(parameter_N)][i]['extracted_num_str_vals'] = None
                                        
                                        #essa key definida aqui coleta o index_counter, o sent_index e o valores num??ricos extra??dos                                        
                                        for output_type in ('num_output', 'str_output'):
                                            self.parameters_extracted['column' + str(parameter_N)][i][output_type] = {}
                                        
                                        #checando se o par??metro ir?? para a extra????o num??rica
                                        if input_parameter in list_numerical_parameter():
                                            #extraindo os par??metros num??ricos com as unidades f??sicas
                                            numerical_params_extracted_from_sent = self.extract_numerical_parameters(sent, sent_index, input_parameter, extract_mode = self.numbers_extraction_mode)
                                            #n??mero de par??metros extraidos
                                            n_num_outputs_extracted = numerical_params_extracted_from_sent['n_num_outputs_extracted']
                                            #unidade SI extra??da
                                            SI_units = numerical_params_extracted_from_sent['SI_units']
                                            #caso tenha sido extra??do algum output num??rico corretamente
                                            print('extracted numerical outputs - n_num_outputs: ', n_num_outputs_extracted, ' ; SI_units: ', SI_units, ' ; sent_len: ', sent_len, '( max_sent_allowed : ', max_token_in_sent, ')')
                                            if n_num_outputs_extracted > 0 and SI_units is not None and sent_len < max_token_in_sent:
                                                #adicionando as SI PUs no nome da coluna que vai ser exportada para a consolidated_DF
                                                self.SI_PUs_dic_to_record[input_parameter] = SI_units
                                                print('> Param??tros num??ricos extra??dos para: ', input_parameter + ' (' + SI_units + ')' )
                                                
                                                #coletando os par??metros extraidos da senten??a
                                                self.parameters_extracted['column' + str(parameter_N)][i]['num_output'] = numerical_params_extracted_from_sent['num_output']
                                                self.parameters_extracted['column' + str(parameter_N)][i]['extracted_num_str_vals'] = numerical_params_extracted_from_sent['extracted_num_str_vals']
                                                
                                                print('> ', numerical_params_extracted_from_sent['num_output'] )
                                                #time.sleep(2)
                                            
                                            #caso nenhum output num??rico tenha sido exportado
                                            else:
                                                print(f'> Nenhum par??metro num??rico foi extra??do para o parameter: {input_parameter}')

                                        #checando se o par??metro ir?? para a extra????o textual                                                
                                        elif input_parameter in list_textual_parameter():
                                            #caso a senten??a seja procurada em lower
                                            sent_modified = sent
                                            if self.lower_sentence_in_textual_search is True:
                                                sent_modified = sent.lower()
                                            textual_params_extracted_from_sent = self.extract_textual_parameters(sent_modified, sent_index, input_parameter, ngram = self.ngram_for_textual_search)
                                            #n??mero de par??metros extraidos
                                            n_str_outputs_extracted = textual_params_extracted_from_sent['n_str_outputs_extracted']
                                            #caso tenha sido extra??do algum output num??rico
                                            if n_str_outputs_extracted > 0 and sent_len < max_token_in_sent:
                                                print('> Param??tros textuais extra??dos: ', input_parameter)
                                                                                                                                                
                                                #coletando os par??metros extraidos da senten??a                                                                                                        
                                                self.parameters_extracted['column' + str(parameter_N)][i]['str_output'] = textual_params_extracted_from_sent['str_output']
                                                self.parameters_extracted['column' + str(parameter_N)][i]['extracted_num_str_vals'] = textual_params_extracted_from_sent['extracted_num_str_vals']
                                                
                                                print('> ', textual_params_extracted_from_sent['str_output'] )
                                                #time.sleep(2)
                                                
                                            else:
                                                print(f'> Nenhum par??metro textual foi extra??do para o parameter: {input_parameter}')
                                            
                                        #caso os par??metros introduzidos (input_parameters) n??o estejam lista para extra????o nas FUN????ES DE EXTRA????O
                                        else:
                                            #listar os par??metros dispon??veis para extra????o
                                            available_inputs = list_numerical_parameter() + list_textual_parameter()
                                            abort_class = error_incompatible_strings_input('column', input_parameter, available_inputs, class_name = self.class_name)
                                            if abort_class is True:
                                                return                                        
                                                                        
                                while True: 
                                    try:  
                                        if self.mode == 'select_sentences':
                                            print('\nDigite o(s) index(es) das senten??as de interesse (digite valores inteiros)')
                                            print('Outros comandos: "+" para ignorar esse PDF e ir para o pr??ximo; "open" para abrir o PDF; e "exit" para sair.')
                                            self.param_val = str(input('Index: '))
                                        
                                        elif self.mode == 'collect_parameters_manual':
                                            print('\nConfirme o(s) valor(es) extraidos ( digite * para confirmar )')
                                            print('Outros comandos: "+" para ignorar esse PDF e ir para o pr??ximo; "open" para abrir o PDF; e "exit" para sair.')
                                            self.param_val = str(input('Confirma: '))
                                            
                                        elif self.mode == 'collect_parameters_automatic':
                                            self.param_val = '*'
                                        
                                        #processando o input                                        
                                        if self.param_val.lower() == 'open':
                                            wb.open_new(self.diretorio + '/DB/' + self.filename + '.pdf')
                                            continue
    
                                        elif self.param_val.lower() == '*':
                                            break
                                        
                                        elif self.param_val.lower() == '+':
                                            break
                                        
                                        elif self.param_val.lower() == 'exit':
                                            print('> Abortando fun????o: DataFrames.get_data')
                                            return                                    
                                        
                                        else:
                                            #caso o modo seja para coletar as senten??as para treinamento ML                                                                                
                                            if self.mode == 'select_sentences':
                                                try:
                                                    selected_sent_key = int(self.param_val)
                                                    #verificando se esse parametro inserido ?? relativo ao valor de key das senten??as coletadas
                                                    self.parameters_extracted['column' + str(parameter_N)][selected_sent_key]
                                                    self.parameters_extracted['column' + str(parameter_N)]['selected_sent_index'] = selected_sent_key
                                                    break
                                                except (TypeError, KeyError):
                                                    print('Erro! Inserir um index v??lido para coletar a senten??a.')
                                                    continue                                                                                            
                                
                                    except ValueError:
                                        print('--------------------------------------------')
                                        print('Erro!')
                                        print('Inserir valor v??lido')
                                        break                                                                                        
                        
                        #se o ??ltimo input introduzido n??o foi "+" (que ?? usado para passar para o proximo PDF)   
                        if self.param_val != '+':
                                            
                            #caso seja modo de coleta de dados
                            convert_to_DF = True
                            if self.mode != 'select_sentences':                                
                                
                                print(self.parameters_extracted)                            

                                #definindo um dic para colocar todos os par??metros coletados
                                #serve para impedir duplicatas quando o modo self.match_all_sent_findings est?? ligado
                                all_values_collected_dic = {}

                                #coletando somente os keys num??ricos do dicion??rio (inteiros) respectivos ?? posi????o da coluna
                                self.columns_numeric_keys = [int( key[ len('column') : ] ) for key in self.parameters_extracted.keys() if ( key[ : len('column') ] == 'column')]

                                #varrendo as colunas
                                for N in self.columns_numeric_keys:
                                    
                                    for output_type in ('num_output', 'str_output'):

                                        #definindo o key para o concatenado
                                        self.parameters_extracted['column' + str(N)][output_type] = {}
                                        #coletando o primeiro valor dos outputs extraidos
                                        first_output = self.parameters_extracted['column' + str(N)][0][output_type]
                                        #check caso tudo seja igual
                                        all_same = True
                                        #olhando somente os valores extraidos das senten??as (sem o sent_index e o index_counter)
                                        vals_extracted_line_0 = self.parameters_extracted['column' + str(N)][0]['extracted_num_str_vals']
                                        #varrendo todas as senten??as que tiveram dados extraidos
                                        for i in [ i for i in self.parameters_extracted['column' + str(N)].keys() if type(i) == int ]:
                                            vals_extracted_next_line = self.parameters_extracted['column' + str(N)][i]['extracted_num_str_vals']
                                            #checando se os valores extra??dos para diferentes linhas batem
                                            if self.match_all_sent_findings is True:
                                                if vals_extracted_next_line == vals_extracted_line_0:
                                                    continue
                                                else:
                                                    all_same = False
                                                    break
                                            else:
                                                #varrendo os par??metros de cada output_type
                                                for param in self.parameters_extracted['column' + str(N)][i][output_type].keys():

                                                    #criando uma lista para coletar todos os valores extraidos para cada par??metro
                                                    try:
                                                        all_values_collected_dic[param]
                                                    except KeyError:                                                        
                                                        all_values_collected_dic[param] = []                                                    

                                                    #caso os resultados num??ricos sejam clusterizados (s?? entra o menor e o maior valor encontrado no artigo)
                                                    if (output_type == 'num_output') and (self.get_avg_num_results is True):

                                                        #varrendo os outputs no formato: [ counter , sent_index , val ]
                                                        for res in self.parameters_extracted['column' + str(N)][i][output_type][param]:
                                                            sent_index = res[1]
                                                            all_values_collected_dic[param].append( round(float( res[2] ), 10) )
                                                            min_val = round(min(all_values_collected_dic[param]), 10)
                                                            max_val = round(max(all_values_collected_dic[param]), 10)
                                                            self.parameters_extracted['column' + str(N)][output_type][param] = [[0 , 
                                                                                                                                 sent_index , 
                                                                                                                                 str(min_val) + ' ' + str(max_val)]]
                                                        #confirmando que foram extraidos par??metros dessa coluna
                                                        self.parameters_extracted['column' + str(N)]['extracted_paramaters_from_column'] = True
                                                        
                                                    #caso todos os resultados entrem
                                                    else:                                                        
                                                        
                                                        #criando um dicion??rio para juntar todos os resultados
                                                        try:
                                                            self.parameters_extracted['column' + str(N)][output_type][param]
                                                        except KeyError:
                                                            index_counter = 0
                                                            self.parameters_extracted['column' + str(N)][output_type][param] = []
                                                        
                                                        #varrendo os outputs no formato: [ counter , sent_index , val ]
                                                        for res in self.parameters_extracted['column' + str(N)][i][output_type][param]:
                                                            sent_index = res[1]
                                                            val = res[2]
                                                            #caso os par??metros sejam extraidos somente um vez
                                                            if self.filter_unique_results == True:
                                                                if val not in all_values_collected_dic[param]:
                                                                    self.parameters_extracted['column' + str(N)][output_type][param].append([ index_counter , sent_index , val])
                                                                    all_values_collected_dic[param].append(val)
                                                                    index_counter += 1
                                                            #caso todos os par??metros textuais de todas as senten??as sejam extra??dos
                                                            else:
                                                                self.parameters_extracted['column' + str(N)][output_type][param].append([ index_counter , sent_index , val])
                                                                index_counter += 1
                                                        #confirmando que foram extraidos par??metros dessa coluna
                                                        self.parameters_extracted['column' + str(N)]['extracted_paramaters_from_column'] = True
                                        
                                        #modo que seleciona os par??metros somente quanto todos os valores extraidos de todas as senten??as sejam iguais
                                        if self.match_all_sent_findings is True:
                                            #print(self.parameters_extracted)
                                            #concatenando os par??metros extraidos em uma key
                                            if all_same is True:
                                                self.parameters_extracted['column' + str(N)][output_type] = first_output
                                                #confirmando que foram extraidos par??metros dessa coluna
                                                self.parameters_extracted['column' + str(N)]['extracted_paramaters_from_column'] = True
                                            else:
                                                convert_to_DF = False                                                
                                        
                                    #apagando as keys de cada senten??a
                                    for i in [ i for i in self.parameters_extracted['column' + str(N)].keys() if type(i) == int ]:
                                        del(self.parameters_extracted['column' + str(N)][i])

                            if convert_to_DF is True:

                                #gerando a DF com os dados colletados
                                #print('\nEXTRACTED FROM THE FILE')
                                #print(self.parameters_extracted)
                                #time.sleep(5)
                                self.convert_parameters_extracted_to_DF()
                            
                            else:
                                print('Erro! Os par??metros extra??dos de multiplas linhas n??o s??o id??nticos.')
                                                                                
                    else:
                        print(f'O PDF {self.filename} j?? foi processado (checar no arquivo ~/Outputs/DataFrames/{self.output_DF_name}.csv)')
                        continue
                    
                #consolidando o report na DF caso seja o ultimo arquivo de procura        
                if self.filename == self.extracted_fragsDF.index.levels[0][-1]:
                    self.generate_search_report()
                    
            else:
                print('Erro! O n??mero de par??metros introduzidos ?? diferente do n??mero de termos coletados (colunas do extracted_DF)')
                print('N??mero de par??metros introduzido: ',  len(self.input_parameters_list) )
                print('N??mero de colunas da extracted_DF: ',  len(self.extracted_fragsDF.columns) / 2)
                print('> Abortando a classe: DataFrames')
                return

        else:
            print('Erro! N??o foi encontrado um DF de fragamentos de artigos.')
            print('> Abortando a classe: DataFrames')
            return



    def convert_parameters_extracted_to_DF(self):
                
        import pandas as pd
        import os
        #import time

        if not os.path.exists(self.diretorio + '/Outputs/DataFrames'):
            os.makedirs(self.diretorio + '/Outputs/DataFrames')
        
        #caso o modo seja de selecionar senten??as
        if self.mode == 'select_sentences':
            #gerando e salvando as DF com as senten??as
            self.analyze_output_results()
            if self.output_results_checked is True:
                self.consolidate_DF()
        
        else:                        
            
            #Caso os ouputs de todos os par??metros (colunas) sejam coletados
            #if False not in [ self.parameters_extracted[key]['extracted_paramaters_from_column'] for key in self.parameters_extracted.keys() if key[ : len('column')] == 'column' ]:                
                    
            #caso a output_DF tenha que ser criada
            if self.create_output_DF is True:
                                                        
                self.output_DF = pd.DataFrame(columns=['Filename', 'Counter'], dtype=object)
                self.output_DF.set_index(['Filename', 'Counter'], inplace=True)
                            
            print(f'Procurando... {self.diretorio}/Outputs/DataFrames/{self.consolidated_DataFrame_name}.csv')
            if not os.path.exists(self.diretorio + f'/Outputs/DataFrames/{self.consolidated_DataFrame_name}.csv'):
                print(f'Criando a DF consolidada... (~/Outputs/DataFrames/{self.consolidated_DataFrame_name}.csv)')                
                self.consolidated_DF = pd.DataFrame(index=[[],[]], dtype=object)
                self.consolidated_DF.index.names = ['Filename', 'Counter']
                self.consolidated_DF.to_csv(self.diretorio + f'/Outputs/DataFrames/{self.consolidated_DataFrame_name}.csv')
            
            #carregando a DF consolidada caso o modo seja de cole????o de par??metros
            else:
                print(f'Abrindo a consolidated DF: {self.diretorio}/Outputs/DataFrames/{self.consolidated_DataFrame_name}.csv')
                self.consolidated_DF = pd.read_csv(self.diretorio + f'/Outputs/DataFrames/{self.consolidated_DataFrame_name}.csv', index_col=[0,1], dtype=object)
                            
            #criando a coluna do Sample_counter no DF consolidado
            if 'Sample_counter' not in self.consolidated_DF.columns:
                self.consolidated_DF['Sample_counter'] = ''                                
            
            #gerando e salvando as DF com os par??metros
            self.analyze_output_results()
            if self.output_results_checked is True:
                self.consolidate_DF()
            
            #else:
            #    #print([ self.parameters_extracted[key]['extracted_paramaters_from_column'] for key in self.parameters_extracted.keys() if key[ : len('column')] == 'column' ])
            #    print('Erro! N??o foram coletados dados para todos os par??metros inseridos.')
            #    print('Passando para o pr??ximo PDF')
            #    #time.sleep(3)
            #    return



    def analyze_output_results(self):

        #determinando todos os par??metros que foram extraidos de todas as colunas
        self.extracted_parameters_list = []
        for N in self.columns_numeric_keys:
            #parametros num??ricos
            if len(self.parameters_extracted['column' + str(N)]['num_output']) > 0:
                num_parameter_extracted_list = self.parameters_extracted['column' + str(N)]['num_output'].keys()
                for parameter in num_parameter_extracted_list:
                    if parameter not in self.extracted_parameters_list:
                        self.extracted_parameters_list.append(parameter)
            
            #parametros textuais
            if len(self.parameters_extracted['column' + str(N)]['str_output']) > 0:
                str_parameter_extracted_list = self.parameters_extracted['column' + str(N)]['str_output'].keys()
                for parameter in str_parameter_extracted_list:
                    if parameter not in self.extracted_parameters_list:
                        self.extracted_parameters_list.append(parameter)

        if self.mode == 'select_sentences':        

            #testa se indexes foram coletados para todas as colunas (par??metros)               
            if None not in [self.parameters_extracted['column' + str(j)]['selected_sent_index'] for j in self.columns_numeric_keys]:                               
                #uma linha pois s?? uma senten??a ?? coletada por PDF
                self.output_results_checked = True
            else:
                print('Erro de extra????o! Nenhum par??metro foi extra??do (max_output_number = 0).')
                self.output_results_checked = False
            
        #modo de coleta de dados
        else:
            #caso tenha tido pelo menos um parametro extraido
            if len( self.extracted_parameters_list ) > 0:
                self.output_results_checked = True
                #fazendo o match para os par??metros        
                self.match_parameters_extracted()
            else:
                print('Erro de extra????o! Nenhum par??metro foi extra??do (max_output_number = 0).')
                self.output_results_checked = False



    def match_parameters_extracted(self):                
        
        #import time
        
        #setando as condi????es para fazer o fill dos outputs
        #o dicion??rio output number carrega o n??mero de outputs por coluna (feature ou par??metro) a ser colocado na DF consolidada
        self.output_number = {}

        #varrendo os par??metros
        for N in self.columns_numeric_keys:                    
            #somente um output ?? extra??do por coluna

            #varrendo as colunas (par??metros)
            for output_parameter in self.extracted_parameters_list:
                #ou ?? numerico
                try:
                    #caso essa coluna tenha tido par??metro num??rico extra??do
                    self.output_number[output_parameter] = len(self.parameters_extracted['column' + str(N)]['num_output'][output_parameter] )
                    #print('column' + str(N), ' ; ', output_parameter, ' : ' , len(self.parameters_extracted['column' + str(N)]['num_output'][output_parameter]) )
                except KeyError:
                    continue                        
                
            #varrendo as colunas (par??metros)
            for output_parameter in self.extracted_parameters_list:    
                #ou ?? textual
                try:                
                    #caso essa coluna tenha tido par??metro num??rico extra??do
                    self.output_number[output_parameter] = len(self.parameters_extracted['column' + str(N)]['str_output'][output_parameter] )
                    #print('column' + str(N), ' ; ', output_parameter, ' : ' , len(self.parameters_extracted['column' + str(N)]['num_output'][output_parameter]) )
                except KeyError:
                    continue

        self.len_set_output_number = len(set(self.output_number.values()))
        self.min_output_number = min(list(set(self.output_number.values())))
        self.max_output_number = max(list(set(self.output_number.values())))
        
        #checando os resultados
        print('* code analysis begin *')
        print('output_number: ', self.output_number)
        print('len_set_output_number = ', self.len_set_output_number)
        print('min_output_number = ', self.min_output_number)
        print('max_output_number = ', self.max_output_number)
        #time.sleep(1)



    def check_output_conditions(self):
        
        #import time
        
        checked = True

        #procurando se o index de PDF j?? tem sample number
        file_tag = self.parameters_extracted['filename']
        try:        
            #caso seja poss??vel converter para inteiro
            self.current_sample_number = int(float(self.consolidated_DF.loc[ ( file_tag , 0 ), 'Sample_counter' ]))             
        #exce????o caso o index n??o exista na DF ou caso o valor de sample_counter seja None
        except KeyError:
            self.consolidated_DF.loc[ ( file_tag , 0 ), 'Sample_counter' ] = 0
            self.current_sample_number = 0
            
        #o n??mero de amostra atualizado ?? definido ou pelo n??mero que j?? est?? na DF consolidada ou pelo n??mero de outputs que est?? sendo atualizado
        self.updated_sample_number = max(self.max_output_number, self.current_sample_number)
        
        print('min_output_number = ', self.min_output_number)
        print('max_output_number = ', self.max_output_number)
        print('current_sample_number = ', self.current_sample_number)
        print('updated_sample_number = ', self.updated_sample_number)
        
        #condi????es para fazer a consolida????o do FULL DF
        
        cond_output_number =  False
        #quando o n??mero de outputs seja tudo igual (ex: temperature_pyrolysis = 2 ; raw_materials = 2)
        if (self.len_set_output_number == 1) and (self.min_output_number > 0):
            cond_output_number = True
        #quando o n??mero de outputs n??o ?? igual (ex: temperature_pyrolysis = 3 ; time_pyrolysis = 1 ; raw_materials = 1)
        #o n??mero de itens do set deve ser 2 
        #o maior valor de output_number deve ser maior que 1
        #o menor valor de output_number deve ser 1 ou 0 
        elif (self.len_set_output_number == 2) and ((self.max_output_number >= 1)) and (self.min_output_number in (0, 1)) :
            cond_output_number = True

        #para atualiza????o no n??mero de amostras
        cond_match_sample_number = False
        #s?? se atualiza quando o n??mero de amostra atualizao for igual ??quele que est?? na DF consolidada
        if self.hold_sample_number is True:
            if self.updated_sample_number == self.current_sample_number:
                cond_match_sample_number = True        
        else:        
            #quando o n??mero de amostra atualizao for igual ??quele que est?? na DF consolidada
            if self.updated_sample_number == self.current_sample_number:
                cond_match_sample_number = True
            #ou quando um valor seja 0 ou 1 e outro maior que 1
            elif self.updated_sample_number in (0, 1) and  self.current_sample_number >= 1:
                cond_match_sample_number = True
            elif self.updated_sample_number >= 1 and  self.current_sample_number in (0, 1):
                cond_match_sample_number = True

        #quando o atributo de hold_samples estive ligado, o algoritmo s?? adiciona par??metros nas amostras que j?? tem sample_counter > 0        
        cond_hold_samples = True                    
        if self.hold_samples is True:
            #s?? atualizar?? se o j?? houver um n??mero de amostra na DF consolidada
            if (self.current_sample_number == 0):
                cond_hold_samples = False
        else:
            pass                
                
        #checando os resultados
        print('cond_output_number: ', cond_output_number)
        print('cond_match_sample_number: ', cond_match_sample_number)
        print('cond_hold_samples: ', cond_hold_samples)
        #time.sleep(2)

        #se alguma das condi????es falhou
        if False in (cond_output_number, cond_match_sample_number, cond_hold_samples):
            checked = False
                            
        return checked



    def consolidate_DF(self):

        print('\nConsolidando a DF...')        

        #import time
        #import pandas as pd
        from FUNCTIONS import save_dic_to_json        
        
        file_tag = self.parameters_extracted['filename']            
        if self.check_output_conditions() is True:
            
            #checando os outputs que ser??o preenchidos de forma repetida
            self.check_outputs_to_fill_in_consolidated_DF()            

            #varrendo os par??metros que ser??o extraidos
            for output_parameter in self.extracted_parameters_list:
                
                #caso j?? exista alguma entrada na consolidated DF para esse file_tag e para esse output_parameter
                try:
                    self.consolidated_DF.loc[ ( file_tag , ) , output_parameter ]
                    if str(self.consolidated_DF.loc[ ( file_tag , 0 ) , output_parameter ]) == 'nan':
                        output_already_in_consolidated_df = False
                    else:
                        output_already_in_consolidated_df = True
                    
                #caso n??o exista                    
                except KeyError:
                    output_already_in_consolidated_df = False
                    
                #varrendo as colunas (par??metros)
                for N in self.columns_numeric_keys:
                    #varrendo os dois par??metros
                    for output_type in ('num_output','str_output'):                    
                        #a inst??ncia possui a quantidade de outputs a ser preenchida na consolidated DF            
                        for counter in range(self.updated_sample_number):
                            try:
                                self.parameters_extracted['column' + str(N)][output_type][output_parameter]
                            except KeyError:
                                continue
                            
                            if self.parameters_extracted['column' + str(N)][output_type][output_parameter] is None:
                                output_val_index = ''
                                output_val = ''
                            
                            elif self.filloutputs[output_parameter] is True and self.parameters_extracted['column' + str(N)][output_type][output_parameter] is not None:
                                output_val_index = self.parameters_extracted['column' + str(N)][output_type][output_parameter][0][1]
                                output_val = self.parameters_extracted['column' + str(N)][output_type][output_parameter][0][2]
                            
                            elif self.filloutputs[output_parameter] is False and self.parameters_extracted['column' + str(N)][output_type][output_parameter] is not None:
                                #caso o par??metro a ser trabalhado esteja no dicion??rio self.parameters_extracted
                                output_val_index = self.parameters_extracted['column' + str(N)][output_type][output_parameter][counter][1]
                                output_val = self.parameters_extracted['column' + str(N)][output_type][output_parameter][counter][2]                            
                        
                            #salvando no output_DF
                            self.output_DF.loc[ ( file_tag , counter ) , output_parameter ] = output_val
                            self.output_DF.loc[ ( file_tag , counter ) , output_parameter + '_index' ] = output_val_index
            
                            if self.mode != 'select_sentences':
                                #salvando na consolidated
                                self.consolidated_DF.loc[ ( file_tag , counter ) , output_parameter ] = output_val
                                self.consolidated_DF.loc[ ( file_tag , counter ) , output_parameter + '_index' ] = output_val_index                          
                                print('> adicionando na consolidated DF - parameter: ', output_parameter, ' ; val:', output_val)
                                #time.sleep(2)
                                                                
                            #salvando o n??mero atualizado de amostras por PDF na DF consolidada
                            self.consolidated_DF.loc[ ( file_tag , counter ) , 'Sample_counter' ] = self.updated_sample_number                        
            
            
                #salvando a consolidated DF caso n??o haja entrada para esse file_tag e output_parameter
                if output_already_in_consolidated_df is False and self.mode != 'select_sentences':
                    
                    #varrendo os par??metros que que j?? foram extraidos para a consolidated_DF e que n??o est??o na self.extracted_parameters_list
                    for output_parameter in self.parameters_in_consolidated_DF_list:
                        if self.filloutputs[output_parameter] == True:                    
                            output_val = self.consolidated_DF.loc[ ( file_tag , 0 ) , output_parameter ]
                            output_val_index = self.consolidated_DF.loc[ ( file_tag , 0 ) , output_parameter + '_index' ]
                            for counter in range(self.updated_sample_number):
                                self.consolidated_DF.loc[ ( file_tag , counter ) , output_parameter ] = output_val
                                self.consolidated_DF.loc[ ( file_tag , counter ) , output_parameter + '_index' ] = output_val_index
                                print('> adicionando na consolidated DF - parameter: ', output_parameter, ' ; val:', output_val)
                    
                    #salvando a DF consolidada
                    save_dic_to_json(self.diretorio + '/Outputs/DataFrames/SI_PUs.json', self.SI_PUs_dic_to_record)
                    self.consolidated_DF.sort_index(level=[0,1], inplace=True)
                    self.consolidated_DF.to_csv(self.diretorio + f'/Outputs/DataFrames/{self.consolidated_DataFrame_name}.csv')
                    print(f'Atualizando DataFrame consolidado para {file_tag}...')

            #salvando a output_DF            
            self.output_DF.to_csv(self.diretorio + f'/Outputs/DataFrames/{self.output_DF_name}_mode_{self.numbers_extraction_mode}_match_{self.match_all_sent_findings}.csv')
            print(f'\nSalvando output_DF para {file_tag}...')

            #contadores para exportar nas DFs de report
            self.search_report_dic['export']['last_PDF_processed'] = self.parameters_extracted['filename']
            self.search_report_dic['export']['total_finds'] += self.max_output_number
            self.search_report_dic['export']['pdf_finds'] += 1
            
            #salvando o DF report
            save_dic_to_json(self.diretorio + f'/Outputs/LOG/counter_SE_report_{self.input_DF_name}_extracted_mode_{self.search_mode}.json', self.search_report_dic)
                    
        else:
            print('\nH?? incompatibilidade entre os outputs (fun????o: check_output_conditions).')
            print(f'DataFrame para o {file_tag} n??o foi consolidada.')
            return

    
    
    def check_outputs_to_fill_in_consolidated_DF(self):
        
        #import time

        #resgatando os par??metros que j?? est??o na consolidated_DF
        #usa-se o slica [ 1 : ] na consolidated_DF porque a primeira coluna ?? de sample_counter
        self.parameters_in_consolidated_DF_list = [parameter for parameter in self.consolidated_DF.columns[ 1: ] if ( (parameter[ -len('_index') : ] != '_index') and (parameter not in self.extracted_parameters_list) ) ]

        full_parameters_list = self.parameters_in_consolidated_DF_list + self.extracted_parameters_list
        print('full_parameters_list:', full_parameters_list)

        #checando se haver?? preenchimento autom??tico de outputs
        self.filloutputs = {}        
        
        #varrendo as colunas (par??metros) tanto para a DF consolidada quanto para a output DF    
        for output_parameter in full_parameters_list:
            
            fillout = False
            
            try:
                #m??ximo n??mero de outputs num??ricos coletados para cada parametro (coluna)
                #print(f'output_number[{output_parameter}]: ', self.output_number[output_parameter], ' ; updated_sample_number: ' , self.updated_sample_number)
                if self.output_number[output_parameter] < self.updated_sample_number:
                    fillout = True
            
            #caso o parametro j?? tenha sido extraido anteriormente e j?? esteja na consolidated_DF
            except KeyError:
                #caso o Sample_Counter seja 1 o Updated_sample_counter seja maior
                if (self.current_sample_number == 1) and self.current_sample_number < self.updated_sample_number:
                    fillout = True
            
            #determinando se ter?? o fillout ou n??o
            if fillout is True:
                self.filloutputs[output_parameter] = True
            else:
                self.filloutputs[output_parameter] = False

        print('filloutputs: ', self.filloutputs)
        print('* code analysis end *')
        #time.sleep(1)
    
    

    def extract_textual_parameters(self, sent, sent_index, parameter, ngram = 1):
        
        from functions_TOKENS import get_nGrams_list
        import regex as re
        #import time
        
        extracted_dic = {}
        
        str_list = []
        term_list_modified, prob_term_list = get_nGrams_list(parameter, ngrams = self.ngram_for_textual_search, min_ngram_appearence = self.min_ngram_appearence, 
                                                             diretorio=self.diretorio)
        #varrendo os termos
        for term in term_list_modified:
            try:
                if re.search(term, sent):
                    
                    #caso seja um regex com espa??o no inicio coleta-se todo o conjunto encontrado sem o primeiro termo
                    if term[ : 2] == '\s' or term[ : 3] == '\\s':
                        term_to_get = re.search(term, sent).captures()[0][ 1 : ]
                    
                    #caso seja um regex sem espa??o coleta-se todo o conjunto encontrado
                    elif term[ : 1] in ('[', '('):
                        term_to_get = re.search(term, sent).captures()[0]
                    
                    #caso n??o seja um regex, coleta-se somente o termo
                    else:
                        term_to_get = term                
                    
                    #adicionando o termo na lista
                    str_list.append(term_to_get)
            #caso haja algum erro com o par??metro regex
            except re._regex_core.error:
                continue
    
        str_list.sort()
        extracted_dic['extracted_num_str_vals'] = str_list
    
        #organizando os dados que ser??o exportados
        extracted_dic['str_output'] = {}
        extracted_dic['str_output'][parameter] = []
        counter = 0
        for str_val in str_list:
            extracted_dic['str_output'][parameter].append([counter, sent_index, str_val])            
            counter += 1
        
        #exportando o n??mero de resultados
        extracted_dic['n_str_outputs_extracted'] = len(str_list)
        
        #print(extracted_dic)
        #time.sleep(10)
        return extracted_dic


    def extract_numerical_parameters(self, text, text_index, parameter, extract_mode = 'first'):
        
        import time
        from functions_PARAMETERS import regex_patt_from_parameter
                
        #obtendo os padr??es regex para encontrar e para n??o encontrar
        parameters_patterns = regex_patt_from_parameter(parameter)
        
        #print('\nparameters_pattern_to_find_in_sent:\n', parameters_patterns['parameters_pattern_to_find_in_sent'])
        #print('\nPU_to_find_regex:\n', parameters_patterns['PU_to_find_regex'])
        #print('\nPU_not_to_find_regex:\n', parameters_patterns['PU_not_to_find_regex'])

        #a primeira tentativa ?? de extrair o padr??o PARAMETROS > N??MEROS
        extracted_dic = self.extract_param_numbers_from_sent(text, text_index, parameter, parameters_patterns, extract_mode = extract_mode, first_to_get = 'parameters')
        
        #a segunda tentativa ?? de extrair o padr??o PARAMETROS > N??MEROS
        if extracted_dic['num_output'] is None and parameters_patterns['parameters_pattern_to_find_in_sent'] is not None:
            extracted_dic = self.extract_param_numbers_from_sent(text, text_index, parameter, parameters_patterns, extract_mode = extract_mode, first_to_get = 'numbers')
        
        print('extracted_dic from sent: ', extracted_dic)
        #if double_num_outputs is True:
        #time.sleep(5)
        return extracted_dic



    def extract_param_numbers_from_sent(self, text, text_index, parameter, parameters_patterns, extract_mode = 'all', first_to_get = 'parameters'):
            
        import regex as re
        import time
    
        extracted_dic = {}
        extracted_dic['n_num_outputs_extracted'] = 0
        extracted_dic['SI_units'] = None
        extracted_dic['num_output'] = None
    
        #definindo a lista com as PUs encontradas na senten??a
        PUs_found_list = []            
        #definindo a lista com os valores num??ricos encontrados na senten??a
        num_list_extracted_raw = []            
        #definindo os par??metros que podem ser extra??dos da senten??a
        parameters_found = []
        
        #existem tres cumulatives cuts
        cumulative_cut_numbers = 0
        cumulative_cut_parameters = 0
        total_n_numbers_to_extract = 0
        PUs_found_list_temp1 = []
        PUs_found_list_temp2 = []
        
        #check para exportar os dados extraidos
        export_extracted = True
        
        #outer while
        counter_outer_while = 0        
        while True:

            #caso esteja em loop infinito, sai-se do loop
            if counter_outer_while > 100:
                break
            
            counter_outer_while += 1
            #print('\n> first_to_get: ', first_to_get)
            #print('\nscanning step (outer while): ', counter_outer_while)
            
            #zerando os checks
            need_to_find_parameter = False
            parameter_found = False
            parameters_extracted_from_sent = None
            match_next_parameters_to_find = None
            
            #determinando o trecho da senten??a a ser encontrado os par??metros
            if first_to_get == 'parameters':
                cut_text = text[ cumulative_cut_numbers : ]
            elif first_to_get == 'numbers':
                cut_text = text[ cumulative_cut_parameters : ]
            #determinando o index do pr??ximo par??metro
            next_parameter_index = len(text)
            
            #caso seja feita a procura por par??metros no texto (ex: 'C content', 'H/C', etc)           
            if parameters_patterns['parameters_pattern_to_find_in_sent'] is not None:
                #parametro precisa ser encontrado
                need_to_find_parameter = True
                parameters_pattern = r'({text_to_find})+'.format(text_to_find = parameters_patterns['parameters_pattern_to_find_in_sent'])
                match_parameters_to_find = re.search(parameters_pattern, cut_text)
                #print('\ncut_text to find parameters: ', cut_text)
                #print('match_parameters_to_find: ', match_parameters_to_find)
                if match_parameters_to_find:
                    #parametro encontrado
                    parameter_found = True
                    #adicionando o corte cumulativo para os par??metros
                    matches_start, matches_end = match_parameters_to_find.span()
                    if first_to_get == 'parameters':
                        cumulative_cut_parameters = cumulative_cut_numbers + matches_end
                        #print('parameter_index: ', cumulative_cut_numbers + matches_end)
                    elif first_to_get == 'numbers':
                        cumulative_cut_parameters += matches_end
                        #print('parameter_index: ', cumulative_cut_parameters)
                    
                    #extraindo os par??metros da senten??a
                    parameters_extracted_from_sent = self.extract_parameters_from_text( match_parameters_to_find.captures()[0] , parameters_patterns)
                    for param in parameters_extracted_from_sent:
                        if param not in parameters_found:
                            parameters_found.append( param )
                            
                    #procurando o pr??ximo par??metro
                    match_next_parameters_to_find = re.search( parameters_pattern, cut_text[ matches_end : ] )
                    #print('text to find next parameters: ', cut_text[ matches_end : ])
                    #print('match_next_parameters_to_find: ', match_next_parameters_to_find)
                    
                    #determinando o index do proximo par??metro na senten??a
                    if match_next_parameters_to_find:
                        if first_to_get == 'parameters':
                            next_parameter_index = cumulative_cut_numbers + matches_end + match_next_parameters_to_find.span()[1]
                            #print('next_parameter_index: ', next_parameter_index)
                            #print('text after next parameter found: ', cut_text[ matches_end + match_next_parameters_to_find.span()[1] : ])
                            pass
                        elif first_to_get == 'numbers':
                            next_parameter_index = cumulative_cut_parameters + match_next_parameters_to_find.span()[1]
                            #print('next_parameter_index: ', next_parameter_index)
                            #print('text after next parameter found: ', cut_text[ matches_end + match_next_parameters_to_find.span()[1] : ])
                            pass
    
                #print('Par??metros extra??dos: ', parameters_found)
            
            #caso tenha que encontrar o par??metro mas ele n??o foi encontrado
            if need_to_find_parameter is True and parameter_found is False:
                break
            #caso tenha que encontrar o par??metro e ele foi encontrado
            elif need_to_find_parameter is True and parameter_found is True:
                if first_to_get == 'parameters':
                    #nesse modo, caso haja par??metros encontrados, a procura pelos n??meros come??a a partir do ultimo par??metro encontrado
                    local_cumulative_cut = cumulative_cut_parameters
                elif first_to_get == 'numbers':
                    #nesse modo, caso haja par??metros encontrados, a procura pelos n??meros come??a a partir do ultimo n??mero encontrado
                    local_cumulative_cut = cumulative_cut_numbers
                #caso haja par??metros encontrados, a quantidade de n??meros a ser encontrada ser?? feita com m??ltiplos dos par??metros (multiplo de 2)
                n_numbers_matches_to_find = len(parameters_extracted_from_sent) * 2
                total_n_numbers_to_extract += n_numbers_matches_to_find
            #caso n??o tenha que se encontrar o par??metro
            else:                
                #caso n??o haja par??metros a serem encontrados, a procura pelos n??meros come??a a partir do ultimo n??mero encontrado
                local_cumulative_cut = cumulative_cut_numbers
                #caso n??o haja par??metros a serem encontrados, a quantidade de n??meros a ser encontrada ser?? de 1, 2 ou 3
                n_numbers_matches_to_find = 3
                total_n_numbers_to_extract = n_numbers_matches_to_find
    
            #inner while
            counter_inner_while = 0
            inner_matches = 0
            while True:            
                
                break_outer_while = False
                counter_inner_while += 1
                #print('\nscanning step (inner while): ', counter_inner_while)
            
                #procurando os n??meros
                cut_text = text[ local_cumulative_cut : ]
                match_numbers_to_find = re.search( parameters_patterns['PU_to_find_regex'] , cut_text )
                #print('\ncut_text to find numbers: ', cut_text)
                #print('match_numbers_to_find: ', match_numbers_to_find)
                
                #caso tenha sido encontrado a PU
                if match_numbers_to_find:
                    
                    matches_start, matches_end = match_numbers_to_find.span()
                    cumulative_cut = matches_end + local_cumulative_cut
                    #caso a posi????o do n??mero na senten??a seja menor que a posi????o do pr??ximo par??metro
                    #print('number_index: ', cumulative_cut)
                    if first_to_get == 'parameters':
                        #print('next_parameter_index: ', next_parameter_index)
                        next_parameter_index = next_parameter_index
                    elif first_to_get == 'numbers':
                        #print('next_parameter_index: ', cumulative_cut_parameters)
                        next_parameter_index = cumulative_cut_parameters
    
                    #caso o n??mero encontrado esteja em index superior ao pr??ximo par??metro econtrado
                    
                    if cumulative_cut > next_parameter_index:
                        #print('parameters_found: ', parameters_found)
                        #print('num_list_extracted_raw: ', num_list_extracted_raw)
                        #caso o par??metro tenha sido encontrado mas o n??mero n??o, rompe-se os dois whiles (inner e outer)
                        if len(parameters_found) > len(num_list_extracted_raw):
                            export_extracted = False
                            break_outer_while = True
                        #print('(breking inner while)')
                        break
    
                    #caso se queira senten??as com um match somente
                    if extract_mode.lower() == 'first':
                        #checando se h?? mais uma parte que bate com o padr??o na mesma senten??a
                        #print('text to find further numbers in sent (first mode): ', text[ cumulative_cut : ])
                        match_numbers_further_in_sent = re.search( parameters_patterns['PU_to_find_regex'] , text[ cumulative_cut : ])
                        if match_numbers_further_in_sent:
                            #print('match_numbers_further_in_sent: ', match_numbers_further_in_sent)
                            break_outer_while = True
                            #print('(breking inner while)')
                            #time.sleep(2)
                            break
                        
                    #checando se h?? termos que podem ou n??o podem ser encontrados antes dos n??meros e PUs (prefixes)
                    prefix_text = text[ cumulative_cut_numbers :  cumulative_cut ]
                    match_prefixes_to_find = re.search(parameters_patterns['pattern_variation_parameter'], prefix_text)
                    #print('prefix_text to search: ', prefix_text)
                    #print('match_prefixes_to_find: ', match_prefixes_to_find)
                    
                    #caso tenha que ser encontrado (ex: increased by)
                    if (match_prefixes_to_find is not None) and parameters_patterns['find_variation_parameter'] is True:
                        #print('Prefix found: ', match_prefixes_to_find is not None, ' ; find_variation: ', parameters_patterns['find_variation_parameter'])
                        #print('passing...')
                        #time.sleep(10)
                        pass
                    
                    #caso n??o tenha que ser encontrado (ex: increased by)                
                    elif (match_prefixes_to_find is None) and parameters_patterns['find_variation_parameter'] is False:
                        #print('Prefix found: ', match_prefixes_to_find is not None, ' ; find_variation: ', parameters_patterns['find_variation_parameter'])
                        #print('passing...')
                        #time.sleep(10)
                        pass
                    
                    else:
                        #adicionando o corte cumulativo para os n??meros
                        if first_to_get == 'parameters':
                            cumulative_cut_numbers = local_cumulative_cut + matches_end
                        elif first_to_get == 'numbers':
                            cumulative_cut_numbers += matches_end
                        #caso haja par??metros extraidos da senten??a, se subtrai o ultimo parametro encontrado
                        if parameters_extracted_from_sent is not None:
                            parameters_found = parameters_found[ :-1]
                            total_n_numbers_to_extract -= n_numbers_matches_to_find
                            if first_to_get == 'numbers':
                                #indo para um index ap??s o primeiro par??metro
                                cumulative_cut_numbers = cumulative_cut_parameters
                        #print('Prefix found: ', match_prefixes_to_find is not None, ' ; find_variation: ', parameters_patterns['find_variation_parameter'])                        
                        #print('continuing...')
                        #print('(breking inner while)')
                        #time.sleep(2)
                        break
                    
                    #encontrando os par??metros com as unidades f??sicas que n??o s??o desejadas
                    #caso seja None, ?? porque a unidade f??sica de interesse ?? combinada (ex: concentra????o - mol L-1, taxa de aumento de temperatura C s-1)
                    if parameters_patterns['PU_not_to_find_regex'] is not None:
                        #caso n??o seja None, ?? porque a unidade f??sica de interesse ?? single (ex: temperatura - C, energia - J)
                        #soma-se 7 no matches_end para checar o restante do texto (ex: C min-1)
                        if first_to_get == 'parameters':
                            begin, end = matches_start + local_cumulative_cut , matches_end + local_cumulative_cut + 7
                        elif first_to_get == 'numbers':
                            begin, end = matches_start + cumulative_cut_numbers , matches_end + cumulative_cut_numbers + 7
                        match_numbers_not_to_find = re.search(parameters_patterns['PU_not_to_find_regex'] , 
                                                              text[ begin : end ] )
                        #print('text to test for PUs_not_to_find: ', text[ begin : end ] )
                        if match_numbers_not_to_find:                            
                            #print('match_numbers_not_to_find: ', match_numbers_not_to_find)
                            #print('(breking inner while)')
                            if first_to_get == 'parameters':
                                #adicionando o corte cumulativo para os n??meros
                                cumulative_cut_numbers = local_cumulative_cut + matches_end                            
                                #atualizando o local_cumulative_cut para a proxima varredura dentro do inner while
                                local_cumulative_cut = cumulative_cut_numbers
                            time.sleep(1)
                            break
                    
                    #checando se padr??es n??mericos que n??o devem ser encontrados esteja no match
                    if first_to_get == 'parameters':
                        begin, end = matches_start + local_cumulative_cut - 3 , matches_end + local_cumulative_cut + 2
                    elif first_to_get == 'numbers':
                        begin, end = matches_start + cumulative_cut_numbers - 3 , matches_end + cumulative_cut_numbers + 2                 
                    match_for_additional_numerical_patterns = re.search(parameters_patterns['aditional_numbers_not_to_find'],
                                                                        text[ begin : end ] )
                    if match_for_additional_numerical_patterns:
                        #print('text to test for additional numerical patterns: ', text[ begin : end ] )
                        ##print('pattern_to_find_additional_numebers: ', parameters_patterns['aditional_numbers_not_to_find'] )
                        #print('match_for_additional_numerical_patterns: ', match_for_additional_numerical_patterns)
                        #adicionando o corte cumulativo para os n??meros
                        if first_to_get == 'parameters':
                            cumulative_cut_numbers = local_cumulative_cut + matches_end                            
                            #atualizando o local_cumulative_cut para a proxima varredura dentro do inner while
                            local_cumulative_cut = cumulative_cut_numbers
                        elif first_to_get == 'numbers':
                            cumulative_cut_numbers += matches_end
                        continue
                    
                    #caso ainda haja number para procurar
                    if inner_matches < n_numbers_matches_to_find:
                        
                        #coletando os n??meros                        
                        match_text = match_numbers_to_find.captures()[0]
                        PUs_found_list_temp1 , num_list_extracted_raw_temp = self.extract_PU_and_numbers_from_text( match_text )    
                        num_list_extracted_raw.extend( num_list_extracted_raw_temp )
                        #coletando as PUs
                        PUs_found_list_temp2.extend( PUs_found_list_temp1 )
                        #print('num_list_extracted_raw: ', num_list_extracted_raw)
                        #print('PUs_found_list_temp2: ', PUs_found_list_temp2)
                        #time.sleep(2)
    
                        #atualizando os inner matches
                        inner_matches += 1                        
                        #adicionando o corte cumulativo para os n??meros
                        if first_to_get == 'parameters':
                            cumulative_cut_numbers = local_cumulative_cut + matches_end
                            #atualizando o local_cumulative_cut para a proxima varredura dentro do inner while
                            local_cumulative_cut = cumulative_cut_numbers
                        elif first_to_get == 'numbers':
                            cumulative_cut_numbers += matches_end
                            local_cumulative_cut = cumulative_cut_numbers
                        #print('inner_matches: ', inner_matches)
                        #print('n_numbers_matches_to_find: ', n_numbers_matches_to_find)
                        continue
                    
                    #saindo do inner                        
                    else:
                        #adicionando o corte cumulativo para os n??meros
                        if first_to_get == 'parameters':
                            cumulative_cut_numbers = local_cumulative_cut + 1
                            #atualizando o local_cumulative_cut para a proxima varredura dentro do inner while
                            local_cumulative_cut = cumulative_cut_numbers
                        elif first_to_get == 'numbers':
                            cumulative_cut_numbers += 1
                            local_cumulative_cut = cumulative_cut_numbers
                        #print('(breking inner while)')
                        break
    
                #caso n??o foi encontrado um padr??o num??rico mas j?? foi encontrado o pr??xima par??metro
                elif match_numbers_to_find is None and match_next_parameters_to_find is not None:
                    export_extracted = False
                    break_outer_while = True
                    #print('(breking inner while)')
                    break
    
                #saindo do inner while e do outer while
                else:
                    for pu in PUs_found_list_temp2:
                        if pu not in PUs_found_list:
                            PUs_found_list.append(pu)
                    break_outer_while = True
                    #print('(breking inner while)')
                    break
                
            #saindo do outer while
            if break_outer_while is True:
                #print('(breking outer while)')
                break
            
    
        #print('Parameters extracted: ', parameters_found)
        #print('Numbers extracted: ', num_list_extracted_raw)
        #print('PUs extracted: ', PUs_found_list)
        #time.sleep(2)
        
        #ap??s ter feito a extra????o no mode "all" ou "first"
        if len(num_list_extracted_raw) > 0 and ( len(num_list_extracted_raw) <= total_n_numbers_to_extract ) and export_extracted is True:
            
            #convertendo as PUs para SI        
            factor_to_multiply , factor_to_add , SI_units = self.extract_PU_to_SI(PUs_found_list)
            #print('factor_to_multiply: ', factor_to_multiply)
            #print('factor_to_add: ', factor_to_add)
            #print('SI_units: ', SI_units)
                    
            #caso n??o tenha sido achada a unidade fisica
            if None in (factor_to_multiply , factor_to_add , SI_units):            
                #print('Erro! Os Par??metros num??ricos foram encontrados, mas houve erro no processamento das unidades f??sicas PUs.')
                return extracted_dic
            
            else:                
                #quantos dados num??ricos foram extraidos
                extracted_dic['n_num_outputs_extracted'] = len(num_list_extracted_raw)
                
                #coletando os dados num??ricos
                num_list_extracted_converted = []
                #varrendo os valores num??ricos capturados
                for num in num_list_extracted_raw:
                    if '&' in num:
                        numbers_interval = re.findall(r'-?[0-9]+\.?[0-9]*', num)
                        first_number = round( float(numbers_interval[0]) * factor_to_multiply + factor_to_add , 10)
                        last_number = round( float(numbers_interval[1]) * factor_to_multiply + factor_to_add , 10)
                        num_list_extracted_converted.append( str(first_number) )
                        num_list_extracted_converted.append( str(last_number) )
                    else:
                        num_list_extracted_converted.append( str( round( float(num) * factor_to_multiply + factor_to_add, 10) ) )
    
                #caso haja procura de par??metro no texto (ex: C#O, C#H, C, O)
                if parameters_patterns['parameters_pattern_to_find_in_sent'] is not None:
                    parameters_list = []
                    try:                        
                        #caso s?? haja um par??metro e a quantidade de n??meros extra??dos pode ser 1
                        if len(parameters_found) == 1 and len(num_list_extracted_converted) == 1:
                            parameters_list = parameters_found
                            extracted_dic['SI_units'] = SI_units
                        #caso s?? haja um par??metro e a quantidade de n??meros extra??dos pode ser 2
                        elif len(parameters_found) == 1 and len(num_list_extracted_converted) == 2:
                            parameters_list = parameters_found * len(num_list_extracted_converted)
                            extracted_dic['SI_units'] = SI_units
                        #o n??mero de par??metros encontrados deve ser igual ?? quantidade de n??meros extra??dos
                        elif len(parameters_found) > 1 and (len(parameters_found) == len(num_list_extracted_converted)):
                            parameters_list = parameters_found
                            extracted_dic['SI_units'] = SI_units
                        #o n??mero de par??metros encontrados deve ser a metade de n??meros extra??dos
                        elif len(parameters_found) > 1 and (2*len(parameters_found) == len(num_list_extracted_converted)):
                            #checando se o formato da senten??a ??: param1 and param2 >>> number1, number 2, and number 3 and number 4
                            #if min(cumulative_cut_numbers_list) > max(cumulative_cut_parameters_list):
                            parameters_list = []
                            for i in [[param]*2 for param in parameters_found]:
                                parameters_list.extend(i)
                            extracted_dic['SI_units'] = SI_units
                            
                    except UnboundLocalError:
                        pass
                
                #caso n??o haja procura de par??metro no texto
                else:                                        
                    extracted_dic['SI_units'] = SI_units
                    #definindo a lista com o par??metro repetido (ex: [ mV , mV , mV ])
                    parameters_list = [parameter] * len(num_list_extracted_converted)
            
            #adicionando os sufixos aos par??metros
            parameters_list = [param + parameters_patterns['parameter_suffix'] for param in parameters_list]
            
            #caso terha sido extra??dos os par??metros corretamente
            print('final SI_units: ', extracted_dic['SI_units'])
            print('final parameters_list: ', parameters_list)
            print('final num_list: ', num_list_extracted_converted)
            if extracted_dic['SI_units']:
                            
                #definindo o dic para coletar os par??metros num??ricos
                extracted_dic['num_output'] = {}                
                    
                #varrendo os resultados extra??dos
                index_counter = 0
                for i in range( len(parameters_list) ):
                    #caso seja a primeira inst??ncia do key
                    try:
                        extracted_dic['num_output'][parameters_list[i]]
                        index_counter += 1
                    except KeyError:
                        index_counter = 0
                        extracted_dic['num_output'][parameters_list[i]] = []
                    
                    extracted_dic['num_output'][parameters_list[i]].append( [ index_counter , text_index, num_list_extracted_converted[i]] )
                                        
                #fazendo o sorting da lista (se faz o sort pois esses valores servir??o para compara????o)
                num_list_extracted_converted.sort()
                #exportando s?? os valores num??ricos
                extracted_dic['extracted_num_str_vals'] = num_list_extracted_converted
    
            else:
                print('\nErro na extra????o de par??metros/n??meros:')
                print('Par??metros extra??dos: ', parameters_found)
                print('N??meros extra??dos: ', num_list_extracted_converted, '\n')
    
        #time.sleep(1)
        return extracted_dic



    '''
    def extract_from_sent_first_parameter_sec_numbers(self, text, text_index, parameter, parameters_patterns, extract_mode = 'all'):

        print('\n> extract_from_sent_first_parameter_sec_numbers')
        
        import regex as re
        import time

        extracted_dic = {}
        extracted_dic['n_num_outputs_extracted'] = 0
        extracted_dic['SI_units'] = None
        extracted_dic['num_output'] = None

        #definindo a lista com as PUs encontradas na senten??a
        PUs_found_list = []            
        #definindo a lista com os valores num??ricos encontrados na senten??a
        num_list_extracted_raw = []            
        #definindo os par??metros que podem ser extra??dos da senten??a
        parameters_found = []
        
        #existem tres cumulatives cuts
        cumulative_cut_numbers  = 0
        cumulative_cut_parameters  = 0
        total_n_numbers_to_extract = 0
        PUs_found_list_temp1 = []
        PUs_found_list_temp2 = []
        
        #check para exportar os dados extraidos
        export_extracted = True
        
        #outer while
        counter_outer_while = 0        
        while True:
            
            counter_outer_while += 1
            print('\n> extract_from_sent_first_parameter_sec_numbers')
            print('\nscanning step (outer while): ', counter_outer_while)
            
            #zerando os checks
            need_to_find_parameter = False
            parameter_found = False
            parameters_extracted_from_sent = None
            match_next_parameters_to_find = None
            
            #determinando o trecho da senten??a a ser encontrado os par??metros
            cut_text = text[ cumulative_cut_numbers : ]
            #determinando o index do pr??ximo par??metro
            next_parameter_index = len(text)
            
            #caso seja feita a procura por par??metros no texto (ex: 'C content', 'H/C', etc)           
            if parameters_patterns['parameters_pattern_to_find_in_sent'] is not None:
                #parametro precisa ser encontrado
                need_to_find_parameter = True
                parameters_pattern = r'({text_to_find})+'.format(text_to_find = parameters_patterns['parameters_pattern_to_find_in_sent'])
                match_parameters_to_find = re.search(parameters_pattern, cut_text)
                print('\ncut_text to find parameters: ', cut_text)
                print('match_parameters_to_find: ', match_parameters_to_find)
                if match_parameters_to_find:
                    #parametro encontrado
                    parameter_found = True
                    #adicionando o corte cumulativo para os par??metros
                    matches_start, matches_end = match_parameters_to_find.span()
                    print('parameter_index: ', cumulative_cut_numbers + matches_end)
                    cumulative_cut_parameters = cumulative_cut_numbers + matches_end
                    
                    #extraindo os par??metros da senten??a
                    parameters_extracted_from_sent = self.extract_parameters_from_text( match_parameters_to_find.captures()[0] , parameters_patterns)
                    for param in parameters_extracted_from_sent:
                        if param not in parameters_found:
                            parameters_found.append( param )
                            
                    #procurando o pr??ximo par??metro
                    match_next_parameters_to_find = re.search( parameters_pattern, cut_text[ matches_end : ] )
                    print('text to find next parameters: ', cut_text[ matches_end : ])
                    print('match_next_parameters_to_find: ', match_next_parameters_to_find)
                    
                    #determinando o index do proximo par??metro na senten??a
                    if match_next_parameters_to_find:
                        next_parameter_index = cumulative_cut_numbers + matches_end + match_next_parameters_to_find.span()[1]
                        print('next_parameter_index: ', next_parameter_index)
                        print('text after next parameter found: ', cut_text[ matches_end + match_next_parameters_to_find.span()[1] : ])

                print('Par??metros extra??dos: ', parameters_found)
            
            #caso tenha que encontrar o par??metro mas ele n??o foi encontrado
            if need_to_find_parameter is True and parameter_found is False:
                break
            #caso tenha que encontrar o par??metro e ele foi encontrado
            elif need_to_find_parameter is True and parameter_found is True:
                #caso haja par??metros encontrados, a procura pelos n??meros come??a a partir do ultimo par??metro encontrado
                local_cumulative_cut = cumulative_cut_parameters
                #caso haja par??metros encontrados, a quantidade de n??meros a ser encontrada ser?? feita com m??ltiplos dos par??metros (multiplo de 2)
                n_numbers_matches_to_find = len(parameters_extracted_from_sent) * 2
                total_n_numbers_to_extract += n_numbers_matches_to_find
            #caso n??o tenha que se encontrar o par??metro
            else:                
                #caso n??o haja par??metros a serem encontrados, a procura pelos n??meros come??a a partir do ultimo n??mero encontrado
                local_cumulative_cut = cumulative_cut_numbers
                #caso n??o haja par??metros a serem encontrados, a quantidade de n??meros a ser encontrada ser?? de 1, 2 ou 3
                n_numbers_matches_to_find = 3
                total_n_numbers_to_extract = n_numbers_matches_to_find

            #inner while
            counter_inner_while = 0
            inner_matches = 0
            while True:
                
                break_outer_while = False
                counter_inner_while += 1
                print('\nscanning step (inner while): ', counter_inner_while)
            
                #procurando os n??meros
                cut_text = text[ local_cumulative_cut : ]
                match_numbers_to_find = re.search( parameters_patterns['PU_to_find_regex'] , cut_text )
                print('\ncut_text to find numbers: ', cut_text)
                print('match_numbers_to_find: ', match_numbers_to_find)
                
                #caso tenha sido encontrado a PU
                if match_numbers_to_find:
                    
                    matches_start, matches_end = match_numbers_to_find.span()
                    cumulative_cut = matches_end + local_cumulative_cut
                    #caso a posi????o do n??mero na senten??a seja menor que a posi????o do pr??ximo par??metro
                    print('number_index: ', cumulative_cut)
                    print('next_parameter_index: ', next_parameter_index)

                    #caso o n??mero encontrado esteja em index superior ao pr??ximo par??metro econtrado
                    if cumulative_cut > next_parameter_index:
                        print('parameters_found: ', parameters_found)
                        print('num_list_extracted_raw: ', num_list_extracted_raw)
                        #caso o par??metro tenha sido encontrado mas o n??mero n??o, rompe-se os dois whiles (inner e outer)
                        if len(parameters_found) > len(num_list_extracted_raw):
                            export_extracted = False
                            break_outer_while = True
                        print('(breking inner while)')
                        break

                    #caso se queira senten??as com um match somente
                    if extract_mode.lower() == 'first':
                        #checando se h?? mais uma parte que bate com o padr??o na mesma senten??a
                        print('text to find further numbers in sent (first mode): ', text[ cumulative_cut : ])
                        match_numbers_further_in_sent = re.search( parameters_patterns['PU_to_find_regex'] , text[ cumulative_cut : ])
                        if match_numbers_further_in_sent:
                            print('match_numbers_further_in_sent: ', match_numbers_further_in_sent)
                            break_outer_while = True
                            print('(breking inner while)')
                            #time.sleep(2)
                            break
                        
                    #checando se h?? termos que podem ou n??o podem ser encontrados antes dos n??meros e PUs (prefixes)
                    prefix_text = text[ cumulative_cut_numbers :  cumulative_cut ]
                    match_prefixes_to_find = re.search(parameters_patterns['pattern_variation_parameter'], prefix_text)
                    print('prefix_text to search: ', prefix_text)
                    print('match_prefixes_to_find: ', match_prefixes_to_find)
                    
                    #caso tenha que ser encontrado (ex: increased by)
                    if (match_prefixes_to_find is not None) and parameters_patterns['find_variation_parameter'] is True:
                        print('Prefix found: ', match_prefixes_to_find is not None, ' ; find_variation: ', parameters_patterns['find_variation_parameter'])
                        print('passing...')
                        #time.sleep(10)
                        pass
                    
                    #caso n??o tenha que ser encontrado (ex: increased by)                
                    elif (match_prefixes_to_find is None) and parameters_patterns['find_variation_parameter'] is False:
                        print('Prefix found: ', match_prefixes_to_find is not None, ' ; find_variation: ', parameters_patterns['find_variation_parameter'])
                        print('passing...')
                        #time.sleep(10)
                        pass
                    
                    else:
                        #adicionando o corte cumulativo para os n??meros
                        cumulative_cut_numbers = local_cumulative_cut + matches_end
                        #caso haja par??metros extraidos da senten??a, se subtrai o ultimo parametro encontrado
                        if parameters_extracted_from_sent is not None:
                            parameters_found = parameters_found[ :-1]
                            total_n_numbers_to_extract -= n_numbers_matches_to_find                            
                        print('Prefix found: ', match_prefixes_to_find is not None, ' ; find_variation: ', parameters_patterns['find_variation_parameter'])                        
                        print('continuing...')
                        print('(breking inner while)')
                        #time.sleep(2)
                        break
                    
                    #encontrando os par??metros com as unidades f??sicas que n??o s??o desejadas
                    #caso seja None, ?? porque a unidade f??sica de interesse ?? combinada (ex: concentra????o - mol L-1, taxa de aumento de temperatura C s-1)
                    if parameters_patterns['PU_not_to_find_regex'] is not None:
                        #caso n??o seja None, ?? porque a unidade f??sica de interesse ?? single (ex: temperatura - C, energia - J)
                        #soma-se 7 no matches_end para checar o restante do texto (ex: C min-1)
                        match_numbers_not_to_find = re.search(parameters_patterns['PU_not_to_find_regex'] , 
                                                              text[ matches_start + local_cumulative_cut : matches_end + local_cumulative_cut + 7 ] )
                        print('text to test for PUs_not_to_find: ', text[ matches_start + local_cumulative_cut : matches_end + local_cumulative_cut + 7 ] )
                        if match_numbers_not_to_find:                            
                            print('match_numbers_not_to_find: ', match_numbers_not_to_find)
                            print('(breking inner while)')
                            #adicionando o corte cumulativo para os n??meros
                            cumulative_cut_numbers = local_cumulative_cut + matches_end                            
                            #atualizando o local_cumulative_cut para a proxima varredura dentro do inner while
                            local_cumulative_cut = cumulative_cut_numbers
                            time.sleep(1)
                            break
                    
                    #checando se padr??es n??mericos que n??o devem ser encontrados esteja no match
                    match_for_additional_numerical_patterns = re.search(parameters_patterns['aditional_numbers_not_to_find'],
                                                                        text[ matches_start + local_cumulative_cut - 3 : matches_end + local_cumulative_cut + 2 ] )
                    if match_for_additional_numerical_patterns:
                        print('text to test for additional numerical patterns: ', text[ matches_start + local_cumulative_cut - 3 : matches_end + local_cumulative_cut + 2 ] )
                        #print('pattern_to_find_additional_numebers: ', parameters_patterns['aditional_numbers_not_to_find'] )
                        print('match_for_additional_numerical_patterns: ', match_for_additional_numerical_patterns)
                        #adicionando o corte cumulativo para os n??meros
                        cumulative_cut_numbers = local_cumulative_cut + matches_end                            
                        #atualizando o local_cumulative_cut para a proxima varredura dentro do inner while
                        local_cumulative_cut = cumulative_cut_numbers
                        continue
                    
                    #caso ainda haja number para procurar
                    if inner_matches < n_numbers_matches_to_find:
                        
                        #coletando os n??meros                        
                        match_text = match_numbers_to_find.captures()[0]
                        PUs_found_list_temp1 , num_list_extracted_raw_temp = self.extract_PU_and_numbers_from_text( match_text )    
                        num_list_extracted_raw.extend( num_list_extracted_raw_temp )
                        #coletando as PUs
                        PUs_found_list_temp2.extend( PUs_found_list_temp1 )
                        print('num_list_extracted_raw: ', num_list_extracted_raw)
                        print('PUs_found_list_temp2: ', PUs_found_list_temp2)
                        #time.sleep(2)

                        #atualizando os inner matches
                        inner_matches += 1                        
                        #adicionando o corte cumulativo para os n??meros
                        cumulative_cut_numbers = local_cumulative_cut + matches_end
                        #atualizando o local_cumulative_cut para a proxima varredura dentro do inner while
                        local_cumulative_cut = cumulative_cut_numbers
                        print('inner_matches: ', inner_matches)
                        print('n_numbers_matches_to_find: ', n_numbers_matches_to_find)
                        continue
                    
                    #saindo do inner                        
                    else:
                        #adicionando o corte cumulativo para os n??meros
                        cumulative_cut_numbers = local_cumulative_cut + 1
                        #atualizando o local_cumulative_cut para a proxima varredura dentro do inner while
                        local_cumulative_cut = cumulative_cut_numbers
                        print('(breking inner while)')
                        break

                elif match_numbers_to_find is None and match_next_parameters_to_find is not None:
                    export_extracted = False
                    break_outer_while = True
                    print('(breking inner while)')
                    break

                #saindo do inner while e do outer while
                else:
                    for pu in PUs_found_list_temp2:
                        if pu not in PUs_found_list:
                            PUs_found_list.append(pu)
                    break_outer_while = True
                    print('(breking inner while)')
                    break
                
            #saindo do outer while
            if break_outer_while is True:
                print('(breking outer while)')
                break
            

        print('Parameters extracted: ', parameters_found)
        print('Numbers extracted: ', num_list_extracted_raw)
        print('PUs extracted: ', PUs_found_list)
        #time.sleep(2)
        
        #ap??s ter feito a extra????o no mode "all" ou "first"
        if len(num_list_extracted_raw) > 0 and ( len(num_list_extracted_raw) <= total_n_numbers_to_extract ) and export_extracted is True:
            
            #convertendo as PUs para SI        
            factor_to_multiply , factor_to_add , SI_units = self.extract_PU_to_SI(PUs_found_list)
            print('factor_to_multiply: ', factor_to_multiply)
            print('factor_to_add: ', factor_to_add)
            print('SI_units: ', SI_units)
                    
            #caso n??o tenha sido achada a unidade fisica
            if None in (factor_to_multiply , factor_to_add , SI_units):            
                print('Erro! Os Par??metros num??ricos foram encontrados, mas houve erro no processamento das unidades f??sicas PUs.')
                return extracted_dic
            
            else:                
                #quantos dados num??ricos foram extraidos
                extracted_dic['n_num_outputs_extracted'] = len(num_list_extracted_raw)
                
                #coletando os dados num??ricos
                num_list_extracted_converted = []
                #varrendo os valores num??ricos capturados
                for num in num_list_extracted_raw:
                    if '&' in num:
                        numbers_interval = re.findall(r'-?[0-9]+\.?[0-9]*', num)
                        first_number = round( float(numbers_interval[0]) * factor_to_multiply + factor_to_add , 10)
                        last_number = round( float(numbers_interval[1]) * factor_to_multiply + factor_to_add , 10)
                        num_list_extracted_converted.append( str(first_number) )
                        num_list_extracted_converted.append( str(last_number) )
                    else:
                        num_list_extracted_converted.append( str( round( float(num) * factor_to_multiply + factor_to_add, 10) ) )

                #caso haja procura de par??metro no texto (ex: C#O, C#H, C, O)
                if parameters_patterns['parameters_pattern_to_find_in_sent'] is not None:
                    parameters_list = []
                    try:                        
                        #caso s?? haja um par??metro e a quantidade de n??meros extra??dos pode ser 1
                        if len(parameters_found) == 1 and len(num_list_extracted_converted) == 1:
                            parameters_list = parameters_found
                            extracted_dic['SI_units'] = SI_units
                        #caso s?? haja um par??metro e a quantidade de n??meros extra??dos pode ser 2
                        elif len(parameters_found) == 1 and len(num_list_extracted_converted) == 2:
                            parameters_list = parameters_found * len(num_list_extracted_converted)
                            extracted_dic['SI_units'] = SI_units
                        #o n??mero de par??metros encontrados deve ser igual ?? quantidade de n??meros extra??dos
                        elif len(parameters_found) > 1 and (len(parameters_found) == len(num_list_extracted_converted)):
                            parameters_list = parameters_found
                            extracted_dic['SI_units'] = SI_units
                        #o n??mero de par??metros encontrados deve ser a metade de n??meros extra??dos
                        elif len(parameters_found) > 1 and (2*len(parameters_found) == len(num_list_extracted_converted)):
                            #checando se o formato da senten??a ??: param1 and param2 >>> number1, number 2, and number 3 and number 4
                            #if min(cumulative_cut_numbers_list) > max(cumulative_cut_parameters_list):
                            parameters_list = []
                            for i in [[param]*2 for param in parameters_found]:
                                parameters_list.extend(i)
                            extracted_dic['SI_units'] = SI_units
                            
                    except UnboundLocalError:
                        pass
                
                #caso n??o haja procura de par??metro no texto
                else:                                        
                    extracted_dic['SI_units'] = SI_units
                    #definindo a lista com o par??metro repetido (ex: [ mV , mV , mV ])
                    parameters_list = [parameter] * len(num_list_extracted_converted)
            
            #adicionando os sufixos aos par??metros
            parameters_list = [param + parameters_patterns['parameter_suffix'] for param in parameters_list]
            
            #caso terha sido extra??dos os par??metros corretamente
            print('final SI_units: ', extracted_dic['SI_units'])
            print('final parameters_list: ', parameters_list)
            print('final num_list: ', num_list_extracted_converted)
            if extracted_dic['SI_units']:
                            
                #definindo o dic para coletar os par??metros num??ricos
                extracted_dic['num_output'] = {}                
                    
                #varrendo os resultados extra??dos
                index_counter = 0
                for i in range( len(parameters_list) ):
                    #caso seja a primeira inst??ncia do key
                    try:
                        extracted_dic['num_output'][parameters_list[i]]
                        index_counter += 1
                    except KeyError:
                        index_counter = 0
                        extracted_dic['num_output'][parameters_list[i]] = []
                    
                    extracted_dic['num_output'][parameters_list[i]].append( [ index_counter , text_index, num_list_extracted_converted[i]] )
                                        
                #fazendo o sorting da lista (se faz o sort pois esses valores servir??o para compara????o)
                num_list_extracted_converted.sort()
                #exportando s?? os valores num??ricos
                extracted_dic['extracted_num_str_vals'] = num_list_extracted_converted
    
            else:
                print('\nErro na extra????o de par??metros/n??meros:')
                print('Par??metros extra??dos: ', parameters_found)
                print('N??meros extra??dos: ', num_list_extracted_converted, '\n')

        #time.sleep(1)
        return extracted_dic



    def extract_from_sent_first_numbers_sec_parameters(self, text, text_index, parameter, parameters_patterns, extract_mode = 'all'):
        
        print('\n> extract_from_sent_first_numbers_sec_parameters')
        
        import regex as re
        import time

        extracted_dic = {}
        extracted_dic['n_num_outputs_extracted'] = 0
        extracted_dic['SI_units'] = None
        extracted_dic['num_output'] = None

        #definindo a lista com as PUs encontradas na senten??a
        PUs_found_list = []            
        #definindo a lista com os valores num??ricos encontrados na senten??a
        num_list_extracted_raw = []            
        #definindo os par??metros que podem ser extra??dos da senten??a
        parameters_found = []
        
        #existem tres cumulatives cuts
        cumulative_cut_numbers  = 0
        cumulative_cut_parameters  = 0
        total_n_numbers_to_extract = 0
        PUs_found_list_temp1 = []
        PUs_found_list_temp2 = []

        #check para exportar os dados extraidos
        export_extracted = True
        
        #outer while
        counter_outer_while = 0
        while True:
            
            counter_outer_while += 1
            print('\n> extract_from_sent_first_numbers_sec_parameters')
            print('\nscanning step (outer while): ', counter_outer_while)
            
            #zerando os checks
            parameter_found = False
            parameters_extracted_from_sent = None
            match_next_parameters_to_find = None
            
            #determinando o trecho da senten??a a ser encontrado os par??metros
            cut_text = text[ cumulative_cut_parameters : ]
            
            #parametro precisa ser encontrado
            parameters_pattern = r'({text_to_find})+'.format(text_to_find = parameters_patterns['parameters_pattern_to_find_in_sent'])
            match_parameters_to_find = re.search(parameters_pattern, cut_text)
            print('\ncut_text to find parameters: ', cut_text)
            print('match_parameters_to_find: ', match_parameters_to_find)

            if match_parameters_to_find:
                #parametro encontrado
                parameter_found = True
                #adicionando o corte cumulativo para os par??metros
                matches_start, matches_end = match_parameters_to_find.span()
                cumulative_cut_parameters += matches_end
                print('parameter_index: ', cumulative_cut_parameters)
                
                #extraindo os par??metros da senten??a
                parameters_extracted_from_sent = self.extract_parameters_from_text( match_parameters_to_find.captures()[0] , parameters_patterns)
                for param in parameters_extracted_from_sent:
                    if param not in parameters_found:
                        parameters_found.append( param )
                        
                #procurando o pr??ximo par??metro
                match_next_parameters_to_find = re.search( parameters_pattern, cut_text[ matches_end : ] )
                print('text to find next parameters: ', cut_text[ matches_end : ])
                print('match_next_parameters_to_find: ', match_next_parameters_to_find)
                
                #determinando o index do proximo par??metro na senten??a
                if match_next_parameters_to_find:
                    print('next_parameter_index: ', matches_end + match_next_parameters_to_find.span()[1])
                    print('text after next parameter found: ', cut_text[ matches_end + match_next_parameters_to_find.span()[1] : ])
                    pass

            print('Par??metros extra??dos: ', parameters_found)
        
            #caso o par??metro n??o foi encontrado
            if parameter_found is False:
                break
            #caso o par??metro foi encontrado
            elif parameter_found is True:
                #caso haja par??metros encontrados, a quantidade de n??meros a ser encontrada ser?? feita com m??ltiplos dos par??metros (multiplo de 2)
                n_numbers_matches_to_find = len(parameters_extracted_from_sent) * 2
                total_n_numbers_to_extract += n_numbers_matches_to_find
                        
            #inner while
            counter_inner_while = 0
            inner_matches = 0
            while True:
                
                break_outer_while = False
                counter_inner_while += 1
                print('\nscanning step (inner while): ', counter_inner_while)
            
                #procurando os n??meros
                cut_text = text[ cumulative_cut_numbers : ]
                match_numbers_to_find = re.search( parameters_patterns['PU_to_find_regex'] , cut_text )
                print('\ncut_text to find numbers: ', cut_text)
                print('match_numbers_to_find: ', match_numbers_to_find)
                
                #caso tenha sido encontrado a PU
                if match_numbers_to_find:
                    
                    matches_start, matches_end = match_numbers_to_find.span()
                    cumulative_cut = matches_end + cumulative_cut_numbers
                    #caso a posi????o do n??mero na senten??a seja menor que a posi????o do par??metro
                    print('number_index: ', cumulative_cut)
                    print('parameter_index: ', cumulative_cut_parameters)                    

                    #caso o n??mero encontrado esteja em index superior ao par??metro
                    if cumulative_cut > cumulative_cut_parameters:
                        print('parameters_found: ', parameters_found)
                        print('num_list_extracted_raw: ', num_list_extracted_raw)
                        #caso o par??metro tenha sido encontrado mas o n??mero n??o, rompe-se os dois whiles (inner e outer)
                        if len(parameters_found) > len(num_list_extracted_raw):
                            export_extracted = False
                            break_outer_while = True
                        print('(breking inner while)')
                        break

                    #caso se queira senten??as com um match somente
                    if extract_mode.lower() == 'first':
                        #checando se h?? mais uma parte que bate com o padr??o na mesma senten??a
                        print('text to find further numbers in sent (first mode): ', text[ cumulative_cut : ])
                        match_numbers_further_in_sent = re.search( parameters_patterns['PU_to_find_regex'] , text[ cumulative_cut : ])
                        if match_numbers_further_in_sent:
                            print('match_numbers_further_in_sent: ', match_numbers_further_in_sent)
                            break_outer_while = True
                            print('(breking inner while)')
                            #time.sleep(2)
                            break
                        
                    #checando se h?? termos que podem ou n??o podem ser encontrados antes dos n??meros e PUs (prefixes)
                    prefix_text = text[ cumulative_cut_numbers :  cumulative_cut ]
                    match_prefixes_to_find = re.search(parameters_patterns['pattern_variation_parameter'], prefix_text)
                    print('prefix_text to search: ', prefix_text)
                    print('match_prefixes_to_find: ', match_prefixes_to_find)
                    
                    #caso tenha que ser encontrado (ex: increased by)
                    if (match_prefixes_to_find is not None) and parameters_patterns['find_variation_parameter'] is True:
                        print('Prefix found: ', match_prefixes_to_find is not None, ' ; find_variation: ', parameters_patterns['find_variation_parameter'])
                        print('passing...')
                        #time.sleep(10)
                        pass
                    
                    #caso n??o tenha que ser encontrado (ex: increased by)                
                    elif (match_prefixes_to_find is None) and parameters_patterns['find_variation_parameter'] is False:
                        print('Prefix found: ', match_prefixes_to_find is not None, ' ; find_variation: ', parameters_patterns['find_variation_parameter'])
                        print('passing...')
                        #time.sleep(10)
                        pass
                    
                    else:
                        #adicionando o corte cumulativo para os n??meros
                        cumulative_cut_numbers += matches_end
                        #caso haja par??metros extraidos da senten??a, se subtrai o ultimo parametro encontrado
                        if parameters_extracted_from_sent is not None:
                            parameters_found = parameters_found[ :-1]
                            total_n_numbers_to_extract -= n_numbers_matches_to_find
                            #indo para um index ap??s o primeiro par??metro
                            cumulative_cut_numbers = cumulative_cut_parameters
                        print('Prefix found: ', match_prefixes_to_find is not None, ' ; find_variation: ', parameters_patterns['find_variation_parameter'])
                        print('continuing...')
                        print('(breking inner while)')
                        #time.sleep(2)
                        break
                    
                    #encontrando os par??metros com as unidades f??sicas que n??o s??o desejadas
                    #caso seja None, ?? porque a unidade f??sica de interesse ?? combinada (ex: concentra????o - mol L-1, taxa de aumento de temperatura C s-1)
                    if parameters_patterns['PU_not_to_find_regex'] is not None:
                        #caso n??o seja None, ?? porque a unidade f??sica de interesse ?? single (ex: temperatura - C, energia - J)
                        #soma-se 7 no matches_end para checar o restante do texto (ex: C min-1)
                        match_numbers_not_to_find = re.search(parameters_patterns['PU_not_to_find_regex'] , 
                                                              text[ matches_start + cumulative_cut_numbers : matches_end + cumulative_cut_numbers + 7 ] )
                        print('text to test for PUs_not_to_find: ', text[ matches_start + cumulative_cut_numbers : matches_end + cumulative_cut_numbers + 7 ] )
                        if match_numbers_not_to_find:                            
                            print('match_numbers_not_to_find: ', match_numbers_not_to_find)
                            print('(breking inner while)')
                            time.sleep(1)
                            break

                    #checando se padr??es n??mericos que n??o devem ser encontrados esteja no match
                    match_for_additional_numerical_patterns = re.search(parameters_patterns['aditional_numbers_not_to_find'] , 
                                                                        text[ matches_start + cumulative_cut_numbers - 3 : matches_end + cumulative_cut_numbers + 2 ] )
                    if match_for_additional_numerical_patterns:
                        print('text to test for additional numerical patterns: ', text[ matches_start + cumulative_cut_numbers - 3 : matches_end + cumulative_cut_numbers + 2 ] )
                        #print('pattern_to_find_additional_numebers: ', parameters_patterns['aditional_numbers_not_to_find'] )
                        print('match_for_additional_numerical_patterns: ', match_for_additional_numerical_patterns)
                        #adicionando o corte cumulativo para os n??meros
                        cumulative_cut_numbers += matches_end
                        continue                    
                    
                    #caso o n??mero de par??metros extra??dos seja menor que o n_numbers_matches_to_find e a posi????o na senten??a seja menor que a posi????o do pr??ximo par??metro
                    print('cumulative_cut: ', cumulative_cut)
                    print('cumulative_cut_parameters: ', cumulative_cut_parameters)
                                        
                    #caso ainda haja num??ricos para procurar
                    if inner_matches < n_numbers_matches_to_find:
                        
                        #coletando os n??meros                        
                        match_text = match_numbers_to_find.captures()[0]
                        PUs_found_list_temp1 , num_list_extracted_raw_temp = self.extract_PU_and_numbers_from_text( match_text )    
                        num_list_extracted_raw.extend( num_list_extracted_raw_temp )
                        #coletando as PUs
                        PUs_found_list_temp2.extend( PUs_found_list_temp1 )
                        print('num_list_extracted_raw: ', num_list_extracted_raw)
                        print('PUs_found_list_temp2: ', PUs_found_list_temp2)
                        #time.sleep(2)

                        #atualizando os inner matches
                        inner_matches += 1                        
                        #adicionando o corte cumulativo para os n??meros
                        cumulative_cut_numbers += matches_end
                        print('inner_matches: ', inner_matches)
                        print('n_numbers_matches_to_find: ', n_numbers_matches_to_find)
                        continue
                    
                    #saindo do inner e continuando com o outer while
                    else:
                        #adicionando o corte cumulativo para os n??meros
                        cumulative_cut_numbers += 1
                        print('(breking inner while)')
                        break
                
                #caso n??o foi encontrado um padr??o num??rico mas j?? foi encontrado o pr??xima par??metro
                elif match_numbers_to_find is None and match_next_parameters_to_find is not None:
                    export_extracted = False
                    break_outer_while = True
                    print('(breking inner while)')
                    break

                #saindo do inner while e do outer while (final)
                else:
                    for pu in PUs_found_list_temp2:
                        if pu not in PUs_found_list:
                            PUs_found_list.append(pu)
                    break_outer_while = True
                    print('(breking inner while)')
                    break
                
            #saindo do outer while
            if break_outer_while is True:
                print('(breking outer while)')
                break

        print('Parameters extracted: ', parameters_found)
        print('Numbers extracted: ', num_list_extracted_raw)
        print('PUs extracted: ', PUs_found_list)
        #time.sleep(2)
        
        #ap??s ter feito a extra????o no mode "all" ou "first"
        if len(num_list_extracted_raw) > 0 and len(num_list_extracted_raw) <= total_n_numbers_to_extract and export_extracted is True:
            
            #convertendo as PUs para SI        
            factor_to_multiply , factor_to_add , SI_units = self.extract_PU_to_SI(PUs_found_list)
            print('factor_to_multiply: ', factor_to_multiply)
            print('factor_to_add: ', factor_to_add)
            print('SI_units: ', SI_units)
                    
            #caso n??o tenha sido achada a unidade fisica
            if None in (factor_to_multiply , factor_to_add , SI_units):            
                print('Erro! Os Par??metros num??ricos foram encontrados, mas houve erro no processamento das unidades f??sicas PUs.')
                return extracted_dic
            
            else:                
                #quantos dados num??ricos foram extraidos
                extracted_dic['n_num_outputs_extracted'] = len(num_list_extracted_raw)
                
                #coletando os dados num??ricos
                num_list_extracted_converted = []
                #varrendo os valores num??ricos capturados
                for num in num_list_extracted_raw:
                    if '&' in num:
                        numbers_interval = re.findall(r'-?[0-9]+\.?[0-9]*', num)
                        first_number = round( float(numbers_interval[0]) * factor_to_multiply + factor_to_add , 10)
                        last_number = round( float(numbers_interval[1]) * factor_to_multiply + factor_to_add , 10)
                        num_list_extracted_converted.append( str(first_number) )
                        num_list_extracted_converted.append( str(last_number) )
                    else:
                        num_list_extracted_converted.append( str( round( float(num) * factor_to_multiply + factor_to_add, 10) ) )

                #caso haja procura de par??metro no texto (ex: C#O, C#H, C, O)
                if parameters_patterns['parameters_pattern_to_find_in_sent'] is not None:
                    parameters_list = []
                    try:                        
                        #caso s?? haja um par??metro e a quantidade de n??meros extra??dos pode ser 1
                        if len(parameters_found) == 1 and len(num_list_extracted_converted) == 1:
                            parameters_list = parameters_found
                            extracted_dic['SI_units'] = SI_units
                        #caso s?? haja um par??metro e a quantidade de n??meros extra??dos pode ser 2
                        elif len(parameters_found) == 1 and len(num_list_extracted_converted) == 2:
                            parameters_list = parameters_found * len(num_list_extracted_converted)
                            extracted_dic['SI_units'] = SI_units
                        #o n??mero de par??metros encontrados deve ser igual ?? quantidade de n??meros extra??dos
                        elif len(parameters_found) > 1 and (len(parameters_found) == len(num_list_extracted_converted)):
                            parameters_list = parameters_found
                            extracted_dic['SI_units'] = SI_units
                        #o n??mero de par??metros encontrados deve ser a metade de n??meros extra??dos
                        elif len(parameters_found) > 1 and (2*len(parameters_found) == len(num_list_extracted_converted)):
                            #checando se o formato da senten??a ??: param1 and param2 >>> number1, number 2, and number 3 and number 4
                            #if min(cumulative_cut_numbers_list) > max(cumulative_cut_parameters_list):
                            parameters_list = []
                            for i in [[param]*2 for param in parameters_found]:
                                parameters_list.extend(i)
                            extracted_dic['SI_units'] = SI_units
                            
                    except UnboundLocalError:
                        pass
                
                #caso n??o haja procura de par??metro no texto
                else:                                        
                    extracted_dic['SI_units'] = SI_units
                    #definindo a lista com o par??metro repetido (ex: [ mV , mV , mV ])
                    parameters_list = [parameter] * len(num_list_extracted_converted)

            #adicionando os sufixos aos par??metros
            parameters_list = [param + parameters_patterns['parameter_suffix'] for param in parameters_list]
            
            #caso terha sido extra??dos os par??metros corretamente
            print('final SI_units: ', extracted_dic['SI_units'])
            print('final parameters_list: ', parameters_list)
            print('final num_list: ', num_list_extracted_converted)
            if extracted_dic['SI_units']:
                            
                #definindo o dic para coletar os par??metros num??ricos
                extracted_dic['num_output'] = {}                
                    
                #varrendo os resultados extra??dos
                index_counter = 0
                for i in range( len(parameters_list) ):
                    #caso seja a primeira inst??ncia do key
                    try:
                        extracted_dic['num_output'][parameters_list[i]]
                        index_counter += 1
                    except KeyError:
                        index_counter = 0
                        extracted_dic['num_output'][parameters_list[i]] = []
                    
                    extracted_dic['num_output'][parameters_list[i]].append( [ index_counter , text_index, num_list_extracted_converted[i]] )
                                        
                #fazendo o sorting da lista (se faz o sort pois esses valores servir??o para compara????o)
                num_list_extracted_converted.sort()
                #exportando s?? os valores num??ricos
                extracted_dic['extracted_num_str_vals'] = num_list_extracted_converted
    
            else:
                print('\nErro na extra????o de par??metros/n??meros:')
                print('Par??metros extra??dos: ', parameters_found)
                print('N??meros extra??dos: ', num_list_extracted_converted, '\n')
        
        #time.sleep(1)
        return extracted_dic
        '''


    #fun????o de extra????o de n??meros e PUs
    def extract_PU_and_numbers_from_text(self, match_text):
        
        import regex as re
        
        num_list_extracted = []
        PUs_list = []
        #texto encontrado
        #print('numerical 1.', match_text)
        #extraindo os n??meros
        #eliminando os n??meros associados ??s PUs inversas (ex: min???1)
        numbers_found_in_text_p1 = re.sub( r'(?<=[a-z]????)[0-9]+', '', match_text )
        #print('numerical 2.', numbers_found_in_text_p1)
        #substituindo o sinal de menos (???) por um caracter que o python reconhe??a como negativo (-)
        numbers_found_in_text_p2 = re.sub( r'\???(?=[0-9])', '-', numbers_found_in_text_p1 )
        #print('numerical 3.', numbers_found_in_text_p2)
        #substituindo o "to" por "&"
        numbers_found_in_text_p3 = re.sub( r'\s+to\s+', '&', numbers_found_in_text_p2 )
        #print('numerical 4.', numbers_found_in_text_p3)
        #encontrando somente os n??meros
        numbers_find_list = re.findall( r'-?[0-9]+\.?[0-9]*&-?[0-9]+\.?[0-9]*(?!\s*[0-9]*\s*\(\s*[0-9]+)|-?[0-9]+\.?[0-9]*(?!\s*[0-9]*\s*\(\s*[0-9]+)' , 
                                       numbers_found_in_text_p3 )
        #print('numerical 5.', numbers_find_list)
        
        #seperando as unidades f??sicas dos n??meros (ex: 2h -> 2 h)
        for result_found in numbers_find_list:
            result_found_filtered1 = re.sub(r'[A-Za-z]', '', result_found)
            result_found_filtered2 = re.sub(r'\s', '', result_found_filtered1)
            num_list_extracted.append(result_found_filtered2)
        
        num_list_extracted_raw_filtered = []
        for raw_num in num_list_extracted:
            try:
                if round(float(raw_num), 10) != 0.0000000000:
                    num_list_extracted_raw_filtered.append(raw_num)
                else:
                    continue
            except ValueError:
                num_list_extracted_raw_filtered.append(raw_num)
        
        #print('Numbers: ', num_list_extracted_raw_filtered)
    
        #procurando a unidade f??sica encontrada para o par??metro introduzido
        #eliminando todos os n??meros com exce????o daqueles associados ??s PUs inversas (ex: 250)
        PUs_found_in_text_p1 = re.sub( r'(?<![a-z]????)[0-9]+', '', match_text )
        #print('numerical 6.', PUs_found_in_text_p1)
        #eliminando todos os sinais de menos (???) com exce????o daqueles associados ??s PUs inversas (ex: ???)
        PUs_found_in_text_p2 = re.sub( r'(?<![a-z])???', '', PUs_found_in_text_p1 )
        #print('numerical 7.', PUs_found_in_text_p2)
        #eliminando os separadores de unidades num??ricas "," ".", "&", "to" ou "and"
        PUs_found_in_text_p3 = re.sub( r'(or|to|and|\,|\.|;|&|\(|\))', '', PUs_found_in_text_p2 )
        #print('numerical 8.', PUs_found_in_text_p3)
        for pu in PUs_found_in_text_p3.split():
            if pu not in PUs_list:
                PUs_list.append(pu)
        #print('PUs: ', PUs_list)
        #time.sleep(2)
        
        return PUs_list , num_list_extracted_raw_filtered


    #fun????o de extra????o de par??metros
    def extract_parameters_from_text(self, match_text, parameters_patterns):
        
        import regex as re
        
        #print('match_parameters_to_find: ', match_parameters_to_find)
        #eliminando os separadores de unidades num??ricas "," ".", ou "and"
        parameter_found_in_text_p1 = re.sub( r'(\s|or|and|\,|\.|;)', '', match_text )
        #print('parameter 1. ', parameter_found_in_text_p1)
        parameters_ = re.findall(parameters_patterns['parameters_to_find_in_sent'], parameter_found_in_text_p1)
        #print('Parameters: ', parameters_)
        
        return parameters_

    
    #fun????o de extra????o de PUs
    def extract_PU_to_SI(self, PU_list):
        
        from functions_PARAMETERS import get_physical_units_converted_to_SI
        
        #caso o par??metro seja adimensional
        if len(PU_list) == 0:
            PU_units_found = ['adimensional']
        #caso o par??metro tenha unidade
        else:
            PU_units_found =[]
            for unit in PU_list:
                if unit not in PU_units_found:
                    PU_units_found.append(unit)
            
        print('Raw PUs encontradas: ', PU_units_found)
        #time.sleep(1)
        
        #adquirindo o factor num??rico de convers??o para colocar as unidades no SI
        factor_to_multiply , factor_to_add , SI_units = get_physical_units_converted_to_SI(PU_units_found)
    
        return factor_to_multiply , factor_to_add , SI_units


    def generate_search_report(self):
        
        import os
        import pandas as pd
        
        #abrindo o SE report            
        if os.path.exists(self.diretorio + '/Settings/SE_inputs.csv'):
            search_report_DF = pd.read_csv(self.diretorio + '/Settings/SE_inputs.csv', index_col = 0)

            search_report_DF.loc[self.input_DF_name, 'Total Extracted'] = self.search_report_dic['export']['total_finds']
            search_report_DF.loc[self.input_DF_name, 'PDF Extracted'] = self.search_report_dic['export']['pdf_finds']
            search_report_DF.loc[self.input_DF_name , 'export_status' ] = 'finished'

            search_report_DF.sort_index(inplace=True)
            search_report_DF.to_csv(self.diretorio + '/Settings/SE_inputs.csv')
            print('Salvando o SE report em ~/Settings/SE_inputs.csv')
                
