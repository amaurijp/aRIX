#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import regex as re
import time
import os

from bs4 import BeautifulSoup as bs # type: ignore
from bs4 import element # type: ignore

from FUNCTIONS import load_dic_from_json
from FUNCTIONS import save_dic_to_json

from functions_TEXTS import break_text_in_sentences
from functions_TEXTS import filter_chars
from functions_TEXTS import check_text_language



class XMLTEXT:

    def __init__(self, xml_file_name : str, 
                 folder_name : str = 'Articles_to_add', 
                 language : str = 'english', 
                 min_len_text : int = 1000, 
                 diretorio : str = None):

        print('\n( Class: XMLTEXT )')

        self.diretorio = diretorio
        self.xml_file_name = xml_file_name
        self.filtered_text = ''
        self.folder_name = folder_name
        self.match_category = True
        self.language = language
        self.min_len_text = min_len_text
        self.proc_error_type = None
        self.sentences = None
        self.publisher = None


    def process_document(self, apply_filters = True):
        
        #print('\n( Function: process_text )')

        
        print(f'\nProcessing: {self.xml_file_name}...')
        with open(self.diretorio + '/' + self.folder_name + '/' + self.xml_file_name + '.xml', 'r', encoding='utf-8') as file:
            self.soup_xml = bs(file, 'lxml-xml', from_encoding="utf-8")
            file.close()

        #identificando o publisher
        coredata_elsevier = self.soup_xml.find('full-text-retrieval-response')
        meta_acs = self.soup_xml.find('journal-meta')

        if meta_acs:
            publisher_name = meta_acs.find('publisher-name').get_text()
            if 'American' in publisher_name and 'Chemical'  in publisher_name and 'Society' in publisher_name:
                self.publisher = 'acs'

        elif coredata_elsevier.attrs['xmlns']:
            if 'elsevier' in coredata_elsevier.attrs['xmlns']:
                self.publisher = 'elsevier'

        else:
            print('\nERRO!\nPublisher não encontrado!')
            return


        #texto raw
        self.raw_text = ''

        if self.publisher == 'acs':

            #varrendo as seções
            sections_tag = self.soup_xml.find_all('sec')
            split_section = None
            last_name_section = None
            
            print('Looking for sections division')
            for sec in sections_tag:
                #try:
                #checando se o tag não é de uma subseção
                if re.search(r'[0-9](?=\.[0-9])', sec.attrs['id']) is None:
                
                    #coletando o nome da seção
                    section_name = sec.title.get_text().lower()
                    print('> section_name: ', section_name)

                    #a seção de introdução sempre aparece como INTRODUCION
                    if 'introduction' in section_name:
                        section_name = 'introduction'
                    
                    #a parte de resultados e discussão pode vir misturada
                    elif ( 'results' in section_name ) and ( 'discussion' in section_name ):
                        section_name = 'results-discussion'
                    
                    #quando os resultados vem separado
                    elif 'results' in section_name:
                        section_name = 'results'
                    
                    #quando a discussão vem separada
                    elif 'discussion' in section_name:
                        section_name = 'discussion'

                    #quando a conclusão vem separada
                    elif 'conclusion' in section_name:
                        section_name = 'conclusion'
                    
                    #a parte de methodologia tem muitas definições diferentes na coleção
                    else:
                        section_name = 'methodology'

                    #caso não seja a mesma seção
                    if section_name in ('introduction', 'results-discussion', 'results', 'discussion', 'conclusion', 'methodology'):
                        
                        if split_section != None:
                            print('> Closing section: ', split_section)
                            self.raw_text += f'The END of the section is here (separated from the XML file). '

                        print('> Opening section: ', section_name, ' ; Last section: ', last_name_section)
                        #print('Concatenating section to text: ', section_name)
                        self.raw_text += f'The BEGIN of the section is here (separated from the XML file) {section_name}. '
                        split_section = section_name
            
                #varrendo os parágrafos
                for p in sec.find_all('p'):
                    self.raw_text += p.get_text() + ' '

                last_name_section = section_name
            
                #se não for possível definir o nome da seção, passa-se para a próxima
                #except (AttributeError, KeyError):
                #    continue
            
            print('> Closing section: ', last_name_section)
            self.raw_text += f'The END of the section is here (separated from the XML file). '
            print('> Closing article...')
        
        
        elif self.publisher == 'elsevier':
            
            #varrendo as seções
            sections_tag = self.soup_xml.find_all('section-title')
            split_section = None
            last_name_section = None
            broke_with_conclusion = False
            broke_with_end_tags = False
            
            print('Looking for sections division')
            for sec in sections_tag:
                #try:
                #coletando o nome da seção
                section_name = sec.get_text().lower()
                print('> section_name: ', section_name)
                
                cond_to_leave1 = 'reference' in section_name
                cond_to_leave2 = 'supplementary' in section_name
                cond_to_leave3 = 'supporting' in section_name
                cond_to_leave4 = 'acknowledgment' in section_name
                if True in (cond_to_leave1, cond_to_leave2, cond_to_leave3, cond_to_leave4):
                    broke_with_end_tags = True
                    break

                #a seção de introdução sempre aparece como INTRODUCION
                if 'introduction' in section_name:
                    section_name = 'introduction'
                
                #a parte de resultados e discussão pode vir misturada
                elif ( 'results' in section_name ) and ( 'discussion' in section_name ):
                    section_name = 'results-discussion'
                
                #quando os resultados vem separado
                elif 'results' in section_name:
                    section_name = 'results'
                
                #quando a discussão vem separada
                elif 'discussion' in section_name:
                    section_name = 'discussion'

                #quando a conclusão vem separada
                elif 'conclusion' in section_name:
                    section_name = 'conclusion'
                
                #a parte de methodologia tem muitas definições diferentes na coleção
                elif ( 'methodology' in section_name ) or ( 'experiment' in section_name ) or ( ( 'materials' in section_name ) and ( 'methods' in section_name ) ):
                    section_name = 'methodology'

                #iniciando uma seção
                if section_name in ('introduction', 'results-discussion', 'results', 'discussion', 'conclusion', 'methodology'):
                    
                    if split_section != None:
                        print('> Closing section: ', split_section)
                        self.raw_text += f'The END of the section is here (separated from the XML file). '

                    print('> Opening section: ', section_name, ' ; Last section: ', last_name_section)
                    #print('Concatenating section to text: ', section_name)
                    self.raw_text += f'The BEGIN of the section is here (separated from the XML file) {section_name}. '
                    split_section = section_name

                #varrendo os parágrafos
                for p in sec.find_next_siblings('para'):
                    self.raw_text += p.get_text() + ' '

                last_name_section = section_name

                #quebra na conclusão
                if 'conclusion' in section_name:
                    broke_with_conclusion = True
                    break
                
                #se não for possível definir o nome da seção, passa-se para a próxima
                #except (AttributeError, KeyError):
                #    continue
            
            if broke_with_conclusion is True or broke_with_end_tags is True:
                print('> Closing section: ', last_name_section)
                self.raw_text += f'The END of the section is here (separated from the XML file). '
                print('> Closing article...')

        #print((self.raw_text))
        #texto filtrado        
        if len(self.raw_text) == 0:
            self.proc_error_type = 'Not_raw_text'

        else:
            if apply_filters is True:
                
                #carregando o log de uso de filtros
                if os.path.exists(self.diretorio + '/Outputs/log/pos_filter_use.json'):
                    filter_log_dic = load_dic_from_json(self.diretorio + '/Outputs/log/pos_filter_use.json')
                else:
                    filter_log_dic = {}

                self.filtered_text, self.len_text, filter_log_dic = filter_chars(self.raw_text, diretorio = self.diretorio, filename = self.xml_file_name, filter_log_dic = filter_log_dic)
                
                #salvando o uso de filtro
                save_dic_to_json(self.diretorio + '/Outputs/log/pos_filter_use.json', filter_log_dic)
                
            else:
                self.filtered_text = self.raw_text
                self.len_text = len(self.raw_text)
            
            #checando a lingua
            if check_text_language( self.raw_text , self.language) is False:
                self.proc_error_type = 'Not_english'
                print(f'O Arquivo {self.xml_file_name} não bate com a língua determinada ({self.language}).')
                print('> Abortando função: XMLTEXT.process_text')
                return
            
            #checando o tamanho mínimo do texto
            if len( self.raw_text ) < self.min_len_text:
                self.proc_error_type = 'Not_min_length'
                print(f'O Arquivo {self.xml_file_name} extraído não possui o tamanho mínimo ({len( self.raw_text )} < {self.min_len_text}).')
                print('> Abortando função: XMLTEXT.process_text')
                return
            
            #encontrando os meta-dados
            self.find_meta_data()

            #quebrando o texto em sentenças
            #print(repr(self.filtered_text))
            self.sentences = break_text_in_sentences(self.filtered_text)
            self.token_list = self.filtered_text.split()
            self.n_tokens = len(self.token_list)
            #usando o número de chars + o de tokens
            self.XML_ID =  str(self.len_text) + '_' + str(self.n_tokens)
            print('Char counter: ', self.len_text)
            print('Token counter: ', self.n_tokens)


    def find_meta_data(self):
        
        if self.publisher == 'acs':
            article_meta = self.soup_xml.find('article-meta')
            self.doi = article_meta.find('article-id').get_text()
            self.title = re.sub(r'\n', '', article_meta.find('article-title').get_text() )
            self.publication_date = article_meta.find('pub-date').year.get_text()

            self.addresses = []
            affs = article_meta.find_all('aff')
            for aff in affs:
                try:
                    self.addresses.append( aff.get_text() )
                
                except AttributeError:
                    continue

        elif self.publisher == 'elsevier':
            coredate = self.soup_xml.find('coredata')
            self.doi = coredate.find('identifier').get_text()
            self.title = re.sub(r'\n', '', coredate.find('title').get_text() )
            self.publication_date = coredate.find('coverDate').get_text()[ : 4]

            self.addresses = []
            affs = self.soup_xml.find_all('affiliation')
            for aff in affs:
                try:
                    self.addresses.append( aff.get_text() )
                
                except AttributeError:
                    continue

        print('> Doc Year: ', self.publication_date, '; DOI: ', self.doi)