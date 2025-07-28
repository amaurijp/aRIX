import ollama # type: ignore
import time
import regex as re # type: ignore

from functions_TEXTS import get_filename_from_sent_index
from functions_TEXTS import get_sent_from_filename_sent_index

class llm(object):

    def __init__(self, model):
        self.model = model

    
    def check_parameters(self, input_parameter : str, input_dic : dict = {}, personality : str = None, index_dic_sents = None, diretorio = None):
        
        print('> LLM input:')
        #print('> personality: ', personality)
        
        #coletando a sentença
        filename = get_filename_from_sent_index(input_dic[input_parameter]['sent_index'], index_dic_sents)
        input_dic[input_parameter]['sent_str'] = get_sent_from_filename_sent_index(input_dic[input_parameter]['sent_index'], filename, diretorio + '/Outputs/sents_filtered')

        
        if input_parameter.lower() == 'biofilm_killing_perc':
            
            def biofilm_killing_perc():

                user_text = f"It was previously extracted that the biological species '{input_dic['biological_species']['val']}' was exposed to or tested with a nanomaterial whose core composition is defined as '{input_dic['nanomaterial_composition']['val']}'. " \
                            f"From the sentence '{input_dic['biofilm_killing_perc']['sent_str']}', it was then extracted that this exposure/test resulted in a " \
                            f"percentage of biofilm killing value (or values) of '{input_dic['biofilm_killing_perc']['val']}'. " \
                            "Considering all the information provided, output the correct percentage of biofilm killing value (or values), " \
                            "taking into account either the biological species or the nanomaterial core composition (or both). " \
                            "If no correlation is found between any of the extracted percentage of biofilm killing values and the nanomaterial core composition, output 'None'. " \
                            "Your output must be between single quote characters (e.g, 'output')."
                
                return user_text

            print('> user_text: ', biofilm_killing_perc())
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": biofilm_killing_perc()}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            
            response = response['message']['content']
            
            print('> LLM response: ', response)
            return response


        elif input_parameter.lower() == 'biological_species':

            def biological_species():
        
                user_text = f"From the sentence '{input_dic['biological_species']['sent_str']}', it was extracted that the biological species '{input_dic['biological_species']['val']}' was exposed to " \
                            f"or tested with the nanomaterial with the core composition defined as: '{input_dic['nanomaterial_composition']['val']}'. " \
                            "Considering all information provided, respond 'yes' or 'no' (and nothing else) for the following question: " \
                            "Does the sentence contain the extracted values and do they follow the pattern described by the context? " \
                            "Your output must be between single quote characters (e.g, 'output')."
                
                return user_text

            print('> user_text: ', biological_species())
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": biological_species()}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            
            response = response['message']['content']
            
            print('> LLM response: ', response)
            return response


        elif input_parameter.lower() == 'microbe_killing_log':

            def microbe_killing_log():
                        
                user_text = f"It was previously extracted that the biological species '{input_dic['biological_species']['val']}' was exposed to or tested with a nanomaterial whose core composition is defined as '{input_dic['nanomaterial_composition']['val']}'. " \
                            f"From the sentence '{input_dic['microbe_killing_log']['sent_str']}', it was then extracted that this exposure/test resulted in a value (or values) of " \
                            f"reduction in the microbial population (in log units) of '{input_dic['microbe_killing_log']['val']}'. " \
                            "Considering all the information provided, output the correct value (or values) for the reduction of the microbial population (in log units), " \
                            "taking into account either the biological species or the nanomaterial core composition (or both). " \
                            "If no correlation is found between any of the extracted reduction values (in log units) and the nanomaterial core composition, output 'None'. " \
                            "Your output must be between single quote characters (e.g, 'output')."
                
                return user_text

            print('> user_text: ', microbe_killing_log())
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": microbe_killing_log()}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            
            response = response['message']['content']
            
            print('> LLM response: ', response)
            return response


        elif input_parameter.lower() == 'microbe_killing_mbc':

            def microbe_killing_mbc():
            
                user_text = f"It was previously extracted that the biological species '{input_dic['biological_species']['val']}' was exposed to or tested with a nanomaterial whose core composition is defined as '{input_dic['nanomaterial_composition']['val']}'. " \
                            f"From the sentence '{input_dic['microbe_killing_mbc']['sent_str']}', it was then extracted that this exposure/test resulted in a " \
                            f"minimum bactericidal concentration (MBC) value (or values) of '{input_dic['microbe_killing_mbc']['val']}'. " \
                            "Considering all the information provided, output the correct MBC value (or values), " \
                            "taking into account either the biological species or the nanomaterial core composition (or both). " \
                            "If no correlation is found between any of the extracted MBC values and the nanomaterial core composition, output 'None'. " \
                            "Your output must be between single quote characters (e.g, 'output')."
                
                return user_text

            print('> user_text: ', microbe_killing_mbc())
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": microbe_killing_mbc()}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            
            mod_text = re.sub(r'µ', 'u', response['message']['content'])
            mod_text = re.sub(r'±', '+/-', mod_text)
            mod_text = re.sub(r'–', '-', mod_text)
            
            print('> LLM response: ', mod_text)
            return mod_text


        elif input_parameter.lower() == 'microbe_killing_mic':

            def microbe_killing_mic():
            
                user_text = f"It was previously extracted that the biological species '{input_dic['biological_species']['val']}' was exposed to or tested with a nanomaterial whose core composition is defined as '{input_dic['nanomaterial_composition']['val']}'. " \
                            f"From the sentence '{input_dic['microbe_killing_mic']['sent_str']}', it was then extracted that this exposure/test resulted in a " \
                            f"minimum inhibitory concentration (MIC) value (or values) of '{input_dic['microbe_killing_mic']['val']}'. " \
                            "Considering all the information provided, output the correct MIC value (or values), " \
                            "taking into account either the biological species or the nanomaterial core composition (or both). " \
                            "If no correlation is found between any of the extracted MIC values and the nanomaterial core composition, output 'None'. " \
                            "Your output must be between single quote characters (e.g, 'output')."
                
                return user_text

            print('> user_text: ', microbe_killing_mic())
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": microbe_killing_mic()}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            
            mod_text = re.sub(r'µ', 'u', response['message']['content'])
            mod_text = re.sub(r'±', '+/-', mod_text)
            mod_text = re.sub(r'–', '-', mod_text)
            
            print('> LLM response: ', mod_text)
            return mod_text


        elif input_parameter.lower() == 'nanomaterial_morphology':

            def nanomaterial_morphology():
        
                user_text = f"From the sentence '{input_dic['nanomaterial_morphology']['sent_str']}', it was extracted that the nanomaterial core composition is " \
                            f"'{input_dic['nanomaterial_composition']['val']}' and the nanomaterial morphology is '{input_dic['nanomaterial_morphology']['val']}'. " \
                            "Considering all information provided, respond just 'yes' or 'no' (and nothing else) for the following question: " \
                            "Does the sentence contain the extracted values and do they follow the pattern described by the context? " \
                            "In case the sentence has several matches for the parameters core composition and morphology, consider if the extracted values are one of them. " \
                            "Your output must be between single quote characters (e.g, 'output')."
                
                return user_text

            print('> user_text: ', nanomaterial_morphology())
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": nanomaterial_morphology()}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            
            response = response['message']['content']
            
            print('> LLM response: ', response)
            return response


        elif input_parameter.lower() == 'nanomaterial_size':

            def nanomaterial_size():
        
                user_text = f"For a nanomaterial with core composition '{input_dic['nanomaterial_composition']['val']}' and morphology '{input_dic['nanomaterial_morphology']['val']}', " \
                            f"it was extracted from the sentence '{input_dic['nanomaterial_size']['sent_str']}' that this nanomaterial has a size value (or values) of '{input_dic['nanomaterial_size']['val']}'. " \
                            "Considering all information provided, output the correct nanomaterial size value considering either the nanomaterial core composition or the morphology (or both). " \
                            "If no correlation is found between any of extracted values for size and the nanomaterial core composition or morpholgy, output 'None'. " \
                            "Your output must be between single quote characters (e.g, 'output')."
                
                return user_text

            print('> user_text: ', nanomaterial_size())
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": nanomaterial_size()}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            mod_text = re.sub(r'µ', 'u', response['message']['content'])
            mod_text = re.sub(r'±', '+/-', mod_text)
            mod_text = re.sub(r'–', '-', mod_text)
            
            print('> LLM response: ', mod_text)
            return mod_text


        elif input_parameter.lower() == 'nanomaterial_surface_area':

            def nanomaterial_surface_area():
        
                user_text = f"For a nanomaterial with core composition '{input_dic['nanomaterial_composition']['val']}' and morphology '{input_dic['nanomaterial_morphology']['val']}', " \
                            f"it was extracted from the sentence '{input_dic['surface_area']['sent_str']}' that this nanomaterial has a surface area value (or values) of '{input_dic['surface_area']['val']}'. " \
                            "Considering all information provided, output the correct nanomaterial surface area value considering either the nanomaterial core composition or the morphology (or both). " \
                            "If no correlation is found between any of extracted values for surface area and the nanomaterial core composition or morpholgy, output 'None'. " \
                            "Your output must be between single quote characters (e.g, 'output')."
                
                return user_text

            print('> user_text: ', nanomaterial_surface_area())
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": nanomaterial_surface_area()}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            
            mod_text = re.sub(r'µ', 'u', response['message']['content'])
            mod_text = re.sub(r'±', '+/-', mod_text)
            mod_text = re.sub(r'–', '-', mod_text)
            
            print('> LLM response: ', mod_text)
            return mod_text


        elif input_parameter.lower() == 'nanomaterial_zeta_potential':
        
            def nanomaterial_zeta_potential():
                
                user_text = f"For a nanomaterial with core composition '{input_dic['nanomaterial_composition']['val']}' and morphology '{input_dic['nanomaterial_morphology']['val']}', " \
                            f"it was extracted from the sentence '{input_dic['zeta_potential']['sent_str']}' that this nanomaterial has a zeta potential value (or values) of '{input_dic['zeta_potential']['val']}'. " \
                            "Considering all information provided, output the correct nanomaterial zeta potential value (or values) considering either the nanomaterial core composition or the morphology (or both). " \
                            "If no correlation is found between any of extracted values for zeta potential and the nanomaterial core composition or morpholgy, output 'None'. " \
                            "Your output must be between single quote characters (e.g, 'output')."
                
                return user_text

            print('> user_text: ', nanomaterial_zeta_potential())
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": nanomaterial_zeta_potential()}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            
            mod_text = re.sub(r'µ', 'u', response['message']['content'])
            mod_text = re.sub(r'±', '+/-', mod_text)
            mod_text = re.sub(r'–', '-', mod_text)
            
            print('> LLM response: ', mod_text)
            return mod_text


        elif input_parameter.lower() == 'toxicity_lc50':

            def toxicity_lc50():
            
                user_text = f"It was previously extracted that the biological species '{input_dic['biological_species']['val']}' was exposed to or tested with a nanomaterial whose core composition is defined as '{input_dic['nanomaterial_composition']['val']}'. " \
                            f"From the sentence '{input_dic['toxicity_lc50']['sent_str']}', it was then extracted that this exposure/test resulted in a " \
                            f"Lethal Concentration 50% (LC50) value (or values) of '{input_dic['toxicity_lc50']['val']}'. " \
                            "Considering all the information provided, please output the correct LC50 value (or values), " \
                            "taking into account either the biological species or the nanomaterial core composition (or both). " \
                            "If no correlation is found between any of the extracted LC50 values and the nanomaterial core composition, output 'None'. " \
                            "Your output must be between single quote characters (e.g, 'output')."
                
                return user_text

            print('> user_text: ', toxicity_lc50())
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": toxicity_lc50()}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            
            mod_text = re.sub(r'µ', 'u', response['message']['content'])
            mod_text = re.sub(r'±', '+/-', mod_text)
            mod_text = re.sub(r'–', '-', mod_text)
            
            print('> LLM response: ', mod_text)
            return mod_text


        elif input_parameter.lower() == 'toxicity_yes_no':

            def toxicity_yes_no():
            
                user_text = f"From the sentence '{input_dic['toxicity_yes_no']['sent_str']}', it was evaluated if there is (or not) a toxicity effect for the endpoint '{input_dic['toxicity_endpoints']['val']}'. " \
                            f"Consider the question: " \
                            f"Is there any evidence in the input sentence of a possible toxic effect for the endpoint '{input_dic['toxicity_endpoints']['val']}' considering the species '{input_dic['biological_species']['val']}'? " \
                            f"Your previous answer for this question was: '{input_dic['toxicity_yes_no']['val']}'. " \
                            "Is your previous answer about the toxicity assessment correct? " \
                            "Your output must be between single quote characters (e.g, 'output')."

                return user_text

            print('> user_text: ', toxicity_yes_no())
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": toxicity_yes_no()}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            
            response = response['message']['content']
            
            print('> LLM response: ', response)
            return response


    
    #######################################################################
    def extract_num_parameter(self, parameter: str, text: str):

        
        if parameter.lower() == 'biofilm_killing_perc':

            def biofilm_killing_perc():
                
                t = "You are a data miner that deals with scientific texts on microbiology and you extract numerical values related with antimicrobial effect. " \
                    "For the text I will input next, output only the numbers associated with percentage values related to an biofilm eradication or disruption. " \
                    "Output the percentage symbol with a blank space after each number found (Ex: '10.1 %'). " \
                    "If no percentage is found, output 'None'. Do not output percentage values not related with antibiofilm effect."
                
                return t

            personality = biofilm_killing_perc()

            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            

            num_params = re.findall(r'[0-9]+\.?[0-9]*\s*%', response['message']['content'])
            
            nums_text = ''
            for val in num_params:
                nums_text += val + ', '
            nums_text = nums_text[ : -2 ]
            
            print('> LLM response: ', nums_text)
            return nums_text
        

        elif parameter.lower() == 'microbe_killing_log':

            def microbe_killing_log():

                t = "You are a data miner that deals with scientific texts on microbiology and you extract numerical values related with antimicrobial effect. " \
                    "For the text I will input next, output only the numbers associated with log values indicating the reduction of the microbial population. " \
                    "Output the log unit (ex: log10 or log or logs) with a blank space after each number found (Ex: '2.2 logs'). " \
                    "If no value is found, output 'None'. Do not output any other values not related the reduction of the microbial population."

                return t

            personality = microbe_killing_log()

            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            
            num_params = re.findall(r'[0-9]+\.?[0-9]*\s*[A-Za-z0-9\-]+', response['message']['content'])
            
            nums_text = ''
            for val in num_params:
                nums_text += val + ', '
            nums_text = nums_text[ : -2 ]
            
            print('> LLM response: ', nums_text)
            return nums_text

        
        elif parameter.lower() == 'microbe_killing_mbc':

            def microbe_killing_mbc():
                t = "You are a data miner that deals with scientific texts on microbiology and you extract numerical values related with antimicrobial effect. " \
                    "For the text I will input next, output only the numbers associated with minimum bactericidal concentration (MBC) values indicating the antimicrobial effect. " \
                    "Output the concentration unit (ex: mg L-1) with a blank space after each number found (Ex: '10.1 mg L-1' or '10.1 mg mL-1' or '10.1 g mL-1' or '10.1 ug mL-1'). " \
                    "If no value is found, output 'None'. Do not output concentration values not related with minimum bactericidal concentration (MBC)."

                return t
            
            personality = microbe_killing_mbc()
            
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            
            mod_text = re.sub(r'µ', 'u', response['message']['content'])
            mod_text = re.sub(r'±', '+/-', mod_text)
            mod_text = re.sub(r'–', '-', mod_text)
            num_params = re.findall(r'[0-9]+\.?[0-9]*\s*[A-Za-z]+\s[A-Za-z0-9\-]+', mod_text)
            
            nums_text = ''
            for val in num_params:
                nums_text += val + ', '
            nums_text = nums_text[ : -2 ]
            
            print('> LLM response: ', nums_text)
            return nums_text


        elif parameter.lower() == 'microbe_killing_mic':

            def microbe_killing_mic():

                t = "You are a data miner that deals with scientific texts on microbiology and you extract numerical values related with antimicrobial effect. " \
                    "For the text I will input next, output only the numbers associated with minimum inhibitory concentration (MIC) values indicating the antimicrobial effect. " \
                    "Output the concentration unit (ex: mg L-1) with a blank space after each number found (Ex: '10.1 mg L-1' or '10.1 mg mL-1' or '10.1 g mL-1' or '10.1 ug mL-1'). " \
                    "If no value is found, output 'None'. Do not output concentration values not related with minimum inhibitory concentration (MIC)."

                return t

            personality = microbe_killing_mic()

            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            
            mod_text = re.sub(r'µ', 'u', response['message']['content'])
            mod_text = re.sub(r'±', '+/-', mod_text)
            mod_text = re.sub(r'–', '-', mod_text)
            num_params = re.findall(r'[0-9]+\.?[0-9]*\s*[A-Za-z]+\s[A-Za-z0-9\-]+', mod_text)
            
            nums_text = ''
            for val in num_params:
                nums_text += val + ', '
            nums_text = nums_text[ : -2 ]
            
            print('> LLM response: ', nums_text)
            return nums_text

        
        elif parameter.lower() == 'microbe_killing_perc':

            def microbe_killing_perc():

                t = "You are a data miner that deals with scientific texts on microbiology and you extract numerical values related with antimicrobial effect. " \
                    "For the text I will input next, output only the numbers associated with percentage values indicating the microbes killing. " \
                    "Output the percentage symbol with a blank space after each number found (Ex: '10.1 %'). " \
                    "If no percentage is found, output 'None'. Do not output percentage values not related with microbe killing."

                return t

            personality = microbe_killing_perc()

            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            

            num_params = re.findall(r'[0-9]+\.?[0-9]*\s*%', response['message']['content'])
            
            nums_text = ''
            for val in num_params:
                nums_text += val + ', '
            nums_text = nums_text[ : -2 ]
            
            print('> LLM response: ', nums_text)
            return nums_text


        elif parameter.lower() == 'nanomaterial_concentration':

            def nanomaterial_concentration():
                    
                t = "You are a data miner that deals with scientific texts on nanotechnology and you extract numerical values related with the concentration of nanomaterials and nanoparticles. " \
                    "For the text I will input next, output only the numbers associated with the concentration of nanomaterials and nanoparticles described in the text. " \
                    "Output the concentration unit (ex: mg L-1) with a blank space after each number found (Ex: '10.1 mg L-1' or '10.1 mg mL-1' or '10.1 g mL-1' or '10.1 ug mL-1'). " \
                    "If no value is found, output 'None'. Do not output concentration values not related with the nanomaterial or nanoparticle described in the text."
                
                return t

            personality = nanomaterial_concentration()

            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            
            mod_text = re.sub(r'µ', 'u', response['message']['content'])
            mod_text = re.sub(r'±', '+/-', mod_text)
            mod_text = re.sub(r'–', '-', mod_text)
            num_params = re.findall(r'[0-9]+\.?[0-9]*\s*[A-Za-z]+\s[A-Za-z0-9\-]+', mod_text)
            
            nums_text = ''
            for val in num_params:
                nums_text += val + ', '
            nums_text = nums_text[ : -2 ]
            
            print('> LLM response: ', nums_text)
            return nums_text


        elif parameter.lower() == 'nanomaterial_size':

            def nanomaterial_size():
                
                t = "You are a data-miner for nanotechnology papers. " \
                    "Your sole task is to extract only the numeric size values of the nanomaterials or nanoparticles mentioned in the passage. " \
                    "*DEFINITIONS* " \
                    "• Nanomaterial / nanoparticle size → any reported linear dimension (diameter, length, width, height, Feret's diameter, " \
                    "hydrodynamic size, etc.) that is explicitly tied to a nano-object (e.g. 'nanoparticle', 'nanotube', 'nanorod', 'CNT', 'NP', 'NPs'). " \
                    "• Valid units → nm, µm, um, Å / angstrom (case-insensitive). " \
                    "• Valid formats → single numbers (10 nm), ranges (10-20 nm), mean ± SD (25 ± 3 nm), scientific notation (1.2 × 10² nm), or approximations ('~15 nm', '≈ 50 nm'). " \
                    "• Non-size numbers → anything tied to concentration, dose, wavelength, voltage, pH, time, statistics, or equipment settings. Ignore these. " \
                    "*RULES* " \
                    "1. The size must be explicitly linked to a nanomaterial term in the same clause or sentence. " \
                    "2. If multiple valid sizes appear, output each one in the order they appear. " \
                    "3. Preserve the original text exactly—do not convert units or round numbers. " \
                    "4. Insert a single blank space between the number string and the unit (e.g. '10.2 nm'). " \
                    "5. Separate multiple results with a comma and one space ('10 nm, 20-30 nm'). " \
                    "6. If no valid size is present, output the single token 'None'. " \
                    "7. Output absolutely nothing else—no explanations, no JSON keys, no newlines except those required by Rule 5 formatting. " \
                    "*EXAMPLES* " \
                    "EXAMPLE 1 " \
                    "text: 'TEM images showed spherical silver nanoparticles with diameters of 12.3 ± 1.8 nm dispersed in water.' " \
                    "your output:  '12.3 ± 1.8 nm' " \
                    "EXAMPLE 2 " \
                    "text: 'Carbon nanotubes (CNTs) had an outer diameter of ~15 nm and lengths reaching 3 µm, while the buffer contained NaCl at 150 mM.' " \
                    "your output: '15 nm, 3 µm' " \
                    "EXAMPLE 3 " \
                    "text: 'No nanoparticle dimensions were reported; only the zeta-potential (-32 mV) was measured.' " \
                    "your output:  'None' " \
                    "-- end examples --"
                
                return t

            personality = nanomaterial_size()

            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            
            mod_text = re.sub(r'µ', 'u', response['message']['content'])
            mod_text = re.sub(r'±', '+/-', mod_text)
            mod_text = re.sub(r'–', '-', mod_text)
            num_params = re.findall(r'[0-9]+\.?[0-9]*\s*[A-Za-z]+', mod_text)
            
            nums_text = ''
            for val in num_params:
                nums_text += val + ', '
            nums_text = nums_text[ : -2 ]
            
            print('> LLM response: ', nums_text)
            return nums_text


        elif parameter.lower() == 'nanomaterial_surface_area':

            def nanomaterial_surface_area():
                
                t = "You are a data miner that deals with scientific texts on nanotechnology and you extract numerical values related with the surface area of nanomaterials and nanoparticles. " \
                    "For the text I will input next, output only the numbers associated with the surface area of nanomaterials and nanoparticles described in the text. " \
                    "Output the surface area unit (ex: m2 g-1) with a blank space after each number found (Ex: '10.1 m2 g-1' or '10.1 m2 mg-1' or '10.1 mm2 kg-1' or '10.1 µm2 g-1'). " \
                    "If no value is found, output 'None'. Do not output surface area values not related with the nanomaterial or nanoparticle described in the text."
                
                return t

            personality = nanomaterial_surface_area()

            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            
            mod_text = re.sub(r'µ', 'u', response['message']['content'])
            mod_text = re.sub(r'±', '+/-', mod_text)
            mod_text = re.sub(r'–', '-', mod_text)
            num_params = re.findall(r'[0-9]+\.?[0-9]*\s*[A-Za-z0-9]+\s[A-Za-z0-9\-]+', mod_text)
            
            nums_text = ''
            for val in num_params:
                nums_text += val + ', '
            nums_text = nums_text[ : -2 ]
            
            print('> LLM response: ', nums_text)
            return nums_text


        elif parameter.lower() == 'nanomaterial_zeta_potential':

            def nanomaterial_zeta_potential():
                
                t = "You are a data miner that deals with scientific texts on nanotechnology and you extract numerical values related with the zeta potential of nanomaterials and nanoparticles. " \
                    "For the text I will input next, output only the numbers associated with the zeta potential of nanomaterials and nanoparticles described in the text. " \
                    "Output the zeta potential unit (ex: mV) with a blank space after each number found (Ex: '-10.1 mV' or '10.1 V' or '10.1 kV' or '-10.1 µV'). " \
                    "If no value is found, output 'None'. Do not output zeta potential values not related with the nanomaterial or nanoparticle described in the text."
                
                return t

            personality = nanomaterial_zeta_potential()

            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            

            mod_text = re.sub(r'µ', 'u', response['message']['content'])
            mod_text = re.sub(r'±', '+/-', mod_text)
            mod_text = re.sub(r'–', '-', mod_text)
            num_params = re.findall(r'\-?\s*[0-9]+\.?[0-9]*\s*[A-Za-z]+', mod_text)
            
            nums_text = ''
            for val in num_params:
                nums_text += val + ', '
            nums_text = nums_text[ : -2 ]
            
            print('> LLM response: ', nums_text)
            return nums_text


        elif parameter.lower() == 'toxicity_ec50':

            def toxicity_ec50():
                
                t = "You are a data miner that deals with scientific texts on the toxicity against living species and you extract numerical values related with the toxicity effect. " \
                    "For the text I will input next, output only the numbers associated with the half-maximal effective concentration (EC50) values indicating the toxicity effect. " \
                    "Output the concentration unit (ex: mg L-1) with a blank space after each number found (Ex: '10.1 mg L-1' or '10.1 mg mL-1' or '10.1 g mL-1' or '10.1 ug mL-1'). " \
                    "If no value is found, output 'None'. Do not output concentration values not related with half-maximal effective concentration (EC50)."
            
                return t

            personality = toxicity_ec50()

            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            
            mod_text = re.sub(r'µ', 'u', response['message']['content'])
            mod_text = re.sub(r'±', '+/-', mod_text)
            mod_text = re.sub(r'–', '-', mod_text)
            num_params = re.findall(r'[0-9]+\.?[0-9]*\s*[A-Za-z]+\s[A-Za-z0-9\-]+', mod_text)
            
            nums_text = ''
            for val in num_params:
                nums_text += val + ', '
            nums_text = nums_text[ : -2 ]
            
            print('> LLM response: ', nums_text)
            return nums_text


        elif parameter.lower() == 'toxicity_lc50':

            def toxicity_lc50():
                
                t = "You are a data miner that deals with scientific texts on the toxicity against living species and you extract numerical values related with the toxicity effect. " \
                    "For the text I will input next, output only the numbers associated with the half-lethal maximal effective concentration (LC50) values indicating the toxicity effect. " \
                    "Output the concentration unit (ex: mg L-1) with a blank space after each number found (Ex: '10.1 mg L-1' or '10.1 mg mL-1' or '10.1 g mL-1' or '10.1 ug mL-1'). " \
                    "If no value is found, output 'None'. Do not output concentration values not related with half-lethal maximal effective concentration (LC50)."

                return t

            personality = toxicity_lc50()

            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
    
            mod_text = re.sub(r'µ', 'u', response['message']['content'])
            mod_text = re.sub(r'±', '+/-', mod_text)
            mod_text = re.sub(r'–', '-', mod_text)
            num_params = re.findall(r'[0-9]+\.?[0-9]*\s*[A-Za-z]+\s[A-Za-z0-9\-]+', mod_text)
            
            nums_text = ''
            for val in num_params:
                nums_text += val + ', '
            nums_text = nums_text[ : -2 ]
            
            print('> LLM response: ', nums_text)
            return nums_text


        elif parameter.lower() == 'toxicity_ld50':

            def toxicity_ld50():
                
                t = "You are a data miner that deals with scientific texts on the toxicity against living species and you extract numerical values related with the toxicity effect. " \
                    "For the text I will input next, output only the numbers associated with the lethal dose 50% (LD50) values indicating the dose of a substance that is lethal to 50% of the tested population. " \
                    "Output the LD50 unit (ex: mg kg-1) with a blank space after each number found (Ex: '10.1 mg kg-1' or '10.1 mg g-1' or '10.1 g mg-1'). " \
                    "If no value is found, output 'None'. Do not output concentration values not related with lethal dose 50% (LD50)."

                return t

            personality = toxicity_ld50()

            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            
            mod_text = re.sub(r'µ', 'u', response['message']['content'])
            mod_text = re.sub(r'±', '+/-', mod_text)
            mod_text = re.sub(r'–', '-', mod_text)
            num_params = re.findall(r'[0-9]+\.?[0-9]*\s*[A-Za-z]+\s[A-Za-z0-9\-]+', mod_text)
            
            nums_text = ''
            for val in num_params:
                nums_text += val + ', '
            nums_text = nums_text[ : -2 ]
            
            print('> LLM response: ', nums_text)
            return nums_text


        elif parameter.lower() == 'toxicity_ic50':

            def toxicity_ic50():

                t = "You are a data miner that deals with scientific texts on the toxicity against living species and you extract numerical values related with the toxicity effect. " \
                    "For the text I will input next, output only the numbers associated with the half-lethal maximal inhibitory concentration (IC50) values indicating the toxicity effect. " \
                    "Output the concentration unit (ex: mg L-1) with a blank space after each number found (Ex: '10.1 mg L-1' or '10.1 mg mL-1' or '10.1 g mL-1' or '10.1 ug mL-1'). " \
                    "If no value is found, output 'None'. Do not output concentration values not related with half-lethal maximal inhibitory concentration (IC50)."

                return t

            personality = toxicity_ic50()

            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            
            mod_text = re.sub(r'µ', 'u', response['message']['content'])
            mod_text = re.sub(r'±', '+/-', mod_text)
            mod_text = re.sub(r'–', '-', mod_text)
            num_params = re.findall(r'[0-9]+\.?[0-9]*\s*[A-Za-z]+\s[A-Za-z0-9\-]+', mod_text)
            
            nums_text = ''
            for val in num_params:
                nums_text += val + ', '
            nums_text = nums_text[ : -2 ]
            
            print('> LLM response: ', nums_text)
            return nums_text


        else:
            return ''



    #######################################################################
    def extract_textual_parameter(self, parameter: str, text: str):

        if parameter.lower() == '2d_materials':
            
            def _2d_materials():

                t = "You are a data miner specialized in nanotechnology. " \
                    "Your task is to extract the core chemical composition of two-dimensional (2D) nanomaterials described in the input sentences. " \
                    "Examples of common 2D nanomaterials include graphene (C), molybdenum disulfide (MoS2), tungsten disulfide (WS2), boron nitride (BN), and other layered compounds. " \
                    "The core composition may also appear as abbreviations or chemical formulas, such as MoS2, WS2, or BN. " \
                    "Output only the core chemical composition(s) identified, using their chemical symbol(s) or formula(s). " \
                    "If more than one 2D nanomaterial is mentioned in a sentence, list each distinct core composition in a Python list format, " \
                    "for example: ['C', 'MoS2', 'WS2', 'BN']. Focus exclusively on 2D nanomaterials and ignore any other types of materials. " \
                    "Output the core composition between single quote characters (e.g, 'C' or 'MoS2' or 'WS2')."
                
                return t
        
            personality = _2d_materials()

            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings


        elif parameter.lower() == 'metallic_materials':

            def metallic_materials():

                t = "You are a data miner specialized in nanotechnology. " \
                    "Your task is to extract the core chemical composition of metallic nanomaterials described in the input sentences. " \
                    "Examples of metallic nanomaterials include gold (Au), silver (Ag), copper (Cu), nickel (Ni), iron (Fe), aluminum (Al), and other metals. " \
                    "The composition may appear as abbreviations or chemical formulas. " \
                    "Output only the core chemical composition(s) using their chemical symbol(s) or formula(s). " \
                    "If more than one metallic nanomaterial is mentioned, list each distinct composition in a Python list format, " \
                    "for example: ['Au', 'Ag', 'Cu']. Focus exclusively on metallic nanomaterials and ignore any other types of materials. " \
                    "Output the core composition between single quote characters (e.g, 'Au' or 'Ag' or 'Cu')."
                
                return t

            personality = metallic_materials()

            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings


        elif parameter.lower() == 'oxides_materials':

            def oxides_materials():

                t = "You are a data miner specialized in nanotechnology. " \
                    "Your task is to extract the core chemical composition of oxide nanomaterials described in the input sentences. " \
                    "Examples of oxide nanomaterials include titanium dioxide (TiO2), zinc oxide (ZnO), silica (SiO2), copper oxide (CuO), iron oxide (Fe2O3), and other metal oxides. " \
                    "The composition may appear as abbreviations or chemical formulas. " \
                    "Output only the core chemical composition(s) using their chemical symbol(s) or formula(s). " \
                    "If more than one oxide nanomaterial is mentioned, list each distinct composition in a Python list format, " \
                    "for example: ['TiO2', 'ZnO', 'SiO2']. Focus exclusively on oxide nanomaterials and ignore any other types of materials. " \
                    "Output the core composition between single quote characters (e.g, 'TiO2' or 'ZnO' or 'SiO2')."
                
                return t

            personality = oxides_materials()

            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings


        elif parameter.lower() == 'qdots_materials':
            
            def qdots_materials():
                
                t = "You are a data miner specialized in nanotechnology. " \
                    "Your task is to extract the core chemical composition of quantum dot nanomaterials described in the input sentences. " \
                    "Examples of quantum dot compositions include CdS, CdSe, CdTe, ZnS, ZnSe, PbS, PbSe, InP, and other semiconductor materials commonly used to form quantum dots. " \
                    "The composition may appear as abbreviations or chemical formulas. " \
                    "Output only the core chemical composition(s) using their chemical symbol(s) or formula(s). " \
                    "If more than one quantum dot nanomaterial is mentioned, list each distinct composition in a Python list format, " \
                    "for example: ['CdSe', 'ZnS', 'PbS']. " \
                    "Focus exclusively on quantum dot nanomaterials and ignore any other types of materials. " \
                    "Output the core composition between single quote characters (e.g, 'CdSe' or 'ZnS' or 'PbS')."
                
                return t

            personality = qdots_materials()

            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings


        elif parameter.lower() == 'toxicity_a_thaliana_all':

            def toxicity_a_thaliana_all():

                t = "You are a sentence analyzer specialized in nanotoxicology. " \
                    "Your task is to evaluate whether there is evidence of toxicity of a nanomaterial tested on the biological species 'Arabidopsis thaliana' in the input sentences. " \
                    "Evidence of toxicity includes: " \
                    "(i) Germination inhibition: reduced or delayed seed germination rates. " \
                    "(ii) Growth suppression: reduced shoot or root elongation, decreased biomass, or overall inhibited growth. " \
                    "(iii) Physiological and biochemical changes: diminished chlorophyll content, impaired photosynthetic efficiency, " \
                    "oxidative stress indicators (e.g., ROS, lipid peroxidation), and altered enzymatic activities (e.g., catalase, peroxidase, superoxide dismutase). " \
                    "(iv) Developmental and morphological changes: leaf deformities, necrosis, chlorosis, abnormal root architecture, or other visible signs of stress. " \
                    "Consider keywords such as 'toxicity', 'adverse effect', 'growth inhibition', 'reduced germination', 'necrosis', 'chlorosis', 'DNA damage', " \
                    "'oxidative stress', or 'significant negative impact' as indications of toxicity. " \
                    "Consider keywords such as 'no observed effect' (NOEL), 'no observed adverse effect' (NOAEL), 'not toxic', and similar terms as indications of non-toxicity. " \
                    "Considering all the information provided, respond only 'yes' or 'no' (and nothing else) to the following question: " \
                    "Is there any evidence in the input sentence of a toxic effect of a nanomaterial against the species 'Arabidopsis thaliana'? " \
                    "If there is no clear evidence of nanomaterial toxicity considering the aspects mentioned above, output 'no'. " \
                    "If the sentence does not deal with toxicity information against 'Arabidopsis thaliana' or if you cannot assess the sentence " \
                    "in regard to toxicity against 'Arabidopsis thaliana', output 'None'. " \
                    "Output 'yes', 'no' or 'None' between single quote characters (e.g, 'yes' or 'no' or 'None')."
                
                return t

            personality = toxicity_a_thaliana_all()
            
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings


        elif parameter.lower() == 'toxicity_a_thaliana_bioaccumulation':


            def toxicity_a_thaliana_bioaccumulation():

                t = "You are a sentence analyzer specialized in nanotoxicology. " \
                    "Your task is to evaluate whether there is evidence of bioaccumulation of a nanomaterial in the biological species 'Arabidopsis thaliana' present in the input sentence. " \
                    "Evidence of bioaccumulation includes elevated nanomaterial concentrations in root, shoot, or leaf tissues relative to the exposure medium, or translocation from roots to shoots. " \
                    "Consider keywords such as 'bioaccumulation', 'bioconcentration', 'uptake', 'root-to-shoot transfer', or quantitative accumulation values as indications of bioaccumulation. " \
                    "Consider keywords such as 'no accumulation', 'no uptake', 'clearance', or 'not detected' as indications of non-bioaccumulation. " \
                    "Considering all the information provided, respond only 'yes' or 'no'. " \
                    "Is there any evidence in the input sentence of bioaccumulation of a nanomaterial in 'Arabidopsis thaliana'? " \
                    "If there is no clear evidence of bioaccumulation, output 'no'. " \
                    "If the sentence does not deal with bioaccumulation in 'Arabidopsis thaliana' or if you cannot assess the sentence for this endpoint, output 'None'. " \
                    "Output 'yes', 'no' or 'None' between single quote characters (e.g, 'yes' or 'no' or 'None')."
                
                return t

            personality = toxicity_a_thaliana_bioaccumulation()
            
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings


        elif parameter.lower() == 'toxicity_a_thaliana_development':

            def toxicity_a_thaliana_development():

                t = "You are a sentence analyzer specialized in nanotoxicology. " \
                    "Your task is to evaluate whether there is evidence of growth suppression caused by a nanomaterial in the biological species 'Arabidopsis thaliana' present in the input sentence. " \
                    "Evidence of growth suppression includes reduced shoot or root length, decreased fresh or dry biomass, stunted seedlings, or overall inhibited growth. " \
                    "Consider keywords such as 'reduced growth', 'decreased biomass', 'shorter roots', 'growth inhibition', or 'stunted seedlings' as indications of toxicity. " \
                    "Consider keywords such as 'growth unaffected', 'normal biomass', or 'no growth inhibition' as indications of non-toxicity. " \
                    "Considering all the information provided, respond only 'yes' or 'no'. " \
                    "Is there any evidence in the input sentence of growth suppression in 'Arabidopsis thaliana' caused by a nanomaterial? " \
                    "If there is no clear evidence of growth suppression, output 'no'. " \
                    "If the sentence does not deal with growth in 'Arabidopsis thaliana' or if you cannot assess the sentence for this endpoint, output 'None'. " \
                    "Output 'yes', 'no' or 'None' between single quote characters (e.g, 'yes' or 'no' or 'None')."

                return t

            personality = toxicity_a_thaliana_development()
            
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings


        elif parameter.lower() == 'toxicity_a_thaliana_enzyme':

            def toxicity_a_thaliana_enzyme():

                t = "You are a sentence analyzer specialized in nanotoxicology. " \
                    "Your task is to evaluate whether there is evidence of enzymatic activity alteration caused by a nanomaterial in the biological species 'Arabidopsis thaliana' present in the input sentence. " \
                    "Evidence includes changes in catalase, peroxidase, superoxide dismutase, glutathione reductase, or other key physiological enzymes. " \
                    "Consider keywords such as 'altered catalase', 'peroxidase induction', 'SOD increase', or 'enzyme inhibition' as indications of toxicity. " \
                    "Consider keywords such as 'enzyme activity unchanged', 'no enzymatic effect', or 'normal catalase levels' as indications of non-toxicity. " \
                    "Considering all the information provided, respond only 'yes' or 'no'. " \
                    "Is there any evidence in the input sentence of enzymatic activity alteration in 'Arabidopsis thaliana' caused by a nanomaterial? " \
                    "If there is no clear evidence of enzymatic alteration, output 'no'. " \
                    "If the sentence does not deal with enzymatic activity in 'Arabidopsis thaliana' or if you cannot assess the sentence for this endpoint, output 'None'. " \
                    "Output 'yes', 'no' or 'None' between single quote characters (e.g, 'yes' or 'no' or 'None')."
                
                return t

            personality = toxicity_a_thaliana_enzyme()
            
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings


        elif parameter.lower() == 'toxicity_a_thaliana_genotox':

            def toxicity_a_thaliana_genotox():

                t = "You are a sentence analyzer specialized in nanotoxicology. " \
                    "Your task is to evaluate whether there is evidence of DNA damage or genotoxicity caused by a nanomaterial in the biological species 'Arabidopsis thaliana' present in the input sentence. " \
                    "Evidence includes comet assay tail moments, DNA strand breaks, micronuclei formation, chromosomal aberrations, or γ-H2AX foci. " \
                    "Consider keywords such as 'DNA damage', 'genotoxic effect', 'comet assay', 'micronuclei', or 'chromosomal aberration' as indications of genotoxicity. " \
                    "Consider keywords such as 'no DNA damage', 'genome integrity unchanged', or 'no genotoxic effect' as indications of non-genotoxicity. " \
                    "Considering all the information provided, respond only 'yes' or 'no'. " \
                    "Is there any evidence in the input sentence of DNA damage in 'Arabidopsis thaliana' caused by a nanomaterial? " \
                    "If there is no clear evidence of genotoxicity, output 'no'. " \
                    "If the sentence does not deal with DNA damage in 'Arabidopsis thaliana' or if you cannot assess the sentence for this endpoint, output 'None'. " \
                    "Output 'yes', 'no' or 'None' between single quote characters (e.g, 'yes' or 'no' or 'None')."
                
                return t

            personality = toxicity_a_thaliana_genotox()
            
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings
        

        elif parameter.lower() == 'toxicity_a_thaliana_germination':

            def toxicity_a_thaliana_germination():

                t = "You are a sentence analyzer specialized in nanotoxicology. " \
                    "Your task is to evaluate whether there is evidence of germination inhibition caused by a nanomaterial in the biological species 'Arabidopsis thaliana' present in the input sentence. " \
                    "Evidence of germination inhibition includes reduced germination rate, delayed germination, or lower percentage of seeds sprouting compared with controls. " \
                    "Consider keywords such as 'reduced germination', 'delayed sprouting', 'germination inhibition', or 'lower seed viability' as indications of toxicity. " \
                    "Consider keywords such as 'no effect on germination', 'germination unaffected', or 'normal sprouting' as indications of non-toxicity. " \
                    "Considering all the information provided, respond only 'yes' or 'no' (and nothing else). " \
                    "Is there any evidence in the input sentence of germination inhibition of 'Arabidopsis thaliana' caused by a nanomaterial? " \
                    "If there is no clear evidence of germination inhibition, output 'no'. " \
                    "If the sentence does not deal with germination in 'Arabidopsis thaliana' or if you cannot assess the sentence for this endpoint, output 'None'. " \
                    "Output 'yes', 'no', or 'None' between single quote characters (e.g., 'yes', 'no', or 'None')."

                return t

            personality = toxicity_a_thaliana_germination()
            
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings
        


        elif parameter.lower() == 'toxicity_a_thaliana_morphology':

            def toxicity_a_thaliana_morphology():

                t = "You are a sentence analyzer specialized in nanotoxicology. " \
                    "Your task is to evaluate whether there is evidence of morphological abnormalities caused by a nanomaterial in the biological species 'Arabidopsis thaliana' present in the input sentence. " \
                    "Evidence includes leaf deformities, necrosis, chlorosis, root hair abnormalities, or other visible structural changes. " \
                    "Consider keywords such as 'necrosis', 'chlorosis', 'leaf deformation', 'abnormal root', or 'morphological alteration' as indications of toxicity. " \
                    "Consider keywords such as 'normal morphology', 'no necrosis', or 'structure unchanged' as indications of non-toxicity. " \
                    "Considering all the information provided, respond only 'yes' or 'no'. " \
                    "Is there any evidence in the input sentence of morphological abnormalities in 'Arabidopsis thaliana' caused by a nanomaterial? " \
                    "If there is no clear evidence of morphological toxicity, output 'no'. " \
                    "If the sentence does not deal with morphology in 'Arabidopsis thaliana' or if you cannot assess the sentence for this endpoint, output 'None'. " \
                    "Output 'yes', 'no' or 'None' between single quote characters (e.g, 'yes' or 'no' or 'None')."
                
                return t

            personality = toxicity_a_thaliana_morphology()
            
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings
        


        elif parameter.lower() == 'toxicity_a_thaliana_oxi_stress':

            def toxicity_a_thaliana_oxi_stress():

                t = "You are a sentence analyzer specialized in nanotoxicology. " \
                    "Your task is to evaluate whether there is evidence of oxidative stress caused by a nanomaterial in the biological species 'Arabidopsis thaliana' present in the input sentence. " \
                    "Evidence of oxidative stress includes elevated reactive oxygen species (ROS), increased lipid peroxidation, protein carbonylation, glutathione depletion, or altered antioxidant enzyme activities. " \
                    "Consider keywords such as 'elevated ROS', 'increased MDA', 'oxidative damage', 'redox imbalance', or 'lipid peroxidation' as indications of oxidative stress. " \
                    "Consider keywords such as 'no oxidative stress', 'ROS unchanged', or 'antioxidant status unchanged' as indications of non-oxidative stress. " \
                    "Considering all the information provided, respond only 'yes' or 'no'. " \
                    "Is there any evidence in the input sentence of oxidative stress in 'Arabidopsis thaliana' caused by a nanomaterial? " \
                    "If there is no clear evidence of oxidative stress, output 'no'. " \
                    "If the sentence does not deal with oxidative stress in 'Arabidopsis thaliana' or if you cannot assess the sentence for this endpoint, output 'None'. " \
                    "Output 'yes', 'no' or 'None' between single quote characters (e.g, 'yes' or 'no' or 'None')."
                
                return t

            personality = toxicity_a_thaliana_oxi_stress()
            
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings
        

        elif parameter.lower() == 'toxicity_a_thaliana_photosynthesis':

            def toxicity_a_thaliana_photosynthesis():

                t = "You are a sentence analyzer specialized in nanotoxicology. " \
                    "Your task is to evaluate whether there is evidence of photosynthetic impairment caused by a nanomaterial in the biological species 'Arabidopsis thaliana' present in the input sentence. " \
                    "Evidence of photosynthetic impairment includes diminished chlorophyll content, reduced photosystem II efficiency, decreased Fv/Fm ratio, or impaired photochemical quenching. " \
                    "Consider keywords such as 'reduced chlorophyll', 'lower photosynthetic efficiency', 'photosystem inhibition', or 'decreased Fv/Fm' as indications of toxicity. " \
                    "Consider keywords such as 'chlorophyll unchanged', 'photosynthesis unaffected', or 'no photosynthetic effect' as indications of non-toxicity. " \
                    "Considering all the information provided, respond only 'yes' or 'no'. " \
                    "Is there any evidence in the input sentence of photosynthetic impairment in 'Arabidopsis thaliana' caused by a nanomaterial? " \
                    "If there is no clear evidence of photosynthetic impairment, output 'no'. " \
                    "If the sentence does not deal with photosynthesis in 'Arabidopsis thaliana' or if you cannot assess the sentence for this endpoint, output 'None'. " \
                    "Output 'yes', 'no' or 'None' between single quote characters (e.g, 'yes' or 'no' or 'None')."
                
                return t

            personality = toxicity_a_thaliana_photosynthesis()
            
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings

        
        elif parameter.lower() == 'toxicity_a_thaliana_seedling_phototropic':

            def toxicity_a_thaliana_seedling_phototropic():

                t = "You are a sentence analyzer specialized in nanotoxicology. " \
                    "Your task is to evaluate whether there is evidence of altered seedling behavior such as phototropism or " \
                    "gravitropism caused by a nanomaterial in the biological species 'Arabidopsis thaliana' present in the input sentence. " \
                    "Evidence includes impaired phototropic bending, altered gravitropic response, or disrupted root waving patterns. " \
                    "Consider keywords such as 'phototropism inhibited', 'gravitropism altered', 'impaired bending', or 'abnormal orientation' as indications of toxicity. " \
                    "Consider keywords such as 'normal phototropism', 'gravitropism unaffected', or 'no orientation change' as indications of non-toxicity. " \
                    "Considering all the information provided, respond only 'yes' or 'no'. " \
                    "Is there any evidence in the input sentence of altered seedling behavior in 'Arabidopsis thaliana' caused by a nanomaterial? " \
                    "If there is no clear evidence of behavioral alteration, output 'no'. " \
                    "If the sentence does not deal with seedling behavior in 'Arabidopsis thaliana' or if you cannot assess the sentence for this endpoint, output 'None'. " \
                    "Output 'yes', 'no' or 'None' between single quote characters (e.g, 'yes' or 'no' or 'None')."
                
                return t

            personality = toxicity_a_thaliana_seedling_phototropic()
            
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )

            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings


        elif parameter.lower() == 'toxicity_c_elegans_all':
            
            def toxicity_c_elegans_all():

                t = "You are a sentence analyzer specialized in nanotoxicology. " \
                    "Your task is to evaluate whether there is evidence of toxicity of a nanomaterial tested on the biological species 'Caenorhabditis elegans' in the input sentences. " \
                    "Evidence of toxicity includes: " \
                    "(i) Acute toxicity (short-term tests): mortality or decreased viability over typical exposure periods (e.g., 24–48 hours). " \
                    "(ii) Chronic toxicity (long-term tests): reduced reproduction (e.g., brood size), diminished growth or body length, delays in development, or other prolonged effects. " \
                    "(iii) Developmental and morphological changes: any reported anatomical malformations, alterations in developmental stages, or general sub-lethal stress. " \
                    "(iv) Physiological and biochemical markers: oxidative stress indicators (e.g., ROS, lipid peroxidation), " \
                    "enzymatic activity shifts (e.g., catalase, superoxide dismutase), genotoxicity (DNA damage, micronuclei formation), and " \
                    "changes in lipid or protein content. " \
                    "(v) Behavioral endpoints: impaired locomotion (e.g., reduced thrashing or body bends), feeding deficits, or other behavioral changes suggesting sub-lethal toxicity. " \
                    "Consider keywords such as 'toxicity', 'adverse effect', 'morbidity', 'reduced reproduction', 'mortality', " \
                    "'significant negative impact', 'hormonal disruption', 'DNA damage', 'bioaccumulation', or 'behavioral change' as indications of toxicity. " \
                    "Consider keywords such as 'no observed effect' (NOEL), 'no observed adverse effect' (NOAEL), 'not toxic', and similar terms " \
                    "as indications of non-toxicity. " \
                    "Considering all the information provided, respond only 'yes' or 'no' (and nothing else) to the following question: " \
                    "Is there any evidence in the input sentence of a toxic effect of a nanomaterial against the species 'Caenorhabditis elegans'? " \
                    "If there is no clear evidence of nanomaterial toxicity considering the aspects mentioned above, output 'no'. " \
                    "If the sentence does not deal with toxicity information against 'Caenorhabditis elegans' or if you cannot assess the sentence " \
                    "in regard to toxicity against 'Caenorhabditis elegans', output 'None'. " \
                    "Output 'yes', 'no' or 'None' between single quote characters (e.g, 'yes' or 'no' or 'None')."
            
                return t
            
            personality = toxicity_c_elegans_all()
            
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings


        elif parameter.lower() == 'toxicity_c_elegans_behavior':

            def toxicity_c_elegans_behavior(): 
        
                t = "You are a sentence analyzer specialized in nanotoxicology. " \
                    "Your task is to evaluate whether there is evidence of behavioral impairment caused by a nanomaterial in the biological species 'Caenorhabditis elegans' present in the input sentence. " \
                    "Evidence includes reduced locomotion, altered thrashing frequency, decreased body bends, impaired pharyngeal pumping, feeding deficits, or other behavioral changes. " \
                    "Consider keywords such as 'impaired locomotion', 'reduced thrashing', 'behavioral change', 'feeding deficit', or 'neuromuscular toxicity' as indications of behavioral impairment. " \
                    "Consider keywords such as 'normal behavior', 'locomotion unaffected', 'feeding unchanged', or 'no behavioral effect' as indications of non-toxicity. " \
                    "Considering all the information provided, respond only 'yes' or 'no'. " \
                    "Is there any evidence in the input sentence of behavioral impairment in 'Caenorhabditis elegans' caused by a nanomaterial? " \
                    "If there is no clear evidence of behavioral toxicity, output 'no'. " \
                    "If the sentence does not deal with behavior in 'Caenorhabditis elegans' or if you cannot assess the sentence for this endpoint, output 'None'. " \
                    "Output 'yes', 'no' or 'None' between single quote characters (e.g, 'yes' or 'no' or 'None')."

                return t

            personality = toxicity_c_elegans_behavior()
            
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings


        elif parameter.lower() == 'toxicity_c_elegans_bioaccumulation':

            def toxicity_c_elegans_bioaccumulation():

                t = "You are a sentence analyzer specialized in nanotoxicology. " \
                    "Your task is to evaluate whether there is evidence of bioaccumulation of a nanomaterial in the biological species 'Caenorhabditis elegans' present in the input sentence. "\
                    "Evidence of bioaccumulation includes terms such as 'bioaccumulation', 'bioconcentration', 'body burden', 'uptake', 'internalization', " \
                    "or a reported increase in internal nanomaterial concentration relative to the exposure medium. " \
                    "Consider keywords such as 'significant accumulation', 'increased body burden', 'biomagnification', or quantitative uptake values as indications of bioaccumulation. " \
                    "Consider keywords such as 'no accumulation', 'no uptake', 'clearance', 'depuration', or 'not detected' as indications of non-bioaccumulation. " \
                    "Considering all the information provided, respond only 'yes' or 'no' (and nothing else) to the following question. " \
                    "Is there any evidence in the input sentence of bioaccumulation of a nanomaterial in 'Caenorhabditis elegans'? " \
                    "If there is no clear evidence of bioaccumulation, output 'no'. " \
                    "If the sentence does not deal with bioaccumulation in 'Caenorhabditis elegans' or if you cannot assess the sentence for this endpoint, output 'None'. " \
                    "Output 'yes', 'no', or 'None' between single quote characters (e.g., 'yes', 'no', or 'None')."
                
                return t

            personality = toxicity_c_elegans_bioaccumulation()
            
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings


        elif parameter.lower() == 'toxicity_c_elegans_development':

            def toxicity_c_elegans_development():

                t = "You are a sentence analyzer specialized in nanotoxicology. " \
                    "Your task is to evaluate whether there is evidence of growth reduction or developmental delay caused by a nanomaterial in the biological species 'Caenorhabditis elegans' present in the input sentence. " \
                    "Evidence includes diminished body length, reduced growth rate, delayed larval development, or prolonged time to reach adulthood. " \
                    "Consider keywords such as 'reduced growth', 'diminished body length', 'developmental delay', or 'reduced size' as indications of toxicity. " \
                    "Consider keywords such as 'growth unaffected', 'normal development', or 'no developmental delay' as indications of non-toxicity. " \
                    "Considering all the information provided, respond only 'yes' or 'no'. " \
                    "Is there any evidence in the input sentence of growth reduction or developmental delay in 'Caenorhabditis elegans' caused by a nanomaterial? " \
                    "If there is no clear evidence of growth or developmental toxicity, output 'no'. " \
                    "If the sentence does not deal with growth or development in 'Caenorhabditis elegans' or if you cannot assess the sentence for this endpoint, output 'None'. " \
                    "Output 'yes', 'no' or 'None' between single quote characters (e.g, 'yes' or 'no' or 'None')."
                
                return t

            personality = toxicity_c_elegans_development()
            
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings


        elif parameter.lower() == 'toxicity_c_elegans_genotox':

            def toxicity_c_elegans_genotox():

                t = "You are a sentence analyzer specialized in nanotoxicology. " \
                    "Your task is to evaluate whether there is evidence of genotoxicity caused by a nanomaterial in the biological species 'Caenorhabditis elegans' present in the input sentence. " \
                    "Evidence of genotoxicity includes DNA strand breaks, comet assay results, micronuclei formation, chromosomal aberrations, or other reported DNA damage endpoints. " \
                    "Consider keywords such as 'DNA damage', 'genotoxic effect', 'micronuclei', 'comet assay tail moment', or 'chromosomal aberration' as indications of genotoxicity.  " \
                    "Consider keywords such as 'no DNA damage', 'no genotoxic effect', or 'genome integrity unchanged' as indications of non-genotoxicity. " \
                    "Considering all the information provided, respond only 'yes' or 'no' to the following question. " \
                    "Is there any evidence in the input sentence of genotoxicity induced by a nanomaterial in 'Caenorhabditis elegans'? " \
                    "If there is no clear evidence of genotoxicity, output 'no'. " \
                    "If the sentence does not deal with genotoxicity in 'Caenorhabditis elegans' or if you cannot assess the sentence for this endpoint, output 'None'. " \
                    "Output 'yes', 'no' or 'None' between single quote characters (e.g, 'yes' or 'no' or 'None')."

                return t

            personality = toxicity_c_elegans_genotox()
            
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings


        elif parameter.lower() == 'toxicity_c_elegans_enzyme':

            def toxicity_c_elegans_enzyme():

                t = "You are a sentence analyzer specialized in nanotoxicology. " \
                    "Your task is to evaluate whether there is evidence of enzymatic activity alteration caused by a nanomaterial in the biological species 'Caenorhabditis elegans' present in the input sentence. " \
                    "Evidence includes changes in catalase, superoxide dismutase, glutathione peroxidase, acetylcholinesterase, or other physiological enzymes. " \
                    "Consider keywords such as 'altered catalase activity', 'SOD increase', 'enzyme inhibition', or 'enzyme induction' as indications of toxicity. " \
                    "Consider keywords such as 'enzyme activity unchanged', 'no enzymatic effect', or 'normal catalase levels' as indications of non-toxicity. " \
                    "Considering all the information provided, respond only 'yes' or 'no'. " \
                    "Is there any evidence in the input sentence of enzymatic activity alteration in 'Caenorhabditis elegans' caused by a nanomaterial? " \
                    "If there is no clear evidence of enzymatic alteration, output 'no'. " \
                    "If the sentence does not deal with enzymatic activity in 'Caenorhabditis elegans' or if you cannot assess the sentence for this endpoint, output 'None'. " \
                    "Output 'yes', 'no' or 'None' between single quote characters (e.g, 'yes' or 'no' or 'None')."

                return t

            personality = toxicity_c_elegans_enzyme()
            
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings


        elif parameter.lower() == 'toxicity_c_elegans_morphology':

            def toxicity_c_elegans_morphology():

                t = "You are a sentence analyzer specialized in nanotoxicology. " \
                    "Your task is to evaluate whether there is evidence of morphological abnormalities caused by a nanomaterial in the biological species 'Caenorhabditis elegans' present in the input sentence. " \
                    "Evidence includes anatomical malformations, cuticle deformities, vulval abnormalities, or general sub-lethal structural changes. " \
                    "Consider keywords such as 'malformation', 'morphological change', 'deformed', 'abnormal anatomy', or 'structural defect' as indications of toxicity. " \
                    "Consider keywords such as 'normal morphology', 'no anatomical changes', or 'structure unchanged' as indications of non-toxicity. " \
                    "Considering all the information provided, respond only 'yes' or 'no'. " \
                    "Is there any evidence in the input sentence of morphological abnormalities in 'Caenorhabditis elegans' caused by a nanomaterial? " \
                    "If there is no clear evidence of morphological toxicity, output 'no'. " \
                    "If the sentence does not deal with morphology in 'Caenorhabditis elegans' or if you cannot assess the sentence for this endpoint, output 'None'. " \
                    "Output 'yes', 'no' or 'None' between single quote characters (e.g, 'yes' or 'no' or 'None')."

                return t

            personality = toxicity_c_elegans_morphology()
            
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings


        elif parameter.lower() == 'toxicity_c_elegans_mortality':

            def toxicity_c_elegans_mortality():

                t = "You are a sentence analyzer specialized in nanotoxicology. " \
                    "Your task is to evaluate whether there is evidence of acute mortality or decreased viability caused by a nanomaterial in the biological species 'Caenorhabditis elegans' present in the input sentence. " \
                    "Evidence of acute toxicity includes increased mortality, reduced survival rate, or decreased viability within typical short-term exposure periods of 24–48 hours. " \
                    "Consider keywords such as 'mortality', 'lethality', 'reduced survival', 'decreased viability', or 'LC50' as indications of acute toxicity. " \
                    "Consider keywords such as 'no mortality', 'survival unchanged', or 'viability unaffected' as indications of non-acute toxicity. " \
                    "Considering all the information provided, respond only 'yes' or 'no'. " \
                    "Is there any evidence in the input sentence of acute mortality or decreased viability of 'Caenorhabditis elegans' caused by a nanomaterial? " \
                    "If there is no clear evidence of acute toxicity, output 'no'. " \
                    "If the sentence does not deal with acute mortality or viability in 'Caenorhabditis elegans' or if you cannot assess the sentence for this endpoint, output 'None'. " \
                    "Output 'yes', 'no' or 'None' between single quote characters (e.g, 'yes' or 'no' or 'None')."

                return t

            personality = toxicity_c_elegans_mortality()
            
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings


        elif parameter.lower() == 'toxicity_c_elegans_oxi_stress':

            def toxicity_c_elegans_oxi_stress():

                t = "You are a sentence analyzer specialized in nanotoxicology. " \
                    "Your task is to evaluate whether there is evidence of oxidative stress caused by a nanomaterial in the biological species 'Caenorhabditis elegans' present in the input sentence. " \
                    "Evidence of oxidative stress includes reactive oxygen species (ROS) generation, lipid peroxidation, protein carbonylation, " \
                    "glutathione depletion, or altered antioxidant enzyme activities such as catalase or superoxide dismutase. " \
                    "Consider keywords such as 'elevated ROS', 'increased lipid peroxidation', 'oxidative damage', 'redox imbalance', " \
                    "or 'antioxidant enzyme depletion' as indications of oxidative stress. " \
                    "Consider keywords such as 'no oxidative stress', 'unchanged ROS', 'no lipid peroxidation', or 'antioxidant status unchanged' as indications of non-oxidative stress. " \
                    "Considering all the information provided, respond only 'yes' or 'no' (and nothing else) to the following question. " \
                    "Is there any evidence in the input sentence of oxidative stress induced by a nanomaterial in 'Caenorhabditis elegans'? " \
                    "If there is no clear evidence of oxidative stress, output 'no'. " \
                    "If the sentence does not deal with oxidative stress in 'Caenorhabditis elegans' or if you cannot assess the sentence for this endpoint, output 'None'. " \
                    "Output 'yes', 'no', or 'None' between single quote characters (e.g., 'yes', 'no', or 'None'). "

                return t

            personality = toxicity_c_elegans_oxi_stress()
            
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings
        

        elif parameter.lower() == 'toxicity_c_elegans_reproduction':

            def toxicity_c_elegans_reproduction():

                t = "You are a sentence analyzer specialized in nanotoxicology. " \
                    "Your task is to evaluate whether there is evidence of reproductive impairment caused by a nanomaterial in the biological species 'Caenorhabditis elegans' present in the input sentence. " \
                    "Evidence of reproductive toxicity includes reduced brood size, decreased egg production, impaired fertility, or embryonic lethality. " \
                    "Consider keywords such as 'reduced reproduction', 'decreased brood size', 'fertility decline', 'embryo lethality', or 'reproductive toxicity' as indications of reproductive impairment. " \
                    "Consider keywords such as 'no effect on reproduction', 'brood size unaffected', or 'fertility unchanged' as indications of non-reproductive toxicity. " \
                    "Considering all the information provided, respond only 'yes' or 'no'. " \
                    "Is there any evidence in the input sentence of reproductive impairment of 'Caenorhabditis elegans' caused by a nanomaterial? " \
                    "If there is no clear evidence of reproductive toxicity, output 'no'. " \
                    "If the sentence does not deal with reproduction in 'Caenorhabditis elegans' or if you cannot assess the sentence for this endpoint, output 'None'. " \
                    "Output 'yes', 'no' or 'None' between single quote characters (e.g, 'yes' or 'no' or 'None')."

                return t

            personality = toxicity_c_elegans_reproduction()
            
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings


        elif parameter.lower() == 'toxicity_d_magna_all':

            def toxicity_d_magna_all():

                t = "You are a sentence analyzer specialized in nanotoxicology. " \
                    "Your task is to evaluate whether there is evidence of toxicity of a nanomaterial tested on the biological species 'Daphnia magna' in the input sentences. " \
                    "Evidence of toxicity includes: " \
                    "(i) Acute toxicity (short-term tests): mortality and immobilization rates over 24–48 hours. " \
                    "(ii) Chronic toxicity (long-term tests): reproductive output (number of offspring), growth or body size changes, " \
                    "time to first brood, and changes in behavior (feeding rate, swimming ability). " \
                    "(iii) Developmental and morphological changes: any reported anatomical malformations, morphological alterations, or general sub-lethal stress. " \
                    "(iv) Physiological and biochemical markers: oxidative stress indicators (e.g., ROS, lipid peroxidation), " \
                    "enzymatic activity shifts (e.g., catalase, superoxide dismutase), genotoxicity (DNA damage, micronuclei formation), and " \
                    "changes in lipid or protein content. " \
                    "(v) Behavioral endpoints: reduced feeding or altered swimming patterns that might suggest sub-lethal toxicity. " \
                    "Consider keywords such as 'toxicity', 'adverse effect', 'morbidity', 'reduced reproduction', 'mortality', 'immobilization', " \
                    "'significant negative impact', 'hormonal disruption', 'DNA damage', 'bioaccumulation', or 'behavioral change' as indications of toxicity. " \
                    "Consider keywords such as 'no observed effect' (NOEL), 'no observed adverse effect' (NOAEL), 'not toxic', and similar terms " \
                    "as indications of non-toxicity. " \
                    "Considering all the information provided, respond only 'yes' or 'no' (and nothing else) to the following question: " \
                    "Is there any evidence in the input sentence of a toxic effect of a nanomaterial against the species 'Daphnia magna'? " \
                    "If there is no clear evidence of nanomaterial toxicity considering the aspects mentioned above, output 'no'. " \
                    "If the sentence does not deal with toxicity information against 'Daphnia magna' or if you cannot assess the sentence " \
                    "in regard to toxicity against 'Daphnia magna', output 'None'. " \
                    "Output 'yes', 'no' or 'None' between single quote characters (e.g, 'yes' or 'no' or 'None')."
                
                return t

            personality = toxicity_d_magna_all()
            
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings


        elif parameter.lower() == 'toxicity_d_magna_behavior':

            def toxicity_d_magna_behavior():

                t = "You are a sentence analyzer specialized in nanotoxicology. " \
                    "Your task is to evaluate whether there is evidence of behavioral impairment caused by a nanomaterial in the biological species 'Daphnia magna' present in the input sentence. " \
                    "Evidence includes reduced feeding rate, altered swimming speed, impaired phototaxis, or abnormal escape responses. " \
                    "Consider keywords such as 'reduced feeding', 'diminished swimming', 'behavioral change', or 'impaired mobility' as indications of toxicity. " \
                    "Consider keywords such as 'normal behavior', 'swimming unaffected', or 'no behavioral effect' as indications of non-toxicity. " \
                    "Considering all the information provided, respond only 'yes' or 'no'. " \
                    "Is there any evidence in the input sentence of behavioral impairment in 'Daphnia magna' caused by a nanomaterial? " \
                    "If there is no clear evidence of behavioral toxicity, output 'no'. " \
                    "If the sentence does not deal with behavior in 'Daphnia magna' or if you cannot assess the sentence for this endpoint, output 'None'. " \
                    "Output 'yes', 'no' or 'None' between single quote characters (e.g, 'yes' or 'no' or 'None')."
                
                return t

            personality = toxicity_d_magna_behavior()
            
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings


        elif parameter.lower() == 'toxicity_d_magna_bioaccumulation':

            def toxicity_d_magna_bioaccumulation():

                t = "You are a sentence analyzer specialized in nanotoxicology. " \
                    "Your task is to evaluate whether there is evidence of bioaccumulation of a nanomaterial in the biological species 'Daphnia magna' present in the input sentence. " \
                    "Evidence of bioaccumulation includes increased internal nanomaterial concentration relative to the exposure medium, or trophic transfer potential. " \
                    "Consider keywords such as 'bioaccumulation', 'bioconcentration', 'body burden', 'internalization', or quantitative uptake values as indications of bioaccumulation. " \
                    "Consider keywords such as 'no accumulation', 'no uptake', 'depuration', or 'not detected' as indications of non-bioaccumulation. " \
                    "Considering all the information provided, respond only 'yes' or 'no'. " \
                    "Is there any evidence in the input sentence of bioaccumulation of a nanomaterial in 'Daphnia magna'? " \
                    "If there is no clear evidence of bioaccumulation, output 'no'. " \
                    "If the sentence does not deal with bioaccumulation in 'Daphnia magna' or if you cannot assess the sentence for this endpoint, output 'None'. " \
                    "Output 'yes', 'no' or 'None' between single quote characters (e.g, 'yes' or 'no' or 'None')."

                return t

            personality = toxicity_d_magna_bioaccumulation()
            
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings
        

        elif parameter.lower() == 'toxicity_d_magna_development':

            def toxicity_d_magna_development():

                t = "You are a sentence analyzer specialized in nanotoxicology. " \
                    "Your task is to evaluate whether there is evidence of growth reduction or body-size change caused by a nanomaterial in the biological species 'Daphnia magna' present in the input sentence. " \
                    "Evidence includes diminished body length, reduced dry weight, slower growth rate, or stunted individuals. " \
                    "Consider keywords such as 'reduced growth', 'smaller body size', 'growth inhibition', or 'stunted daphnids' as indications of toxicity. " \
                    "Consider keywords such as 'growth unaffected', 'normal size', or 'no growth inhibition' as indications of non-toxicity. " \
                    "Considering all the information provided, respond only 'yes' or 'no'. " \
                    "Is there any evidence in the input sentence of growth reduction in 'Daphnia magna' caused by a nanomaterial? " \
                    "If there is no clear evidence of growth toxicity, output 'no'. " \
                    "If the sentence does not deal with growth in 'Daphnia magna' or if you cannot assess the sentence for this endpoint, output 'None'. " \
                    "Output 'yes', 'no' or 'None' between single quote characters (e.g, 'yes' or 'no' or 'None')."

                return t

            personality = toxicity_d_magna_development()
            
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings


        elif parameter.lower() == 'toxicity_d_magna_enzyme':

            def toxicity_d_magna_enzyme():

                t = "You are a sentence analyzer specialized in nanotoxicology. " \
                    "Your task is to evaluate whether there is evidence of enzymatic activity alteration caused by a nanomaterial in the biological species 'Daphnia magna' present in the input sentence. " \
                    "Evidence includes changes in catalase, superoxide dismutase, glutathione-S-transferase, acetylcholinesterase, or other key enzymes. " \
                    "Consider keywords such as 'altered catalase', 'SOD increase', 'enzyme inhibition', or 'enzyme induction' as indications of toxicity. " \
                    "Consider keywords such as 'enzyme activity unchanged', 'no enzymatic effect', or 'normal catalase levels' as indications of non-toxicity. " \
                    "Considering all the information provided, respond only 'yes' or 'no'. " \
                    "Is there any evidence in the input sentence of enzymatic activity alteration in 'Daphnia magna' caused by a nanomaterial? " \
                    "If there is no clear evidence of enzymatic alteration, output 'no'. " \
                    "If the sentence does not deal with enzymatic activity in 'Daphnia magna' or if you cannot assess the sentence for this endpoint, output 'None'. " \
                    "Output 'yes', 'no' or 'None' between single quote characters (e.g, 'yes' or 'no' or 'None')."
                
                return t

            personality = toxicity_d_magna_enzyme()
            
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings


        elif parameter.lower() == 'toxicity_d_magna_genotox':

            def toxicity_d_magna_genotox():

                t = "You are a sentence analyzer specialized in nanotoxicology. " \
                    "Your task is to evaluate whether there is evidence of genotoxicity caused by a nanomaterial in the biological species 'Daphnia magna' present in the input sentence. " \
                    "Evidence includes DNA strand breaks, comet assay results, micronuclei formation, or chromosomal aberrations. " \
                    "Consider keywords such as 'DNA damage', 'genotoxic effect', 'comet assay', 'micronuclei', or 'chromosomal aberration' as indications of genotoxicity. " \
                    "Consider keywords such as 'no DNA damage', 'no genotoxic effect', or 'genome integrity unchanged' as indications of non-toxicity. " \
                    "Considering all the information provided, respond only 'yes' or 'no'. " \
                    "Is there any evidence in the input sentence of genotoxicity in 'Daphnia magna' caused by a nanomaterial? " \
                    "If there is no clear evidence of genotoxicity, output 'no'. " \
                    "If the sentence does not deal with DNA damage in 'Daphnia magna' or if you cannot assess the sentence for this endpoint, output 'None'. " \
                    "Output 'yes', 'no' or 'None' between single quote characters (e.g, 'yes' or 'no' or 'None')."
                
                return t

            personality = toxicity_d_magna_genotox()
            
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings


        elif parameter.lower() == 'toxicity_d_magna_morphology':

            def toxicity_d_magna_morphology():

                t = "You are a sentence analyzer specialized in nanotoxicology. " \
                    "Your task is to evaluate whether there is evidence of developmental or morphological abnormalities caused by a nanomaterial in the biological species 'Daphnia magna' present in the input sentence. " \
                    "Evidence includes carapace deformities, abnormal antennae, delayed molting, or other reported anatomical changes. " \
                    "Consider keywords such as 'malformation', 'deformed', 'abnormal morphology', or 'delayed development' as indications of toxicity. " \
                    "Consider keywords such as 'normal morphology', 'no deformities', or 'structure unchanged' as indications of non-toxicity. " \
                    "Considering all the information provided, respond only 'yes' or 'no'. " \
                    "Is there any evidence in the input sentence of developmental or morphological abnormalities in 'Daphnia magna' caused by a nanomaterial? " \
                    "If there is no clear evidence of developmental toxicity, output 'no'. " \
                    "If the sentence does not deal with morphology in 'Daphnia magna' or if you cannot assess the sentence for this endpoint, output 'None'. " \
                    "Output 'yes', 'no' or 'None' between single quote characters (e.g, 'yes' or 'no' or 'None')."
                
                return t

            personality = toxicity_d_magna_morphology()
            
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings
        

        elif parameter.lower() == 'toxicity_d_magna_mortality':

            def toxicity_d_magna_mortality():

                t = "You are a sentence analyzer specialized in nanotoxicology. " \
                    "Your task is to evaluate whether there is evidence of acute mortality or immobilization caused by a nanomaterial in the biological species 'Daphnia magna' present in the input sentence. " \
                    "Evidence of acute toxicity includes increased mortality, immobilization, or reduced survival within 24–48 h exposure periods. " \
                    "Consider keywords such as 'mortality', 'lethality', 'immobilization', 'LC50', or 'reduced survival' as indications of acute toxicity. " \
                    "Consider keywords such as 'no mortality', 'survival unchanged', or 'immobilization not observed' as indications of non-toxicity. " \
                    "Considering all the information provided, respond only 'yes' or 'no'. " \
                    "Is there any evidence in the input sentence of acute mortality or immobilization in 'Daphnia magna' caused by a nanomaterial? " \
                    "If there is no clear evidence of acute toxicity, output 'no'. " \
                    "If the sentence does not deal with acute mortality or immobilization in 'Daphnia magna' or if you cannot assess the sentence for this endpoint, output 'None'. " \
                    "Output 'yes', 'no' or 'None' between single quote characters (e.g, 'yes' or 'no' or 'None')."
                
                return t

            personality = toxicity_d_magna_mortality()
            
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings


        elif parameter.lower() == 'toxicity_d_magna_oxi_stress':

            def toxicity_d_magna_oxi_stress():

                t = "You are a sentence analyzer specialized in nanotoxicology. " \
                    "Your task is to evaluate whether there is evidence of oxidative stress caused by a nanomaterial in the biological species 'Daphnia magna' present in the input sentence. " \
                    "Evidence of oxidative stress includes elevated reactive oxygen species (ROS), increased lipid peroxidation, protein carbonylation, or glutathione depletion. " \
                    "Consider keywords such as 'elevated ROS', 'increased MDA', 'oxidative damage', or 'lipid peroxidation' as indications of oxidative stress. " \
                    "Consider keywords such as 'no oxidative stress', 'ROS unchanged', or 'antioxidant status normal' as indications of non-toxicity. " \
                    "Considering all the information provided, respond only 'yes' or 'no'. " \
                    "Is there any evidence in the input sentence of oxidative stress in 'Daphnia magna' caused by a nanomaterial? " \
                    "If there is no clear evidence of oxidative stress, output 'no'. " \
                    "If the sentence does not deal with oxidative stress in 'Daphnia magna' or if you cannot assess the sentence for this endpoint, output 'None'. " \
                    "Output 'yes', 'no' or 'None' between single quote characters (e.g, 'yes' or 'no' or 'None')."
                
                return t

            personality = toxicity_d_magna_oxi_stress()
            
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings


        elif parameter.lower() == 'toxicity_d_magna_reproduction':

            def toxicity_d_magna_reproduction():

                t = "You are a sentence analyzer specialized in nanotoxicology. " \
                    "Your task is to evaluate whether there is evidence of reproductive impairment caused by a nanomaterial in the biological species 'Daphnia magna' present in the input sentence. " \
                    "Evidence of reproductive toxicity includes reduced number of offspring, smaller broods, delayed first brood, or decreased fecundity. " \
                    "Consider keywords such as 'reduced reproduction', 'fewer neonates', 'brood size decline', or 'reproductive toxicity' as indications of toxicity. " \
                    "Consider keywords such as 'no effect on reproduction', 'brood size unaffected', or 'fecundity unchanged' as indications of non-toxicity. " \
                    "Considering all the information provided, respond only 'yes' or 'no'. " \
                    "Is there any evidence in the input sentence of reproductive impairment of 'Daphnia magna' caused by a nanomaterial? " \
                    "If there is no clear evidence of reproductive toxicity, output 'no'. " \
                    "If the sentence does not deal with reproduction in 'Daphnia magna' or if you cannot assess the sentence for this endpoint, output 'None'. " \
                    "Output 'yes', 'no' or 'None' between single quote characters (e.g, 'yes' or 'no' or 'None')."
                
                return t

            personality = toxicity_d_magna_reproduction()
            
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings


        elif parameter.lower() == 'toxicity_d_rerio_all':

            def toxicity_d_rerio_all():

                t = "You are a sentence analyzer specialized in nanotoxicology. " \
                    "Your task is to evaluate whether there is evidence of toxicity of a nanomaterial tested on the biological species 'Danio rerio' in the input sentences. " \
                    "Evidence of toxicity includes: " \
                    "(i) Acute toxicity (short-term tests): mortality, abnormal hatching rates, or other acute effects over typical exposure periods (e.g., 24–96 hours). " \
                    "(ii) Chronic toxicity (long-term tests): reduced growth or body length, altered reproductive output (e.g., number of eggs laid or hatchlings), " \
                    "and prolonged effects on physiology (e.g., organ development). " \
                    "(iii) Developmental and morphological changes: any reported anatomical malformations, deformities, or general sub-lethal stress such as abnormal swim bladder inflation. " \
                    "(iv) Physiological and biochemical markers: oxidative stress indicators (e.g., ROS, lipid peroxidation), " \
                    "enzymatic activity shifts (e.g., catalase, superoxide dismutase), genotoxicity (DNA damage, micronuclei formation), and " \
                    "changes in lipid or protein content. " \
                    "(v) Behavioral endpoints: altered swimming patterns, feeding behavior, or any other observed sub-lethal behavioral changes. " \
                    "Consider keywords such as 'toxicity', 'adverse effect', 'morbidity', 'reduced reproduction', 'mortality', 'hatching failure', " \
                    "'significant negative impact', 'hormonal disruption', 'DNA damage', 'bioaccumulation', or 'behavioral change' as indications of toxicity. " \
                    "Consider keywords such as 'no observed effect' (NOEL), 'no observed adverse effect' (NOAEL), 'not toxic', and similar terms " \
                    "as indications of non-toxicity. " \
                    "Considering all the information provided, respond only 'yes' or 'no' (and nothing else) to the following question: " \
                    "Is there any evidence in the input sentence of a toxic effect of a nanomaterial against the species 'Danio rerio'? " \
                    "If there is no clear evidence of nanomaterial toxicity considering the aspects mentioned above, output 'no'. " \
                    "If the sentence does not deal with toxicity information against 'Danio rerio' or if you cannot assess the sentence " \
                    "in regard to toxicity against 'Danio rerio', output 'None'. " \
                    "Output 'yes', 'no' or 'None' between single quote characters (e.g, 'yes' or 'no' or 'None')."
                
                return t

            personality = toxicity_d_rerio_all()
            
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings


        elif parameter.lower() == 'toxicity_d_rerio_behavior':

            def toxicity_d_rerio_behavior():

                t = "You are a sentence analyzer specialized in nanotoxicology. " \
                    "Your task is to evaluate whether there is evidence of behavioral impairment caused by a nanomaterial in the biological species 'Danio rerio' present in the input sentence. " \
                    "Evidence includes altered swimming speed, erratic movement, reduced feeding, impaired predator avoidance, or other behavioral changes. " \
                    "Consider keywords such as 'altered swimming', 'reduced feeding', 'behavioral change', or 'impaired locomotion' as indications of toxicity. " \
                    "Consider keywords such as 'normal behavior', 'swimming unaffected', or 'no behavioral effect' as indications of non-toxicity. " \
                    "Considering all the information provided, respond only 'yes' or 'no'. " \
                    "Is there any evidence in the input sentence of behavioral impairment in 'Danio rerio' caused by a nanomaterial? " \
                    "If there is no clear evidence of behavioral toxicity, output 'no'. " \
                    "If the sentence does not deal with behavior in 'Danio rerio' or if you cannot assess the sentence for this endpoint, output 'None'. " \
                    "Output 'yes', 'no' or 'None' between single quote characters (e.g, 'yes' or 'no' or 'None')."

                return t

            personality = toxicity_d_rerio_behavior()
            
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings


        elif parameter.lower() == 'toxicity_d_rerio_bioaccumulation':

            def toxicity_d_rerio_bioaccumulation():

                t = "You are a sentence analyzer specialized in nanotoxicology. " \
                    "Your task is to evaluate whether there is evidence of bioaccumulation of a nanomaterial in the biological species 'Danio rerio' present in the input sentence. " \
                    "Evidence of bioaccumulation includes elevated nanomaterial concentrations in tissues, organ-specific uptake, or trophic transfer potential. " \
                    "Consider keywords such as 'bioaccumulation', 'bioconcentration', 'body burden', 'internalization', or quantitative uptake values as indications of bioaccumulation. " \
                    "Consider keywords such as 'no accumulation', 'no uptake', 'depuration', or 'not detected' as indications of non-bioaccumulation. " \
                    "Considering all the information provided, respond only 'yes' or 'no'. " \
                    "Is there any evidence in the input sentence of bioaccumulation of a nanomaterial in 'Danio rerio'? " \
                    "If there is no clear evidence of bioaccumulation, output 'no'. " \
                    "If the sentence does not deal with bioaccumulation in 'Danio rerio' or if you cannot assess the sentence for this endpoint, output 'None'. " \
                    "Output 'yes', 'no' or 'None' between single quote characters (e.g, 'yes' or 'no' or 'None')."
                
                return t

            personality = toxicity_d_rerio_bioaccumulation()
            
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings
        

        elif parameter.lower() == 'toxicity_d_rerio_development':

            def toxicity_d_rerio_development():

                t = "You are a sentence analyzer specialized in nanotoxicology. " \
                    "Your task is to evaluate whether there is evidence of growth suppression caused by a nanomaterial in the biological species 'Danio rerio' present in the input sentence. " \
                    "Evidence of growth suppression includes reduced body length, decreased weight, stunted larvae, or overall inhibited growth. " \
                    "Consider keywords such as 'reduced growth', 'shorter body length', 'stunted larvae', or 'growth inhibition' as indications of toxicity. " \
                    "Consider keywords such as 'growth unaffected', 'normal size', or 'no growth inhibition' as indications of non-toxicity. " \
                    "Considering all the information provided, respond only 'yes' or 'no'. " \
                    "Is there any evidence in the input sentence of growth suppression in 'Danio rerio' caused by a nanomaterial? " \
                    "If there is no clear evidence of growth toxicity, output 'no'. " \
                    "If the sentence does not deal with growth in 'Danio rerio' or if you cannot assess the sentence for this endpoint, output 'None'. " \
                    "Output 'yes', 'no' or 'None' between single quote characters (e.g, 'yes' or 'no' or 'None')."
                
                return t

            personality = toxicity_d_rerio_development()
            
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings


        elif parameter.lower() == 'toxicity_d_rerio_enzyme':

            def toxicity_d_rerio_enzyme():

                t = "You are a sentence analyzer specialized in nanotoxicology. " \
                    "Your task is to evaluate whether there is evidence of enzymatic activity alteration caused by a nanomaterial in the biological species 'Danio rerio' present in the input sentence. " \
                    "Evidence includes changes in catalase, superoxide dismutase, glutathione-S-transferase, acetylcholinesterase, or other key enzymes. " \
                    "Consider keywords such as 'altered catalase', 'SOD increase', 'enzyme inhibition', or 'enzyme induction' as indications of toxicity. " \
                    "Consider keywords such as 'enzyme activity unchanged', 'no enzymatic effect', or 'normal catalase levels' as indications of non-toxicity. " \
                    "Considering all the information provided, respond only 'yes' or 'no'. " \
                    "Is there any evidence in the input sentence of enzymatic activity alteration in 'Danio rerio' caused by a nanomaterial? " \
                    "If there is no clear evidence of enzymatic alteration, output 'no'. " \
                    "If the sentence does not deal with enzymatic activity in 'Danio rerio' or if you cannot assess the sentence for this endpoint, output 'None'. " \
                    "Output 'yes', 'no' or 'None' between single quote characters (e.g, 'yes' or 'no' or 'None')."
                
                return t

            personality = toxicity_d_rerio_enzyme()
            
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings
        

        elif parameter.lower() == 'toxicity_d_rerio_genotox':

            def toxicity_d_rerio_genotox():

                t = "You are a sentence analyzer specialized in nanotoxicology. " \
                    "Your task is to evaluate whether there is evidence of genotoxicity caused by a nanomaterial in the biological species 'Danio rerio' present in the input sentence. " \
                    "Evidence includes DNA strand breaks, comet assay results, micronuclei formation, γ-H2AX foci, or chromosomal aberrations. " \
                    "Consider keywords such as 'DNA damage', 'genotoxic effect', 'comet assay', 'micronuclei', or 'chromosomal aberration' as indications of genotoxicity. " \
                    "Consider keywords such as 'no DNA damage', 'no genotoxic effect', or 'genome integrity unchanged' as indications of non-toxicity. " \
                    "Considering all the information provided, respond only 'yes' or 'no'. " \
                    "Is there any evidence in the input sentence of genotoxicity in 'Danio rerio' caused by a nanomaterial? " \
                    "If there is no clear evidence of genotoxicity, output 'no'. " \
                    "If the sentence does not deal with DNA damage in 'Danio rerio' or if you cannot assess the sentence for this endpoint, output 'None'. " \
                    "Output 'yes', 'no' or 'None' between single quote characters (e.g, 'yes' or 'no' or 'None')."
                
                return t

            personality = toxicity_d_rerio_genotox()
            
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings
        

        elif parameter.lower() == 'toxicity_d_rerio_morphology':

            def toxicity_d_rerio_morphology():

                t = "You are a sentence analyzer specialized in nanotoxicology. " \
                    "Your task is to evaluate whether there is evidence of developmental or morphological deformities caused by a nanomaterial in the biological species 'Danio rerio' present in the input sentence. " \
                    "Evidence includes spinal curvature, edema, craniofacial malformations, abnormal swim bladder inflation, or other anatomical changes. " \
                    "Consider keywords such as 'malformation', 'deformity', 'spinal curvature', 'pericardial edema', or 'abnormal swim bladder' as indications of toxicity. " \
                    "Consider keywords such as 'normal morphology', 'no deformities', or 'structure unchanged' as indications of non-toxicity. " \
                    "Considering all the information provided, respond only 'yes' or 'no'. " \
                    "Is there any evidence in the input sentence of developmental or morphological deformities in 'Danio rerio' caused by a nanomaterial? " \
                    "If there is no clear evidence of developmental toxicity, output 'no'. " \
                    "If the sentence does not deal with morphology in 'Danio rerio' or if you cannot assess the sentence for this endpoint, output 'None'. " \
                    "Output 'yes', 'no' or 'None' between single quote characters (e.g, 'yes' or 'no' or 'None')."
                
                return t

            personality = toxicity_d_rerio_morphology()
            
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings


        elif parameter.lower() == 'toxicity_d_rerio_mortality':

            def toxicity_d_rerio_mortality():

                t = "You are a sentence analyzer specialized in nanotoxicology. " \
                    "Your task is to evaluate whether there is evidence of acute mortality or abnormal hatching caused by a nanomaterial in the biological species 'Danio rerio' present in the input sentence. " \
                    "Evidence of acute toxicity includes increased mortality, abnormal or delayed hatching, reduced survival, or other effects within 24–96 h exposure periods. " \
                    "Consider keywords such as 'mortality', 'lethality', 'hatching failure', 'reduced survival', or 'LC50' as indications of acute toxicity. " \
                    "Consider keywords such as 'no mortality', 'normal hatching', or 'survival unchanged' as indications of non-toxicity. " \
                    "Considering all the information provided, respond only 'yes' or 'no'. " \
                    "Is there any evidence in the input sentence of acute mortality or abnormal hatching in 'Danio rerio' caused by a nanomaterial? " \
                    "If there is no clear evidence of acute toxicity, output 'no'. " \
                    "If the sentence does not deal with acute mortality or hatching in 'Danio rerio' or if you cannot assess the sentence for this endpoint, output 'None'. " \
                    "Output 'yes', 'no' or 'None' between single quote characters (e.g, 'yes' or 'no' or 'None')."
                
                return t

            personality = toxicity_d_rerio_mortality()
            
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings


        elif parameter.lower() == 'toxicity_d_rerio_oxi_stress':

            def toxicity_d_rerio_oxi_stress():

                t = "You are a sentence analyzer specialized in nanotoxicology. " \
                    "Your task is to evaluate whether there is evidence of oxidative stress caused by a nanomaterial in the biological species 'Danio rerio' present in the input sentence. " \
                    "Evidence of oxidative stress includes elevated reactive oxygen species (ROS), increased lipid peroxidation, protein carbonylation, glutathione depletion, or altered antioxidant enzyme activities. " \
                    "Consider keywords such as 'elevated ROS', 'increased MDA', 'oxidative damage', 'redox imbalance', or 'lipid peroxidation' as indications of oxidative stress. " \
                    "Consider keywords such as 'no oxidative stress', 'ROS unchanged', or 'antioxidant status normal' as indications of non-toxicity. " \
                    "Considering all the information provided, respond only 'yes' or 'no'. " \
                    "Is there any evidence in the input sentence of oxidative stress in 'Danio rerio' caused by a nanomaterial? " \
                    "If there is no clear evidence of oxidative stress, output 'no'. " \
                    "If the sentence does not deal with oxidative stress in 'Danio rerio' or if you cannot assess the sentence for this endpoint, output 'None'. " \
                    "Output 'yes', 'no' or 'None' between single quote characters (e.g, 'yes' or 'no' or 'None')."
                
                return t

            personality = toxicity_d_rerio_oxi_stress()
            
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings
        

        elif parameter.lower() == 'toxicity_d_rerio_reproduction':

            def toxicity_d_rerio_reproduction():

                t = "You are a sentence analyzer specialized in nanotoxicology. " \
                    "Your task is to evaluate whether there is evidence of reproductive impairment caused by a nanomaterial in the biological species 'Danio rerio' present in the input sentence. " \
                    "Evidence of reproductive toxicity includes reduced number of eggs laid, lower fertilization rate, decreased hatchling count, or impaired gonadal development. " \
                    "Consider keywords such as 'reduced reproduction', 'fewer eggs', 'fertility decline', or 'reproductive toxicity' as indications of toxicity. " \
                    "Consider keywords such as 'no effect on reproduction', 'egg production normal', or 'fertility unchanged' as indications of non-toxicity. " \
                    "Considering all the information provided, respond only 'yes' or 'no'. " \
                    "Is there any evidence in the input sentence of reproductive impairment in 'Danio rerio' caused by a nanomaterial? " \
                    "If there is no clear evidence of reproductive toxicity, output 'no'. " \
                    "If the sentence does not deal with reproduction in 'Danio rerio' or if you cannot assess the sentence for this endpoint, output 'None'. " \
                    "Output 'yes', 'no' or 'None' between single quote characters (e.g, 'yes' or 'no' or 'None')."
                
                return t

            personality = toxicity_d_rerio_reproduction()
            
            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             #{"role": "assistant", "content": llm_responses},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0, 'top_p': 0.3},
                                   stream = False
                                   )
            
            findings = re.findall(r"(?<=\')[A-Za-z0-9\.\s]+(?=\')", response['message']['content'])
            
            print('> LLM response: ', findings)
            return findings


        else:
            return ''