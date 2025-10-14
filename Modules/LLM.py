import ollama# type: ignore
import time
import regex as re # type: ignore

from PROMPT import prompt
from functions_TEXTS import get_filename_from_sent_index
from functions_TEXTS import get_sent_from_filename_sent_index

class llm(object):

    def __init__(self, model):
        
        self.prompt = prompt()

        if model in [  ollama.list()['models'][i]['model'] for i in range(len(ollama.list()['models'])) ]:        
            self.model = model
            self.client = ollama.Client()
        
        else:
            print(f'\nERRO! LLM {model} não encontrado na bibioteca OLLAMA.\n')
            time.sleep(5)


    
    def chat(self, input_text, personality: str = '', think: str = False, param_type: str = 'numerical'):

        #modelos com reasoning
        print('> chat with ', self.model)
        thinking = ''
        if self.model in ("gpt-oss:20b", "qwen3:30b", "qwen3:32b"):
            thinking = " Your final output MUST be generated after the <think\> </think\> window."

        response = self.client.chat(model = self.model,
                                    messages=[{"role": "system", "content": personality + thinking},
                                                #{"role": "assistant", "content": llm_responses},
                                                {"role": "user", "content": input_text}
                                    ],
                                    options={'temperature': 0, 'top_p': 0, 'num_ctx': 23456 },
                                    think = think,                            
                                    stream = False
                                    )

        #print('> LLM response: ', response['message']['content'] )
        
        findings = []
        if param_type.lower() == 'numerical':
            
            response_ = re.sub(r"(?s)\<think\>.*?\</think\>\s*", '', response['message']['content']  )
            response_ = re.sub(r'µ', 'u', response_)
            response_ = re.sub(r'±', '+/-', response_)
            response_ = re.sub(r'–', '-', response_)
            findings = re.findall(r'(?:(?:\-\s?)?\d+(?:\.\d+)?\s?)?(?:(?:\-\s?)?\d+(?:\.\d+)?)(?:\s[A-Za-z%°][A-Za-z]?[0-9\-]*)+', response_)
            print('> LLM response proc: ', findings)

            '''
            nums_text = ''
            for val in findings:
                nums_text += val + ', '
            nums_text = nums_text[ : -2 ]'''

        elif param_type.lower() == 'textual':
            
            response_ = re.sub(r"(?s)\<think\>.*?\</think\>\s*", '', response['message']['content']  )
            findings = re.findall(r"(?<=\')[A-Za-z0-9/\.\s\-\(\)δ]+(?=\')", response_)
            print('> LLM response proc: ', findings)

        
        return findings
    

    
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

            
            user_text = biofilm_killing_perc()
            print('> user_text: ', user_text)

            return self.chat(user_text, personality = personality, param_type = 'numerical')



        elif input_parameter.lower() == 'biological_species':

            def biological_species():
        
                user_text = f"From the sentence '{input_dic['biological_species']['sent_str']}', it was extracted that the biological species '{input_dic['biological_species']['val']}' was exposed to " \
                            f"or tested with the nanomaterial with the core composition defined as: '{input_dic['nanomaterial_composition']['val']}'. " \
                            "Considering all information provided, respond 'yes' or 'no' (and nothing else) for the following question: " \
                            "Does the sentence contain the extracted values and do they follow the pattern described by the context? " \
                            "Your output must be between single quote characters (e.g, 'output')."
                
                return user_text

            user_text = biological_species()
            print('> user_text: ', user_text)

            return self.chat(user_text, personality = personality, param_type = 'textual')


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

            user_text = microbe_killing_log()
            print('> user_text: ', user_text)

            return self.chat(user_text, personality = personality, param_type = 'numerical')


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

            user_text = microbe_killing_mbc()
            print('> user_text: ', user_text)

            return self.chat(user_text, personality = personality, param_type = 'numerical')


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

            user_text = microbe_killing_mic()
            print('> user_text: ', user_text)

            return self.chat(user_text, personality = personality, param_type = 'numerical')


        elif input_parameter.lower() == 'nanomaterial_morphology':

            def nanomaterial_morphology():
        
                user_text = f"From the sentence '{input_dic['nanomaterial_morphology']['sent_str']}', it was extracted that the nanomaterial core composition is " \
                            f"'{input_dic['nanomaterial_composition']['val']}' and the nanomaterial morphology is '{input_dic['nanomaterial_morphology']['val']}'. " \
                            "Considering all information provided, respond just 'yes' or 'no' (and nothing else) for the following question: " \
                            "Does the sentence contain the extracted values and do they follow the pattern described by the context? " \
                            "In case the sentence has several matches for the parameters core composition and morphology, consider if the extracted values are one of them. " \
                            "Your output must be between single quote characters (e.g, 'output')."
                
                return user_text

            user_text = nanomaterial_morphology()
            print('> user_text: ', user_text)

            return self.chat(user_text, personality = personality, param_type = 'textual')


        elif input_parameter.lower() == 'nanomaterial_size':

            def nanomaterial_size():
        
                user_text = f"For a nanomaterial with core composition '{input_dic['nanomaterial_composition']['val']}' and morphology '{input_dic['nanomaterial_morphology']['val']}', " \
                            f"it was extracted from the sentence '{input_dic['nanomaterial_size']['sent_str']}' that this nanomaterial has a size value (or values) of '{input_dic['nanomaterial_size']['val']}'. " \
                            "Considering all information provided, output the correct nanomaterial size value considering either the nanomaterial core composition or the morphology (or both). " \
                            "If no correlation is found between any of extracted values for size and the nanomaterial core composition or morpholgy, output 'None'. " \
                            "Your output must be between single quote characters (e.g, 'output')."
                
                return user_text

            user_text = nanomaterial_size()
            print('> user_text: ', user_text)

            return self.chat(user_text, personality = personality, param_type = 'numerical')


        elif input_parameter.lower() == 'nanomaterial_surface_area':

            def nanomaterial_surface_area():
        
                user_text = f"For a nanomaterial with core composition '{input_dic['nanomaterial_composition']['val']}' and morphology '{input_dic['nanomaterial_morphology']['val']}', " \
                            f"it was extracted from the sentence '{input_dic['surface_area']['sent_str']}' that this nanomaterial has a surface area value (or values) of '{input_dic['surface_area']['val']}'. " \
                            "Considering all information provided, output the correct nanomaterial surface area value considering either the nanomaterial core composition or the morphology (or both). " \
                            "If no correlation is found between any of extracted values for surface area and the nanomaterial core composition or morpholgy, output 'None'. " \
                            "Your output must be between single quote characters (e.g, 'output')."
                
                return user_text

            user_text = nanomaterial_surface_area()
            print('> user_text: ', user_text)

            return self.chat(user_text, personality = personality, param_type = 'numerical')


        elif input_parameter.lower() == 'nanomaterial_zeta_potential':
        
            def nanomaterial_zeta_potential():
                
                user_text = f"For a nanomaterial with core composition '{input_dic['nanomaterial_composition']['val']}' and morphology '{input_dic['nanomaterial_morphology']['val']}', " \
                            f"it was extracted from the sentence '{input_dic['zeta_potential']['sent_str']}' that this nanomaterial has a zeta potential value (or values) of '{input_dic['zeta_potential']['val']}'. " \
                            "Considering all information provided, output the correct nanomaterial zeta potential value (or values) considering either the nanomaterial core composition or the morphology (or both). " \
                            "If no correlation is found between any of extracted values for zeta potential and the nanomaterial core composition or morpholgy, output 'None'. " \
                            "Your output must be between single quote characters (e.g, 'output')."
                
                return user_text

            user_text = nanomaterial_zeta_potential()
            print('> user_text: ', user_text)

            return self.chat(user_text, personality = personality, param_type = 'numerical')


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

            user_text = toxicity_lc50()
            print('> user_text: ', user_text)

            return self.chat(user_text, personality = personality, param_type = 'numerical')


        elif input_parameter.lower() == 'toxicity_yes_no':

            def toxicity_yes_no():
            
                user_text = f"From the sentence '{input_dic['toxicity_yes_no']['sent_str']}', it was evaluated if there is (or not) a toxicity effect for the endpoint '{input_dic['toxicity_endpoints']['val']}'. " \
                            f"Consider the question: " \
                            f"Is there any evidence in the input sentence of a possible toxic effect for the endpoint '{input_dic['toxicity_endpoints']['val']}' considering the species '{input_dic['biological_species']['val']}'? " \
                            f"Your previous answer for this question was: '{input_dic['toxicity_yes_no']['val']}'. " \
                            "Is your previous answer about the toxicity assessment correct? " \
                            "Your output must be between single quote characters (e.g, 'output')."

                return user_text

            user_text = toxicity_yes_no()
            print('> user_text: ', user_text)

            return self.chat(user_text, personality = personality, param_type = 'textual')



    #######################################################################
    def extract_num_parameter(self, parameter: str, text: str):

        
        if parameter.lower() == 'biofilm_killing_perc':
            return self.chat(text, personality = self.prompt.extract_num_param('biofilm_killing_perc'), param_type = 'numerical')
        
        elif parameter.lower() == 'microbe_killing_log':
            return self.chat(text, personality = self.prompt.extract_num_param('microbe_killing_log'), param_type = 'numerical')

        elif parameter.lower() == 'microbe_killing_mbc':
            return self.chat(text, personality = self.prompt.extract_num_param('microbe_killing_mbc'), param_type = 'numerical')

        elif parameter.lower() == 'microbe_killing_mic':
            return self.chat(text, personality = self.prompt.extract_num_param('microbe_killing_mic'), param_type = 'numerical')
        
        elif parameter.lower() == 'microbe_killing_perc':
            return self.chat(text, personality = self.prompt.extract_num_param('microbe_killing_perc'), param_type = 'numerical')

        elif parameter.lower() == 'nanomaterial_concentration':
            return self.chat(text, personality = self.prompt.extract_num_param('nanomaterial_concentration'), param_type = 'numerical')

        elif parameter.lower() == 'nanomaterial_size':
            return self.chat(text, personality = self.prompt.extract_num_param('nanomaterial_size'), param_type = 'numerical')

        elif parameter.lower() == 'nanomaterial_surface_area':
            return self.chat(text, personality = self.prompt.extract_num_param('nanomaterial_surface_area'), param_type = 'numerical')

        elif parameter.lower() == 'nanomaterial_zeta_potential':
            return self.chat(text, personality = self.prompt.extract_num_param('nanomaterial_zeta_potential'), param_type = 'numerical')

        elif parameter.lower() == 'sofc_powerdensity':
            return self.chat(text, personality = self.prompt.extract_num_param('sofc_powerdensity'), param_type = 'numerical')

        elif parameter.lower() == 'temperature':
            return self.chat(text, personality = self.prompt.extract_num_param('temperature'), param_type = 'numerical')

        elif parameter.lower() == 'toxicity_ec50':
            return self.chat(text, personality = self.prompt.extract_num_param('toxicity_ec50'), param_type = 'numerical')

        elif parameter.lower() == 'toxicity_lc50':
            return self.chat(text, personality = self.prompt.extract_num_param('toxicity_lc50'), param_type = 'numerical')

        elif parameter.lower() == 'toxicity_ld50':
            return self.chat(text, personality = self.prompt.extract_num_param('toxicity_ld50'), param_type = 'numerical')

        elif parameter.lower() == 'toxicity_ic50':
            return self.chat(text, personality = self.prompt.extract_num_param('toxicity_ic50'), param_type = 'numerical')

        else:
            return ''



    #######################################################################
    def extract_textual_parameter(self, parameter: str, text: str):

        if parameter.lower() == '2d_materials':
            return self.chat(text, personality = self.prompt.extract_textual_param('2d_materials'), param_type = 'textual')

        elif parameter.lower() == 'metallic_materials':
            return self.chat(text, personality = self.prompt.extract_textual_param('metallic_materials'), param_type = 'textual')

        elif parameter.lower() == 'oxide_materials':
            return self.chat(text, personality = self.prompt.extract_textual_param('oxide_materials'), param_type = 'textual')

        elif parameter.lower() == 'qdots_materials':
            return self.chat(text, personality = self.prompt.extract_textual_param('qdots_materials'), param_type = 'textual')

        elif parameter.lower() == 'sofc_anode_composition':
            return self.chat(text, personality = self.prompt.extract_textual_param('sofc_anode_composition'), param_type = 'textual')

        elif parameter.lower() == 'toxicity_a_thaliana_all':
            return self.chat(text, personality = self.prompt.extract_textual_param('toxicity_a_thaliana_all'), param_type = 'textual')

        elif parameter.lower() == 'toxicity_a_thaliana_bioaccumulation':
            return self.chat(text, personality = self.prompt.extract_textual_param('toxicity_a_thaliana_bioaccumulation'), param_type = 'textual')

        elif parameter.lower() == 'toxicity_a_thaliana_development':
            return self.chat(text, personality = self.prompt.extract_textual_param('toxicity_a_thaliana_development'), param_type = 'textual')

        elif parameter.lower() == 'toxicity_a_thaliana_enzyme':
            return self.chat(text, personality = self.prompt.extract_textual_param('toxicity_a_thaliana_enzyme'), param_type = 'textual')

        elif parameter.lower() == 'toxicity_a_thaliana_genotox':
            return self.chat(text, personality = self.prompt.extract_textual_param('toxicity_a_thaliana_genotox'), param_type = 'textual')
        
        elif parameter.lower() == 'toxicity_a_thaliana_germination':
            return self.chat(text, personality = self.prompt.extract_textual_param('toxicity_a_thaliana_germination'), param_type = 'textual')

        elif parameter.lower() == 'toxicity_a_thaliana_morphology':
            return self.chat(text, personality = self.prompt.extract_textual_param('toxicity_a_thaliana_morphology'), param_type = 'textual')
        
        elif parameter.lower() == 'toxicity_a_thaliana_oxi_stress':
            return self.chat(text, personality = self.prompt.extract_textual_param('toxicity_a_thaliana_oxi_stress'), param_type = 'textual')
        
        elif parameter.lower() == 'toxicity_a_thaliana_photosynthesis':
            return self.chat(text, personality = self.prompt.extract_textual_param('toxicity_a_thaliana_photosynthesis'), param_type = 'textual')

        elif parameter.lower() == 'toxicity_a_thaliana_seedling_phototropic':
            return self.chat(text, personality = self.prompt.extract_textual_param('toxicity_a_thaliana_seedling_phototropic'), param_type = 'textual')

        elif parameter.lower() == 'toxicity_c_elegans_all':
            return self.chat(text, personality = self.prompt.extract_textual_param('toxicity_c_elegans_all'), param_type = 'textual')

        elif parameter.lower() == 'toxicity_c_elegans_behavior':
            return self.chat(text, personality = self.prompt.extract_textual_param('toxicity_c_elegans_behavior'), param_type = 'textual')

        elif parameter.lower() == 'toxicity_c_elegans_bioaccumulation':
            return self.chat(text, personality = self.prompt.extract_textual_param('toxicity_c_elegans_bioaccumulation'), param_type = 'textual')

        elif parameter.lower() == 'toxicity_c_elegans_development':
            return self.chat(text, personality = self.prompt.extract_textual_param('toxicity_c_elegans_development'), param_type = 'textual')

        elif parameter.lower() == 'toxicity_c_elegans_genotox':
            return self.chat(text, personality = self.prompt.extract_textual_param('toxicity_c_elegans_genotox'), param_type = 'textual')

        elif parameter.lower() == 'toxicity_c_elegans_enzyme':
            return self.chat(text, personality = self.prompt.extract_textual_param('toxicity_c_elegans_enzyme'), param_type = 'textual')

        elif parameter.lower() == 'toxicity_c_elegans_morphology':
            return self.chat(text, personality = self.prompt.extract_textual_param('toxicity_c_elegans_morphology'), param_type = 'textual')

        elif parameter.lower() == 'toxicity_c_elegans_mortality':
            return self.chat(text, personality = self.prompt.extract_textual_param('toxicity_c_elegans_mortality'), param_type = 'textual')

        elif parameter.lower() == 'toxicity_c_elegans_oxi_stress':
            return self.chat(text, personality = self.prompt.extract_textual_param('toxicity_c_elegans_oxi_stress'), param_type = 'textual')
        
        elif parameter.lower() == 'toxicity_c_elegans_reproduction':
            return self.chat(text, personality = self.prompt.extract_textual_param('toxicity_c_elegans_reproduction'), param_type = 'textual')

        elif parameter.lower() == 'toxicity_d_magna_all':
            return self.chat(text, personality = self.prompt.extract_textual_param('toxicity_d_magna_all'), param_type = 'textual')

        elif parameter.lower() == 'toxicity_d_magna_behavior':
            return self.chat(text, personality = self.prompt.extract_textual_param('toxicity_d_magna_behavior'), param_type = 'textual')

        elif parameter.lower() == 'toxicity_d_magna_bioaccumulation':
            return self.chat(text, personality = self.prompt.extract_textual_param('toxicity_d_magna_bioaccumulation'), param_type = 'textual')
        
        elif parameter.lower() == 'toxicity_d_magna_development':
            return self.chat(text, personality = self.prompt.extract_textual_param('toxicity_d_magna_development'), param_type = 'textual')

        elif parameter.lower() == 'toxicity_d_magna_enzyme':
            return self.chat(text, personality = self.prompt.extract_textual_param('toxicity_d_magna_enzyme'), param_type = 'textual')

        elif parameter.lower() == 'toxicity_d_magna_genotox':
            return self.chat(text, personality = self.prompt.extract_textual_param('toxicity_d_magna_genotox'), param_type = 'textual')

        elif parameter.lower() == 'toxicity_d_magna_morphology':
            return self.chat(text, personality = self.prompt.extract_textual_param('toxicity_d_magna_morphology'), param_type = 'textual')
        
        elif parameter.lower() == 'toxicity_d_magna_mortality':
            return self.chat(text, personality = self.prompt.extract_textual_param('toxicity_d_magna_mortality'), param_type = 'textual')

        elif parameter.lower() == 'toxicity_d_magna_oxi_stress':
            return self.chat(text, personality = self.prompt.extract_textual_param('toxicity_d_magna_oxi_stress'), param_type = 'textual')

        elif parameter.lower() == 'toxicity_d_magna_reproduction':
            return self.chat(text, personality = self.prompt.extract_textual_param('toxicity_d_magna_reproduction'), param_type = 'textual')

        elif parameter.lower() == 'toxicity_d_rerio_all':
            return self.chat(text, personality = self.prompt.extract_textual_param('toxicity_d_rerio_all'), param_type = 'textual')

        elif parameter.lower() == 'toxicity_d_rerio_behavior':
            return self.chat(text, personality = self.prompt.extract_textual_param('toxicity_d_rerio_behavior'), param_type = 'textual')

        elif parameter.lower() == 'toxicity_d_rerio_bioaccumulation':
            return self.chat(text, personality = self.prompt.extract_textual_param('toxicity_d_rerio_bioaccumulation'), param_type = 'textual')        

        elif parameter.lower() == 'toxicity_d_rerio_development':
            return self.chat(text, personality = self.prompt.extract_textual_param('toxicity_d_rerio_development'), param_type = 'textual')

        elif parameter.lower() == 'toxicity_d_rerio_enzyme':
            return self.chat(text, personality = self.prompt.extract_textual_param('toxicity_d_rerio_enzyme'), param_type = 'textual')
        
        elif parameter.lower() == 'toxicity_d_rerio_genotox':
            return self.chat(text, personality = self.prompt.extract_textual_param('toxicity_d_rerio_genotox'), param_type = 'textual')

        elif parameter.lower() == 'toxicity_d_rerio_morphology':
            return self.chat(text, personality = self.prompt.extract_textual_param('toxicity_d_rerio_morphology'), param_type = 'textual')

        elif parameter.lower() == 'toxicity_d_rerio_mortality':
            return self.chat(text, personality = self.prompt.extract_textual_param('toxicity_d_rerio_mortality'), param_type = 'textual')

        elif parameter.lower() == 'toxicity_d_rerio_oxi_stress':
            return self.chat(text, personality = self.prompt.extract_textual_param('toxicity_d_rerio_oxi_stress'), param_type = 'textual')
        
        elif parameter.lower() == 'toxicity_d_rerio_reproduction':
            return self.chat(text, personality = self.prompt.extract_textual_param('toxicity_d_rerio_reproduction'), param_type = 'textual')

        else:
            return ''