import ollama # type: ignore
import time
import regex as re # type: ignore

class llm(object):

    def __init__(self, model):
        self.model = model


    def extract_num_parameter(self, parameter: str, text: str):

        if parameter == 'microbe_log_reduction':

            personality = 'You are a data miner that deals with scientific texts on microbiology and you extract numerical values related with antimicrobial effect.'
            rules = "For the text I will input next, output only the numbers associated with log values indicating the reduction of the microbial population. \
                    Output the log unit (ex: log10 or log or logs) with a blank space after each number found (Ex: '2.2 logs'). \
                    If no value is found, output 'None'. Do not output any other values not related the reduction of the microbial population."

            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             {"role": "assistant", "content": rules},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0.3, 'top_p': 0.3},
                                   stream = False
                                   )
            
            print('LLM response: ', response['message']['content'])
            
            num_params = re.findall(r'[0-9]+\.?[0-9]*\s*[A-Za-z0-9\-]+', response['message']['content'])
            
            nums_text = 'begining '
            for val in num_params:
                nums_text += val + ', '
            nums_text = nums_text[ : -2 ]
            nums_text += ' ending'
            print('LLM nums_text: ', nums_text)
            return nums_text
        
        elif parameter == 'microbe_mic_inhibition':

            personality = 'You are a data miner that deals with scientific texts on microbiology and you extract numerical values related with antimicrobial effect.'
            rules = "For the text I will input next, output only the numbers associated with minimum inhibitory concentration (MIC) values indicating the antimicrobial effect. \
                    Output the concentration unit (ex: mg L-1) with a blank space after each number found (Ex: '10.1 mg L-1' or '10.1 mg mL-1' or '10.1 g mL-1' or '10.1 ug mL-1'). \
                    If no value is found, output 'None'. Do not output concentration values not related with minimum inhibitory concentration (MIC)."

            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             {"role": "assistant", "content": rules},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0.3, 'top_p': 0.3},
                                   stream = False
                                   )
            
            print('LLM response: ', response['message']['content'])
            
            mod_text = re.sub(r'µ', 'u', response['message']['content'])
            num_params = re.findall(r'[0-9]+\.?[0-9]*\s*[A-Za-z]+\s[A-Za-z0-9\-]+', mod_text)
            
            nums_text = 'begining '
            for val in num_params:
                nums_text += val + ', '
            nums_text = nums_text[ : -2 ]
            nums_text += ' ending'
            print('LLM nums_text: ', nums_text)
            return nums_text

        elif parameter == 'microbe_percentage_killing':

            personality = 'You are a data miner that deals with scientific texts on microbiology and you extract numerical values related with antimicrobial effect.'
            rules = "For the text I will input next, output only the numbers associated with percentage values indicating the microbes killing. \
                    Output the percentage symbol with a blank space after each number found (Ex: '10.1 %'). \
                    If no percentage is found, output 'None'. Do not output percentage values not related with microbe killing."

            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             {"role": "assistant", "content": rules},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0.3, 'top_p': 0.3},
                                   stream = False
                                   )
            
            print('LLM response: ', response['message']['content'])

            num_params = re.findall(r'[0-9]+\.?[0-9]*\s*%', response['message']['content'])
            
            nums_text = 'begining '
            for val in num_params:
                nums_text += val + ', '
            nums_text = nums_text[ : -2 ]
            nums_text += ' ending'
            print('LLM nums_text: ', nums_text)
            return nums_text

        elif parameter == 'nanomaterial_concentration':

            personality = 'You are a data miner that deals with scientific texts on nanotechnology and you extract numerical values related with the concentration of nanomaterials and nanoparticles.'
            rules = "For the text I will input next, output only the numbers associated with the concentration of nanomaterials and nanoparticles described in the text. \
                    Output the concentration unit (ex: mg L-1) with a blank space after each number found (Ex: '10.1 mg L-1' or '10.1 mg mL-1' or '10.1 g mL-1' or '10.1 ug mL-1'). \
                    If no value is found, output 'None'. Do not output concentration values not related with the nanomaterial or nanoparticle described in the text."

            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             {"role": "assistant", "content": rules},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0.3, 'top_p': 0.3},
                                   stream = False
                                   )
            
            print('LLM response: ', response['message']['content'])
            
            mod_text = re.sub(r'µ', 'u', response['message']['content'])
            num_params = re.findall(r'[0-9]+\.?[0-9]*\s*[A-Za-z]+\s[A-Za-z0-9\-]+', mod_text)
            
            nums_text = 'begining '
            for val in num_params:
                nums_text += val + ', '
            nums_text = nums_text[ : -2 ]
            nums_text += ' ending'
            print('LLM nums_text: ', nums_text)
            return nums_text
        
        elif parameter == 'nanomaterial_size':

            personality = 'You are a data miner that deals with scientific texts on nanotechnology and you extract numerical values related with the size of nanomaterials and nanoparticles.'
            rules = "For the text I will input next, output only the numbers associated with the size of nanomaterials and nanoparticles described in the text. \
                    Output the size unit (ex: nm) with a blank space after each number found (Ex: '10.1 nm' or '10.1 um' or '10.1 µm'). \
                    If no value is found, output 'None'. Do not output size values not related with the nanomaterial or nanoparticle described in the text."

            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             {"role": "assistant", "content": rules},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0.3, 'top_p': 0.3},
                                   stream = False
                                   )
            
            print('LLM response: ', response['message']['content'])
            
            mod_text = re.sub(r'µ', 'u', response['message']['content'])
            num_params = re.findall(r'[0-9]+\.?[0-9]*\s*[A-Za-z]+', mod_text)
            
            nums_text = 'begining '
            for val in num_params:
                nums_text += val + ', '
            nums_text = nums_text[ : -2 ]
            nums_text += ' ending'
            print('LLM nums_text: ', nums_text)
            return nums_text

        elif parameter == 'nanomaterial_surface_area':

            personality = 'You are a data miner that deals with scientific texts on nanotechnology and you extract numerical values related with the surface area of nanomaterials and nanoparticles.'
            rules = "For the text I will input next, output only the numbers associated with the surface area of nanomaterials and nanoparticles described in the text. \
                    Output the surface area unit (ex: m2 g-1) with a blank space after each number found (Ex: '10.1 m2 g-1' or '10.1 m2 mg-1' or '10.1 mm2 kg-1' or '10.1 µm2 g-1'). \
                    If no value is found, output 'None'. Do not output surface area values not related with the nanomaterial or nanoparticle described in the text."

            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             {"role": "assistant", "content": rules},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0.3, 'top_p': 0.3},
                                   stream = False
                                   )
            
            print('LLM response: ', response['message']['content'])
            
            mod_text = re.sub(r'µ', 'u', response['message']['content'])
            num_params = re.findall(r'[0-9]+\.?[0-9]*\s*[A-Za-z0-9]+\s[A-Za-z0-9\-]+', mod_text)
            
            nums_text = 'begining '
            for val in num_params:
                nums_text += val + ', '
            nums_text = nums_text[ : -2 ]
            nums_text += ' ending'
            print('LLM nums_text: ', nums_text)
            return nums_text

        elif parameter == 'nanomaterial_zeta_potential':

            personality = 'You are a data miner that deals with scientific texts on nanotechnology and you extract numerical values related with the zeta potential of nanomaterials and nanoparticles.'
            rules = "For the text I will input next, output only the numbers associated with the zeta potential of nanomaterials and nanoparticles described in the text. \
                    Output the zeta potential unit (ex: mV) with a blank space after each number found (Ex: '-10.1 mV' or '10.1 V' or '10.1 kV' or '-10.1 µV'). \
                    If no value is found, output 'None'. Do not output zeta potential values not related with the nanomaterial or nanoparticle described in the text."

            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             {"role": "assistant", "content": rules},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0.3, 'top_p': 0.3},
                                   stream = False
                                   )
            
            print('LLM response: ', response['message']['content'])
            
            mod_text = re.sub(r'µ', 'u', response['message']['content'])
            num_params = re.findall(r'\-?\s*[0-9]+\.?[0-9]*\s*[A-Za-z]+', mod_text)
            
            nums_text = 'begining '
            for val in num_params:
                nums_text += val + ', '
            nums_text = nums_text[ : -2 ]
            nums_text += ' ending'
            print('LLM nums_text: ', nums_text)
            return nums_text

        elif parameter == 'toxicity_ec50':

            personality = 'You are a data miner that deals with scientific texts on the toxicity against living species and you extract numerical values related with the toxicity effect.'
            rules = "For the text I will input next, output only the numbers associated with the half-maximal effective concentration (EC50) values indicating the toxicity effect. \
                    Output the concentration unit (ex: mg L-1) with a blank space after each number found (Ex: '10.1 mg L-1' or '10.1 mg mL-1' or '10.1 g mL-1' or '10.1 ug mL-1'). \
                    If no value is found, output 'None'. Do not output concentration values not related with half-maximal effective concentration (EC50)."

            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             {"role": "assistant", "content": rules},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0.3, 'top_p': 0.3},
                                   stream = False
                                   )
            
            print('LLM response: ', response['message']['content'])
            
            mod_text = re.sub(r'µ', 'u', response['message']['content'])
            num_params = re.findall(r'[0-9]+\.?[0-9]*\s*[A-Za-z]+\s[A-Za-z0-9\-]+', mod_text)
            
            nums_text = 'begining '
            for val in num_params:
                nums_text += val + ', '
            nums_text = nums_text[ : -2 ]
            nums_text += ' ending'
            print('LLM nums_text: ', nums_text)
            return nums_text

        elif parameter == 'toxicity_lc50':

            personality = 'You are a data miner that deals with scientific texts on the toxicity against living species and you extract numerical values related with the toxicity effect.'
            rules = "For the text I will input next, output only the numbers associated with the half-lethal maximal effective concentration (LC50) values indicating the toxicity effect. \
                    Output the concentration unit (ex: mg L-1) with a blank space after each number found (Ex: '10.1 mg L-1' or '10.1 mg mL-1' or '10.1 g mL-1' or '10.1 ug mL-1'). \
                    If no value is found, output 'None'. Do not output concentration values not related with half-lethal maximal effective concentration (LC50)."

            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             {"role": "assistant", "content": rules},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0.3, 'top_p': 0.3},
                                   stream = False
                                   )
            
            print('LLM response: ', response['message']['content'])
            
            mod_text = re.sub(r'µ', 'u', response['message']['content'])
            num_params = re.findall(r'[0-9]+\.?[0-9]*\s*[A-Za-z]+\s[A-Za-z0-9\-]+', mod_text)
            
            nums_text = 'begining '
            for val in num_params:
                nums_text += val + ', '
            nums_text = nums_text[ : -2 ]
            nums_text += ' ending'
            print('LLM nums_text: ', nums_text)
            return nums_text

        elif parameter == 'toxicity_ld50':

            personality = 'You are a data miner that deals with scientific texts on the toxicity against living species and you extract numerical values related with the toxicity effect.'
            rules = "For the text I will input next, output only the numbers associated with the lethal dose 50% (LD50) values indicating the dose of a substance that is lethal to 50% of the tested population. \
                    Output the LD50 unit (ex: mg kg-1) with a blank space after each number found (Ex: '10.1 mg kg-1' or '10.1 mg g-1' or '10.1 g mg-1'). \
                    If no value is found, output 'None'. Do not output concentration values not related with lethal dose 50% (LD50)."

            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             {"role": "assistant", "content": rules},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0.3, 'top_p': 0.3},
                                   stream = False
                                   )
            
            print('LLM response: ', response['message']['content'])
            
            mod_text = re.sub(r'µ', 'u', response['message']['content'])
            num_params = re.findall(r'[0-9]+\.?[0-9]*\s*[A-Za-z]+\s[A-Za-z0-9\-]+', mod_text)
            
            nums_text = 'begining '
            for val in num_params:
                nums_text += val + ', '
            nums_text = nums_text[ : -2 ]
            nums_text += ' ending'
            print('LLM nums_text: ', nums_text)
            return nums_text

        elif parameter == 'toxicity_ic50':

            personality = 'You are a data miner that deals with scientific texts on the toxicity against living species and you extract numerical values related with the toxicity effect.'
            rules = "For the text I will input next, output only the numbers associated with the half-lethal maximal inhibitory concentration (IC50) values indicating the toxicity effect. \
                    Output the concentration unit (ex: mg L-1) with a blank space after each number found (Ex: '10.1 mg L-1' or '10.1 mg mL-1' or '10.1 g mL-1' or '10.1 ug mL-1'). \
                    If no value is found, output 'None'. Do not output concentration values not related with half-lethal maximal inhibitory concentration (IC50)."

            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             {"role": "assistant", "content": rules},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0.3, 'top_p': 0.3},
                                   stream = False
                                   )
            
            print('LLM response: ', response['message']['content'])
            
            mod_text = re.sub(r'µ', 'u', response['message']['content'])
            num_params = re.findall(r'[0-9]+\.?[0-9]*\s*[A-Za-z]+\s[A-Za-z0-9\-]+', mod_text)
            
            nums_text = 'begining '
            for val in num_params:
                nums_text += val + ', '
            nums_text = nums_text[ : -2 ]
            nums_text += ' ending'
            print('LLM nums_text: ', nums_text)
            return nums_text


        else:
            return text



    def extract_textual_num_parameter(self, parameter: str, text: str):

        if parameter == 'species_percentage_killing':

            response = ollama.generate(
                                model= self.model,
                                prompt = "For the text I will input next, output only the microbe species and percentage values indicating the microbes killing. \
                                            Organize the output as: 'microbe species': 'killing percentage %'. If no percentage is found, output 'None'. \
                                            Input text: {input_text}".format(input_text = text),
                                format = 'json',
                                options={'temperature': 0.3, 'top_p': 0.3},
                                stream = False
                                )
            print(response['reponse'])
            return response['response']
        
        else:
            return text



    def extract_textual_parameter(self, parameter: str, text: str):

        if parameter == 'toxicity_yes_no':

            personality = 'You are a data miner that deals with scientific texts on the toxicity against living species and you analyze if the text describe any toxic effect on the living species.'
            rules = "For the text I will input next, output only the numbers associated with the half-lethal maximal inhibitory concentration (IC50) values indicating the toxicity effect. \
                    Output the concentration unit (ex: mg L-1) with a blank space after each number found (Ex: '10.1 mg L-1' or '10.1 mg mL-1' or '10.1 g mL-1' or '10.1 ug mL-1'). \
                    If no value is found, output 'None'. Do not output concentration values not related with half-lethal maximal inhibitory concentration (IC50)."

            response = ollama.chat(model = self.model,
                                   messages=[{"role": "system", "content": personality},
                                             {"role": "assistant", "content": rules},
                                             {"role": "user", "content": text}
                                   ],
                                   options={'temperature': 0.3, 'top_p': 0.3},
                                   stream = False
                                   )
            
            print('LLM response: ', response['message']['content'])
            return 'begining ' + response['message']['content'] + ' ending'

        else:
            return text