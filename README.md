## Corpus processing
First paste the documents (PDF, XML or a single Web of Science CSV report) into folder ".../Articles_to_add/". Then use command "process_collection.py" to:

1 - Index each input document (PDF, XML or Web of Science CSV report) and convert it to plaintext format.

2 - Separate (if possible) each document section (Introduction, Methodology and Results) and export to ".../Outputs/sections/" (does not work when a Web of Science CSV report is used as input, since it has only the article abstract).

3 - Split and index each sentence and export to ".../Outputs/sents_raw/".

4 - Filter sentences (with REGEX scripts) and export to ".../Outputs/sents_filtered/".

5 - Find unique tokens and export to ".../Outputs/ngrams/".

6 - Generate the document-token matrix (TFIDF) and export to ".../Outputs/tfidf/".

7 - Generate document-topic matrices via Latent Semantic Analysis (LSA) and Latent Dirichlet Allocation approaches and export to ".../Outputs/models/".

8 - Generate word-vectors embeddings via Word2-vec approach and export to ".../Outputs/wv/".

9 - Train the section filter (for Introduction, Methodology and Results sections) with convolutional neuron networks (CNNs) and export to ".../Outputs/models/".




## Search routines are defined in file ".../Settings/SE_inputs.csv". 
After corpus processing, search routines can be set to identify and extract relevant information from the corpus.
Description of each field present in ".../Settings/SE_inputs.csv" file goes bellow:

1 - *filename:* insert the name of the ".csv" file that will be generated and saved on path ".../Outputs/extracted/". This .csv file contains the indexed documents in which relevant information was found after the program scanned the entire corpus. Use the following range of characters as input:* "A-Z", "a-z", "0-9" and "_". Input example: "green_synt_01".

2 - *index_list_name:* insert the name of an already generated ".csv" file present in folder ".../Outputs/extracted/". The program will search only for the documents indexed in the ".csv" file. If a complete scan over the whole corpus is wanted, insert "None".

3 - *file_type:* insert the text format to be processed. Use "pdf", "xml" or "txt".

4 - *parameter_to_extract:* choose between one of the many parameters configured in the program. The extraction depends on REGEX patterns and LLM prompts previously set. The current list of available parameters is available on https://github.com/amaurijp/arix_v2/blob/main/parameters_to_extract.txt. To set more parameters (numerical or categorical), files "functions_PARAMETERS.py" and "LLM.py" must be manipulated on folder ".../Modules/".

5 - *extraction_mode:* use "collect_parameters_automatic" to automatically extract the parameters through a combination of REGEX patterns and LLM prompts.

6 - *scan_sent_by_sent:* insert "True" if the user wants to scan the corpus at a sentence level, i.e., the input document will be sentences. The argument "False" will make the full article as the input document.

7 - *numbers_extraction_mode:* use the entry "all" to extract all available numerical parameters from the input document. This argument is only considered by the program if a numerical parameter is inserted in the "parameter_to_extract" field.

8 - *filter_unique_results:* insert "True" to prevent repeated parameters extracted from each input document. For example, concentration values 4.3 and 6.7 would be extracted from the input document "The GO concentration was of 4.3 mg mL-1 for sample A, 4.3 mg mL-1 for sample B and 6.7 mg mL-1 for sample C." If "False", values 4.3, 4.3 and 6.7 would be extracted instead.

9 - *literal_entry:* insert a REGEX pattern to be matched in the input document during the corpus scan. Use the format "s(regex_pattern)e". Example:* the entry s([Mm]aterials?)e will match for "Material", "Materials", "material" or "materials" in the input document. The match attempt will be done using function regex.search( regex_pattern , input_document ). If the user does not want to find matches using the regex search engine (Lm), insert "None" into this field.

10 - *semantic_entry:* choose between one of the many categories containing semantically related terms to be matched in the input document. The current list of available categories is available on https://github.com/amaurijp/arix_v2/blob/main/categories_for_semantic_search.txt. The terms present in each category are found during corpus processing by calculating cosine similarities of the word-vector embeddings and are recorded in the json file ".../Inputs/ner_rules.json". The match attempt will be done using function regex.search( cat_term , input_document ). If the user does not want to find matches using the semantic search engine (Sm), insert "None" into this field.

11 - *search_token_by_token:* insert "True" if the user wants to search for occurrences at a token level. Each token (token\textsubscript{i}) separated by the space character " " in the input document will be split and introduced in function regex.search( text_to_be_found , token\textsubscript{i} ). Argument "False" will make the input document go through the function regex.search( text_to_be_found , input_document ). This argument is only considered by the program if an entry is provided for the Sm engine in the field "semantic_entry".

12 - *lower_sentence_for_semantic:* use "True" to convert the input document to lowercase when the search is made using the "semantic_entry". The match attempt will be then with regex.search( cat_term , input_document.lower() ). Argument "False" will make the input document to keep the its original format. This argument is only considered by the program if an entry is provided for the Sm engine in the field "semantic_entry".

13 - *lda_sents_topic:* insert a topic vector to be matched with vectors present in the document-topic matrix obtained via Latent Dirichlet Allocation (LDA) approach, where the document is at sentence level. Use the entry in the format "( mag\textsubscript{i} * topic\textsubscript{i})", where "topic\textsubscript{i}" is one of the "d" topics defined during the LDA training and "mag" is the magnitude of this specific topic. For example, considering that \textit{d} = 100, the entry "(1, 3, 5, 99)" generates a topic vector array in the format:* [1, 0, 1, 0, 1, 0, ..., 1, 0]. The magnitudes of the vector components can be modified as "(2*1, -1*3, 3*5, 2*99)" $\rightarrow$ [2, 0, -1, 0, 3, 0, ..., 2, 0]. If the user does not want to find matches using the LDA topic search engine (Tm) for sentences, insert "None" into this field.

14 - *lda_articles_topic:* insert a topic vector to be matched with vectors present in the document-topic matrix obtained via Latent Dirichlet Allocation (LDA) approach, where the document is at article level. Use the entry in the same format as described for the field lda_sents_topic. If the user does not want to find matches using the LDA topic search engine (Tm) for articles, insert "None" into this field.

15 - *lsa_sents_topic:* insert a topic vector to be matched with vectors present in the document-topic matrix obtained via Latent Semantic Analysis (LSA) approach, where the document is at sentence level. Use the entry in the same format as described for the field lda_sents_topic. If the user does not want to find matches using the LSA topic search engine (Tm) for sentences, insert "None" into this field.

16 - *lsa_articles_topic:* insert a topic vector to be matched with vectors present in the document-topic matrix obtained via Latent Semantic Analysis (LSA) approach, where the document is at article level. Use the entry in the same format as described for the field lda_sents_topic. If the user does not want to find matches using the LSA topic search engine (Tm) for articles, insert "None" into this field.

17 - *topic_search_mode:* use the entry "cosine" to analyze matches based on the cosine value between the topic vector introduced in the Tm engine and the document-topic vectors present in the document-topic matrix. This value will be applied for both LDA and LSA topic engines and this argument is only considered by the program if an entry is provided for the Tm engine in fields "lda_sents_topic", "lda_articles_topic", "lsa_sents_topic" and "lsa_articles_topic".

18 - *cos_thr:* insert the cosine value threshold (between 0 and 1) above which a match will be found between the topic vector introduced in the Tm engine and the document-topic vectors present in the document-topic matrix. This value will be applied for both LDA and LSA topic engines and this argument is only considered by the program if an entry is provided for the Tm engine in fields "lda_sents_topic", "lda_articles_topic", "lsa_sents_topic" and "lsa_articles_topic".

19 - *num_param:* choose between one of the many numerical parameters set in the file ".../Modules/functions_PARAMETERS.py". A search will be performed to locate the units related to certain parameters in the input document. For example, if entry "nanomaterial_size" is used, input documents containing text fragments such as "50 nm" or "0.4 um" will be selected by the program. The current list of available numerical parameters is available on https://github.com/amaurijp/arix_v2/blob/main/parameters_to_extract.txt. 

20 - *filter_section:* choose one of the filters trained during corpus processing to identify whether the input document belongs to the introduction part of the article, methodology, results, or abstract. The filters are based on convolutional neuron networks (CNN) and their training depends on a successful identification of the articles section parts during corpus processing. Possible entries set in the program are "abstract", "introduction", "methodology", "results", and "conclusion". If the user does not want to use section filters, insert "None" into this field.
