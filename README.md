# Corpus Processing

First, paste the document files (**PDF**, **XML**, or a **single Web of Science CSV report**) into the folder `.../Articles_to_add/`.

Then, use the command `.../process_collection.py` to:

1. **Index each input document** (**PDF**, **XML**, or **Web of Science CSV report**) and convert it to plaintext format.
2. **Separate** (if possible) each document section (**Introduction, Methodology, and Results**) and export to `.../Outputs/sections/` (this does not work when a Web of Science CSV report is used as input, since it contains only the article abstract).
3. **Split and index each sentence** and export to `.../Outputs/sents_raw/`.
4. **Filter sentences** (using **REGEX** scripts) and export to `.../Outputs/sents_filtered/`.
5. **Find unique tokens** and export to `.../Outputs/ngrams/`.
6. **Generate the document-token matrix** (**TF-IDF**) and export to `.../Outputs/tfidf/`.
7. **Generate document-topic matrices** via **Latent Semantic Analysis (LSA)** and **Latent Dirichlet Allocation (LDA)** approaches, then export to `.../Outputs/models/`.
8. **Generate word-vector embeddings** via the **Word2Vec** approach and export to `.../Outputs/wv/`.
9. **Train the section filter** (for **Introduction, Methodology, and Results** sections) with **convolutional neural networks (CNNs)** and export to `.../Outputs/models/`.

---

## Definitions in `.../process_collection.py`

To perform corpus processing, some basic inputs must be provided in `.../process_collection.py`.

### Required Inputs:

1. **`article_file_type`** – Choose the input document file type: `"webscience_csv_report"`, `"pdf"`, or `"xml"`.
2. **`min_token_appearance_in_corpus`** – Set the minimum token appearance count to be considered in the models (**default: 10**).
3. **`tfidf_batch_size`** – Set the batch size for **TF-IDF matrix generation** (**default: 1000**).
4. **`words_to_test_wv_models`** – Introduce sample tokens to evaluate the results of **word-vector embeddings**. Example (for a corpus on **nanotechnology**): `"nanoparticles", "nanosheets", "aunps", "gold", "nps", "cu", "copper", "molybdenum"`.

---

## After Corpus Processing: Search Routines

After corpus processing, **search routines** can be set to identify and extract relevant information from the corpus. Each field present in `.../Settings/SE_inputs.csv` is described below:

### 1. `filename`
- **Description**: Name of the `.csv` file to be generated and saved at `.../Outputs/extracted/`.
- **Allowed characters**: `"A-Z"`, `"a-z"`, `"0-9"`, and `"_"`.
- **Example**: `"green_synt_01"`.

### 2. `index_list_name`
- Name of an **already generated `.csv` file** in `.../Outputs/extracted/`.
- If a **complete corpus scan** is needed, use `"None"`.

### 3. `file_type`
- Insert the **text format** to be processed: `"pdf"`, `"xml"`, `"txt"`.

### 4. `parameter_to_extract`
- Choose from the available parameters **configured in the program**.
- **Extraction depends on** **REGEX patterns** and **LLM prompts**.
- **Current list of parameters**: [🔗 Available Parameters](https://github.com/amaurijp/arix_v2/blob/main/parameters_to_extract.txt).
- To add **new parameters**, modify `"functions_PARAMETERS.py"` and `"LLM.py"` in `.../Modules/`.

### 5. `extraction_mode`
- Use `"collect_parameters_automatic"` to **automatically extract** parameters using a **combination of REGEX patterns and LLM prompts**.

### 6. `scan_sent_by_sent`
- **`True`** → Scan **at sentence level**.
- **`False`** → The **full article** will be processed.

### 7. `numbers_extraction_mode`
- Use `"all"` to extract **all available numerical parameters**.

### 8. `filter_unique_results`
- **`True`** → Prevents repeated parameters from being extracted.
- **Example**: Given the sentence: `"The GO concentration was 4.3 mg/mL for sample A, 4.3 mg/mL for sample B, and 6.7 mg/mL for sample C."`
  - If **`True`** → Extracts **4.3, 6.7**.
  - If **`False`** → Extracts **4.3, 4.3, 6.7**.

### 9. `literal_entry`
- Insert a **REGEX pattern** to be matched during the corpus scan.
- **Format**: `s(regex_pattern)e`.
- **Example**: `s([Mm]aterials?)e` – Matches `"Material"`, `"Materials"`, `"material"`, `"materials"`.
- If no **REGEX search** is needed, insert `"None"`.

### 10. `semantic_entry`
- Choose from **predefined semantic categories**.
- **Current categories**: [🔗 Available Categories](https://github.com/amaurijp/arix_v2/blob/main/categories_for_semantic_search.txt).
- If no **semantic search** is needed, insert `"None"`.

---

## Additional Search Parameters

| Parameter                  | Description |
|----------------------------|-------------|
| **`search_token_by_token`** | **`True`** → Searches occurrences **at a token level**. |
| **`lower_sentence_for_semantic`** | **`True`** → Converts input **to lowercase** for semantic search. |
| **`lda_sents_topic`** | Insert a **topic vector** to be matched with **LDA sentence-level** topics. |
| **`lda_articles_topic`** | Insert a **topic vector** to be matched with **LDA article-level** topics. |
| **`lsa_sents_topic`** | Insert a **topic vector** to be matched with **LSA sentence-level** topics. |
| **`lsa_articles_topic`** | Insert a **topic vector** to be matched with **LSA article-level** topics. |
| **`topic_search_mode`** | Use **"cosine"** for **cosine similarity-based** topic matching. |
| **`cos_thr`** | **Set cosine similarity threshold** (0 to 1). |
| **`num_param`** | Choose a **numerical parameter** for extraction. |
| **`filter_section`** | Use **trained CNN filters** to identify sections (*introduction, methodology, results*). |

---

## Notes:
- The field `numbers_extraction_mode` is only considered by the program if a numerical parameter is inserted in the `parameter_to_extract`.
- For the `semantic_entry`, the terms present in each category [🔗 Available Categories](https://github.com/amaurijp/arix_v2/blob/main/categories_for_semantic_search.txt) are found during corpus processing by calculating cosine similarities of the word-vector embeddings and are recorded in the json file `.../Inputs/ner_rules.json`. The match attempt will be done using function `regex.search( cat_term , input_document )`.
- If `search_token_by_token` is `True`, each token (token_i) separated by the space character `\s` in the input document will be split and introduced in function `regex.search( text_to_be_found , token_i )`. Argument `False` will make the input document go through the function `regex.search( text_to_be_found , input_document )`. This field is only considered by the program if an entry is provided in the field `semantic_entry`.
- The field `lower_sentence_for_semantic` is only considered by the program if an entry is provided in the field `semantic_entry`.
- For fields `lda_sents_topic`, `lda_articles_topic`, `lsa_sents_topic`, and `lsa_articles_topic`, use the entry in the format `( mag_i * topic_i)`, where `topic_i` is one of the `d` topics defined during the LDA training and `mag` is the magnitude of this specific topic. For example, considering that `d = 100`, the entry `(1, 3, 5, 99)` generates a topic vector array in the format: `[1, 0, 1, 0, 1, 0, ..., 1, 0]`. The magnitudes of the vector components can be modified as `(2*1, -1*3, 3*5, 2*99)` → `[2, 0, -1, 0, 3, 0, ..., 2, 0]`. If the user does not want to find matches using the LDA topic search engine for sentences, insert `None` in the field respective field (`lda_sents_topic`, `lda_articles_topic`, `lsa_sents_topic`, and `lsa_articles_topic`).
- The value set in `cos_thr` will be applied for both LDA and LSA topic engines and this argument is only considered by the program if an entry is provided for the Tm engine in fields `lda_sents_topic`, `lda_articles_topic`, `lsa_sents_topic`, and `lsa_articles_topic`.
- The **program uses convolutional neural networks (CNNs)** to classify document sections.
- **Topic-based searches** rely on **LDA & LSA document-topic matrices**.
- **LLM prompts** assist in **automatic parameter extraction**.
---
