# BioASQ Task 1 - b: Biomedical Semantic Question Answering (Task 13b)

## Task description
- Large-scale biomedical semantic indexing and question answering
- Task goal: Biomedical questions (English) + expert (reference) answers.
  - The participants have to respond with (retrieve) relevant articles, and snippets from designated resources, as well as exact and "ideal" answers
- Document, passage, entity retrieval – you can pick one or many!
  - "clarified this with Prof. Knees: it will be sufficient to implement Phase A (list of articles and a list of snippets to be returned), which aligns well with the classical IR systems you’re expected to build. However, if you decide to also work on Phases A+ and/or B, you can earn extra points for your efforts!"
- General info: https://participants-area.bioasq.org/general_information/Task13b/

## Course deiliverables
https://tuwel.tuwien.ac.at/pluginfile.php/4422884/mod_resource/content/1/Exercise%20Introduction%202025.pdf
1. a “traditional” IR model, cf. Crash Course IR Block
2. a “neural” NLP representation learning approach
3. a “neural” re-ranking model
4. report - https://www.overleaf.com/2214395211rbsnzbrtrtjc#424aba

## Deadlines
- 04/25 CLEF registration -- DONE
- 05/06 submit abstract -- only if we wanna submit to the lab
- 05/13 submit project -- only if we wanna submit to the lab
- 05/22 report submission in tuwel -- final deadline

## General Info
- **dataset**
    - training and test biomedical questions, in English, along with gold standard (reference) answers
    - more than 5,000 training questions
    - test dataset released in batches, 300 in total
- **task: respond to test question**
    - types of questions
        - yes/no
        - factoid (named entity retrival, no synonyms)
        - list (also ner, different return)
        - short text summarizing
    - required answers
        - phase A: relevant articals and text snippents
            - list of at most 10 relevant documents
                - in English, from designated article repositories
                - ordered by decreasing confidence
                - one list per question
                - may contain articles from multiple designated repositories
                - list will actually contain unique article *identifiers* (obtained from the repositories)
            - list of at most 10 relevant text snippets
                - from the returned articles
                - ordered by decreasing confidence
                - single list returned, but may contain any number (or no) snippets from any of the returned articles
                - snippet represented: id of article + id of starting section +  offset of first character + id of ending section + off set of last character
                - also string has to be returned
        - phase B
            - exact answers (e.g., named entities in the case of factoid questions)
            - 'ideal' answers (English paragraph-sized summaries)
- **Evaluation measures**
    - given a set of golden items (e.g., articles), and a set of items returned by a system:
        - precision
        - recall
        - F-measure
    - given a set of questions → mean precision, mean recall, and mean F-measure
    - for snippets: article-snippet overlap
- PubMed
    - *relevant articles* are to be retrieved from [PubMed](http://www.ncbi.nlm.nih.gov/pubmed) Annual Baseline Repository for 2025
    - id is url
    - access articles through API (how to can be downloaded from website)
