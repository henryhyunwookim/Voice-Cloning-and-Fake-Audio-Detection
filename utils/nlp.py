import numpy as np
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from gensim.models.fasttext import FastText


def convert_words_to_vectors(input_words, reference_vectors, dimension):
    output_vectors = []
    for words in input_words:
        word_vector = np.zeros(dimension)
        for word in words.split():
            try:
                word_vector += reference_vectors[word.lower()]
            except KeyError:
                pass
        
        output_vectors.append(word_vector)

    return np.array(output_vectors)


def get_word_vectors(dataframe, word_col, vectorizer='fasttext',
                     to_process_text=True, remove_stopwords=True, lemmatize=False, stem=False):
    if to_process_text:
        to_convert = dataframe[word_col].apply(
            process_text,
            remove_stopwords=remove_stopwords,
            lemmatize=lemmatize,
            stem=stem
        )
    else:
        to_convert = dataframe[word_col]

    if vectorizer == 'fasttext':
        fasttext = FastText(sentences=to_convert.apply(
            lambda x: [word.lower() for word in x.split()]
            ))
        dimension = fasttext.vector_size
        word_vectors = convert_words_to_vectors(to_convert, fasttext, dimension)

    return word_vectors


def update_str_col(dataframe, column, mapping_dict):
    for i, job_title in enumerate(dataframe[column]):
        converted = []
        for word in job_title.split():
            converted.append(convert_terms(word, mapping_dict))
        dataframe.loc[i, column] = " ".join(converted)

    return dataframe


def convert_terms(word, convert_terms_dict):
    to_return = word
    for _from, _to in convert_terms_dict.items():
        if word == _from:
            to_return = _to
    
    return to_return


def process_text(text,
                 remove_stopwords=True,
                 lemmatize=True,
                 stem=True):
    
    stop_words = []
    if remove_stopwords:
        stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    processed = []
    tokens = word_tokenize(text.translate(str.maketrans("", "", string.punctuation)))
    for token in tokens:
        if token not in stop_words:
            if lemmatize:
                lemma = lemmatizer.lemmatize(token)
                if stem:
                    stem = stemmer.stem(token)
                    processed.append(stem)
                else:
                    processed.append(lemma)
            else:
                processed.append(token)
    
    return " ".join(processed)


def get_relevant_terms(word_list, term):
    unique_words = list(set(" ".join(word_list).split()))
    relevant_words = [word for word in unique_words if term in word]

    return relevant_words