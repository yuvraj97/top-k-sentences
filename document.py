from typing import Dict, List, Tuple, Union
import numpy as np
from collections import defaultdict
import nltk
from nltk.stem import WordNetLemmatizer
import re

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def upperCase(word: str) -> float:
    """
    :param word: string
    :return: proportion of Upper case letters
    """

    count_upper = 0
    for c in word:
        if c.isupper():
            count_upper += 1
    return count_upper / len(word)

def getSentencesIndices(paragraph: str) -> List[Tuple[int, int]]:
    """
    It returns the List of indices range for each sentence in the paragraph
    :param paragraph: string
    :return: List[Tuple[start_index, end_index]]
    """

    indices: List[Tuple[int, int]] = []
    prev = 0
    for i in range(1, len(paragraph)):
        if (i + 2 < len(paragraph) and
           paragraph[i] == '.' and
           paragraph[i + 1] == " " and
           paragraph[i + 2].isupper()):
            indices.append((prev, i))
            prev = i + 2
    indices.append((prev, len(paragraph)))
    return indices

class Document:

    def __init__(self, js: dict):
        """
        Here we receive a json formatted document.
        To score each sentence, we treat sections as a document,
        and these sections together creates whole corpus of documents.
        Here we use TF-IDF approach to give weight to every word for each document (section)

        :param js: represents document (template):  {
             "sections": [
                {
                    "text": "..."
                },
                {
                    "text": "..."
                }
            ]
        }
        """

        self._sections: List[Tuple[str, Dict[str, float]]] = []
        self._idf: defaultdict[str, int] = defaultdict(lambda: 0)
        self._weights: List[Dict[str, float]] = []
        self._sentence_weights: Dict[int, Tuple[Union[np.ndarray, float], str]] = {}
        self._stem = WordNetLemmatizer().lemmatize
        stop_tags = ["DT", "CC", "IN", "WDT", "WRB", "TO", "CD", "EX", "RBR", "PRP", "WP", "PRP$", "NNP", "MD"]
        self._tokenize = lambda sentence: [
            self._stem(word) for (word, tag) in nltk.pos_tag(nltk.word_tokenize(sentence))
            if re.search('[a-zA-Z]', tag) is not None and
            re.search('[a-zA-Z]', word) is not None and
            tag not in stop_tags and len(word) > 1
        ]

        self.__initialize(js)

    def __initialize(self, js: dict):
        """
        Here we initialize the TF for every documents (sections) with there word's frequency.
        Then we initialize the IDF for all words in the corpus (Document).
        Then we initialize the weights (TF-IDF) for every documents (sections) words.
        Then we give weights to every sentence in whole corpus (Document)

        Time Complexity: O(N)
        Space Complexity: O(N)
        Where N is length of corpus (Document)

        :param js: (representing document)
        :return:
        """

        for section in js["sections"]:
            section = section["text"]
            if section == "": continue
            tf: Dict[str, float] = defaultdict(lambda: 0)
            n_words = 0
            for paragraph in section.lower().split("\n"):
                for word in self._tokenize(paragraph):
                    tf[word] += 1
                    n_words += 1

            for word in tf:
                tf[word] = tf[word] / n_words

            self._sections.append((section, tf))

        self.__initialize_idf()
        self.__initialize_weights()
        self.__initialize_sentence_weighting()

    def __initialize_idf(self):
        """
        Here we initialize the IDF score for all words in the corpus (Document).

        Time Complexity: O(N)
        Space Complexity: O(N)
        Where N is length of corpus (Document)

        :return:
        """

        for section, tf in self._sections:
            for word in tf:
                self._idf[word] += 1
        for word in self._idf:
            self._idf[word] = np.log(len(self._sections) / self._idf[word])

    def __initialize_weights(self):
        """
        By using previously calculated TF and IDF scores,
        we initialize the weights (TF-IDF) for every documents (sections) words.

        Time Complexity: O(N)
        Space Complexity: O(N)
        Where N is length of corpus (Document)

        :return:
        """

        for section, tf in self._sections:
            weights: Dict[str, float] = {}
            for word in tf:
                weights[word] = tf[word] * self._idf[word]
            self._weights.append(weights)

    def __initialize_sentence_weighting(self):
        """
        Here we give weights to every possible sentence.
        Criteria of weighting a sentence:
            For every sentence we take weight of all possible words in that sentence,
            Where the weight for word W is p_U * STD + TF-IDF[W].
                Here STD is the standard deviation of TF-IDF values, it's the spread of TF-IDF.
                p_U is the proportion of uppercase letters in the word.
            Then we take all the weights that are above 1 standard deviation, and took average of all those weights.
            and this average weight is that sentence's weight

        Time Complexity: O(N)
        Space Complexity: O(N)
        Where N is length of corpus (Document)

        :return:
        """

        sentence_i = 0
        std = np.std([self._idf[w] for w in self._idf])
        for index, (section, tf) in enumerate(self._sections):
            for paragraph in section.split("\n"):
                for (start_index, end_index) in getSentencesIndices(paragraph):
                    sentence = paragraph[start_index: end_index]
                    weights = []
                    for word in nltk.word_tokenize(sentence):
                        word_s = self._stem(word.lower())
                        if word_s in self._weights[index]:
                            word_U = upperCase(word)
                            weights.append(word_U * std + self._weights[index][word_s])
                    if len(weights) == 0: continue
                    # weights = np.array(weights)
                    avg = np.mean(weights)
                    if not np.isnan(avg):
                        self._sentence_weights[sentence_i] = (avg, sentence)
                    sentence_i += 1

    @staticmethod
    def __get_all_k_sentences__(indices, length, D, sentences_indices):
        k = len(indices)

        for kth in range(k - 1, -1, -1):
            if indices[kth] + 1 >= length:
                continue
            if ((kth - 1 >= 0 and indices[kth - 1] >= indices[kth] + 1) or
                    (kth + 1 < k and indices[kth] + 1 >= indices[kth + 1])):
                continue
            if kth - 1 >= 0 and indices[kth] + 1 - indices[kth - 1] > D:
                continue

            indices[kth] += 1

            if tuple(indices) not in sentences_indices:
                Document.__get_all_k_sentences__(indices, length, D, sentences_indices)
                sentences_indices.add(tuple(indices))

            indices[kth] -= 1

    @staticmethod
    def __get_best_sentence__(indices, sentencesD):
        max_weight, best_set = 0, []
        for idx in indices:
            weight = sum([sentencesD[i][0] for i in idx])
            if weight > max_weight:
                max_weight = weight
                best_set = idx
        return best_set

    def get_top_k_sentence(self, k: int, D: int):
        """
        This method gives us top 'k' sentence from the Document
        while maintaining the maximum distance (D)

        Because we don't want sentences to be farther then each other,
        consecutive sentences must have some maximum distance (D).

        :param k: (top k sentences)
        :param D: (maximum distance between two consecutive sentences)
        :return:
        """

        sentences_indices = set()
        initial_idx = [_ for _ in range(k)]
        sentences_indices.add(tuple(initial_idx))
        print(
            initial_idx,
            len(self._sentence_weights),
            D
        )

        self.__get_all_k_sentences__(
            [_ for _ in range(k)],
            len(self._sentence_weights),
            D,
            sentences_indices
        )

        top_k_sentences_indices = self.__get_best_sentence__(
            sentences_indices,
            self._sentence_weights
        )
        return [(idx, self._sentence_weights[idx]) for idx in top_k_sentences_indices]
