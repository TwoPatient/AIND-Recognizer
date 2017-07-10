import warnings
import math
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    for x, length in test_set.get_all_Xlengths().values():
        probability = {}
        guess = ''
        highest_probability = -math.inf
        for word, model in models.items():
            try:
                probability[word] = model.score(x, length)
            except Exception as e:
                probability[word] = -math.inf
            # If probability is highest that becomes guess
            if probability[word] > highest_probability:
                highest_probability = probability[word]
                guess = word

        probabilities.append(probability)
        guesses.append(guess)

    return probabilities, guesses
