import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ 
    Recognizes test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Likelihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    all_word_Xlengths = test_set.get_all_Xlengths()

    for test_word in all_word_Xlengths:

        X, lengths = all_word_Xlengths[test_word]
        probs_dict = {}
        for w in models:
            try:
                model = models[w]
                probs_dict[w] = model.score(X, lengths)
            except:
                pass

        guesses.append(max(probs_dict, key=probs_dict.get))
        probabilities.append(probs_dict)

    return probabilities, guesses
