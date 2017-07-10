import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        best_BIC = math.inf
        best_model = self.base_model(self.n_constant)

        number_features = len(self.X[0])
        # N is the number of data points
        N = len(self.X)
        log_N = math.log(N)

        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n)
                # L is the likelihood of the fitted model
                log_L = model.score(self.X, self.lengths)
                # p is the number of parameters
                p = (n ** 2) + (2 * number_features * n) - 1
                # Calculate BIC for n
                BIC = -2 * log_L + p * log_N

                # The lower the BIC value the better the model
                if (BIC < best_BIC):
                    best_BIC = BIC
                    best_model = model
            except Exception as e:
                print (e)
                continue

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_DIC = -math.inf
        best_model = self.base_model(self.n_constant)

        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                # Work out score for i
                model = self.base_model(n)
                log_L = model.score(self.X, self.lengths)

                # Work out score for all but i
                all_but_i = 0
                for word, x in self.hwords.items():
                    if word != self.this_word:
                        all_but_i += model.score(x[0],x[1])

                # Calculate DIC for n
                DIC = log_L - (1/(len(self.hwords)-1)) * all_but_i

                # The higher the DIC value the better the model
                if DIC > best_DIC:
                    best_DIC = DIC
                    best_model = model
                
            except Exception as e:
                print (e)
                continue

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        best_CV = -math.inf
        best_model = self.base_model(self.n_constant)

        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n)
                
                n_splits=3
                word_sequences = self.sequences
                
                if len(word_sequences) < n_splits:
                    break

                split_method = KFold(n_splits=n_splits)

                sum_scores = 0
                number_scores = 0

                for cv_train_idx, cv_test_idx in split_method.split(word_sequences):
                    X_train, lengths_train = combine_sequences(cv_train_idx, word_sequences)
                    X_test, lengths_test = combine_sequences(cv_test_idx, word_sequences)

                    # Fit model with training set
                    model.fit(X_train, lengths_train)
                    # Score model with test set
                    sum_scores += model.score(X_test, lengths_test)
                    number_scores += 1
                # Calculate average of all the scores for n
                score = sum_scores / number_scores

                # The higher the CV value the better the model
                if score > best_CV:
                    best_CV = score
                    best_model = model

            except Exception as e:
                print (e)
                continue

        return best_model