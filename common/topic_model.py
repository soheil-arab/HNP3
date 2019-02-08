__author__ = 'Abbas'
import numpy as np
from scipy.stats import entropy as entropy
from common import helper


class TopicModel:
    def __init__(self, default_beta, default_eta):
        self.eta = dict()
        self.vocab_size = len(default_eta)
        self.beta = default_beta
        # Value of eta before adding any document
        self.default_eta = default_eta
        self.sum_eta = sum(default_eta)
        self.min_distance = 0.0001

    # @classmethod
    # def init_using_eta(eta, beta):
    #     topicModel = TopicModel()
    #     topicModel.eta = eta
    #     topicModel.size = len(eta)
    #     topicModel.beta = beta
    #     topicModel.default_eta = eta
    #     topicModel.sum_eta = sum(eta)
    #     topicModel.number_of_samples = 1000
    #     return topicModel

    def log_likelihood(self, document):
        return helper.document_log_likelihood(self.eta,
                                              document['words_num'], document['words'],
                                              self.sum_eta, self.default_eta)

    def update(self, document):
        for word_idx in document['words'].keys():
            self.eta[word_idx] = self.eta.get(word_idx, 0) + document['words'][word_idx]
        self.sum_eta += document['words_num']

    def update_kernel(self, beta):
        self.beta = beta

    def get_mean_of_topic(self):
        return self.eta

    def get_eta_as_list(self):
        eta_1 = np.zeros((self.vocab_size, 1))
        for i in range(self.vocab_size):
            eta_1[i] = self.default_eta[i]+self.eta.get(i,0)
        eta_1 /= np.sum(eta_1)
        return eta_1

    def distance(self, topic):
        eta_1 = np.zeros((self.vocab_size, 1))
        eta_2 = np.zeros((self.vocab_size, 1))
        for i in range(self.vocab_size):
            eta_1[i]=self.default_eta[i]+self.eta.get(i,0)
            eta_2[i] = self.default_eta[i]+topic.eta.get(i,0)
        eta_1/=np.sum(eta_1)
        eta_2 /= np.sum(eta_2)
        return (entropy(eta_1,eta_2)+entropy(eta_2,eta_1))/2