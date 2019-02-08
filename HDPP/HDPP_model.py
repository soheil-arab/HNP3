import pickle
from HDPP.particle import Particle
import numpy as np
from copy import deepcopy
from numpy import random as random


# from scipy.cluster.hierarchy import fclusterdata
# from topic_model import TopicModel


# from threading import Thread
def dist(topic1, topic2):
    return topic1.distance(topic2)


class NP3:
    def __init__(self, name, number_of_user, adjacency_matrix,
                 hyper_params,
                 number_of_particles, block_size, number_of_samples):

        self.name = name
        self.number_of_user = number_of_user
        self.adjacency_matrix = adjacency_matrix

        self.hyperparams = hyper_params

        self.number_of_particles = number_of_particles
        self.block_size = block_size
        self.number_of_samples = number_of_samples

        self.expected_mu = list()
        self.expected_alpha = list()
        # The threshold for effective sample size to resample
        self.thresh = int(number_of_particles / 2) - 1
        # Number of processed docs
        self.number_of_doc = 0
        # The counter for saving models iteratively
        self.counter = 0

        self.initial_log_weight = np.log(1.0 / number_of_particles)
        self.particles = [
            Particle(name=self.name, user_num=self.number_of_user, adjacency_matrix=self.adjacency_matrix,
                     hyper_params=hyper_params, initial_log_weight=self.initial_log_weight,
                     block_size=self.block_size, number_of_samples=self.number_of_samples)
            for i in range(number_of_particles)]

        #########################################################################
    def add_event(self, event):
        self.number_of_doc += 1
        for idx, particle in enumerate(self.particles):
            particle.update(event)
            event = particle.events[len(particle.events) - 1]
            print('Particle #{0}: s={1} , z={2}, l={3}'.format(idx, event['parent'], event['topic'], event['locality']))
        self.expected_mu.append(self.mu_expectation(event['time']))
        self.expected_alpha.append(self.alpha_expectation())
        self.resample_particles()

            #########################################################################

    def resample_particles(self):
        log_weights = list()
        for idx, particle in enumerate(self.particles):
            log_weights.append(particle.log_particle_weight)
        log_weights = np.array(log_weights)
        mx = np.max(log_weights)
        log_weights -= mx
        weights = np.exp(log_weights) / np.sum(np.exp(log_weights))

        print("Effective weight is:{0}".format(1 / np.sum(weights ** 2)))
        if 1 / np.sum(weights ** 2) < self.thresh:
            print("Resampling needed.")
            indices = random.choice(len(weights), size=len(weights), p=weights)
            print("These samples selected:{0}".format(indices))
            temp_particles = self.particles
            for i in range(len(self.particles)):
                self.particles[i] = deepcopy(temp_particles[indices[i]])
                # self.particles[i] = temp_particles[indices[i]].get_a_copy()
                self.particles[i].log_particle_weight = np.log(1 / len(self.particles))
        else:
            for i in range(len(self.particles)):
                self.particles[i].log_particle_weight = np.log(weights[i])

        #### For debugging
        num_of_topics = 0
        for idx, particle in enumerate(self.particles):
            num_of_topics += weights[idx] * particle.number_of_topics
        print("expected number of topics:{0}".format(num_of_topics))

                #########################################################################

    def alpha_expectation(self):
        mx = self.particles[0].log_particle_weight
        for particle in self.particles:
            if particle.log_particle_weight > mx:
                mx = particle.log_particle_weight
        alpha = np.ndarray((self.number_of_user, self.number_of_user))
        for j in range(self.number_of_user):
            for i in range(self.number_of_user):
                alpha[i][j] = 0
                weight_sum = 0
                if self.adjacency_matrix[i][j] != 0:
                    for particle in self.particles:
                        w_particle = np.exp(particle.log_particle_weight - mx)
                        alpha[i][j] += w_particle * (self.hyperparams['a'] + particle.counts[i][j]) / (
                            self.hyperparams['b'] + particle.omega_u.get(i, 0))
                        weight_sum += w_particle
                    alpha[i][j] /= weight_sum
        return alpha

        #########################################################################

    def mu_expectation(self, time):
        mu = np.zeros((self.number_of_user, 1))
        mx = self.particles[0].log_particle_weight
        for particle in self.particles:
            if particle.log_particle_weight > mx:
                mx = particle.log_particle_weight
        for u in range(self.number_of_user):
            weight_sum = 0
            for particle in self.particles:
                w_particle = np.exp(particle.log_particle_weight - mx)
                mu[u] += w_particle * (self.hyperparams['c'] + particle.counts[u][u]) / (
                    self.hyperparams['d'] + time)
                weight_sum += w_particle
            mu[u] /= weight_sum
        return mu

    def compute_log_likelihood(self, time, user, document):
        mx = self.particles[0].log_particle_weight
        for particle in self.particles:
            if particle.log_particle_weight > mx:
                mx = particle.log_particle_weight
        weights = np.zeros(len(self.particles))
        for idx, particle in enumerate(self.particles):
            weights[idx] = np.exp(particle.log_particle_weight - mx)
        weights /= np.sum(weights)
        log_likelihood = 0.0
        for idx, particle in enumerate(self.particles):
            log_likelihood += np.log(weights[idx]) + particle.compute_likelihood(time, user, document)[1]
        return log_likelihood

        #########################################################################

    def compute_time_log_likelihood(self, events):
        mx = self.particles[0].log_particle_weight
        for particle in self.particles:
            if particle.log_particle_weight > mx:
                mx = particle.log_particle_weight
        weights = np.zeros(len(self.particles))
        for idx, particle in enumerate(self.particles):
            weights[idx] = np.exp(particle.log_particle_weight - mx)
        weights /= np.sum(weights)
        particle_log_likelihood = list()
        for idx, particle in enumerate(self.particles):
            particle_log_likelihood.append(np.log(weights[idx]) + particle.compute_time_likelihood(events))
        total_log_likelihood = np.zeros(len(events))
        for i in range(len(events)):
            mx = -1 * np.inf
            for p, particle in enumerate(self.particles):
                if particle_log_likelihood[p][i] > mx:
                    mx = particle_log_likelihood[p][i]
            for p, particle in enumerate(self.particles):
                total_log_likelihood[i] += np.exp(particle_log_likelihood[p][i] - mx)
            total_log_likelihood[i] = np.log(total_log_likelihood[i]) + mx
        # log_likelihood = np.sum(total_log_likelihood)
        return total_log_likelihood

        #########################################################################

    def global_topic_distribution_expectation(self):
        G_0 = dict()
        weight_sum = 0
        # for particle in self.particles:
        particle = self.particles[0]
        w_particle = np.exp(particle.log_particle_weight)
        for k in range(particle.number_of_topics):
            topic_k = particle.topics[k]
            topic_added = 0
            topic_weight = w_particle * np.exp(particle.log_m_k[k]) / (particle.zeta + np.exp(particle.log_m_0))
            for topic in G_0:
                if topic_k.distance(topic) < 1e-10:
                    G_0[topic] += topic_weight
                    if topic.sum_eta < topic.sum_eta:
                        G_0[topic_k] = G_0[topic]
                        del G_0[topic]
                    topic_added = 1
                    break
            if topic_added == 0:
                G_0[topic_k] = topic_weight
        weight_sum += w_particle
        for topic in G_0:
            G_0[topic] /= weight_sum
        return G_0

        #########################################################################

    def save(self):
        f = open('../results/' + self.name + '/HDPP/estimated_alpha_' + str(self.number_of_doc) + '.pk', 'wb')
        pickle.dump(self.expected_alpha, f)
        f.close()
        f = open('../results/' + self.name + '/HDPP/estimated_mu_' + str(self.number_of_doc) + '.pk', 'wb')
        pickle.dump(self.expected_mu, f)
        f.close()
        for idx, particle in enumerate(self.particles):
            particle.save(idx)

    def estimate_parent(self, event_number):
        probs = dict()
        for p in range(len(self.particles)):
            parent = self.particles[p].events[event_number]["parent"]
            probs[parent] = probs.get(parent, 0.0) + np.exp(self.particles[p].log_particle_weight)

        mx = -1 * np.inf
        most_probable_parent = -1
        for parent in probs:
            if probs[parent] > mx:
                mx = probs[parent]
                most_probable_parent = parent
        return most_probable_parent

        #########################################################################

    def load(self, number_of_events):
        self.number_of_doc = number_of_events
        f = open('../results/' + self.name + '/HDPP/estimated_alpha_' + str(self.number_of_doc) + '.pk', 'rb')
        self.expected_alpha = pickle.load(f)
        f.close()
        f = open('../results/' + self.name + '/HDPP/estimated_mu_' + str(self.number_of_doc) + '.pk', 'rb')
        self.expected_mu = pickle.load(f)
        f.close()
        self.particles = [
            Particle(self.adjacency_matrix, self.a, self.hyperparams['b'] , self.hyperparams['c'], self.hyperparams['d'], self.gamma, self.zeta, self.name,
                     self.epsilon, self.kernel_mean, self.kernel_var,
                     local_topic_decreasing_coefficient=self.local_topic_decreasing_coefficient,
                     global_topic_decreasing_coefficient=self.global_topic_decreasing_coefficient,
                     initial_log_weight=self.initial_log_weight, user_num=self.number_of_user,
                     block_size=self.block_size,
                     vocab_size=self.vocab_size, default_eta=self.default_eta,
                     number_of_samples=self.number_of_samples)
            for i in range(self.number_of_particles)]
        for p in range(len(self.particles)):
            print("going to load particle {0}".format(p))
            f = open('../results/' + self.name + '/HDPP/particle' + str(p) + '_' + str(number_of_events) + '.pk',
                     'rb')
            loaded_particle = pickle.load(f)
            f.close()
            loaded_events = loaded_particle["events"]
            # for key in loaded_particle:
            #     print(key)
            # ins += self.block_size
            self.particles[p].add_analyzed_events(loaded_events, loaded_particle["topics"],
                                                  loaded_particle["log_m_ku"], loaded_particle["log_m_k"],
                                                  loaded_particle["log_m_u0"], loaded_particle["log_m_0"],
                                                  loaded_particle["omega_u"],
                                                  loaded_particle["old_events_omega_u"],
                                                  loaded_particle["log_particle_weight"])
