import pickle
from Hawkes.hawkes_particle import HawkesParticle
import numpy as np
from copy import deepcopy
from numpy import random as random


# from threading import Thread


class HawkesModel:
    def __init__(self, name, number_of_user, adjacency_matrix,
                 hyper_params,
                 number_of_particles, block_size):
        self.name = name
        self.hyper_params = hyper_params
        self.adjacency_matrix = adjacency_matrix
        # Number of users of dataset
        self.number_of_user = number_of_user
        self.block_size = block_size

        self.expected_mu = list()
        self.expected_alpha = list()
        # The threshold for effective sample size to resample
        self.thresh = int(number_of_particles / 2) - 1
        # Number of processed docs
        self.number_of_doc = 0
        # The counter for saving models iteratively
        self.counter = 0

        initial_log_weight = np.log(1.0 / number_of_particles)
        self.particles = [
            HawkesParticle(name=self.name, user_num=number_of_user, adjacency_matrix = self.adjacency_matrix,
                           hyper_params=self.hyper_params,
                           initial_log_weight=initial_log_weight, block_size=self.block_size)
            for i in range(number_of_particles)]

    def add_event(self, event):
        time = event['time']
        user= event['user']
        document = event['document']
        log_weights = list()
        for idx, particle in enumerate(self.particles):
            particle.update(time, user, document)
            log_weights.append(particle.log_particle_weight)
            event = particle.events[len(particle.events) - 1]
            print('Particle #{0}: s={1}'.format(idx, event["parent"], ))
        log_weights = np.array(log_weights)
        mx = np.max(log_weights)
        log_weights -= mx
        weights = np.exp(log_weights) / np.sum(np.exp(log_weights))
        # print(1 / np.sum(weights ** 2))
        # print('max log weight is %f'%np.max(log_weights))
        self.expected_mu.append(self.mu_expectation(time))
        self.expected_alpha.append(self.alpha_expectation())
        print("Effective weight is:{0}".format(1 / np.sum(weights ** 2)))
        if 1 / np.sum(weights ** 2) < self.thresh:
            print("Resampling needed.")
            # print("log-Weights before resampling are:{0}".format(log_weights))
            self.resample_particles(weights)
        else:
            for i in range(len(self.particles)):
                self.particles[i].log_particle_weight = np.log(weights[i])

                # Print for debugging
                # new_weights = list()
                # for particle in self.particles:
                # new_weights.append(np.exp(particle.log_particle_weight))
                # print("new Weights are:{0}".format(new_weights))

    def resample_particles(self, weights):
        indices = random.choice(len(weights), size=len(weights), p=weights)
        print("These samples selected:{0}".format(indices))
        temp_particles = self.particles
        for i in range(len(self.particles)):
            self.particles[i] = deepcopy(temp_particles[indices[i]])
            # self.particles[i] = temp_particles[indices[i]].get_a_copy()
            self.particles[i].log_particle_weight = np.log(1 / len(self.particles))

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
                        alpha[i][j] += w_particle * (self.hyper_params['a'] + particle.counts[i][j]) / (
                            self.hyper_params['b'] + particle.omega_u.get(i, 0))
                        weight_sum += w_particle
                    alpha[i][j] /= weight_sum
        return alpha

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
                mu[u] += w_particle * (self.hyper_params['c'] + particle.counts[u][u]) / (
                    self.hyper_params['d'] + time)
                weight_sum += w_particle
            mu[u] /= weight_sum
        return mu

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

    def save(self):
        f = open('../results/' + self.name + '/hawkes/estimated_alpha' + '.pk', 'wb')
        pickle.dump(self.expected_alpha, f)
        f.close()
        f = open('../results/' + self.name + '/hawkes/estimated_mu' + '.pk', 'wb')
        pickle.dump(self.expected_mu, f)
        f.close()
        for idx, particle in enumerate(self.particles):
            particle.save(idx, self.counter)
        self.counter += 1
