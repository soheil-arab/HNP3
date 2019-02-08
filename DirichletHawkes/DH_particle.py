from __future__ import division, print_function
# from copy import deepcopy
import numpy as np
from numpy import random as random
from collections import deque
from common.topic_model import TopicModel
import pickle
# import six.moves.cPickle as pickle
import sys
# import pyximport

# pyximport.install(setup_args={'include_dirs': np.get_include()})
# from HNP3 import helper
from math import exp, log


class Particle:
    def __init__(self, name, user_num, hyper_params,
                 initial_log_weight, number_of_samples, block_size):
        self.name = name
        self.gamma = hyper_params['gamma']

        self.block_size = block_size
        self.kernel_mean = hyper_params['kernel_mean']
        self.kernel_var = hyper_params['kernel_var']
        self.default_eta = hyper_params['default_eta']
        kernel_sigma = np.sqrt(self.kernel_var)
        self.threshold = -np.log(hyper_params['epsilon']) / np.max([self.kernel_mean - 2 * kernel_sigma, 1e-5])

        # Number of samples to estimate the posterior of the topics kernel parameter
        self.number_of_samples = number_of_samples

        # Log of the particle weight
        self.log_particle_weight = initial_log_weight

        self.user_num = user_num
        self.number_of_docs = 0

        self.topics_u = list()
        for i in range(user_num):
            self.topics_u.append(list())

        self.events = dict()
        self.active_set = list()
        for u in range(user_num):
            self.active_set.append(deque())

        self.first_active_event = np.zeros(shape=(user_num, 1), dtype=int)

        self.log_m_uk = list()
        for i in range(user_num):
            self.log_m_uk.append(list())
        self.log_m_u0 = -1 * np.inf * np.ones(self.user_num)
        self.last_update_time = np.zeros(shape=(user_num, 1))

        self.omega_u = dict()
        self.old_events_num_uk = list()
        for u in range(self.user_num):
            self.old_events_num_uk.append(dict())

        self.document_log_likelihood_per_topic = list()

        self.sum_of_time_diff = list()
        for u in range(user_num):
            self.sum_of_time_diff.append(dict())
        self.topic_u_lenghts = np.zeros(user_num, dtype=int)

    # Update the particle after adding a new event by adding a new (z,s,l) to the particle
    def update(self, time, user, document):
        # Remove old events from the active set
        ###################################################
        # print('1-->len={0},last_len={1}'.format(len(self.topics_u[25]), self.topic_u_lenghts[25]))
        # for u in range(self.user_num):
        #     if len(self.topics_u[u]) < self.topic_u_lenghts[u]:
        #         print('1-->len={0},len_last={1}'.format(len(self.topics_u[u]), self.topic_u_lenghts[u]))
        #         sys.exit()
        ###################################################

        self.deque_old_events(time)
        ###################################################
        # print('2-->len={0},last_len={1}'.format(len(self.topics_u[25]), self.topic_u_lenghts[25]))
        # for u in range(self.user_num):
        #     if len(self.topics_u[u]) < self.topic_u_lenghts[u]:
        #         print('2-->len={0},len_last={1}'.format(len(self.topics_u[u]), self.topic_u_lenghts[u]))
        #         sys.exit()
        ###################################################

        log_prob, document_log_likelihood = self.compute_likelihood(time, user, document)
        ###################################################
        # print('3-->len={0},last_len={1}'.format(len(self.topics_u[25]), self.topic_u_lenghts[25]))
        # for u in range(self.user_num):
        #     if len(self.topics_u[u]) < self.topic_u_lenghts[u]:
        #         print('3-->len={0},len_last={1}'.format(len(self.topics_u[u]), self.topic_u_lenghts[u]))
        #         sys.exit()
        ###################################################
        self.number_of_docs += 1
        # print("New log_particle_weight is:{0}".format(self.log_particle_weight + document_log_likelihood))
        self.log_particle_weight += document_log_likelihood
        # **************** Sampling the s and z and r ***********************
        mx = np.max(log_prob)
        prob = np.exp(log_prob[:, 0] - mx)
        prob = prob / np.sum(prob)
        # print('prob.shape={0}'.format(prob.shape))
        s = random.choice(len(prob), size=1, p=prob)[0] + self.first_active_event[user][0]
        if s != self.number_of_docs - 1:
            z = self.events[s]["topic"]
        else:
            z = len(self.topics_u[user])

        # print('s={0},self.number_of_docs={1},z={2},len(self.topics_u[user])={3}'.format(s, self.number_of_docs, z,
        #                                                                                 len(self.topics_u[user])))
        new_event = dict()
        new_event['time'] = time
        new_event['user'] = user
        new_event['document'] = document
        new_event['parent'] = s
        new_event['topic'] = z
        new_event['document_number'] = self.number_of_docs - 1
        self.active_set[user].append(new_event)
        # self.events_of_users[user] = temp_deque
        self.events[self.number_of_docs - 1] = new_event
        ###################################################
        # print('4-->len={0},last_len={1}'.format(len(self.topics_u[25]), self.topic_u_lenghts[25]))
        # for u in range(self.user_num):
        #     if len(self.topics_u[u]) < self.topic_u_lenghts[u]:
        #         print('4-->len={0},len_last={1}'.format(len(self.topics_u[u]), self.topic_u_lenghts[u]))
        #         sys.exit()
        ###################################################
        # Updating topics
        if s == self.number_of_docs - 1:
            self.topics_u[user].append(TopicModel(default_beta=self.kernel_mean, default_eta=self.default_eta))
            self.topic_u_lenghts[user] += 1
            self.log_m_uk[user].append(-1 * np.inf)
            self.sum_of_time_diff[user][z] = 0
        ###################################################
        # print('5-->len={0},last_len={1}'.format(len(self.topics_u[25]), self.topic_u_lenghts[25]))
        # for u in range(self.user_num):
        #     if len(self.topics_u[u]) < self.topic_u_lenghts[u]:
        #         print('5-->len={0},len_last={1}'.format(len(self.topics_u[u]), self.topic_u_lenghts[u]))
        #         sys.exit()
        ###################################################
        self.update_weight_of_topic(z, user)
        ###################################################
        # print('6-->len={0},last_len={1}'.format(len(self.topics_u[25]), self.topic_u_lenghts[25]))
        # for u in range(self.user_num):
        #     if len(self.topics_u[u]) < self.topic_u_lenghts[u]:
        #         print('6-->len={0},len_last={1}'.format(len(self.topics_u[u]), self.topic_u_lenghts[u]))
        #         sys.exit()
        ###################################################
        # print("z={0},len(topics_u[user])={1}".format(z,len(self.topics_u[user])))
        self.topics_u[user][z].update(document)
        ###################################################
        # print('7-->len={0},last_len={1}'.format(len(self.topics_u[25]), self.topic_u_lenghts[25]))
        # for u in range(self.user_num):
        #     if len(self.topics_u[u]) < self.topic_u_lenghts[u]:
        #         print('7-->len={0},len_last={1}'.format(len(self.topics_u[u]), self.topic_u_lenghts[u]))
        #         sys.exit()
        ###################################################
        # self.document_log_likelihood_per_topic[z] = helper.document_log_likelihood(self.topics[z].eta,
        # self.topics[z].size,
        # document['length'],
        # document['words'],
        # self.topics[z].sum_eta)

        # # Updating the counts
        # self.counts[self.events[s]['user']][user] += 1

        # Updating the sum_of_time_diffs
        if s != self.number_of_docs - 1:
            self.sum_of_time_diff[user][z] = self.sum_of_time_diff[user].get(z, 0) + time - self.events[s]['time']
            self.update_beta(user, z, time)
        ###################################################
        # print('8-->len={0},last_len={1}'.format(len(self.topics_u[25]), self.topic_u_lenghts[25]))
        # for u in range(self.user_num):
        #     if len(self.topics_u[u]) < self.topic_u_lenghts[u]:
        #         print('8-->len={0},len_last={1}'.format(len(self.topics_u[u]), self.topic_u_lenghts[u]))
        #         sys.exit()
        ###################################################
        # Updating the weight of the particle
        # self.update_weight(document, time, user)

        # Updating the kernel
        # self.update_beta(user, z, time)
        # if self.number_of_docs % 100 == 0:
        #     for u in range(self.user_num):
        #         for z in range(len(self.topics_u[user])):
        #             self.update_beta(user, z, time)
        # self.update_beta(user, z, time)
        # print("New Beta is:{0}".format(self.topics[z].beta))
        # print('%f', (sys_time()-t_s))
        # sys.stdout.flush()
        return s, z

    def compute_likelihood(self, time, user, document):
        # update the weight of tables of restaurants
        self.compute_topic_weights(time, user)

        number_of_active_docs = self.number_of_docs - self.first_active_event[user][0]
        # ****************Computing the omega_u and the document log likelihood***********

        # a new topic model with no documents
        tmp_topic = TopicModel(default_beta=self.kernel_mean,
                               default_eta=self.default_eta)

        self.document_log_likelihood_per_topic = list()
        for idx, topic in enumerate(self.topics_u[user]):
            self.document_log_likelihood_per_topic.append(topic.log_likelihood(document))
            # self.document_log_likelihood_per_topic.append(
            # helper.document_log_likelihood(topic.eta, topic.size, document['length'], document['words'],
            # topic.sum_eta, topic.default_eta))
        self.document_log_likelihood_per_topic.append(tmp_topic.log_likelihood(document))
        log_probs = np.zeros(shape=(number_of_active_docs + 1, 1))
        # print('user={0},self.first_active_event[user]={1},self.number_of_docs={2}'.format(user,self.first_active_event[user][0],self.number_of_docs))
        for s in range(self.first_active_event[user], self.number_of_docs + 1):
            if s == self.number_of_docs:
                log_probs[s - self.first_active_event[user]] = np.log(self.gamma) + \
                                                               self.document_log_likelihood_per_topic[-1]
                # print('log(gamma)={0},document_log_likelihood_per_topic[-1]={1}'.format(np.log(self.gamma),
                #                                                                         self.document_log_likelihood_per_topic[
                #                                                                             -1]))
            else:
                user_s = self.events[s]["user"]
                if user_s == user:
                    z_s = self.events[s]["topic"]
                    t_s = self.events[s]["time"]
                    log_probs[s - self.first_active_event[user]] = log(0.0001) - self.topics_u[user][z_s].beta * (
                        time - t_s) + \
                                                                   self.document_log_likelihood_per_topic[z_s]
                    # print('beta={0},time - t_s={1},document_log_likelihood_per_topic[{2}]={3}'.
                    #       format(self.topics_u[user][z_s].beta, time - t_s, z_s,
                    #              self.document_log_likelihood_per_topic[z_s]))
                else:
                    log_probs[s - self.first_active_event[user]] = -1 * np.inf
        # print('user={0},self.first_active_event[user]={1}'.format(user, self.first_active_event[user]))
        # for s in range(self.first_active_event[user], self.number_of_docs + 1):
        #     print('{0}'.format(log_probs[s - self.first_active_event[user]][0][0]),end=',')
        # print()

        # **************** updating the particle weight ***********************
        mx = np.max(log_probs)
        prob = np.exp(log_probs - mx)
        # for s in range(self.first_active_event[user], self.number_of_docs + 1):
        #     print('{0}'.format(prob[s - self.first_active_event[user]][0][0]), end=',')
        # print()
        document_log_likelihood = np.log(sum(prob)) + mx
        return log_probs, document_log_likelihood[0]

    def deque_old_events(self, time):
        for user in range(self.user_num):
            while len(self.active_set[user]) > 1 and time - self.active_set[user][0]['time'] > self.threshold:
                user = self.active_set[user][0]['user']
                topic = self.active_set[user][0]['topic']
                self.old_events_num_uk[user][topic] = self.old_events_num_uk[user].get(topic, 0) + 1
                self.active_set[user].popleft()
            if len(self.active_set[user]) != 0:
                self.first_active_event[user] = self.active_set[user][0]['document_number']
            else:
                self.first_active_event[user] = 0

    # Updates the weight of topics for each user and in the up-level restaurant
    # after passing time-last_update_time time and adding no new event in this interval
    def compute_topic_weights(self, time, user):
        last_time = self.last_update_time[user]
        self.log_m_u0[user] = 0
        m_0 = 0
        for k in range(len(self.log_m_uk[user])):
            self.log_m_uk[user][k] -= self.topics_u[user][k].beta * (time - last_time)
            m_0 += np.exp(self.log_m_uk[user][k])
        self.log_m_u0[user] = np.log(m_0)
        self.last_update_time[user] = time

    # Update the weight of topic z for user u after observing a new event in that topic
    def update_weight_of_topic(self, z, user):  # m_u_k(t)
        # if len(self.log_m_uk[user]) < z:
        #     print('An error occured. Wrong topic number.')
        # else:
        #     if len(self.log_m_uk[user])<z+1:
        #         self.log_m_uk[user].append(0)
        #     else:
        #         print('user={0},z={1},len(self.log_m_uk[user])={2}'.format(user,z,len(self.log_m_uk[user])))
        self.log_m_uk[user][z] = log(exp(self.log_m_uk[user][z]) + 1)
        self.log_m_u0[user] = log(exp(self.log_m_u0[user]) + 1)

    def compute_omega_u_between_two_time(self, t_last, time):
        sum_omega_u = dict()
        for user in range(self.user_num):
            for event in self.active_set[user]:
                k = event["topic"]
                beta_k = self.topics_u[user][k].beta
                t_e = event["time"]
                sum_omega_u[user] = sum_omega_u.get(user, 0) + 1 / beta_k * (
                    exp(-beta_k * (t_last - t_e)) - exp(-beta_k * (time - t_e)))
                # print('user={0},k={1},1/beta_k={2},zarib={3},sum_omega_u[user]={4}'.format(user, k, 1 / beta_k, exp(
                #     -beta_k * (t_last - t_e)) - exp(-beta_k * (time - t_e)), sum_omega_u[user]))
        return sum_omega_u

    def update_beta(self, user, k, time):
        scale = self.kernel_var / self.kernel_mean  # only to avoid computational errors
        shape = self.kernel_mean ** 2 / self.kernel_var
        beta_sample = random.gamma(shape=shape, scale=scale, size=self.number_of_samples)
        sample_mean = np.mean(beta_sample)
        sample_var = np.var(beta_sample)
        # print('sample_mean={0},sample_varince={1}'.format(sample_mean, sample_var))
        # beta_sample /= 100  # only to avoid computational errors
        weight = np.zeros(self.number_of_samples)
        for i in range(self.number_of_samples):
            weight[i] = self.compute_beta_log_likelihood(user, k, beta_sample[i], time)
        # temp = weight
        weight -= np.max(weight)
        weight = np.exp(weight)
        weight /= np.sum(weight)
        beta = beta_sample.dot(weight)
        # print('weighted_beta_sample_mean={0}'.format(beta))
        self.topics_u[user][k].update_kernel(beta)
        return beta

    def compute_beta_log_likelihood(self, user, k, beta, time):
        result = -1 * beta * self.sum_of_time_diff[user].get(k, 0)
        for event in self.active_set[user]:
            if event["topic"] == k:
                result -= (1 / beta) * (1 - exp(-1 * beta * (time - event['time'])))

        result += self.old_events_num_uk[user].get(k, 0) / beta
        return result

    def compute_time_likelihood(self, events):
        N = len(events)
        log_likelihood = np.zeros((N, 1))
        for event in events:
            event['time'] -= self.events[0]['time']
        last_t = self.events[self.number_of_docs - 1]['time']
        for idx, event in enumerate(events):
            # print('Computing time_log_likelihood for event {0} started.'.format(idx))
            u = event['user']
            t = event['time']
            # if t == last_t:
            #     t += 1e-6
            # print('u={0},t={1},last_t={2}'.format(u, t, last_t))
            omega_u = self.compute_omega_u_between_two_time(last_t, t)
            omega = 0
            for v in range(self.user_num):
                omega += omega_u.get(v, 0)
                # print('DirichletHawkes-->v:{0},np.sum(alpha[v, :]):{1},omega_u[v]={2},omega={3}'.format(v, np.sum(alpha[v, :]),
                #                                                                    omega_u[v],omega))
            beta = self.kernel_mean
            print('DirichletHawkes-->omega1:{0}'.format(omega))
            for i in range(idx):
                e = events[i]
                t_e = e['time']
                omega += (1 / beta) * (
                    np.exp(-1 * beta * (last_t - t_e)) - np.exp(-1 * beta * (t - t_e)))
                # else:
                #     omega += np.sum(alpha[v, :]) * 1 / beta
            print('DirichletHawkes-->omega2:{0}'.format(omega))
            lambda_u = 0
            lambda_u += self.gamma
            for e in self.active_set[u]:
                t_s = e['time']
                k = e['topic']
                u_s = e['user']
                if u_s == u:
                    # print("Lambda-->event_num={0},beta:{1}, (t-t_s)={2}, val={3}".format(e['document_number'],
                    #                                                                      self.topics_u[u][k].beta,
                    #                                                                      t - t_s,
                    #                                                                      1e-4 * np.exp(
                    #                                                                          -1 * self.topics_u[u][
                    #                                                                              k].beta * (t - t_s))))
                    lambda_u += 1e-5*np.exp(-1*self.topics_u[u][k].beta*(t-t_s))

            for s in range(idx):
                source = events[s]
                u_s = source['user']
                t_s = source['time']
                if u_s == u:
                    lambda_u += 1e-5*np.exp(-1 * beta * (t - t_s))
            log_likelihood[idx] = -1 * ((t - last_t) * (self.gamma * self.user_num) + 1e-5 * omega) + np.log(lambda_u)
            print('DirichletHawkes:(t - last_t):{0}, (self.gamma * self.user_num):{1},omega:{2},log(lambda):{3}'.
                  format((t - last_t), (self.gamma * self.user_num), omega, np.log(lambda_u)))
            last_t = event['time']
            print('DirichletHawkes:event {0} processed. log_likelihood={1}'.format(idx, log_likelihood[idx]))
        return log_likelihood

    def add_analyzed_events(self, analyzed_events, topics_u,
                            log_m_uk, log_m_u0, omega_u, old_events_num_uk, log_particle_weight):
        self.topics_u = topics_u
        self.log_m_uk = log_m_uk
        self.log_m_u0 = log_m_u0

        time = 0
        keys = list(analyzed_events.keys())
        keys.sort()
        for key in keys:
            event = analyzed_events[key]
            self.number_of_docs += 1
            user = event['user']
            self.events[self.number_of_docs - 1] = event
            self.active_set[user].append(event)
            time = np.max((event["time"], time))

        self.deque_old_events(time)
        # self.first_active_event[user] = self.active_set[0]['document_number']

        # self.omega_u = omega_uv
        self.omega_u = omega_u
        self.old_events_num_uk = old_events_num_uk
        self.log_particle_weight = log_particle_weight

        # self.document_log_likelihood_per_topic = list()

        # self.sum_of_time_diff = dict()

    def save(self, idx, counter):
        user = self.events[174]['user']
        parent = self.events[174]['parent']
        k = self.events[174]['topic']
        # print(
        #     '##########################################\nuser = {0},parent = {1} , topic ={2},'
        #     ' len(model.topics_u[user])={3},self.topic_u_lenghts[user]={4}\n###################################'.format(
        #         user, parent, k, len(
        #             self.topics_u[user]),self.topic_u_lenghts[user]))
        save_data = dict()
        save_data["log_particle_weight"] = self.log_particle_weight
        save_data["events"] = dict()
        for i in range((counter + 1) * self.block_size):
            save_data["events"][i] = dict()
            save_data["events"][i]['time'] = self.events[i]['time']
            save_data["events"][i]['user'] = self.events[i]['user']
            save_data["events"][i]['topic'] = self.events[i]['topic']
            save_data["events"][i]['parent'] = self.events[i]['parent']
            save_data["events"][i]['document_number'] = self.events[i]['document_number']

        save_data["topics_u"] = self.topics_u
        save_data["log_m_ku"] = self.log_m_uk
        save_data["log_m_u0"] = self.log_m_u0

        save_data["omega_u"] = self.omega_u
        save_data["old_events_num_uk"] = self.old_events_num_uk

        f = open('../results/' + self.name + '/DirichletHawkes/particle' + str(idx) + '_' + str(self.number_of_docs) + '.pk', 'wb')
        pickle.dump(save_data, f)
        f.close()
