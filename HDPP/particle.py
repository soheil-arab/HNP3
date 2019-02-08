from __future__ import division, print_function
import numpy as np
from numpy import random as random
from collections import deque
from common.topic_model import TopicModel
from common import helper
from math import exp, log
import pickle
import math
import sys


class Particle(object):
    def __init__(self, name, user_num, adjacency_matrix,
                 hyper_params, initial_log_weight,
                 number_of_samples=128, block_size=200):
        self.name = name
        self.user_num = user_num
        self.adjacency_matrix = adjacency_matrix

        self.a = hyper_params['a']
        self.b = hyper_params['b']
        self.c = hyper_params['c']
        self.d = hyper_params['d']
        self.gamma = hyper_params['gamma']
        self.zeta = hyper_params['zeta']

        self.default_eta = hyper_params['default_eta']
        self.kessi = hyper_params['local_topic_decreasing_coefficient']
        self.g_kessi = hyper_params['global_topic_decreasing_coefficient']

        self.kernel_mean = hyper_params['kernel_mean']
        self.kernel_var = hyper_params['kernel_var']
        kernel_sigma = np.sqrt(self.kernel_var)
        self.threshold = -np.log(hyper_params['epsilon']) / np.max([self.kernel_mean - 2 * kernel_sigma, 1e-2])

        self.block_size = block_size
        # Number of samples to estimate the posterior of the topics kernel parameter
        self.number_of_samples = number_of_samples

        # Log of the particle weight
        self.log_particle_weight = initial_log_weight

        self.number_of_topics = 0
        self.current_event_num = 0

        self.counts = np.zeros(shape=(user_num, user_num), dtype=int)

        self.topics = list()

        self.events = dict()
        self.active_set = deque()
        self.first_active_event = 0

        self.log_m_ku = dict()
        self.log_m_k = dict()
        self.log_m_u0 = -1 * np.inf * np.ones(self.user_num)
        self.log_m_0 = -1 * np.inf
        self.last_update_time = 0

        self.omega_u = dict()
        self.old_events_omega_u = list()
        for u in range(self.user_num):
            self.old_events_omega_u.append(dict())

        self.document_log_likelihood_per_topic = list()

        self.sum_of_time_diff = dict()
        self.active_events = dict()

    # Update the particle after adding a new event by adding a new (z,s,l) to the particle
    def update(self, event):
        time = event['time']
        user = event['user']
        document = event['document']
        # Remove old events from the active set
        self.deque_old_events(time)

        log_prob, document_log_likelihood = self.compute_likelihood(time, user, document)

        self.current_event_num += 1
        number_of_active_docs = self.current_event_num - self.first_active_event

        self.log_particle_weight += document_log_likelihood

        # **************** Sampling the s and z and l ***********************
        s, z, l = self.sample(log_prob)

        # **************** Updating events list ***********************
        new_event = dict()
        new_event['time'] = time
        new_event['user'] = user
        new_event['document'] = document
        new_event['topic'] = z
        new_event['parent'] = s
        new_event['document_number'] = self.current_event_num - 1
        new_event['locality'] = l
        if s != self.current_event_num - 1:
            new_event['parent_user'] = self.events[s]['user']
        else:
            new_event['parent_user'] = user
        self.active_set.append(new_event)
        self.events[self.current_event_num - 1] = new_event

        # ********************* Updating the counts *********************
        self.counts[self.events[s]['user']][event['user']] += 1

        # ******************** Updating topics **************************
        if z == self.number_of_topics:
            self.number_of_topics += 1
            new_topic = TopicModel(default_eta=self.default_eta,
                                   default_beta=self.kernel_mean)
            self.topics.append(new_topic)
            self.log_m_ku[z] = -1 * np.inf * np.ones(self.user_num)
            self.log_m_k[z] = -1 * np.inf
            self.sum_of_time_diff[z] = 0

        self.update_weight_of_topic(z, user, l)
        self.topics[z].update(document)

        # Updating the sum_of_time_diffs
        if s != self.current_event_num - 1:
            self.sum_of_time_diff[z] += time - self.events[s]['time']

        # Updating the kernel
        self.active_events[z] = 1
        if self.current_event_num % 100 == 0:
            for z in self.active_events:
                self.update_beta(z, time)
            self.active_events = dict()
        return s, z

    def sample(self, log_prob):
        mx = np.max(log_prob)
        prob = np.exp(log_prob - mx)
        prob = prob / np.sum(prob)
        index = random.choice(len(prob), size=1, p=prob)[0]

        number_of_active_docs = self.current_event_num - self.first_active_event
        l = 0
        if index < number_of_active_docs - 1:
            s = index + self.first_active_event
            z = self.events[s]['topic']
        else:
            s = self.current_event_num - 1
            if index < number_of_active_docs + self.number_of_topics - 1:
                z = index - (number_of_active_docs - 1)
                l = 0  # The topic have been selected from the local restaurant
            else:
                z = index - (number_of_active_docs - 1) - self.number_of_topics
                l = 1
        return s, z, l

    def compute_likelihood(self, time, user, document):
        # update the weight of tables of restaurants
        self.compute_topic_weights(time)

        number_of_active_docs = self.current_event_num - self.first_active_event + 1
        last_t = 0
        if self.current_event_num > 0:
            last_t = self.events[self.current_event_num - 1]["time"]
        # ****************Computing the omega_u and the document log likelihood***********
        self.omega_u = self.compute_omega_u(last_t)

        # a new topic model with no documents
        tmp_topic = TopicModel(default_eta=self.default_eta, default_beta=self.kernel_mean)

        self.document_log_likelihood_per_topic = list()
        for idx, topic in enumerate(self.topics):
            self.document_log_likelihood_per_topic.append(topic.log_likelihood(document))
            # self.document_log_likelihood_per_topic.append(
            #     helper.document_log_likelihood(topic.eta, topic.vocab_size, document['length'], document['words'],
            #                                    topic.sum_eta, topic.default_eta))
        self.document_log_likelihood_per_topic.append(tmp_topic.log_likelihood(document))
        # self.document_log_likelihood_per_topic.append(
        #     helper.document_log_likelihood(tmp_topic.eta, tmp_topic.vocab_size, document['length'], document['words'],
        #                                    tmp_topic.sum_eta, tmp_topic.default_eta))
        # **************** Computing the topic and ancestor parts of posterior for different s and z seperately *
        s_log_prob = np.zeros(number_of_active_docs)

        for s in range(self.first_active_event, self.current_event_num + 1):
            if s == self.current_event_num:
                if self.current_event_num > 0:
                    cnt = self.counts[user][user]
                    s_log_prob[s - self.first_active_event] = log(self.c + cnt) - (log(self.d + last_t))
                    # print('self_s_log_prob = {0},self_s_log_prob[-1] = {1}'.format(log(self.c + cnt) - (log(self.d + last_t)),s_log_prob[-1]))
                else:
                    s_log_prob[s - self.first_active_event] = 0
            else:
                user_s = self.events[s]["user"]

                if self.adjacency_matrix[user_s][user] > 0:
                    z_s = self.events[s]["topic"]
                    t_s = self.events[s]["time"]
                    s_log_prob[s - self.first_active_event] = -1 * self.topics[z_s].beta * (time - t_s) + \
                                                              log(self.a + self.counts[user_s][user]) - \
                                                              log(self.b + self.omega_u.get(user_s, 0))
                    # print('z_s={0},beta={1},time-t_s={2},self.counts[user_s][user]={3},self.omega_u.get(user_s)={4},log_part={5}'.format(z_s,self.topics[z_s].beta,time - t_s,self.counts[user_s][user],self.omega_u.get(user_s, 0),log(self.a + self.counts[user_s][user]) - \
                    #                                           log(self.b + self.omega_u.get(user_s, 0))))
                    # if s - self.first_active_event==0:
                    #     print("beta = {0}, time-t_s={1},self.a={2},self.b={3},"
                    #           "self.counts[user_s][user]={4},self.omega_u.get(user_s, 0)={5},part1={6}, part2={7}".
                    #           format(self.topics[z_s].beta,time - t_s,self.a,self.b,self.counts[user_s][user],self.omega_u.get(user_s, 0),-1 * self.topics[z_s].beta * (time - t_s),log(self.a + self.counts[user_s][user]) - \
                    #                                           log(self.b + self.omega_u.get(user_s, 0))))
                else:
                    s_log_prob[s - self.first_active_event] = -1 * np.inf

        z_log_prob = np.zeros(shape=(self.number_of_topics + 1, 2))
        for z in range(self.number_of_topics + 1):
            if z < self.number_of_topics:
                # z_log_prob[z][0] += self.document_log_likelihood_per_topic[z]
                z_log_prob[z][0] = (self.log_m_ku[z][user]) - log(exp(self.log_m_u0[user]) + self.gamma)
            else:
                z_log_prob[z][0] = -1 * np.inf
            if z < self.number_of_topics:
                # z_log_prob[z][1] += self.document_log_likelihood_per_topic[z]
                z_log_prob[z][1] = np.log(self.gamma) - log(
                    exp(self.log_m_u0[user]) + self.gamma) \
                                   + (self.log_m_k[z]) - np.log(exp(self.log_m_0) + self.zeta)
            else:
                # z_log_prob[z][1] += self.document_log_likelihood_per_topic[z]
                z_log_prob[z][1] = log(self.zeta) + log(self.gamma) - log(
                    self.zeta + exp(self.log_m_0)) - log(exp(self.log_m_u0[user]) + self.gamma)
                # print(
                # 'log(self.zeta)={0},log(self.gamma)={1},log(exp(self.log_m_u0[user]) + self.gamma)={2}, log(self.zeta + exp(self.log_m_0))={3},z_log_prob[z][1]={4}'.format(
                #     log(self.zeta), log(self.gamma), log(exp(self.log_m_u0[user]) + self.gamma),
                #     log(self.zeta + exp(self.log_m_0)),z_log_prob[z][1]))

        # **************** computing the complete posterior for any z and s and r ***********************

        log_prob = np.zeros(number_of_active_docs - 1 + self.number_of_topics + (self.number_of_topics + 1))
        for i in range(number_of_active_docs - 1):
            z_s = self.events[i + self.first_active_event]["topic"]
            log_prob[i] = s_log_prob[i] + self.document_log_likelihood_per_topic[z_s]
            # if i==0:
            # print('mutu_excited:z_s={0},s_log_prob[{1}]={2},doc_log_likelihood={3}'.format(z_s, i, s_log_prob[i],
            #                                                                                self.document_log_likelihood_per_topic[
            #                                                                                    z_s]))

        for k in range(self.number_of_topics):
            log_prob[k + number_of_active_docs - 1] = s_log_prob[-1] + z_log_prob[k][0] + \
                                                      self.document_log_likelihood_per_topic[k]
            log_prob[k + self.number_of_topics + number_of_active_docs - 1] = s_log_prob[-1] + z_log_prob[k][1] + \
                                                                              self.document_log_likelihood_per_topic[k]
            # print('self_excited:z={0},l=0,s_log_prob={1},doc_log_likelihood={2},z_log_prob={3}'.format(k,
            #                                                                                            s_log_prob[-1],
            #                                                                                            self.document_log_likelihood_per_topic[
            #                                                                                                k],
            #                                                                                            z_log_prob[k][
            #                                                                                                0]))
            # print('self_excited:z={0},,l=1,s_log_prob={1},doc_log_likelihood={2},z_log_prob={3}'.format(k,
            #                                                                                            s_log_prob[-1],
            #                                                                                            self.document_log_likelihood_per_topic[
            #                                                                                                k],
            #                                                                                            z_log_prob[k][
            #                                                                                                1]))
        log_prob[-1] = s_log_prob[-1] + z_log_prob[-1][1] + self.document_log_likelihood_per_topic[
            self.number_of_topics]
        # print(
        #     'self_excited:z={0},l=1,s_log_prob={1},doc_log_likelihood={2},z_log_prob={3}'.format(self.number_of_topics,
        #                                                                                          s_log_prob[-1],
        #                                                                                          self.document_log_likelihood_per_topic[
        #                                                                                              self.number_of_topics],
        #                                                                                          z_log_prob[-1][
        #                                                                                              1]))
        # **************** computing likelihood ***********************
        mx = np.max(log_prob)
        prob = np.exp(log_prob - mx)
        # print('****************Prob**************')
        # for i in range(len(prob)):
        #     print(prob[i],end=',')
        # print()
        document_log_likelihood = np.log(sum(prob)) + mx
        return log_prob, document_log_likelihood

    def deque_old_events(self, time):
        while len(self.active_set) > 1 and time - self.active_set[0]['time'] > self.threshold:
            user = self.active_set[0]['user']
            topic = self.active_set[0]['topic']
            self.old_events_omega_u[user][topic] = self.old_events_omega_u[user].get(topic, 0) + 1
            self.active_set.popleft()
        if len(self.active_set) != 0:
            self.first_active_event = self.active_set[0]['document_number']
        else:
            self.first_active_event = 0

    # Updates the weight of topics for each user and in the up-level restaurant
    # after passing time-last_update_time time and adding no new event in this interval
    def compute_topic_weights(self, time):
        for topic in self.log_m_ku.keys():
            self.log_m_ku[topic] -= self.kessi * (time - self.last_update_time)
            self.log_m_k[topic] -= self.g_kessi * (time - self.last_update_time)
        self.log_m_u0 -= self.kessi * (time - self.last_update_time)
        self.log_m_0 -= self.g_kessi * (time - self.last_update_time)
        self.last_update_time = time

    # Update the weight of topic z after observing a new event in that topic
    def update_weight_of_topic(self, z, user, l):  # m_u_k(t)
        self.log_m_ku[z][user] = log(exp(self.log_m_ku[z][user]) + 1)
        self.log_m_u0[user] = log(exp(self.log_m_u0[user]) + 1)
        # r=1 indicates that the topic is selected from the up-level restaurant
        if l == 1:
            self.log_m_k[z] = log(exp(self.log_m_k[z]) + 1)
            self.log_m_0 = log(exp(self.log_m_0) + 1)

    # This function returns a dict that for each pair of user u returns the
    # sum_{e\inD_u}1/beta_{z_e}(1-exp(beta_{z_e}(time-t_e))) which is used in computing
    # the likelihood of time of events in the Hawkes process.
    def compute_omega_u(self, time):
        betas = np.zeros(len(self.topics))
        for idx, topic in enumerate(self.topics):
            betas[idx] = topic.beta

        active_events_omega = helper.compute_omega_u(list(self.active_set), time, betas)
        for user in range(self.user_num):
            for k in self.old_events_omega_u[user]:
                active_events_omega[user] = active_events_omega.get(user, 0) + self.old_events_omega_u[user][k] / \
                                                                               self.topics[k].beta
        # ####################################
        # total_events_omega = dict()
        # for idx in self.events:
        #     event = self.events[idx]
        #     user = event['user']
        #     k = event["topic"]
        #     beta_k = betas[k]
        #     t = event["time"]
        #     total_events_omega[user] = total_events_omega.get(user, 0) + 1 / beta_k * (
        #         1 - exp(-beta_k * (time - t)))
        # for user in range(self.user_num):
        #     print('u={0},total_events_omega={1},omega_using_old={2},diff={3}'.format(user,
        #                                                                              total_events_omega.get(user, 0),
        #                                                                              active_events_omega.get(user, 0),
        #                                                                              total_events_omega.get(user, 0)-active_events_omega.get(user, 0)))

        return active_events_omega

    def compute_omega_u_between_two_time(self, t_last, time):
        sum_omega_u = dict()
        for event in self.active_set:
            user = event['user']
            k = event["topic"]
            beta_k = self.topics[k].beta
            t_e = event["time"]
            sum_omega_u[user] = sum_omega_u.get(user, 0) + 1 / beta_k * (
                exp(-beta_k * (t_last - t_e)) - exp(-beta_k * (time - t_e)))
        return sum_omega_u

    def update_beta(self, k, time):
        scale = self.kernel_var / self.kernel_mean  # only to avoid computational errors
        shape = self.kernel_mean ** 2 / self.kernel_var
        beta_sample = random.gamma(shape=shape, scale=scale, size=self.number_of_samples)
        sample_mean = np.mean(beta_sample)
        sample_var = np.var(beta_sample)
        # print('sample_mean={0},sample_varince={1}'.format(sample_mean, sample_var))
        # beta_sample /= 100  # only to avoid computational errors
        betas = np.zeros(len(self.topics))
        for idx, topic in enumerate(self.topics):
            betas[idx] = topic.beta
        weight = np.zeros(self.number_of_samples)
        for i in range(self.number_of_samples):
            betas[k] = beta_sample[i]
            weight[i] = self.compute_beta_log_likelihood(k, betas, time)
            if math.isnan(weight[i]):
                print('Nan found in weights. beta is:', betas[k])
                sys.exit()
            if i%1000==0:
                print('{0}th sample processed.'.format(i))
        print()
        # temp = weight
        weight -= np.max(weight)
        weight = np.exp(weight)
        weight /= np.sum(weight)
        beta = beta_sample.dot(weight)
        # print('weighted_beta_sample_mean={0}'.format(beta))
        self.topics[k].update_kernel(beta)
        return beta

    def compute_beta_log_likelihood(self, k, betas, time):
        result = -1 * betas[k] * self.sum_of_time_diff[k]
        omega_u = self.compute_omega_u(time)
        for user in omega_u:
            if math.isnan(omega_u[user]):
                print('omega_u is nan. k ={0}, betas={1}, omega={2}'.format(k, betas, omega_u))
            for user_prime in range(self.user_num):
                if self.adjacency_matrix[user][user_prime] != 0:
                    result -= (self.a + self.counts[user][user_prime]) * log(self.b + omega_u[user])
        # print('beta_particle={0}'.format(betas))
        # result += helper.compute_beta_log_likelihood(list(self.active_set), time, betas, self.user_num, self.a, self.b,
        #                                              self.adjacency_matrix, self.counts)
        # print('after={0}'.format(result))
        return result

        # for u in range(self.user_num):
        #     if not u in omega_u:
        #         continue
        #     for v in range(self.user_num):
        #         if self.adjacency_matrix[u][v] != 0:
        #             result -= (self.a + self.counts[u][v]) * log(self.b + omega_u[u])
        # return result

    def estimate_mu(self):
        time = self.events[self.current_event_num - 1]['time']
        mean_counts = 0
        for u in range(self.user_num):
            mean_counts += self.counts[u][u]
        mean_counts /= self.user_num
        mu = np.zeros((self.user_num, 1))
        for u in range(self.user_num):
            mu[u] = (self.c + self.counts[u][u]) / (
                self.d + time)
        return mu

    def estimate_alpha(self):
        time = self.events[self.current_event_num - 1]['time']
        omega_u = self.compute_omega_u(time)
        alpha = np.zeros((self.user_num, self.user_num))
        for u in range(self.user_num):
            for v in range(self.user_num):
                if self.adjacency_matrix[u][v] != 0:
                    alpha[u][v] = (self.a + self.counts[u][v]) / (
                        self.b + omega_u.get(u, 0))
        return alpha

    def compute_time_likelihood(self, events):
        mu = self.estimate_mu()
        alpha = self.estimate_alpha()
        N = len(events)
        log_likelihood = np.zeros((N, 1))
        for event in events:
            event['time'] -= self.events[0]['time']
        last_t = self.events[self.current_event_num - 1]['time']
        # print('self.number_of_docs - 1]={0},last_t={1}'.format(self.number_of_docs - 1,self.events[self.number_of_docs - 1]['time']))
        for idx, event in enumerate(events):
            u = event['user']
            t = event['time']
            omega_u = self.compute_omega_u_between_two_time(last_t, t)
            omega = 0
            for v in range(self.user_num):
                omega += np.sum(alpha[v, :]) * omega_u.get(v, 0)
                # print('HDPP-->v:{0},np.sum(alpha[v, :]):{1},omega_u[v]={2},omega={3}'.format(v, np.sum(alpha[v, :]),
                #                                                                    omega_u[v],omega))
            beta = self.kernel_mean
            print('HDPP-->omega1:{0}'.format(omega))
            for i in range(idx):
                e = events[i]
                v = e['user']
                t_e = e['time']
                # if beta * (t - t_e)<100:
                omega += np.sum(alpha[v, :]) * (1 / beta) * (
                    np.exp(-1 * beta * (last_t - t_e)) - np.exp(-1 * beta * (t - t_e)))
                # else:
                #     omega += np.sum(alpha[v, :]) * 1 / beta
            print('HDPP-->omega2:{0}'.format(omega))
            lambda_u = 0
            lambda_u += mu[u]
            print('mu[u]={0}'.format(mu[u]))
            print('lambda1={0}'.format(lambda_u))
            for s in range(self.first_active_event, self.current_event_num):
                source = self.events[s]
                u_s = source['user']
                t_s = source['time']
                k = source['topic']
                if self.adjacency_matrix[u_s][u]:
                    #     if s==23195:
                    #         print("u_s:{0},u={1},k={2}".format(u_s, u, k))
                    #         old_counts = 0
                    #         for k in self.old_events_omega_u[u_s]:
                    #             old_counts+=self.old_events_omega_u[u_s][k]
                    #         print('u_s={0},counts[u_s][u]={1},omega[u_s]={2},old_events_num={3}'.format(u_s,
                    #                                                                                     self.counts[:,u],
                    #                                                                                     self.compute_omega_u(
                    #                                                                                         self.events[
                    #                                                                                             self.number_of_docs - 1][
                    #                                                                                             'time'])[
                    #                                                                                         u_s],old_counts))
                    #     print("Lambda-->event_num={0},u_s={5},beta:{1}, (t-t_s)={2}, alpha={3}, val={4}".format(source['document_number'],
                    #                                                                          self.topics[k].beta, t - t_s,alpha[u_s][u],
                    #                                                                          alpha[u_s][u] * exp(
                    #                                                                              -1 * self.topics[
                    #                                                                                  k].beta * (
                    #                                                                                  t - t_s)),u_s))
                    lambda_u += alpha[u_s][u] * exp(-1 * self.topics[k].beta * (t - t_s))
                    # print('s={0},u_s={1},log(lambda)={2}'.format(s,u_s, np.log(lambda_u)))
            print('lambda2={0}'.format(lambda_u))
            for s in range(idx):
                source = events[s]
                u_s = source['user']
                t_s = source['time']
                if self.adjacency_matrix[u_s][u] != 0:
                    lambda_u += alpha[u_s][u] * exp(-1 * beta * (t - t_s))
            log_likelihood[idx] = -1 * ((t - last_t) * np.sum(mu) + omega) + np.log(lambda_u)
            print('lambda3={0}'.format(lambda_u))
            print('HDPP:(t - last_t):{0}, np.sum(mu):{1},omega:{2},log(lambda):{3}'.format((t - last_t), np.sum(mu),
                                                                                           omega, np.log(lambda_u)))
            last_t = t
            print('HDPP:event {0} processed. log_likelihood={1}'.format(idx, log_likelihood[idx]))
        return log_likelihood

    def add_analyzed_events(self, analyzed_events, topics,
                            log_m_ku, log_m_k, log_m_u0,
                            log_m_0, omega_u, old_events_omega_u, log_particle_weight):
        self.topics = topics
        self.number_of_topics = len(topics)
        self.log_m_ku = log_m_ku
        self.log_m_k = log_m_k
        self.log_m_u0 = log_m_u0
        self.log_m_0 = log_m_0

        time = 0
        keys = list(analyzed_events.keys())
        keys.sort()
        for key in keys:
            self.current_event_num += 1
            event = analyzed_events[key]
            # self.number_of_topics = np.max((self.number_of_topics, event['topic']))
            user = event['user']
            parent_user = event['parent_user']
            self.counts[parent_user][user] += 1
            self.events[self.current_event_num - 1] = event
            self.active_set.append(event)
            time = event["time"]
            self.last_update_time = time
            k = event['topic']
            if not k in self.sum_of_time_diff:
                self.sum_of_time_diff[k] = 0
            if event['parent'] != event['document_number']:
                t_s = self.events[event['parent']]['time']
                k = event['topic']
                self.sum_of_time_diff[k] = self.sum_of_time_diff.get(k, 0) + (time - t_s)
        # print('number_of_topics = {0}, sum_of_time_diff={1}'.format(self.number_of_topics, self.sum_of_time_diff))
        # print('self.number_of_docs={0}, len(self.events)={1}'.format(self.number_of_docs,len(self.events)))

        self.deque_old_events(time)
        self.first_active_event = self.active_set[0]['document_number']

        self.omega_u = omega_u
        self.old_events_omega_u = old_events_omega_u
        self.log_particle_weight = log_particle_weight

    def add_analyzed_events_old(self, analyzed_events, topics,
                                log_m_ku, log_m_k, log_m_u0,
                                log_m_0, omega_u, old_events_omega_u, log_particle_weight):
        self.topics = topics
        self.number_of_topics = len(topics)
        self.log_m_ku = log_m_ku
        self.log_m_k = log_m_k
        self.log_m_u0 = log_m_u0
        self.log_m_0 = log_m_0

        time = 0
        keys = list(analyzed_events.keys())
        keys.sort()
        for key in keys:
            self.current_event_num += 1
            event = analyzed_events[key]
            # self.number_of_topics = np.max((self.number_of_topics, event['topic']))
            user = event['user']
            parent_user = event['parent_user']
            self.counts[parent_user][user] += 1
            self.events[self.current_event_num - 1] = event
            self.active_set.append(event)
            time = event["time"]
            self.last_update_time = time
            k = event['topic']
            if not k in self.sum_of_time_diff:
                self.sum_of_time_diff[k] = 0
            if event['parent'] != event['document_number']:
                t_s = self.events[event['parent']]['time']
                k = event['topic']
                self.sum_of_time_diff[k] = self.sum_of_time_diff.get(k, 0) + (time - t_s)
        # print('number_of_topics = {0}, sum_of_time_diff={1}'.format(self.number_of_topics, self.sum_of_time_diff))
        # print('self.number_of_docs={0}, len(self.events)={1}'.format(self.number_of_docs,len(self.events)))

        self.deque_old_events(time)
        self.first_active_event = self.active_set[0]['document_number']

        self.omega_u = omega_u
        self.old_events_omega_u = old_events_omega_u
        self.log_particle_weight = log_particle_weight

    def save(self, idx, prefix=''):
        save_data = dict()
        save_data["log_particle_weight"] = self.log_particle_weight
        save_data["events"] = dict()
        # for i in range(counter * self.block_size, (counter + 1) * self.block_size):
        for i in range(self.current_event_num):
            save_data["events"][i] = dict()
            save_data["events"][i]['time'] = self.events[i]['time']
            save_data["events"][i]['user'] = self.events[i]['user']
            save_data["events"][i]['topic'] = self.events[i]['topic']
            save_data["events"][i]['parent'] = self.events[i]['parent']
            save_data["events"][i]['document_number'] = self.events[i]['document_number']
            save_data["events"][i]['parent_user'] = self.events[i]['parent_user']
            save_data["events"][i]['locality'] = self.events[i]['locality']

        save_data["topics"] = self.topics
        save_data["log_m_ku"] = self.log_m_ku
        save_data["log_m_k"] = self.log_m_k
        save_data["log_m_u0"] = self.log_m_u0
        save_data["log_m_0"] = self.log_m_0

        save_data["omega_u"] = self.omega_u
        save_data["old_events_omega_u"] = self.old_events_omega_u

        # f = open('../results/' + self.name + '/HDPP/particle' + str(idx) + '_' + str(counter) + '.pk', 'wb')
        f = open('../results/' + self.name + '/HDPP/' + prefix + 'particle' + str(idx) + '_' + str(
            self.current_event_num) + '.pk', 'wb')
        pickle.dump(save_data, f)
        f.close()

        # def get_a_copy(self):
        #     pickled = pickle.dumps(self)
        #     print(type(pickled))
        #     result = pickle.loads(pickled)
        #     return result
