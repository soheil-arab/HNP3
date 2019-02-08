import numpy as np
from numpy import random as random
from collections import deque
import pickle
from math import exp, log


class HawkesParticle:
    def __init__(self, name, user_num, adjacency_matrix,
                 hyper_params, initial_log_weight,
                 block_size):
        self.name = name
        self.adjacency_matrix = adjacency_matrix
        self.a = hyper_params['a']
        self.b = hyper_params['b']
        self.c = hyper_params['c']
        self.d = hyper_params['d']

        self.block_size = block_size
        self.kernel_mean = hyper_params['kernel_mean']
        self.kernel_var = hyper_params['kernel_var']
        kernel_sigma = np.sqrt(self.kernel_var)
        self.threshold = -np.log(hyper_params['epsilon']) / np.max([self.kernel_mean - 2 * kernel_sigma, 1e-2])
        self.beta = self.kernel_mean

        # Log of the particle weight
        self.log_particle_weight = initial_log_weight

        self.user_num = user_num
        self.number_of_docs = 0

        self.counts = np.zeros(shape=(user_num, user_num), dtype=int)

        self.events = dict()
        self.active_set = deque()
        self.first_active_event = 0

        self.log_m_ku = dict()
        self.log_m_k = dict()
        self.log_m_u0 = -1 * np.inf * np.ones(self.user_num)
        self.log_m_0 = -1 * np.inf
        self.last_update_time = 0

        self.omega_u = dict()
        self.old_events_omega_u = dict()

        self.sum_of_time_diff = 0

    # Update the particle after adding a new event by adding a new (z,s,l) to the particle
    def update(self, time, user, document):
        self.number_of_docs += 1

        # Remove old events from the active set
        self.deque_old_events(time)

        number_of_active_docs = self.number_of_docs - self.first_active_event

        # ****************Computing the omega_u and the document log likelihood***********
        old_omega_u = self.omega_u
        last_t = 0
        if number_of_active_docs > 1:
            last_t = self.events[self.number_of_docs - 2]["time"]

        self.omega_u = self.compute_omega_u(last_t)
        # **************** Computing the topic and ancestor parts of posterior for different s and z seperately *
        log_prob = np.zeros(number_of_active_docs)

        for s in range(self.first_active_event, self.number_of_docs):
            if s == self.number_of_docs - 1:
                log_prob[s - self.first_active_event] = log(self.c + self.counts[user][user]) - (log(self.d + last_t))
            else:
                user_s = self.events[s]["user"]
                if self.adjacency_matrix[user_s][user] > 0:
                    t_s = self.events[s]["time"]
                    log_prob[s - self.first_active_event] = -1 * self.beta * (time - t_s) + \
                                                            log(self.a + self.counts[user_s][user]) - \
                                                            log(self.b + self.omega_u.get(user_s, 0))
                else:
                    log_prob[s - self.first_active_event] = -1 * np.inf

        # **************** updating the particle weight ***********************
        mx = np.max(log_prob)
        prob = np.exp(log_prob - mx)
        event_log_likelihood = np.log(sum(prob)) + mx
        # print("New log_particle_weight is:{0}".format(self.log_particle_weight + document_log_likelihood))
        self.log_particle_weight += event_log_likelihood

        # **************** Sampling the s and z and r ***********************
        prob = prob / np.sum(prob)
        index = random.choice(len(prob), size=1, p=prob)[0]
        l = 0
        if index < number_of_active_docs - 1:
            s = index + self.first_active_event
        else:
            s = self.number_of_docs - 1

        new_event = dict()
        new_event['time'] = time
        new_event['user'] = user
        new_event['parent'] = s
        new_event['document_number'] = self.number_of_docs - 1
        if s != self.number_of_docs - 1:
            new_event['parent_user'] = self.events[s]['user']
        else:
            new_event['parent_user'] = user
        # print(new_event)
        # print(prob)
        # temp_deque = self.events_of_users.get(user, deque())
        self.active_set.append(new_event)
        # self.events_of_users[user] = temp_deque
        self.events[self.number_of_docs - 1] = new_event

        # Updating the counts
        self.counts[self.events[s]['user']][user] += 1

        # Updating the sum_of_time_diffs
        if s != self.number_of_docs - 1:
            self.sum_of_time_diff += time - self.events[s]['time']

        return s

    def deque_old_events(self, time):
        while len(self.active_set) > 1 and time - self.active_set[0]['time'] > self.threshold:
            user = self.active_set[0]['user']
            self.old_events_omega_u[user] = self.old_events_omega_u.get(user, 0) + 1 / self.beta
            self.active_set.popleft()
        if len(self.active_set) != 0:
            self.first_active_event = self.active_set[0]['document_number']
        else:
            self.first_active_event = 0

    # This function returns a dict that for each pair of user u returns the
    # sum_{e\inD_u}1/beta_{z_e}(1-exp(beta_{z_e}(time-t_e))) which is used in computing
    # the likelihood of time of events in the Hawkes process.
    def compute_omega_u(self, time):
        sum_omega_u = dict()
        for event in self.active_set:
            user = event['user']
            t = event["time"]
            # if self.beta * (time - t)<10:
            #     print('time:{0},t:{1}'.format(time, t))
            sum_omega_u[user] = sum_omega_u.get(user, 0) + 1 / self.beta * (
                1 - exp(-self.beta * (time - t)))
        for user in self.old_events_omega_u:
            sum_omega_u[user] = sum_omega_u.get(user, 0) + self.old_events_omega_u[user]
            # if sum_omega_u[(user_s, user)]<0:
            # print('K_s is %f and beta_k is %f'%(k_s,beta_k))
        return sum_omega_u

    def compute_omega_u_between_two_time(self, t_last, time):
        sum_omega_u = dict()
        for event in self.active_set:
            user = event['user']
            t_e = event["time"]
            # print('user={0},t_e={1},t_last={2},time={3}'.format(user,t_e,t_last,time))
            # print('beta={0},t_last_t_e={1},t-t_e={2}'.format(self.beta,t_last-t_e, time-t_e))
            sum_omega_u[user] = sum_omega_u.get(user, 0) + 1 / self.beta * (
                exp(-self.beta * (t_last - t_e)) - exp(-self.beta * (time - t_e)))
        return sum_omega_u

    # def compute_time_likelihood(self, time, user):
    #     # update the weight of tables of restaurants
    #     number_of_active_docs = self.number_of_docs - self.first_active_event + 1
    #     last_t = 0
    #     if self.number_of_docs > 0:
    #         last_t = self.events[self.number_of_docs - 1]["time"]
    #     # ****************Computing the omega_u and the document log likelihood***********
    #     self.omega_u = self.compute_omega_u(last_t)
    #     # **************** Computing the topic and ancestor parts of posterior for different s and z seperately *
    #     s_log_prob = np.zeros(number_of_active_docs)
    #
    #     for s in range(self.first_active_event, self.number_of_docs + 1):
    #         if s == self.number_of_docs:
    #             if self.number_of_docs > 0:
    #                 cnt = self.counts[user][user]
    #                 s_log_prob[s - self.first_active_event] = log(self.c + cnt) - (log(self.d + last_t))
    #             else:
    #                 s_log_prob[s - self.first_active_event] = 0
    #         else:
    #             user_s = self.events[s]["user"]
    #
    #             if self.adjacency_matrix[user_s][user] > 0:
    #                 t_s = self.events[s]["time"]
    #                 s_log_prob[s - self.first_active_event] = -1 * self.beta * (time - t_s) + \
    #                                                           log(self.a + self.counts[user_s][user]) - \
    #                                                           log(self.b + self.omega_u.get(user_s, 0))
    #                 # if s - self.first_active_event == 0:
    #                 #     print("beta = {0}, time-t_s={1},self.a={2},self.b={3},"
    #                 #           "self.counts[user_s][user]={4},self.omega_u.get(user_s, 0)={5},part1={6}, part2={7}".
    #                 #           format(self.beta, time - t_s, self.a, self.b, self.counts[user_s][user],
    #                 #                  self.omega_u.get(user_s, 0), -1 * self.beta * (time - t_s),
    #                 #                  log(self.a + self.counts[user_s][user]) - \
    #                 #                  log(self.b + self.omega_u.get(user_s, 0))))
    #             else:
    #                 s_log_prob[s - self.first_active_event] = -1 * np.inf
    #
    #     # **************** computing likelihood ***********************
    #     mx = np.max(s_log_prob)
    #     prob = np.exp(s_log_prob - mx)
    #     event_log_likelihood = np.log(sum(prob)) + mx
    #     return event_log_likelihood

    def estimate_mu(self):
        time = self.events[self.number_of_docs - 1]['time']
        mean_counts = 0
        for u in range(self.user_num):
            mean_counts += self.counts[u][u]
        mean_counts /= self.user_num
        # print("hawkes_mu_estimation-->time:{0},mean(counts[u][u])={1}".format(time, mean_counts))
        mu = np.zeros((self.user_num, 1))
        for u in range(self.user_num):
            mu[u] = (self.c + self.counts[u][u]) / (
                self.d + time)
        # print('Hawkes:sum(mu):{0}'.format(np.sum(mu)))
        return mu

    def estimate_alpha(self):
        time = self.events[self.number_of_docs - 1]['time']
        omega_u = self.compute_omega_u(time)
        alpha = np.zeros((self.user_num, self.user_num))
        for u in range(self.user_num):
            for v in range(self.user_num):
                alpha[u][v] = (self.a + self.counts[u][v]) / (
                    self.b + omega_u.get(u, 0))
        return alpha

    def compute_time_likelihood(self, events):
        mu = self.estimate_mu()
        alpha = self.estimate_alpha()
        # print('Hawkes-->mean(alpha):{0}'.format(np.mean(alpha)))
        # T = events[-1]['time']
        N = len(events)
        log_likelihood = np.zeros((N, 1))
        last_t = self.events[self.number_of_docs - 1]['time']
        for idx, event in enumerate(events):
            # print(idx)
            u = event['user']
            t = event['time']
            # print('idx={0},t={1},u={2}',idx,t,u)
            omega_u = self.compute_omega_u_between_two_time(last_t, t)
            omega = 0
            for v in range(self.user_num):
                omega += np.sum(alpha[v, :]) * omega_u.get(v, 0)
                # print('Hawkes-->v:{0},np.sum(alpha[v, :]):{1},omega_u[v]={2}'.format(omega, np.sum(alpha[v, :]),
                #                                                                      omega_u.get(v,0)))
            print('Hawkes-->omega1:{0}'.format(omega))
            beta = self.kernel_mean
            for i in range(idx):
                e = events[i]
                v = e['user']
                t_e = e['time']
                omega += np.sum(alpha[v, :]) * (1 / beta) * (
                    np.exp(-1 * beta * (last_t - t_e)) - np.exp(-1 * beta * (t - t_e)))
            print('Hawkes-->omega2:{0}'.format(omega))
            lmbd = 0
            lmbd += mu[u]
            # print('mu[u]={0}'.format(mu[u]))
            # print('lambda1={0}'.format(lmbd))
            for s in range(self.first_active_event, self.number_of_docs):
                source = self.events[s]
                u_s = source['user']
                t_s = source['time']
                if self.adjacency_matrix[u_s][u] != 0:
                    # if s==23195:
                    #     print("u_s:{0},u={1}".format(u_s, u))
                    #     print('u_s={0},counts[u_s]={1},omega[u_s]={2},old_events_num={3}'.format(u_s, self.counts[:,u],
                    #                                                              self.compute_omega_u(self.events[
                    #                                                                                       self.number_of_docs - 1][
                    #                                                                                       'time'])[
                    #                                                                  u_s],self.old_events_omega_u[u_s]*self.beta))
                    # print("Lambda-->event_num={0},u_s={4},beta:{1}, (t-t_s)={2}, alpha={3}, val={4}".format(
                    #     source['document_number'],
                    #     self.beta, t - t_s, alpha[u_s][u],
                    #     alpha[u_s][u] * exp(
                    #         -1 * self.beta * (
                    #             t - t_s)),u_s))
                    val = alpha[u_s][u] * exp(-1 * self.beta * (t - t_s))
                    lmbd += val
                    # if np.log(val)>-10:
                    #     print('s={0},u_s={1},alpha={2},log(val)={3}'.format(s,u_s,alpha[u_s][u],np.log(val)))
            # print('lambda2={0}'.format(lmbd))
            for s in range(idx):
                source = events[s]
                u_s = source['user']
                t_s = source['time']
                if self.adjacency_matrix[u_s][u] != 0:
                    lmbd += alpha[u_s][u] * exp(-1 * beta * (t - t_s))
            log_likelihood[idx] = -1 * ((t - last_t) * np.sum(mu) + omega) + np.log(lmbd)
            # print('lambda3={0}'.format(lmbd))
            print('Hawkes:(t - last_t):{0}, np.sum(mu):{1},omega:{2},log(lambda):{3}'.format((t - last_t), np.sum(mu),
                                                                                             omega, np.log(lmbd)))
            last_t = t
            # print('event {0} processed. log_likelihood={1}'.format(idx, log_likelihood[idx]))
        return log_likelihood

    def add_analyzed_events(self, analyzed_events, omega_u, old_events_mega_u, log_particle_weight):

        time = 0
        keys = list(analyzed_events.keys())
        keys.sort()
        for key in keys:
            event = analyzed_events[key]
            self.number_of_docs += 1
            user = event['user']
            parent_user = event['parent_user']
            self.counts[parent_user][user] += 1
            self.events[self.number_of_docs - 1] = event
            self.active_set.append(event)
            time = np.max((analyzed_events[key]["time"], time))

        self.deque_old_events(time)
        self.first_active_event = self.active_set[0]['document_number']

        self.omega_u = omega_u
        self.old_events_omega_u = old_events_mega_u
        self.log_particle_weight = log_particle_weight

    def save(self, idx, counter):
        save_data = dict()
        save_data["log_particle_weight"] = self.log_particle_weight
        save_data["events"] = dict()
        # for i in range(counter * self.block_size, (counter + 1) * self.block_size):
        for i in range((counter + 1) * self.block_size):
            save_data["events"][i] = dict()
            save_data["events"][i]['time'] = self.events[i]['time']
            save_data["events"][i]['user'] = self.events[i]['user']
            save_data["events"][i]['parent'] = self.events[i]['parent']
            save_data["events"][i]['document_number'] = self.events[i]['document_number']
            save_data["events"][i]['parent_user'] = self.events[i]['parent_user']

        save_data["omega_u"] = self.omega_u
        save_data["old_events_omega_u"] = self.old_events_omega_u

        f = open('../results/' + self.name + '/hawkes/particle' + str(idx) + '_' + str(self.number_of_docs) + '.pk',
                 'wb')
        pickle.dump(save_data, f)
        f.close()

        # def get_a_copy(self):
        #     result = pickle.loads(pickle.dumps(self, -1))
        #     return result
