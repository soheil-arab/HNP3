# from __future__ import division
import numpy as np
import math as mth
import matplotlib as mpl
import sys

# from HDPP.topic_model import TopicModel

# mpl.use("qt4agg")
import matplotlib.pyplot as plt

num_of_events = 10000
number_of_user = 6
number_of_adj = 3
vocab_size = 20
document_length = 100
max_topic_num = 1000

a = 2
b = 20
c = 1
d = 20
epsilon = 16e-8
# kernel_mean = 0.01
# kernel_var = 1e-5
gamma = 10
zeta = 10
topic_decreasing_coefficient = 2.5e-4

name = "Test9"  # input("Enter Test Name: ")
# if_draw_plot = input("Do you want to draw plots?(True if want, False if do not want) ") == "True"

# topics
H_param = np.random.uniform(0, 1, vocab_size) + 1
etas = np.random.dirichlet(H_param, max_topic_num)
betas = np.zeros((max_topic_num, 1))
betas[0] = 2
betas[1] = 1.5
betas[2] = 1
betas[3] = 0.5
number_per_topic = np.zeros(4)
for t in range(max_topic_num):
    num = t % 4
    for i in range(vocab_size):
        if np.floor(i / 5) != num:
        # if np.floor(i / 5) != 0:
            etas[t][i] = 1e-7
    # etas[t,:] = etas[int(t/4),:]
    betas[t] = betas[num]
    # betas[t] = betas[0]

    # if t<12:
    #     print("t={0},eta={1}".format(t,etas[t]))
# betas = [0.0001, 0.001, 0.01, 0.1, 1]
# scale = kernel_var / kernel_mean  # only to avoid computational errors
# shape = kernel_mean ** 2 / kernel_var
# betas = np.random.gamma(shape=shape, scale=scale, size=max_topic_num)
np.savetxt('../data/' + name + '_beta.txt', betas, delimiter=',')
np.savetxt('../data/' + name + '_eta.txt', etas, delimiter=',')
# alpha & mu
alpha = np.zeros((number_of_user, number_of_user))

for u in range(number_of_user):
    idx = np.random.choice(number_of_user, number_of_adj, replace=False)
    for i in range(number_of_adj):
        alpha[u][idx[i]] = np.random.gamma(a, 1 / b, 1)
    if alpha[u][u] != 0:
        for j in range(number_of_user):
            if j != u and alpha[u][j] == 0:
                alpha[u][j] = alpha[u][u]
                break
        alpha[u][u] = 0
# alpha = np.transpose(alpha)
mu = np.random.gamma(c, 1 / d, number_of_user)

print("mus and alphas are generated")

# Saving alphas and mus
np.savetxt('../data/' + name + '_alpha.txt', alpha, delimiter=',')
np.savetxt('../data/' + name + '_mu.txt', mu, delimiter=',')
print("mus and alphas are saved")

# Generating Events using Ogatta's method
events = []
lambdas_per_topic = np.zeros((max_topic_num, number_of_user))  # intensities
local_restaurant_count = np.zeros((number_of_user, max_topic_num))
global_restaurant_count = np.zeros(max_topic_num)
current_time = 0
num_of_topics = 0
mu_errors = list()
alpha_errors = list()


def sample_in_log(probabilities):
    probabilities -= np.max(probabilities)
    probabilities = np.exp(probabilities)
    probabilities /= sum(probabilities)
    return np.random.choice(a=len(probabilities), size=1, p=probabilities)[0]


def sample_parent(event):
    prev_events_log_intensity = np.zeros(len(events) + 1)
    for i in range(len(events)):
        prev_events_log_intensity[i] = np.log(alpha[events[i]["user"]][event["user"]]) - (betas[events[i]["topic"]]) * (
            current_time - events[i]["time"])
    prev_events_log_intensity[-1] = np.log(mu[event["user"]])
    parent = sample_in_log(prev_events_log_intensity)
    return parent


def sample_topic(event):
    if event["parent"] < len(events):
        parent = event["parent"]
        return events[parent]["topic"]
    else:
        user = event["user"]
        probs = np.zeros(max_topic_num + 1)
        probs[:-1] = local_restaurant_count[user]
        probs[-1] = gamma
        topic = np.random.choice(a=len(probs), size=1, p=probs / sum(probs))[0]
        if topic < len(probs) - 1:
            local_restaurant_count[user][topic] += 1
            return topic
        else:
            topics_num = 0
            for i in range(max_topic_num):
                if global_restaurant_count[i] == 0:
                    topics_num = i
                    break
            probs = np.zeros(topics_num + 1)
            probs[:-1] = global_restaurant_count[0:topics_num]
            probs[-1] = zeta
            topic = np.random.choice(a=len(probs), size=1, p=probs / sum(probs))[0]
            global_restaurant_count[topic] += 1
            local_restaurant_count[user][topic] += 1
            return topic


events_file = open('../data/' + name + '_events.txt', 'w')
counts = np.zeros((number_of_user, number_of_user))
estimated_mu = np.zeros((number_of_user, 1))
estimated_alpha = np.zeros((number_of_user, number_of_user))


def compute_mu_and_alpha_errors(time):
    sum_omega_u = dict()
    for event in events:
        user = event['user']
        # triggering_event = events[event["parent"]]
        # user_s = triggering_event["user"]
        # k_s = triggering_event["topic"]
        k = event["topic"]
        # beta_k = betas[k_s]
        beta_k = betas[k]
        # t_s = triggering_event["time"]
        t = event["time"]
        sum_omega_u[user] = sum_omega_u.get(user, 0) + 1 / beta_k * (
            1 - np.exp(-beta_k * (time - t)))
    for i in range(number_of_user):
        estimated_mu[i] = (c + counts[i][i]) / (d + current_time)
        for j in range(number_of_user):
            estimated_alpha[i][j] = (a + counts[i][j]) / (b + sum_omega_u.get(i, 0))
    mu_error = 0
    alpha_error = 0
    num_of_alpha_elements = 0
    for i in range(number_of_user):
        mu_error += np.absolute(mu[i] - estimated_mu[i])
        for j in range(number_of_user):
            if alpha[i][j] > 0:
                alpha_error += np.absolute(alpha[i][j] - estimated_alpha[i][j])
                num_of_alpha_elements += 1
    mu_error /= number_of_user
    alpha_error /= num_of_alpha_elements
    alpha_errors.append(alpha_error)
    mu_errors.append(mu_error)


while len(events) < num_of_events:
    print(len(events))
    total_lambda = sum(mu) + np.sum(lambdas_per_topic)
    next_event_time = np.random.exponential(1 / total_lambda)
    current_time += next_event_time
    # Updating topic weights and lambdas to the current time values
    local_restaurant_count *= np.exp(-next_event_time * topic_decreasing_coefficient)
    global_restaurant_count *= np.exp(-next_event_time * topic_decreasing_coefficient)
    for i in range(max_topic_num):
        lambdas_per_topic[i, :] *= mth.exp(-next_event_time * betas[i])

    lambda_u = mu + np.sum(lambdas_per_topic, axis=0)
    new_total_lambda = sum(lambda_u)
    a_uniform_sample = np.random.uniform(0, 1, 1)
    if a_uniform_sample < new_total_lambda / total_lambda:
        event = dict()
        event["time"] = current_time
        event["user"] = np.random.choice(a=number_of_user, size=1, p=lambda_u / new_total_lambda)[0]
        event["parent"] = sample_parent(event)
        if event["parent"] == len(events):
            counts[event["user"]][event["user"]] += 1
        else:
            counts[events[event["parent"]]["user"]][event["user"]] += 1
        event["topic"] = sample_topic(event)
        number_per_topic[event['topic']%4]+=1
        event["document"] = np.random.multinomial(document_length, etas[event["topic"]], 1)[0]
        for u in range(number_of_user):
            lambdas_per_topic[event["topic"]][u] += alpha[event["user"]][u]
        events.append(event)
        # Printing the generated event to the file###################
        doc_len = 0
        for i in range(len(event["document"])):
            if not event["document"][i] == 0:
                doc_len += 1
        event_str = "{0:.4f}\t{1}\t{2}\t0\t0\t{3}\t0\t{4}".format(event["time"], event["parent"], event["topic"],
                                                                  event["user"], doc_len)
        print(event_str)
        for i in range(len(event["document"])):
            if not event["document"][i] == 0:
                event_str += "\t{0}:{1}".format(i, event["document"][i])
        events_file.write(event_str + "\n")
        compute_mu_and_alpha_errors(current_time)
events_file.close()
for i in range(number_of_user):
    for j in range(number_of_user):
        sys.stdout.write(str(counts[i][j]) + ",")
    sys.stdout.write("\n")
for i in range(4):
    print('number_per_topic[{0}]={1}'.format(i,number_per_topic[i]))
# plt.plot(range(len(mu_errors)), mu_errors)
# plt.title("Mean Absolute Error of mu")
# plt.show()
# plt.plot(range(len(alpha_errors)), alpha_errors)
# plt.title("Mean Absolute Error of alpha")
# plt.show()
