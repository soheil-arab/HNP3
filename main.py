from __future__ import division
from HDPP.HDPP_model import NP3
from DirichletHawkes.DH_model import DirichletHawkes
from Hawkes.hawkes_model import HawkesModel
from common.input_reader import InputReader
import numpy as np
import pickle
import sys

if len(sys.argv) < 3:
    print('Wrong Arguments.\n Correct format is: python main.py MethodName DatasetName')
    print('MethodName can be:\n HDPP\n DirichletHawkes\n Hawkes')
    print('DatasetName can be:\n EventRegistry\n Synthetic')
    exit(1)
methodName = sys.argv[1]
dataset_name = sys.argv[2]
if methodName != 'HDPP' and methodName != 'DirichletHawkes' and methodName != 'Hawkes':
    print('Wrong Arguments. MethodName can be:\n HDPP or DirichletHawkes or Hawkes')
    exit(1)
if dataset_name != 'EventRegistry'and methodName != 'Synthetic':
    print('Wrong Arguments. DatasetName can be:\n EventRegistry or Synthetic')
    exit(1)
print('Going to run ' + methodName + " on " + dataset_name + " dataset.")

# ****************************** Setting Hyperparameters ******************************
hyper_params = dict()
if methodName == 'HDPP':
    hyper_params['a'] = 0.00016
    hyper_params['b'] = 16
    hyper_params['c'] = 0.16
    hyper_params['d'] = 16

    hyper_params['kernel_mean'] = 0.1
    hyper_params['kernel_var'] = 0.001
    hyper_params['epsilon'] = 16e-8

    hyper_params['gamma'] = 1
    hyper_params['zeta'] = 1
    hyper_params['local_topic_decreasing_coefficient'] = 2.5e-4
    hyper_params['global_topic_decreasing_coefficient'] = 2.5e-4
elif methodName == 'DirichletHawkes':
    hyper_params['epsilon'] = 16e-8
    hyper_params['kernel_mean'] = 0.1
    hyper_params['kernel_var'] = 1e-3
    hyper_params['gamma'] = 0.01
else:
    hyper_params['a'] = 0.00016
    hyper_params['b'] = 16
    hyper_params['c'] = 0.16
    hyper_params['d'] = 16

    hyper_params['kernel_mean'] = 0.1
    hyper_params['kernel_var'] = 0.001
    hyper_params['epsilon'] = 16e-8

if methodName == 'HDPP' or methodName == 'DirichletHawkes':
    with open('../data/' + dataset_name + '/wordcount.pickle', 'rb') as default_eta_file:
        default_eta_dict = pickle.load(default_eta_file)
        default_eta = np.zeros(len(default_eta_dict))
        for i in range(len(default_eta_dict)):
            default_eta[i] = default_eta_dict[i]
        default_eta /= sum(default_eta)
        default_eta *= 1
        hyper_params['default_eta'] = default_eta

number_of_particles = 8
start_event_number = 0
number_of_samples = 10
block_size = 200

# ****************************** Setting Dataset Dependent Parameters ******************************
if dataset_name == 'Synthetic':
    number_of_user = 1000
else:
    number_of_user = 100
if dataset_name == 'EventRegistry':
    adjacency_matrix = np.ones((number_of_user, number_of_user), dtype=int)
else:
    with open('../data/' + dataset_name + '/adjacency_matrix.pickle', 'rb') as adjacency_matrix_file:
        adjacency_matrix = pickle.load(adjacency_matrix_file)
# ****************************** Initializing Model ******************************
if methodName == 'DirichletHawkes':
    my_model = DirichletHawkes(name=dataset_name, number_of_user=number_of_user, hyper_params=hyper_params,
                               number_of_particles=number_of_particles, number_of_samples=number_of_samples,
                               block_size=block_size)
elif methodName == 'Hawkes':
    my_model = HawkesModel(name=dataset_name, number_of_user=number_of_user, adjacency_matrix=adjacency_matrix,
                           hyper_params=hyper_params, number_of_particles=number_of_particles, block_size=block_size)
else:
    my_model = NP3(name=dataset_name, number_of_user=number_of_user, adjacency_matrix=adjacency_matrix,
                   hyper_params=hyper_params,
                   number_of_particles=number_of_particles, number_of_samples=number_of_samples, block_size=block_size)

with open('data/' + dataset_name + '/events.csv', 'r') as data_file, \
        open('logs/' + methodName + "/" + dataset_name, 'w') as log_file:
    data_reader = InputReader(data_file, start_event_number)
    if start_event_number != 0:
        my_model.load(start_event_number)
    idx = start_event_number
    event = data_reader.provide_data()
    while not (event is None):
        log_file.write("Event {0}:".format(idx))
        idx += 1
        event = data_reader.provide_data()
        my_model.add_event(event)
