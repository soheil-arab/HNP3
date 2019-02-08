__author__ = 'Abbas'


class InputReader:
    def __init__(self, data_file, start_number=0):
        self.start_number = start_number
        self.data_file = data_file
        self.current_line_number = 0
        self.source_id_map = dict()
        self.user_counter = 0
        self.start_time = 0
        if start_number != 0:
            for i in range(start_number):
                self.provide_data()

    def provide_data(self):
        line = self.data_file.readline()
        if not line:
            return None
        self.current_line_number += 1
        event = dict()
        splited = line.split('\t')

        if self.current_line_number == 0:
            splited = line.split('\t')
            print("Iteration {0}:".format(self.current_line_number))
            time = float(splited[0])
            self.start_time = time
        splited = line.split('\t')

        # Fetching the time of event
        time = float(splited[0])
        time -= self.start_time
        event['time'] = time

        user_id = splited[2]
        if not (user_id in self.source_id_map):
            self.source_id_map[user_id] = self.user_counter
            self.user_counter += 1
        user_id = self.source_id_map[user_id]

        event['user'] = user_id

        document = dict()
        document['words'] = dict()
        document['named_entities'] = dict()
        document['words_num'] = 0
        document['named_entity_num'] = 0
        doc_len = 0
        words_length = int(splited[5])
        for i in range(6, 6 + words_length):
            word_idx, freq = splited[i].split(':')
            document['words'][int(word_idx)] = int(freq)
            document['words_num'] += int(freq)

        named_entities_length = int(splited[6 + words_length])
        start = 7 + words_length
        finish = start + named_entities_length
        for i in range(start, finish):
            word_idx, freq = splited[i].split(':')
            document['named_entities'][int(word_idx)] = int(freq)
            document['named_entity_num'] += int(freq)

        event['document'] = document
        return event
