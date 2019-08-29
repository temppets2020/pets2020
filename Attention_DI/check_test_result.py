import time


class token_id:
    def __init__(self, document_id, start):
        self.document_id = document_id
        self.start = start

    def __eq__(self, other):
        return hasattr(other, 'document_id') and hasattr(other,
                                                         'start') and other.document_id == self.document_id and other.start == self.start

    def __hash__(self):
        return hash(self.document_id + self.start)


class label_pair:
    def __init__(self, predicted_label, gold_label):
        self.predicted_label = predicted_label
        self.gold_label = gold_label


with open('./test_data/dernoncourt_test.txt') as wf:
    dernoncourt_dataset_map = dict()
    for line in wf:
        if len(line) > 6:
            token = line.strip().split(' ')[0]
            document_id = line.strip().split(' ')[1]
            start = line.strip().split(' ')[2]
            predicted_label = line.strip().split(' ')[-1]
            gold_label = line.strip().split(' ')[-2]

            dernoncourt_dataset_map[token_id(document_id, start)] = label_pair(predicted_label, gold_label)

# with open('./test_data/dernoncourt_test.txt') as wf:
#     dernoncourt_dataset = []
#     for line in wf:
#         if len(line) > 6:
#             token = line.strip().split(' ')[0]
#             document_id = line.strip().split(' ')[1]
#             start = line.strip().split(' ')[2]
#             predicted_label = line.strip().split(' ')[-1]
#             gold_label = line.strip().split(' ')[-2]
#             data_tuple = (token, document_id, start, predicted_label, gold_label)
#             dernoncourt_dataset.append(data_tuple)
#
with open('./test_data/bert_test.txt') as wf:
    bert_dataset = []
    for line in wf:
        token = line.strip().split('\t')[0]
        document_id = line.strip().split('\t')[1]
        start = line.strip().split('\t')[2]
        predicted_label = line.strip().split('\t')[-2]
        gold_label = line.strip().split('\t')[-1]
        data_tuple = (token, document_id, start, predicted_label, gold_label)
        bert_dataset.append(data_tuple)

#
# # token document start predicted gold_label
# assert len(dernoncourt_dataset) > len(bert_dataset)
# # output_file_destination = './output/debuging/ensamble_result.txt'
# output_file_destination = './test_data/ensamble_result.txt'
# with open('./output/debuging/ensamble_result.txt', 'w') as wf:
#     outer_loop_range = len(dernoncourt_dataset)
#     inner_loop_range = len(bert_dataset)
#     # outer_loop_range = 10
#     # inner_loop_range = 10
#
#     print('Starting....')
#     count = 0
#     for i in range(0, outer_loop_range):
#         token = dernoncourt_dataset[i][0]
#         predicted_label = dernoncourt_dataset[i][3]
#         gold_label = dernoncourt_dataset[i][4]
#         start_time = time.time()
#         for j in range(0, inner_loop_range):
#             if dernoncourt_dataset[i][1] == bert_dataset[j][1] and dernoncourt_dataset[i][2] == bert_dataset[j][2]:
#                 # print("line {}".format(i))
#                 if dernoncourt_dataset[i][3] == 'O' and bert_dataset[j][3] is not 'O':
#                     count = count + 1
#                     predicted_label = bert_dataset[j][3]
#                     gold_label = bert_dataset[j][4]
#
#                 # wf.write("{}\t{}\t{}\t{}\t{}\n".format(dernoncourt_dataset[i][0], dernoncourt_dataset[i][1],
#                 #                                        dernoncourt_dataset[i][2], bert_dataset[j][3]
#                 #                                        , bert_dataset[j][4]))
#                 # else:
#                 #     wf.write("{}\t{}\t{}\t{}\t{}\n".format(dernoncourt_dataset[i][0], dernoncourt_dataset[i][1],
#                 #                                        dernoncourt_dataset[i][2], dernoncourt_dataset[i][3],
#                 #                                        dernoncourt_dataset[i][4]))
#
#         wf.write("{}\t{}\t{}\t{}\t{}\n".format(token, dernoncourt_dataset[i][1],
#                                                dernoncourt_dataset[i][2], predicted_label, gold_label))
#         print("Time elapsed on data {} {}s".format(i, time.time() - start_time))
#
#     print("Total ensamble result {}".format(count))


# output_file_destination = './output/debuging/ensamble_result.txt'
output_file_destination = './test_data/ensamble_result_bert_dernoncourt.txt'
with open(output_file_destination, 'w') as wf:
    outer_loop_range = len(bert_dataset)
    # inner_loop_range = len(bert_dataset)
    # outer_loop_range = 10
    # inner_loop_range = 10

    print('Starting....')
    count = 0
    for i in range(0, outer_loop_range):
        token = bert_dataset[i][0]
        predicted_label = bert_dataset[i][3]
        gold_label = bert_dataset[i][4]
        document_id = bert_dataset[i][1]
        start = bert_dataset[i][2]
        start_time = time.time()
        dernoncourt_predicted = dernoncourt_dataset_map.get(token_id(document_id, start))

        if predicted_label is 'O' and dernoncourt_predicted is not None and dernoncourt_predicted.predicted_label is not 'O':
            print("Ensambling... {}\t{}\t{}".format(token, predicted_label, dernoncourt_predicted.predicted_label))
            predicted_label = dernoncourt_predicted.predicted_label
            count = count + 1
            # gold_label = dernoncourt_predicted.gold_label
        #
        # if predicted_label is not 'O' and dernoncourt_predicted is not None and dernoncourt_predicted is 'O':
        #     predicted_label = dernoncourt_predicted.predicted_label
        #     gold_label = dernoncourt_predicted.gold_label
        #

        wf.write("{}\t{}\t{}\t{}\t{}\n".format(token, bert_dataset[i][1],
                                               bert_dataset[i][2], predicted_label, gold_label))
        print("Time elapsed on data {} {}s".format(i, time.time() - start_time))

    print("Total ensamble result {}".format(count))
