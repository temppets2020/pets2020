test_file = "./output/results/label_test.txt"
o_tag = 'O'

with open(test_file) as wf:
    dataset = []
    for line in wf:
        token = line.strip().split('\t')[0]
        document_id = line.strip().split('\t')[1]
        start = line.strip().split('\t')[2]
        predicted_label = line.strip().split('\t')[-2]
        if predicted_label != o_tag:
            predicted_label = predicted_label.split('-')[1]
        gold_label = line.strip().split('\t')[-1]
        if gold_label != o_tag:
            gold_label = gold_label.split('-')[1]
        data_tuple = (token, document_id, start, predicted_label, gold_label)
        dataset.append(data_tuple)


# with open(test_file) as wf:
#     dataset = []
#     for line in wf:
#         if (len(line) > 1):
#             token = line.strip().split(' ')[0]
#             document_id = line.strip().split(' ')[1]
#             start = line.strip().split(' ')[2]
#             predicted_label = line.strip().split(' ')[-1]
#             if predicted_label != o_tag:
#                 predicted_label = predicted_label.split('-')[1]
#             gold_label = line.strip().split(' ')[-2]
#             if gold_label != o_tag:
#                 gold_label = gold_label.split('-')[1]
#             data_tuple = (token, document_id, start, predicted_label, gold_label)
#             dataset.append(data_tuple)

total_tag_count = dict()
tag_false_positive = dict()
tag_false_negative = dict()
tag_true_positive = dict()
total_token_count = 0
total_correct_count = 0
for data in dataset:
    total_tag_count[data[4]] = total_tag_count.get(data[4]) + 1 if total_tag_count.get(data[4]) is not None else 0
    # calculate true positive
    total_token_count = total_token_count + 1
    if data[3] == data[4]:
        total_correct_count = total_correct_count + 1

    if data[3] == data[4] and data[4] != o_tag:
        tag_true_positive[data[3]] = tag_true_positive.get(data[3]) + 1 if tag_true_positive.get(
            data[3]) is not None else 0

    # calculate false_negative
    if data[3] != data[4] and data[4] != o_tag:
        tag_false_negative[data[3]] = tag_false_negative.get(data[3]) + 1 if tag_false_negative.get(
            data[3]) is not None else 0

    # calculate false positive

    if data[3] != data[4] and data[4] == o_tag and data[3] != o_tag:
        tag_false_positive[data[3]] = tag_false_positive.get(data[3]) + 1 if tag_false_positive.get(
            data[3]) is not None else 0

TP = 0
FP = 0
FN = 0

for keys, value in total_tag_count.items():
    TP = TP + tag_true_positive.get(keys) if tag_true_positive.get(keys) is not None else TP + 0
    FP = FP + tag_false_positive.get(keys) if tag_false_positive.get(keys) is not None else FP + 0
    FN = FN + tag_false_negative.get(keys) if tag_false_negative.get(keys) is not None else FN + 0

accuracy = total_correct_count / total_token_count * 100
precision = TP / (TP + FP) * 100 if (FP + TP) != 0 else 0
recall = TP / (TP + FN) * 100 if (FN + TP) != 0 else 0
F1 = ((2 * precision * recall) / (precision + recall)) if (precision + recall) != 0 else 0
print("accuracy:{:6.3f}\tprecision:{:6.3f}\trecall:{:6.3f}\tFB1:{:6.3f}".format(accuracy, precision, recall, F1))

precision = 0
recall = 0
F1 = 0

for keys, value in total_tag_count.items():
    if keys != o_tag:
        TP = tag_true_positive.get(keys) if tag_true_positive.get(keys) is not None else 0
        FP = tag_false_positive.get(keys) if tag_false_positive.get(keys) is not None else 0
        FN = tag_false_negative.get(keys) if tag_false_negative.get(keys) is not None else 0

        precision = TP / (TP + FP) * 100 if (FP + TP) != 0 else 0
        recall = TP / (TP + FN) * 100 if (FN + TP) != 0 else 0
        F1 = ((2 * precision * recall) / (precision + recall)) if (precision + recall) != 0 else 0
        print("{:17s}:\tprecision:{:6.4f}\trecall:{:6.4f}\tFB1:{:6.4f}\t{}".format(keys, precision, recall, F1, value))
