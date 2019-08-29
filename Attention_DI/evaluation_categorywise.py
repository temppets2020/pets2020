test_file = "./output/results/i2b2_bert_test.txt"
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

category_map = {
    'doctor': 'name',
    'patient': 'name',
    'username': 'name',
    'age': 'age',
    'phone': 'contact',
    'fax': 'contact',
    'email': 'contact',
    'medicalrecord': 'id',
    'idnum': 'id',
    'device': 'id',
    'healthplan': 'id',
    'bioid': 'id',
    'hospital': 'location',
    'city': 'location',
    'state': 'location',
    'street': 'location',
    'zip': 'location',
    'country': 'location',
    'organization': 'location',
    'location_other': 'location',
    'profession': 'profession',
    'date': 'date',
    'O':'O'
}

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



main_category_fp = dict()
main_category_fn = dict()
main_category_tp = dict()

for keys, values in total_tag_count.items():
    if keys != o_tag:
        main_category_fp[category_map.get(keys.lower())] = (main_category_fp.get(
            category_map.get(keys.lower())) if main_category_fp.get(
            category_map.get(keys.lower())) is not None else 0) + (
                                                               tag_false_positive.get(keys) if tag_false_positive.get(
                                                                   keys) is not None else 0)

        main_category_fn[category_map.get(keys.lower())] = (main_category_fn.get(
            category_map.get(keys.lower())) if main_category_fn.get(
            category_map.get(keys.lower())) is not None else 0) + (
                                                               tag_false_negative.get(keys) if tag_false_negative.get(
                                                                   keys) is not None else 0)

        main_category_tp[category_map.get(keys.lower())] = (main_category_tp.get(
            category_map.get(keys.lower())) if main_category_tp.get(
            category_map.get(keys.lower())) is not None else 0) + (tag_true_positive.get(keys) if tag_true_positive.get(
            keys) is not None else 0)


print(tag_true_positive)
print(main_category_tp)

print(tag_false_negative)
print(main_category_fn)

print(tag_false_positive)
print(main_category_fp)


TP = 0
FP = 0
FN = 0

assert len(main_category_fn) == len(main_category_fp)
assert len(main_category_fp) == len(main_category_tp)

main_category_keys = main_category_fp.items()
for key, values in main_category_keys:
    main_category = key
    if main_category is not None:
        TP = TP + main_category_tp.get(main_category) if main_category_tp.get(main_category) is not None else TP + 0
        FP = FP + main_category_fp.get(main_category) if main_category_fp.get(main_category) is not None else FP + 0
        FN = FN + main_category_fn.get(main_category) if main_category_fn.get(main_category) is not None else FN + 0


# FN = FN + 934
FN = FN + 459

accuracy = total_correct_count / total_token_count * 100
precision = TP / (TP + FP) * 100 if (FP + TP) != 0 else 0
recall = TP / (TP + FN) * 100 if (FN + TP) != 0 else 0
F1 = ((2 * precision * recall) / (precision + recall)) if (precision + recall) != 0 else 0
print("accuracy:{:6.3f}\tprecision:{:6.3f}\trecall:{:6.3f}\tFB1:{:6.3f}".format(accuracy, precision, recall, F1))



precision = 0
recall = 0
F1 = 0


for keys, value in main_category_keys:
    if keys != o_tag:
        TP = main_category_tp.get(keys) if main_category_tp.get(keys) is not None else 0
        FP = main_category_fp.get(keys) if main_category_fp.get(keys) is not None else 0
        FN = main_category_fn.get(keys) if main_category_fn.get(keys) is not None else 0

        precision = TP / (TP + FP) * 100 if (FP + TP) != 0 else 0
        recall = TP / (TP + FN) * 100 if (FN + TP) != 0 else 0
        F1 = ((2 * precision * recall) / (precision + recall)) if (precision + recall) != 0 else 0
        print("{:17s}:\tprecision:{:6.4f}\trecall:{:6.4f}\tFB1:{:6.4f}\t{}".format(keys.upper(), precision, recall, F1, value))
