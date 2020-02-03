import numpy as np


def best_attribute(data, attributes, base_entropy, break_tie='deterministic'):
    igs = []
    names = []
    for i, attribute_name in enumerate(attributes):
        entropies, possible_vals, sample_sizes = attribute_entropy(data, attribute_name)
        igs.append(information_gain(base_entropy, entropies, sample_sizes, len(data)))
        names.append(attribute_name)

    igs = np.array(igs)

    if break_tie == 'deterministic':
        max_index = np.argmax(igs)
    else:
       max_index = np.random.choice(np.flatnonzero(igs == igs.max()))

    return names[max_index], igs[max_index]


def information_gain(base_entropy, attribute_value_entropies, sample_sizes, data_len):
    frac = sample_sizes / data_len
    return base_entropy - np.sum(frac * attribute_value_entropies)


def attribute_entropy(data, attribute_name):
    possible_vals = data.attributes[attribute_name].possible_vals
    entropies = []
    sample_sizes = []

    for i, attribute_value in enumerate(possible_vals):
        data_sample = data.get_row_subset(attribute_name, attribute_value)
        attribute_value_entropy = entropy(data_sample)
        entropies.append(attribute_value_entropy)
        sample_sizes.append(len(data_sample))

    return np.array(entropies), possible_vals, np.array(sample_sizes)


def entropy(data):
    total = len(data)
    labels_column = data.get_column('label')
    labels, counts = np.unique(labels_column, return_counts=True)
    probabilities = counts / total

    if len(labels) == 1:
        return 0

    entropy_val = -np.sum(np.log(probabilities) * probabilities)

    if entropy_val > 0:
        return entropy_val

    return 0


def most_common(data):
    labels = data.get_column('label')
    unique, pos = np.unique(labels, return_inverse=True)
    counts = np.bincount(pos)
    maxpos = counts.argmax()

    return unique[maxpos]


def all_the_same_label(data):
    labels = data.get_column('label')
    return all(x == labels[0] for x in labels)


def remove_attribute(attributes, attribute):
    return [a for a in attributes if a != attribute]