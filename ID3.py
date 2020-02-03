import utils
from node import Node, Branch


class ID3Classifier:
    def __init__(self, max_depth=-1):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, data):
        self.tree = self._learn_tree(data, data.attributes, -1)

    def predict(self, data):
        y_pred = []
        for data_point in data.raw_data:
            y_pred.append(self._walk_down(self.tree, data_point))
        return y_pred

    def _learn_tree(self, data, attributes, depth):
        if not len(attributes) or depth + 1 == self.max_depth:
            leaf = Node()
            leaf.set_is_leaf(utils.most_common(data))
            return leaf

        if utils.all_the_same_label(data):
            leaf = Node()
            label = data.raw_data[0][0]
            leaf.set_is_leaf(label)
            return leaf

        base_entropy = utils.entropy(data)
        attribute_name, attribute_ig = utils.best_attribute(data, attributes, base_entropy)
        attribute = data.attributes[attribute_name]
        root = Node(attribute_name, attribute.possible_vals, attribute.index)
        depth += 1

        for attribute_value in root.possible_vals:
            b = Branch(attribute_value)
            root.add_branch(b)
            data_sample = data.get_row_subset(attribute_name, attribute_value)

            if not len(data_sample):
                leaf = Node()
                leaf.set_is_leaf(utils.most_common(data))
                b.set_child(leaf)
            else:
                attributes = utils.remove_attribute(attributes, attribute_name)
                b.set_child(self._learn_tree(data_sample, attributes, depth))

        return root

    def _walk_down(self, node, point):
        if node.name == "leaf":
            return node.value
        if node.branches:
            for b in node.branches:
                if b.value == point[node.index]:
                    return self._walk_down(b.child, point)

        return node.value