class Node:
    def __init__(self, name="leaf", vals=None, index=-1):
        self.name = name
        self.possible_vals = vals
        self.index = index
        self.branches = []
        self.leaf = None
        self.value = None

    def set_is_leaf(self, value):
        self.leaf = True
        self.value = value

    def add_branch(self, b):
        self.branches.append(b)


class Branch:
    def __init__(self, value):
        self.value = value
        self.child = None

    def set_child(self, child):
        self.child = child