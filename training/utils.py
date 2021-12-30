class AvgValue:
    def __init__(self):
        self._sum = 0
        self._count = 0

    def reset(self):
        self._sum = 0
        self._count = 0

    def update(self, value):
        self._sum += value
        self._count += 1

    def get(self):
        if self._count:
            return self._sum / self._count
        return 0


def add_weight_decay(named_parameters, weight_decay, no_decay_names=('bias', 'norm')):
    params = list(named_parameters)
    grouped_parameters = [{'params': [p for n, p in params if not any(nd in n for nd in no_decay_names)],
                           'weight_decay': weight_decay},
                          {'params': [p for n, p in params if any(nd in n for nd in no_decay_names)],
                           'weight_decay': 0.0}]

    return grouped_parameters