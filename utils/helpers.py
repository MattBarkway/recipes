from itertools import islice


def get_gen_at_index(gen, index):
    return next(islice(gen, index, None))
