import numpy as np


def expand_macro_set(macro_dict):
    lower_bound = 10
    upper_bound = max(list(macro_dict.keys())) + 1
    for macro_label in range(lower_bound, upper_bound):
        macro_primitives = list(macro_dict[macro_label])
        for larger_macro in range(macro_label+1, upper_bound):
            larger_macro_sequence = list(macro_dict[larger_macro])
            new_seq = []
            for val in larger_macro_sequence:
                if val == macro_label:
                    new_seq += macro_primitives
                else:
                    new_seq.append(val)
            macro_dict[larger_macro] = tuple(new_seq)
    return macro_dict