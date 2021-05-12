CHAR_LIST = ['<sos>', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
             'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '-', "'", '.', '_', '+', ' ', '<eos>']

char_map = dict([(v, i) for i, v in enumerate(CHAR_LIST)])
rev_char_map = dict([(i, v) for i, v in enumerate(CHAR_LIST)])
