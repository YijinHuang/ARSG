CHAR_LIST = ['blank', 'UNK', 'sil', 'ah', 's', 'uw', 'm', 'f', 'aa', 'r', 'ih', 'z', 'ae', 'p', 'uh', 'l', 'ch', 'ey', 'sh', 'n', 'w', 'eh', 'er', 'hh', 'k', 'iy', 'ng', 'd', 'dx', 'aw', 'ay', 'v', 'ow', 'b', 'th', 'g', 'y', 'jh', 'dh', 't', 'oy']

char_map = dict([(v, i) for i, v in enumerate(CHAR_LIST)])
rev_char_map = dict([(i, v) for i, v in enumerate(CHAR_LIST)])
