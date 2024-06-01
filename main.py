import itertools
import numpy as np
import card
import scoring

from scoring import compute_value
from card import get_card
from collections import Counter
import functools
import time

def sanity_checks():
	# 29, optimal hand
	starter_card = np.int8(get_card('5H'))
	hand = np.array([get_card('5D'), get_card('5S'), get_card('5C'), get_card('JH')], dtype=np.int8)
	assert compute_value(starter_card, hand, False) == 29
	print(compute_value(starter_card, hand, False))

	# 10, pair + 2 15s
	starter_card = np.int8(get_card('5D'))
	hand = np.array([get_card('AH'), get_card('3H'), get_card('5H'), get_card('7H')], dtype=np.int8)
	assert compute_value(starter_card, hand, False) == 10
	print(compute_value(starter_card, hand, False))

	# 5 pt flush
	starter_card = np.int8(get_card('AD'))
	hand = np.array([get_card('KD'), get_card('QD'), get_card('TD'), get_card('8D')], dtype=np.int8)
	assert compute_value(starter_card, hand, False) == 5
	print(compute_value(starter_card, hand, False))

	# 5 pt flush
	starter_card = np.int8(get_card('AD'))
	hand = np.array([get_card('KD'), get_card('QD'), get_card('TD'), get_card('8D')], dtype=np.int8)
	assert compute_value(starter_card, hand, True) == 5
	print(compute_value(starter_card, hand, True))

	# 4 pt flush
	starter_card = np.int8(get_card('AS'))
	hand = np.array([get_card('KD'), get_card('QD'), get_card('TD'), get_card('8D')], dtype=np.int8)
	assert compute_value(starter_card, hand, False) == 4
	print(compute_value(starter_card, hand, False))

	# no flush
	starter_card = np.int8(get_card('AS'))
	hand = np.array([get_card('KD'), get_card('QD'), get_card('TD'), get_card('8D')], dtype=np.int8)
	assert compute_value(starter_card, hand, True) == 0
	print(compute_value(starter_card, hand, True))

	starter_card = np.int8(get_card('KD'))
	hand = np.array([get_card('KH'), get_card('QH'), get_card('QD'), get_card('JS')], dtype=np.int8)
	assert compute_value(starter_card, hand, True) == 16
	print(compute_value(starter_card, hand, True))

	starter_card = np.int8(get_card('AD'))
	hand = np.array([get_card('KH'), get_card('QH'), get_card('QD'), get_card('JS')], dtype=np.int8)
	assert compute_value(starter_card, hand, True) == 8
	print(compute_value(starter_card, hand, True))


sanity_checks()

# STARTER_CARDS = 6

# def get_ranks(rank_array):
# 	return [rank - i for i, rank in enumerate(rank_array)]

# rank_array_values = [get_ranks(el) for el in itertools.combinations(np.arange(13 + STARTER_CARDS - 1), STARTER_CARDS)]
# rank_array_values = [el for el in rank_array_values if max(Counter(el).values()) <= 4]
# print(rank_array_values[:5])
# print(len(rank_array_values))

# rank_array = rank_array_values[0]


# suit_counts = [count_pair[1] for count_pair in sorted(Counter(rank_array).items(), key=lambda x: x[0])]

# suits = np.arange(4)
# suit_combinations = [
# 	list(itertools.chain.from_iterable(combos)) for combos in
# 	itertools.product(*[itertools.combinations(suits, suit_count) for suit_count in suit_counts])
# 	]

# suit_combinations = np.array(suit_combinations, dtype=np.int8)
# card_combinations = suit_combinations << 4
# for i, rank in enumerate(rank_array):
# 	card_combinations[:, i] |= rank



# print(card_combinations[:10])




