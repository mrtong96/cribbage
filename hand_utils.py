import itertools
import numpy as np
import card
import scoring
from numba import jit
from collections import Counter

from numba.typed import Dict
from numba import types


# jit can't sort stuff
# https://stackoverflow.com/questions/23926670/numba-sorting-an-array-in-place
@jit(nopython=True)
def inplace_sort(arr):
    ndim = arr.shape[0]
    for i in range(ndim):
        for j in range(i+1,ndim):
            if arr[i] > arr[j]:
                tmp = arr[i]
                arr[i] = arr[j]
                arr[j] = tmp
    return arr

def get_ranks(rank_array: list[int]):
	return [rank - i for i, rank in enumerate(rank_array)]

def compute_rank_array_values(num_cards):
    rank_array_values = [get_ranks(el) for el in itertools.combinations(np.arange(13 + num_cards - 1), num_cards)]
    rank_array_values = [el for el in rank_array_values if max(Counter(el).values()) <= 4]
    return rank_array_values

@jit(nopython=True)
def get_suit_reduction(card_combo):
    suits = np.zeros(4, dtype=np.int16)
    for cur_card in card_combo:
        suits[card.get_suit(cur_card)] |= 1 << card.get_rank(cur_card)
    inplace_sort(suits)

    result = np.int64(0)
    for i in range(4):
        result |= suits[i] << (i * 16)
    return result

def compute_suit_count_to_truncated_suit_vectors(rank_array_values):
    # power of suit reductions... so I don't have to compute this for all nCr(6, 2) * nCr(52, 6) combinations
    suit_count_to_truncated_suit_vectors = dict()
    
    # get counts of the suits
    possible_suit_count_values = Counter(
        [tuple(item[1] for item in sorted(Counter(rank_array).items(), key=lambda x: x[0]))
        for rank_array in rank_array_values]
    )
    
    for possible_suit_count in possible_suit_count_values:
        # enumerate through all possible combinations of suits
        suits = np.arange(4)
        suit_combinations = [
        	list(itertools.chain.from_iterable(combos)) for combos in
        	itertools.product(*[itertools.combinations(suits, suit_count) for suit_count in possible_suit_count])
        ]
    
        # build some sample cards
        suit_combinations = np.array(suit_combinations, dtype=np.int8)
        card_combinations = suit_combinations << 4
        index = 0
        for card_rank, suit_count in enumerate(possible_suit_count):
            for i in range(suit_count):
                card_combinations[:, index] |= card_rank
                index += 1
    
        # do the suit reduction
        suit_reductions = [get_suit_reduction(card_combo) for card_combo in card_combinations]
        suit_reduction_count = Counter(suit_reductions)
        seen_reductions = set()
    
        # compute a map of `representative_suit -> weight`
        truncated_suit_mask = []
        truncated_suit_weights = []
        for i, suit_reduction in enumerate(suit_reductions):
            if suit_reduction not in seen_reductions:
                seen_reductions.add(suit_reduction)
                truncated_suit_mask.append(i)
                truncated_suit_weights.append(suit_reduction_count[suit_reduction])
    
        # add it to the dict
        suit_count_to_truncated_suit_vectors[possible_suit_count] = (
            suit_combinations[truncated_suit_mask],
            np.array(truncated_suit_weights)
        )

    return suit_count_to_truncated_suit_vectors

deck = card.get_deck()

# AKs, AKo, AA
# mapped to [0->n]
@jit(nopython=True)
def get_discard_int(card1, card2):
    rank1 = card1 & 0xF
    rank2 = card2 & 0xF
    suit1 = card1 >> 4
    suit2 = card2 >> 4

    lower_rank = min(rank1, rank2)
    higher_rank = max(rank1, rank2)

    # pairs, 0, 14, 
    if lower_rank == higher_rank:
        return lower_rank * 14
    # suited
    elif suit1 == suit2:
        return lower_rank * 13 + higher_rank
    # off-suited
    else:
        return higher_rank * 13 + lower_rank

discard_int_to_index = Dict.empty(
    key_type=types.int32,
    value_type=types.int32,
)
discard_index_to_int = Dict.empty(
    key_type=types.int32,
    value_type=types.int32,
)
discard_ints = sorted(set([get_discard_int(card1, card2) for card1, card2 in itertools.combinations(deck, 2)]))
for i, discard_int in enumerate(discard_ints):
    discard_int_to_index[discard_int] = i
    discard_index_to_int[i] = discard_int
