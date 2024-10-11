from numba import jit
import numpy as np
import itertools

SUITS = 'SHCD'
RANKS = 'A23456789TJQK'

@jit(nopython=True)
def get_card(card: str) -> np.int8:
    rank = card[0]
    suit = card[1]
    return np.int8((SUITS.index(suit) << 4) | (RANKS.index(rank)))

@jit(nopython=True)
def get_rank_suit(card: np.int8) -> np.int8:
    rank = card & 0xF
    suit = (card >> 4) & 0x3
    return f"{RANKS[rank]}{SUITS[suit]}"

@jit(nopython=True)
def get_rank(card: np.int8) -> np.int8:
    return card & 0xF

@jit(nopython=True)
def get_suit(card: np.int8) -> np.int8:
    return (card >> 4) & 0x3

@jit(nopython=True)
def get_value(card: np.int8) -> np.int8:
    value = (card & 0xF) + 1
    return min(value, 10)

# @jit(nopython=True)
def get_deck():
    return np.array(list([get_card(f"{rank}{suit}") for suit, rank in itertools.product(SUITS, RANKS)]), dtype=np.int8)

def get_deck_by_ranks():
    return np.array(list([get_card(f"{rank}{suit}") for rank, suit in itertools.product(RANKS, SUITS)]), dtype=np.int8)
