from numba import jit
import numpy as np
import itertools
import card
import functools
import hand_utils
from numba.typed import Dict
from numba import types

HAND_CARDS = 4
rank_array_values = np.array(hand_utils.compute_rank_array_values(HAND_CARDS))
SCORE_DTYPE = np.int32

@jit(nopython=True)
def is_hand_flush(hand):
    hand_suits = (hand >> 4)
    return hand_suits[0] == hand_suits[1] == hand_suits[2] == hand_suits[3]

@jit(nopython=True)
def score_flush(starter_card, hand, is_crib):
    if is_hand_flush(hand):
        starter_suit = (starter_card >> 4)
        if starter_suit == hand[0] >> 4:
            return SCORE_DTYPE(5)
        elif not is_crib:
            return SCORE_DTYPE(4)
    return SCORE_DTYPE(0)

@jit(nopython=True)
def score_flush_starter_card(starter_card, hand_suit, is_crib):
    starter_suit = (starter_card >> 4)
    if starter_suit == hand_suit:
        return SCORE_DTYPE(5)
    elif not is_crib:
        return SCORE_DTYPE(4)
    return SCORE_DTYPE(0)

@jit(nopython=True)
def get_card_values(starter_card, hand):
    values = np.zeros(5, dtype=np.int8)
    values[0] = card.get_value(starter_card)
    for i in range(4):
        values[i+1] = card.get_value(hand[i])
    return values

@jit(nopython=True)
def score_fifteens(values):
    # if the total of all 5 cards is <=15, we are done
    sum_values = np.sum(values)
    if sum_values == 15:
        return SCORE_DTYPE(2)
    elif sum_values < 15:
        return SCORE_DTYPE(0)

    fifteen_score = SCORE_DTYPE(0)

    # three/two cards
    for i_1 in range(5):
        # four cards
        if sum_values - values[i_1] == 15:
            fifteen_score += SCORE_DTYPE(2)
        for i_2 in range(i_1+1,5):
            two_total = values[i_1] + values[i_2]
            # two cards
            if two_total == 15:
                fifteen_score += SCORE_DTYPE(2)
            # three cards
            if sum_values - two_total == 15:
                fifteen_score += SCORE_DTYPE(2)

    return fifteen_score
    
@jit(nopython=True)
def score_runs(rank_counts):
    runs_total = SCORE_DTYPE(0)
    last_run_start = -1
    num_runs = 1
    for rank, rank_count in enumerate(rank_counts):
        # count number of duplicates
        if rank_count > 0:
            num_runs *= rank_count
            # if new run, start counting
            if last_run_start == -1:
                last_run_start = rank
        elif rank_count == 0:
            if last_run_start != -1 and rank - last_run_start >= 3:
                runs_total += num_runs * (rank - last_run_start)
            # I'm lazy
            last_run_start = -1
            num_runs = 1
    return runs_total

@jit(nopython=True)
def score_pairs(rank_counts):
    pair_score = SCORE_DTYPE(0)

    for rank_count in rank_counts[:13]:
        if rank_count >= 2:
            pair_score += (rank_count) * (rank_count - 1)
    return pair_score

# while playing we should be able to score runs
@jit(nopython=True)
def score_play_run(cards):
    for start_card_index in range(cards.shape[0] - 2):
        total_cards = cards.shape[0] - start_card_index
        values = cards = sorted(set(removed_cards[start_card_index:] & 0xF))
        if len(values) == total_cards and max(values) - min(values) == total_cards:
            return np.int8(total_cards)
    return np.int8(0)

@jit(nopython=True)
def score_play_pairs(cards):
    card_values = cards & 0xF
    for neg_num_cards in range(-4, -1):
        num_cards = neg_num_cards * -1
        if len(set(card_values[neg_num_cards:])) == 1:
            # 2->2, 3->6, 4->12
            return np.int8(num_cards * (num_cards - 1))
    return np.int8(0)

@jit(nopython=True)
def score_play_values(cards):
    total_values = np.sum([card.get_value(cur_card) for cur_card in cards])
    if total_values == 15:
        return np.int8(2)
    elif total_values == 31:
        return np.int8(2)
    else:
        return np.int8(0)a
        

@jit(nopython=True)
def score_nob(starter_card, hand):    
    # if the jack is in the hand in the right suit
    if (starter_card & 0xF0) | 0xA in hand:
        return SCORE_DTYPE(1)
    return SCORE_DTYPE(0)

@jit(nopython=True)
def compute_value(starter_card, hand, is_crib):
    # do it one longer than 13 for nice edge condition behavior
    rank_counts = np.zeros(14, dtype=np.int8)
    rank_counts[starter_card & 0xF] = 1
    for rank in hand & 0xF:
        rank_counts[rank] += 1

    total = score_nob(starter_card, hand)
    total += score_flush(starter_card, hand, is_crib)
    values = get_card_values(starter_card, hand)
    total += score_fifteens(values)
    total += score_runs(rank_counts)
    total += score_pairs(rank_counts)

    return total

@jit(nopython=True)
def get_scoring_ranks_int(scoring_card_ranks, starter_card_rank):
    # -1 is the error code
    if np.all(starter_card_rank == scoring_card_ranks):
        return -1

    sorted_card_ranks = np.sort(scoring_card_ranks)
    
    return np.int32(
        (starter_card_rank << 16) |
        (sorted_card_ranks[0] << 12) |
        (sorted_card_ranks[1] << 8) |
        (sorted_card_ranks[2] << 4) |
        (sorted_card_ranks[3])
    )

@jit(nopython=True)
def compute_non_suit_score(scoring_card_ranks, starter_card_rank):
    # do it one longer than 13 for nice edge condition behavior
    rank_counts = np.zeros(14, dtype=np.int8)
    rank_counts[starter_card_rank] += 1
    for rank in scoring_card_ranks:
        rank_counts[rank] += 1

    values = np.zeros(5, dtype=np.int8)
    for i, scoring_card_rank in enumerate(scoring_card_ranks):
        values[i] = card.get_value(scoring_card_rank)
    values[4] = card.get_value(starter_card_rank)

    total = SCORE_DTYPE(0)
    total += score_fifteens(values)
    total += score_runs(rank_counts)
    total += score_pairs(rank_counts)

    return total


# to compute scores faster, we can just cache the scores without suit information
# (score not including scores from flushes and jacks/nobs)
def get_scoring_rank_int_to_suitless_score():
    scoring_rank_int_to_suitless_score = Dict.empty(
        key_type=types.int32,
        value_type=types.int32,
    )
    
    for rank_array_values_row in rank_array_values:
        for starter_card_rank in np.arange(13, dtype=np.int8):
            scoring_rank_int = get_scoring_ranks_int(rank_array_values_row, starter_card_rank)
            if scoring_rank_int == -1:
                continue
            score = compute_non_suit_score(rank_array_values_row, starter_card_rank)
            scoring_rank_int_to_suitless_score[scoring_rank_int] = score
    return scoring_rank_int_to_suitless_score

@jit(nopython=True)
def get_possible_starter_ranks(non_starter_cards):
    possible_starter_ranks = np.full(13, 4, dtype=SCORE_DTYPE)
    for rank in (non_starter_cards & 0xF):
        possible_starter_ranks[rank] -= 1
    return possible_starter_ranks

@jit(nopython=True)
def get_possible_starter_suits(non_starter_cards):
    possible_starter_suits = np.full(4, 13, dtype=SCORE_DTYPE)
    for suit in (non_starter_cards >> 4):
        possible_starter_suits[suit] -= 1
    return possible_starter_suits

# very optimized/unfriendly function for computing the expected value of a possible hand
# super optimized because this gets called ~1M times or so per batch eval pass
@jit(nopython=True)
def compute_expected_value(hand, is_crib, possible_starter_ranks, possible_starter_suits, scoring_rank_int_to_suitless_score):
    last_rank = -1
    rank_total = 0
    running_total = SCORE_DTYPE(0)
    hand_is_flush = is_hand_flush(hand)

    # Do this logic here because we reduce the number of int creations by ~50x... it's a lot
    hand_ranks = hand & 0xF
    sorted_hand_ranks = np.sort(hand_ranks)
    hand_rank_int_base = np.int32(
        (sorted_hand_ranks[0] << 12) |
        (sorted_hand_ranks[1] << 8) |
        (sorted_hand_ranks[2] << 4) |
        (sorted_hand_ranks[3])
    )

    # see if we have a 4 of a kind
    skip_rank = -1
    if sorted_hand_ranks[0] == sorted_hand_ranks[1] == sorted_hand_ranks[2] == sorted_hand_ranks[3]:
        skip_rank = sorted_hand_ranks[0]
    # look up the runs/15s/pairs score for the set of 5 cards via a dictionary
    for card_rank in range(13):
        if card_rank != skip_rank:
            rank_score = scoring_rank_int_to_suitless_score[(card_rank << 16) | hand_rank_int_base]
            running_total += possible_starter_ranks[card_rank] * rank_score
    
    # nob scoring for the hand
    for hand_card in hand:
        # if we have a jack
        if hand_card & 0xF == 0xA:
            # figure out how many possible starter cards have the same matching suit
            running_total += possible_starter_suits[hand_card >> 4]

    
    # flush scoring for the hand
    if hand_is_flush:
        hand_suit = hand[0] >> 4        
        # All 5-card flushes
        running_total += 5 * possible_starter_suits[hand_suit]
        if not is_crib:
            # count cases where the starter card doesn't match the hand suit
            running_total += 4 * (np.sum(possible_starter_suits) - possible_starter_suits[hand_suit])

    return running_total
