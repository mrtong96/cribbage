from numba import jit
import numpy as np
import itertools
import card
import functools

@jit(nopython=True)
def score_flush(starter_card, hand, is_crib):
    hand_suits = (hand >> 4)
    if hand_suits[0] == hand_suits[1] == hand_suits[2] == hand_suits[3]:
        starter_suit = (starter_card >> 4)
        if starter_suit == hand_suits[0]:
            return np.int8(5)
        elif not is_crib:
            return np.int8(4)
    return np.int8(0)

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
        return np.int8(2)
    elif sum_values < 15:
        return np.int8(0)

    fifteen_score = np.int8(0)

    # three/two cards
    for i_1 in range(5):
        # four cards
        if sum_values - values[i_1] == 15:
            fifteen_score += 2
        for i_2 in range(i_1+1,5):
            two_total = values[i_1] + values[i_2]
            # two cards
            if two_total == 15:
                fifteen_score += 2
            # three cards
            if sum_values - two_total == 15:
                fifteen_score += 2

    return fifteen_score
    
@jit(nopython=True)
def score_runs(rank_counts):
    runs_total = np.int8(0)
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
    pair_score = np.int8(0)

    for rank_count in rank_counts[:13]:
        if rank_count >= 2:
            pair_score += (rank_count) * (rank_count - 1)
    return pair_score

@jit(nopython=True)
def score_nob(starter_card, hand):
    starter_suit = starter_card >> 4
    if (starter_suit << 4) | 10 in hand:
        return np.int8(1)
    
    return np.int8(0)

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

deck = card.get_deck_by_ranks()
@jit(nopython=True)
def compute_expected_value(hand, is_crib, non_starter_cards):
    last_rank = -1
    rank_total = 0
    running_total = 0

    # do it one longer than 13 for nice edge condition behavior
    rank_counts = np.zeros(14, dtype=np.int8)
    for rank in hand & 0xF:
        rank_counts[rank] += 1
    card_values = get_card_values(np.int8(0), hand)

    for starter_card in deck:
        if starter_card in non_starter_cards:
            continue

        card_rank = card.get_rank(starter_card)
        if card_rank != last_rank:
            # substitue the rank counts of the starter card
            if last_rank != -1:
                rank_counts[last_rank] -= 1
            rank_counts[starter_card & 0xF] += 1
            # substite the value of the starter card
            card_values[0] = card.get_value(starter_card)

            rank_total = score_fifteens(card_values) + score_runs(rank_counts) + score_pairs(rank_counts)
            last_rank = card_rank

        total = 0
        total += score_nob(starter_card, hand)
        total += score_flush(starter_card, hand, is_crib)
        total += rank_total

        running_total += total
    return running_total


