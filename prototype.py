# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.9.16
# ---

# %% tags=[] execution={"iopub.status.busy": "2024-04-23T17:45:07.774048Z", "iopub.execute_input": "2024-04-23T17:45:07.774402Z", "iopub.status.idle": "2024-04-23T17:45:07.778955Z", "shell.execute_reply.started": "2024-04-23T17:45:07.774366Z", "shell.execute_reply": "2024-04-23T17:45:07.778122Z"} trusted=true
# cribbage code

# %% tags=[] execution={"iopub.status.busy": "2024-04-23T17:45:07.780078Z", "iopub.execute_input": "2024-04-23T17:45:07.780452Z", "iopub.status.idle": "2024-04-23T17:45:08.675102Z", "shell.execute_reply.started": "2024-04-23T17:45:07.780424Z", "shell.execute_reply": "2024-04-23T17:45:08.674369Z"} trusted=true
from numba import jit
import numpy as np
import itertools

# %% tags=[] execution={"iopub.status.busy": "2024-04-23T17:45:08.676078Z", "iopub.execute_input": "2024-04-23T17:45:08.676467Z", "iopub.status.idle": "2024-04-23T17:45:08.691520Z", "shell.execute_reply.started": "2024-04-23T17:45:08.676438Z", "shell.execute_reply": "2024-04-23T17:45:08.690902Z"} trusted=true
# cards stuff

SUITS = 'SHCD'
RANKS = 'A23456789TJQK'

@jit(nopython=True)
def get_card(card):
    rank = card[0]
    suit = card[1]
    return np.int8((SUITS.index(suit) << 4) | (RANKS.index(rank)))

@jit(nopython=True)
def get_rank_suit(card):
    rank = card & 0xF
    suit = (card >> 4) & 0x3
    return f"{RANKS[rank]}{SUITS[suit]}"

@jit(nopython=True)
def get_rank(card):
    return card & 0xF

@jit(nopython=True)
def get_suit(card):
    return (card >> 4) & 0x3

@jit(nopython=True)
def get_value(card):
    value = (card & 0xF) + 1
    return min(value, 10)

# @jit(nopython=True)
def get_deck():
    return np.array(list([get_card(f"{rank}{suit}") for suit, rank in itertools.product(SUITS, RANKS)]), dtype=np.int8)



# %% tags=[] execution={"iopub.status.busy": "2024-04-23T17:45:08.693041Z", "iopub.execute_input": "2024-04-23T17:45:08.693538Z", "iopub.status.idle": "2024-04-23T17:45:11.677625Z", "shell.execute_reply.started": "2024-04-23T17:45:08.693510Z", "shell.execute_reply": "2024-04-23T17:45:11.676895Z"} trusted=true
@jit(nopython=True)
def score_flush(starter_card, hand, is_crib):
    hand_suits = (hand >> 4) & 0x3
    if hand_suits[0] == hand_suits[1] == hand_suits[2] == hand_suits[3]:
        starter_suit = (starter_card >> 4) & 0x3
        if starter_suit == hand_suits[0]:
            return np.int8(5)
        elif not is_crib:
            return np.int8(4)
    return np.int8(0)

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

@jit(nopython=True)
def score_fifteens(starter_card, hand):
    values = np.zeros(5, dtype=np.int8)
    values[0] = get_value(starter_card)
    for i in range(4):
        values[i+1] = get_value(hand[i])

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
def score_nob(rank_counts, starter_card, hand):
    if rank_counts[10] == 0:
        return np.int8(0)
    
    if starter_card & 0xF == 10:
        return np.int8(2)
    
    starter_suit = (starter_card >> 4) & 0x3
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

    total = score_nob(rank_counts, starter_card, hand)
    total += score_flush(starter_card, hand, is_crib)
    total += score_fifteens(starter_card, hand)
    total += score_runs(rank_counts)
    total += score_pairs(rank_counts)

    return total

deck = get_deck()
@jit(nopython=True)
def compute_expected_value(hand, is_crib):
    last_rank = -1
    rank_total = 0
    num_cards_checked = 0
    running_total = 0
    for starter_card in deck:
        if starter_card in hand:
            continue
        
        card_rank = get_rank(starter_card)
        if card_rank != last_rank:
            # do it one longer than 13 for nice edge condition behavior
            rank_counts = np.zeros(14, dtype=np.int8)
            rank_counts[starter_card & 0xF] = 1
            for rank in hand & 0xF:
                rank_counts[rank] += 1

            rank_total = score_fifteens(starter_card, hand) + score_runs(rank_counts) + score_pairs(rank_counts)
            last_rank = card_rank

        total = 0
        total += score_nob(rank_counts, starter_card, hand)
        total += score_flush(starter_card, hand, is_crib)
        total += rank_total

        running_total += total
        num_cards_checked += 1
    return running_total, num_cards_checked



# %% tags=[] execution={"iopub.status.busy": "2024-04-23T17:45:11.678659Z", "iopub.execute_input": "2024-04-23T17:45:11.679069Z", "iopub.status.idle": "2024-04-23T17:45:13.544031Z", "shell.execute_reply.started": "2024-04-23T17:45:11.679039Z", "shell.execute_reply": "2024-04-23T17:45:13.543318Z"} trusted=true
hand = np.array([get_card('5D'), get_card('5S'), get_card('5C'), get_card('JH')], dtype=np.int8)
# %timeit compute_expected_value(hand, False)

# %% tags=[] execution={"iopub.status.busy": "2024-04-23T17:45:13.545075Z", "iopub.execute_input": "2024-04-23T17:45:13.545418Z", "iopub.status.idle": "2024-04-23T17:45:13.878062Z", "shell.execute_reply.started": "2024-04-23T17:45:13.545388Z", "shell.execute_reply": "2024-04-23T17:45:13.877347Z"} trusted=true
total = 0

for card in deck:
    if card in hand:
        continue

    total += compute_value(card, hand, False)
total

# %% tags=[] execution={"iopub.status.busy": "2024-04-23T17:45:13.879025Z", "iopub.execute_input": "2024-04-23T17:45:13.879365Z", "iopub.status.idle": "2024-04-23T17:45:13.883804Z", "shell.execute_reply.started": "2024-04-23T17:45:13.879337Z", "shell.execute_reply": "2024-04-23T17:45:13.883190Z"} trusted=true
compute_expected_value(hand, False)

# %% tags=[] execution={"iopub.status.busy": "2024-04-23T17:45:13.884660Z", "iopub.execute_input": "2024-04-23T17:45:13.884957Z", "iopub.status.idle": "2024-04-23T17:45:13.887758Z", "shell.execute_reply.started": "2024-04-23T17:45:13.884930Z", "shell.execute_reply": "2024-04-23T17:45:13.887136Z"} trusted=true
# deck = get_deck()


# %% tags=[] execution={"iopub.status.busy": "2024-04-23T17:45:13.889617Z", "iopub.execute_input": "2024-04-23T17:45:13.889934Z", "iopub.status.idle": "2024-04-23T17:45:13.901566Z", "shell.execute_reply.started": "2024-04-23T17:45:13.889908Z", "shell.execute_reply": "2024-04-23T17:45:13.900936Z"} trusted=true
# 29
starter_card = np.int8(get_card('5H'))
hand = np.array([get_card('5D'), get_card('5S'), get_card('5C'), get_card('JH')], dtype=np.int8)

print(compute_value(starter_card, hand, False))

# 10, pair + 2 15s
starter_card = np.int8(get_card('5D'))
hand = np.array([get_card('AH'), get_card('3H'), get_card('5H'), get_card('7H')], dtype=np.int8)

print(compute_value(starter_card, hand, False))

# 5 pt flush
starter_card = np.int8(get_card('AD'))
hand = np.array([get_card('KD'), get_card('QD'), get_card('TD'), get_card('8D')], dtype=np.int8)

print(compute_value(starter_card, hand, False))

# 5 pt flush
starter_card = np.int8(get_card('AD'))
hand = np.array([get_card('KD'), get_card('QD'), get_card('TD'), get_card('8D')], dtype=np.int8)

print(compute_value(starter_card, hand, True))

# 4 pt flush
starter_card = np.int8(get_card('AS'))
hand = np.array([get_card('KD'), get_card('QD'), get_card('TD'), get_card('8D')], dtype=np.int8)

print(compute_value(starter_card, hand, False))

# no flush
starter_card = np.int8(get_card('AS'))
hand = np.array([get_card('KD'), get_card('QD'), get_card('TD'), get_card('8D')], dtype=np.int8)

print(compute_value(starter_card, hand, True))

starter_card = np.int8(get_card('KD'))
hand = np.array([get_card('KH'), get_card('QH'), get_card('QD'), get_card('JS')], dtype=np.int8)

print(compute_value(starter_card, hand, True))

starter_card = np.int8(get_card('AD'))
hand = np.array([get_card('KH'), get_card('QH'), get_card('QD'), get_card('JS')], dtype=np.int8)

print(compute_value(starter_card, hand, True))


# %% tags=[] execution={"iopub.status.busy": "2024-04-23T17:45:13.902420Z", "iopub.execute_input": "2024-04-23T17:45:13.902725Z", "iopub.status.idle": "2024-04-23T17:45:19.793987Z", "shell.execute_reply.started": "2024-04-23T17:45:13.902698Z", "shell.execute_reply": "2024-04-23T17:45:19.793215Z"} trusted=true
# %timeit _ = compute_value(starter_card, hand, True)

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

