
import card
import scoring
import numpy as np
from numba import jit

# Also TODO: refactor host -> delaer at some point... played too much tractor recently
# TODO: look up enums
game_states = [
    'INIT', 'DEALT', 'CRIB_SELECTED', 'SHARED_CARD_REVEALED', 'ALL_CARDS_PLAYED',
    'NON_HOST_HAND_SCORED', 'HOST_HAND_SCORED', 'HOST_CRIB_SCORED', 'ROUND_OVER'
]
GAME_STATES = {state: i for i, state in enumerate(game_states)}

PASS = None

@jit(nopython=True)
def remove_card(cards: np.array[np.int8], to_remove: np.int8):
    return np.array([card for card in cards if card != to_remove], dtype=np.int8)

class PlayerRoundState():
    def __init__(self, is_host: bool, starting_cards: np.array[np.int8], played_cards=np.array[np.int8], shared_card=np.int8):
        self.is_host = is_host
        self.non_crib_cards = None
        self.hand_cards = starting_cards
        self.played_cards = played_cards
        # technically we don't see the shared card until after the crib is set... just don't let the player know
        self.shared_card = shared_card

    def remove_card_for_crib(card: np.int8):
        self.hand_cards = remove_card(self.hand_cards)
        return card

    def play_card(card: np.int8, play_index: int):
        self.hand_cards = remove_card(self.hand_cards)
        self.played_cards[play_index] = card

    def set_non_crib_cards(self, non_crib_cards):
        self.non_crib_cards = non_crib_cards

class Player():
    def __init__(self, verbose=100):
        self.points = 0
        self.has_won = False
        self.player_round_state = None
        self.verbose = verbose

    def increase_score(self, points):
        self.points += points
        if self.points >= 121:
            self.has_won = True

    def set_player_round_state(self, player_round_state):
        self.player_round_state = player_round_state

    # select cards for the crib... for now do it randomly
    def select_cards_for_crib():
        if self.verbose >= 100:
            print(f'is_host={self.player_round_state.is_host}, selecting cards for crib')

        crib_cards = np.random.choice(self.player_round_state.hand_cards, size=2, replace=False)
        for crib_card in crib_cards:
            self.player_round_state.remove_card_for_crib(crib_card)

        if self.verbose >= 100:
            print(f'is_host={self.player_round_state.is_host}, picked cards for crib: {crib_cards}')

        self.player_round_state.set_non_crib_cards(np.copy(self.player_round_state.hand_cards))
        
        return crib_cards

    # for now, pick a random card
    def play_card(current_total: np.int8):
        # if we have no cards, pass
        if len(self.player_round_state.hand_cards) == 0:
            return PASS
        
        np.random.shuffle(self.player_round_state.hand_cards)

        # check through each card to see if the total is under 31
        for index, hand_card in enumerate(self.player_round_state.hand_cards):
            card_value = card.get_value(hand_card)
            if card_value + current_total <= 31:
                if self.verbose >= 100:
                    print(f'is_host={self.player_round_state.is_host}, played card: {play_card}')
                self.player_round_state.remove_card_for_crib(crib_card)
                return play_card

        # if we can't find a card, just pass
        return PASS

class RoundState():
    def __init__(self, host_player, non_host_player):
        self.host_player = host_player
        self.non_host_player = non_host_player
        
        self.round_state = GAME_STATES['INIT']        
        self.deck = None
        self.crib_cards = None
        self.shared_card = None
        self.play_card_index = None
        self.play_card_current_run_index = None
        self.play_card_total = None
        self.play_history = None

    def play_round(self):
        routines = [
            self.init_state,
            self.select_crib,
            self.reveal_shared_card,
            self.play_cards,
            self.score_results,
        ]

        for routine in routines:
            routine()

            if self.round_state == GAME_STATES['ROUND_OVER']:
                if self.verbose >= 100:
                    print('round is over')
                break
    
    def init_state(self):
        if self.round_state != GAME_STATES['INIT']:
            raise RuntimeError(f"wrong state: {self.round_state}")

        # shuffle the deck
        self.deck = np.array(card.get_deck(), dtype=np.int8)
        np.random.shuffle(self.deck)

        # deal the cards
        played_cards = np.zeros(8, dtype=np.int8)
        host_cards = self.deck[:6]
        non_host_cards = self.deck[6:12]
        shared_card = self.deck[12]

        host_player_round_state = PlayerRoundState(host_cards, played_cards, shared_card)
        non_host_player_round_state = PlayerRoundState(non_host_cards, played_cards, shared_card)

        self.host_player.set_player_round_state(host_player_round_state)
        self.non_host_player.set_player_round_state(non_host_player_round_state)
        self.shared_card = shared_card

        self.round_state = GAME_STATES['DEALT']

    def select_crib(self):
        if self.round_state != GAME_STATES['DEALT']:
            raise RuntimeError(f"wrong state: {self.round_state}")

        host_crib_cards = self.host_player.select_cards_for_crib()
        non_host_crib_cards = self.non_host_player.select_cards_for_crib()

        self.crib_cards = np.concatenate(host_crib_cards, non_host_crib_cards)
        self.round_state = GAME_STATES['CRIB_SELECTED']

    def reveal_shared_card(self):
        if self.round_state != GAME_STATES['CRIB_SELECTED']:
            raise RuntimeError(f"wrong state: {self.round_state}")

        # TODO: double check that the right player is getting the points incremented
        if card.get_rank == 10:
            self.host_player.increase_score(2)
            if self.host_player.has_won:
                self.round_state = GAME_STATES['ROUND_OVER']
                return

        self.round_state = GAME_STATES['SHARED_CARD_REVEALED']

    def play_cards(self):
        if self.round_state != GAME_STATES['SHARED_CARD_REVEALED']:
            raise RuntimeError(f"wrong state: {self.round_state}")
        
        self.play_card_index = 0
        self.play_card_current_run_index = 0
        self.play_card_total = 0
        self.play_history = []
        self.turn_index = 0

        # this shouldn't last more than 20 turns...
        for i in range(20):
            # person without the crib goes first
            current_player = self.non_host_player if self.turn_index % 2 == 0 else self.host_player

            player_card = current_player.play_card(self.play_card_total)
            self.play_history.append(player_card)
            if self.verbose >= 100:
                print(f'is_host={current_player.player_round_state.is_host}, played card: {play_card}')

            # play a card, add scores
            if player_card != PASS:
                self.play_card_total += card.get_value(player_card)
                self.play_card_index += 1

                # score things
                play_cards = current_player.player_round_state.played_cards[self.play_card_current_run_index: self.play_card_index]
                for score_method in [scoring.score_play_run, scoring.score_play_pairs, scoring.score_play_values]:
                    current_player.increase_score(score_method(play_cards))
                    if current_player.has_won:
                        self.round_state = GAME_STATES['ROUND_OVER']
                        return
            # if there are two passes, reset some things and try again
            else:
                if len(self.play_history) >= 2 and self.play_history[-2] == PASS:
                    next_player = self.host_player if self.turn_index % 2 == 0 else self.non_host_player

                    # two passes, give the other player a point for go if we didn't give them 2 for 31
                    if self.player_card_total != 31:
                        next_player.increase_score(1)
                        if next_player.has_won:
                            self.round_state = GAME_STATES['ROUND_OVER']
                            return

                    self.play_card_current_run_index = self.play_card_index

                    # if we've played all 8 cards and checked for passes, it's time to finish things up
                    if self.play_card_index == 8:
                        break
            # how many turns did we take
            self.turn_index += 1

        if self.play_card_index != 8:
            raise RuntimeError(f"too many turns without a conclusion, something is  {self.play_card_index}")
            
        self.round_state = GAME_STATES['ALL_CARDS_PLAYED']

    def score_results(self):
        if self.round_state != GAME_STATES['ALL_CARDS_PLAYED']:
            raise RuntimeError(f"wrong state: {self.round_state}")

        non_host_score = scoring.compute_value(self.shared_card, self.non_host_player.player_round_state.non_crib_cards, False)
        self.non_host_player.increase_score(non_host_score)
        if self.non_host_player.has_won:
            self.round_state = GAME_STATES['ROUND_OVER']
            return
        else:
            self.round_state = GAME_STATES['NON_HOST_HAND_SCORED']

        host_score = scoring.compute_value(self.shared_card, self.host_player.player_round_state.non_crib_cards, False)
        self.host_player.increase_score(host_score)
        if self.host_player.has_won:
            self.round_state = GAME_STATES['ROUND_OVER']
            return
        else:
            self.round_state = GAME_STATES['HOST_HAND_SCORED']

        crib_score = scoring.compute_value(self.shared_card, self.crib_cards, True)
        self.host_player.increase_score(crib_score)
        if self.host_player.has_won:
            self.round_state = GAME_STATES['ROUND_OVER']
            return
        else:
            self.round_state = GAME_STATES['HOST_CRIB_SCORED']

    def finish_round(self):
        if self.round_state != GAME_STATES['HOST_CRIB_SCORED']:
            raise RuntimeError(f"wrong state: {self.round_state}")
        self.round_state = GAME_STATES['ROUND_OVER']

class GameState():
    def __init__(self):
        # TODO: implement host picking
        pass



