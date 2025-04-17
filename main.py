import random
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt

class UnoEnv:
    def __init__(self, hand_size=5):
        self.colors = ['Red', 'Blue', 'Green', 'Yellow']
        self.numbers = list(range(10)) * 2
        self.actions = ['Skip', 'Reverse', 'Draw Two'] * 2
        self.hand_size = hand_size
        self.max_hand_size = 20
        self.max_turns = 500
        self.reset()

    def reset(self):
        self.deck = self._create_deck()
        random.shuffle(self.deck)

        self.player_hand = [self.deck.pop() for _ in range(self.hand_size)]
        self.opponent_hand = [self.deck.pop() for _ in range(self.hand_size)]

        self.discard_pile = [self.deck.pop()]
        while self.discard_pile[-1].split()[1] in ['Skip', 'Reverse', 'Draw Two']:
            self.deck.insert(0, self.discard_pile.pop())
            self.discard_pile.append(self.deck.pop())

        self.current_player = 0
        self.done = False
        self.turn_count = 0
        return self._get_state()

    def _create_deck(self):
        deck = []
        for color in self.colors:
            for number in self.numbers:
                deck.append(f"{color} {number}")
            for action in self.actions:
                deck.append(f"{color} {action}")
        return deck

    def _get_state(self) -> Dict:
        return {
            'player_hand': tuple(sorted(self.player_hand)),
            'opponent_hand_size': len(self.opponent_hand),
            'top_card': self.discard_pile[-1],
            'current_player': self.current_player,
            'deck_size': len(self.deck),
            'can_play': any(self._is_valid_move(card) for card in self.player_hand)
        }

    def _is_valid_move(self, card: str) -> bool:
        if card is None:
            return False
        top_card = self.discard_pile[-1].split()
        card_parts = card.split()
        return (card_parts[0] == top_card[0] or card_parts[1] == top_card[1])

    def step(self, action) -> Tuple[Dict, float, bool, Dict]:
        if self.done:
            raise ValueError("Game is over. Call reset() to start a new game.")

        reward = 0
        self.turn_count += 1

        if self.turn_count > self.max_turns:
            self.done = True
            return self._get_state(), -0.5, self.done, {'reason': 'max_turns'}

        if action == 'draw':
            if any(self._is_valid_move(card) for card in self.player_hand):
                reward = -0.3

            if not self.deck:
                self._reshuffle_discard_pile()
                if not self.deck:
                    self.done = True
                    return self._get_state(), -1.0, self.done, {'reason': 'deck_empty'}

            drawn_card = self.deck.pop()
            self.player_hand.append(drawn_card)
            reward += 0.1 if self._is_valid_move(drawn_card) else -0.1
        else:
            try:
                card = self.player_hand[action]
                if not self._is_valid_move(card):
                    reward = -0.5
                else:
                    self.player_hand.pop(action)
                    self.discard_pile.append(card)
                    reward += 0.3

                    card_value = card.split()[1]
                    if card_value == 'Skip':
                        reward += 0.5
                    elif card_value == 'Reverse':
                        pass
                    elif card_value == 'Draw Two':
                        self._force_opponent_draw(2)
                        reward += 0.7

                    if len(self.player_hand) == 1:
                        reward += 1.0
            except IndexError:
                reward = -1.0

        if not self.player_hand:
            self.done = True
            reward = 10.0
            return self._get_state(), reward, self.done, {'reason': 'player_win'}

        self._opponent_turn()

        if not self.opponent_hand:
            self.done = True
            reward = -5.0
            return self._get_state(), reward, self.done, {'reason': 'opponent_win'}

        reward += self._calculate_strategic_rewards()

        return self._get_state(), reward, self.done, {}

    def _reshuffle_discard_pile(self):
        if len(self.discard_pile) > 1:
            self.deck = self.discard_pile[:-1]
            random.shuffle(self.deck)
            self.discard_pile = [self.discard_pile[-1]]

    def _force_opponent_draw(self, num_cards: int):
        for _ in range(num_cards):
            if not self.deck:
                self._reshuffle_discard_pile()
                if not self.deck:
                    break
            self.opponent_hand.append(self.deck.pop())

    def _opponent_turn(self):
        valid_moves = [i for i, card in enumerate(self.opponent_hand)
                       if self._is_valid_move(card)]

        if valid_moves:
            action_cards = [i for i in valid_moves
                            if self.opponent_hand[i].split()[1] in ['Skip', 'Draw Two']]
            if action_cards:
                played_card = self.opponent_hand.pop(action_cards[0])
            else:
                played_card = self.opponent_hand.pop(valid_moves[-1])

            self.discard_pile.append(played_card)
            card_value = played_card.split()[1]
            if card_value == 'Skip':
                return
            elif card_value == 'Draw Two':
                self._force_opponent_draw(2)
        else:
            if self.deck:
                drawn_card = self.deck.pop()
                self.opponent_hand.append(drawn_card)
                if self._is_valid_move(drawn_card):
                    self.opponent_hand.remove(drawn_card)
                    self.discard_pile.append(drawn_card)

        self.current_player = 0

    def _calculate_strategic_rewards(self) -> float:
        reward = 0.0
        card_diff = len(self.opponent_hand) - len(self.player_hand)
        reward += card_diff * 0.05
        if len(self.player_hand) > 5:
            reward -= 0.02 * (len(self.player_hand) - 5)
        return reward


class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.95,
                 exploration_rate=1.0, exploration_decay=0.9995,
                 min_exploration=0.01):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration
        self.q_table = defaultdict(lambda: np.zeros(env.max_hand_size + 1))
        self.training_stats = {
            'episodes': [],
            'rewards': [],
            'exploration_rates': [],
            'win_rates': []
        }

    def _state_to_key(self, state: Dict) -> Tuple:
        return (
            tuple(sorted(state['player_hand'])),
            state['top_card'],
            state['opponent_hand_size'],
            state['can_play']
        )

    def get_action(self, state: Dict) -> int:
        possible_actions = len(state['player_hand']) + 1  # +1 for draw

        if np.random.random() < self.exploration_rate:
            return np.random.randint(0, possible_actions)
        else:
            state_key = self._state_to_key(state)
            q_values = self.q_table[state_key][:possible_actions]

            if state['can_play']:
                q_values[-1] = -np.inf if any(self.env._is_valid_move(card)
                                              for card in state['player_hand']) else q_values[-1]

            return np.argmax(q_values)

    def update(self, state: Dict, action: int, reward: float,
               next_state: Dict, done: bool):
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)

        current_q = self.q_table[state_key][action]
        max_next_q = np.max(self.q_table[next_state_key]) if not done else 0

        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * max_next_q)
        self.q_table[state_key][action] = new_q

        if done:
            self.exploration_rate = max(self.min_exploration, self.exploration_rate * self.exploration_decay)

    def train(self, num_episodes=1000):
        wins = 0

        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.get_action(state)
                next_state, reward, done, info = self.env.step(action)
                self.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

            if info.get('reason') == 'player_win':
                wins += 1

            # Logging training stats every 100 episodes
            if (episode + 1) % 100 == 0:
                win_rate = wins / (episode + 1)
                self.training_stats['episodes'].append(episode + 1)
                self.training_stats['rewards'].append(total_reward)
                self.training_stats['exploration_rates'].append(self.exploration_rate)
                self.training_stats['win_rates'].append(win_rate)
                print(f"Episode {episode + 1}: Win rate: {win_rate:.2f}, Exploration rate: {self.exploration_rate:.3f}")

        final_win_rate = wins / num_episodes
        print(f"\nFinal Win Rate after {num_episodes} episodes: {final_win_rate:.2f}")
        return final_win_rate
    
env = UnoEnv()
agent = QLearningAgent(env)
final_win_rate = agent.train(num_episodes=5000)

episodes = agent.training_stats['episodes']
win_rates = agent.training_stats['win_rates']
exploration_rates = agent.training_stats['exploration_rates']

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(episodes, win_rates, label='Win Rate')
plt.xlabel('Episodes')
plt.ylabel('Win Rate')
plt.title('Win Rate over Time')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(episodes, exploration_rates, label='Exploration Rate', color='orange')
plt.xlabel('Episodes')
plt.ylabel('Exploration Rate')
plt.title('Exploration Decay over Time')
plt.legend()

plt.tight_layout()
plt.show()


