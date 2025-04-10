import random
from collections import defaultdict

class UnoEnv:
    def __init__(self, hand_size=5):
        self.colors = ['Red', 'Blue', 'Green', 'Yellow']
        self.numbers = list(range(10)) + ['Skip', 'Reverse', 'Draw Two']
        self.hand_size = hand_size
        self.max_hand_size = 20  # Set a reasonable maximum hand size
        self.reset()
    
    def reset(self):
        """Initialize a new game"""
        self.deck = self._create_deck()
        random.shuffle(self.deck)
        
        self.player_hand = [self.deck.pop() for _ in range(self.hand_size)]
        self.opponent_hand = [self.deck.pop() for _ in range(self.hand_size)]
        
        self.discard_pile = [self.deck.pop()]
        while ' '.join(self.discard_pile[-1].split()[1:]) in ['Skip', 'Reverse', 'Draw Two']:
            self.deck.insert(0, self.discard_pile.pop())
            self.discard_pile.append(self.deck.pop())
        
        self.current_player = 0  # 0 for player, 1 for opponent
        self.done = False
        self.turn_count = 0  # Initialize turn counter
        return self._get_state()
    
    def _create_deck(self):
        """Create a standard Uno deck (simplified)"""
        deck = []
        for color in self.colors:
            for number in self.numbers:
                deck.append(f"{color} {number}")
        return deck * 2  # Two of each card
    
    def _get_state(self):
        """Return current game state"""
        return {
            'player_hand': tuple(sorted(self.player_hand)),
            'opponent_hand': len(self.opponent_hand),  # Only track opponent hand size for state
            'top_card': self.discard_pile[-1],
            'current_player': self.current_player
        }
    
    def _is_valid_move(self, card):
        """Check if a card can be played"""
        if card is None:  # Draw action
            return False
            
        top_parts = self.discard_pile[-1].split()
        top_color = top_parts[0]
        top_value = ' '.join(top_parts[1:])
        
        card_parts = card.split()
        card_color = card_parts[0]
        card_value = ' '.join(card_parts[1:])
        
        return card_color == top_color or card_value == top_value
    
    def step(self, action):
        """Execute one action in the environment"""

        print(f"\nTurn {self.turn_count}")
        print(f"Player hand size: {len(self.player_hand)}")
        print(f"Opponent hand size: {len(self.opponent_hand)}")
        print(f"Deck size: {len(self.deck)}")
        print(f"Discard size: {len(self.discard_pile)}")

        if self.done:
            raise ValueError("Game is over, call reset() to start a new game")
        
        reward = 0  # Initialize reward
        self.turn_count += 1
        
        # Check for maximum turns
        if self.turn_count > 500:
            self.done = True
            return self._get_state(), -0.1, self.done, {}
        
        if action == 'draw':
            # Check if player could have played instead of drawing
            if any(self._is_valid_move(card) for card in self.player_hand):
                reward = -0.5  # Penalty for drawing when valid moves exist
            
            # Handle deck reshuffling if needed
            if not self.deck:
                if len(self.discard_pile) > 1:
                    self.deck = self.discard_pile[:-1]
                    random.shuffle(self.deck)
                    self.discard_pile = [self.discard_pile[-1]]
                else:
                    self.done = True
                    return self._get_state(), -0.5, self.done, {}
            
            drawn_card = self.deck.pop()
            self.player_hand.append(drawn_card)
        else:
            # Player plays a card
            try:
                card = self.player_hand[action]
                if not self._is_valid_move(card):
                    reward = -0.5  # Penalty for illegal move
                else:
                    self.player_hand.pop(action)
                    self.discard_pile.append(card)
                    
                    # Handle special cards
                    card_value = ' '.join(card.split()[1:])
                    if card_value == 'Skip':
                        self.current_player = 0  # Player gets another turn
                    elif card_value == 'Reverse':
                        pass  # No effect in 2-player game
                    elif card_value == 'Draw Two':
                        # Opponent draws two cards
                        for _ in range(2):
                            if not self.deck:
                                if len(self.discard_pile) > 1:
                                    self.deck = self.discard_pile[:-1]
                                    random.shuffle(self.deck)
                                    self.discard_pile = [self.discard_pile[-1]]
                                else:
                                    break
                            if self.deck:
                                self.opponent_hand.append(self.deck.pop())
                        reward = 0.2  # Reward for forcing opponent to draw
            except IndexError:
                reward = -1.0  # Penalty for invalid action
        
        # Check if player won
        if not self.player_hand:
            self.done = True
            reward = 1.0
            return self._get_state(), reward, self.done, {}
        
        # Opponent's turn (simple rule-based AI)
        self._opponent_turn()
        
        # Check if opponent won
        if not self.opponent_hand:
            self.done = True
            reward = -1.0
            return self._get_state(), reward, self.done, {}
        
        # Check for stalemate
        if (not any(self._is_valid_move(card) for card in self.player_hand) and 
            not any(self._is_valid_move(card) for card in self.opponent_hand) and
            not self.deck and len(self.discard_pile) == 1):
            self.done = True
            reward = -0.1
            return self._get_state(), reward, self.done, {}
        
        return self._get_state(), reward, self.done, {}
    
    def _opponent_turn(self):
        """Simple rule-based opponent"""
        valid_moves = [i for i, card in enumerate(self.opponent_hand) 
                      if self._is_valid_move(card)]
        
        if valid_moves:
            # Play first valid card (could be made smarter)
            played_card = self.opponent_hand.pop(valid_moves[0])
            self.discard_pile.append(played_card)
            
            # Handle special cards
            card_value = ' '.join(played_card.split()[1:])
            if card_value == 'Skip':
                return  # Opponent gets another turn
            elif card_value == 'Reverse':
                pass  # No effect in 2-player game
            elif card_value == 'Draw Two':
                # Player draws two cards
                for _ in range(2):
                    if self.deck:
                        self.player_hand.append(self.deck.pop())
        else:
            # Draw a card
            if self.deck:
                self.opponent_hand.append(self.deck.pop())

            else:
            # Can't draw, can't play - stalemate will be detected in step()
                pass

        self.current_player = 0  # Switch back to player


import numpy as np
from collections import defaultdict

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, 
                 exploration_rate=1.0, exploration_decay=0.995):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        # Initialize Q-table with max possible actions
        self.q_table = defaultdict(lambda: np.zeros(env.max_hand_size + 1))  # +1 for draw action
    
    def _state_to_key(self, state):
        """Convert state to a hashable key for Q-table"""
        return (state['player_hand'], state['top_card'], state['opponent_hand'])
    
    def get_action(self, state):
        """Select action using ε-greedy policy"""
        possible_actions = len(state['player_hand']) + 1  # +1 for draw
        
        if np.random.random() < self.exploration_rate:
            # Explore: random valid action
            return np.random.randint(0, possible_actions)
        else:
            # Exploit: best known action (only considering valid actions)
            state_key = self._state_to_key(state)
            q_values = self.q_table[state_key][:possible_actions]  # Only consider valid actions
            return np.argmax(q_values)
    
    def update(self, state, action, reward, next_state, done):
        """Update Q-table using Q-learning update rule"""
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        
        possible_next_actions = len(next_state['player_hand']) + 1 if not done else 0
        
        current_q = self.q_table[state_key][action]
        max_next_q = np.max(self.q_table[next_state_key][:possible_next_actions]) if not done else 0
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        self.q_table[state_key][action] = new_q
        
        # Decay exploration rate
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(0.01, self.exploration_rate)  # Keep some exploration

    def train_agent(self, env, agent, episodes=1):
        wins = 0
        losses = 0
    
        for episode in range(1, episodes + 1):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                # Get action from agent
                action = agent.get_action(state)
                
                # Map action index to game action
                if action == len(state['player_hand']):  # Last action is draw
                    game_action = 'draw'
                else:
                    game_action = action
                
                # Execute action
                next_state, reward, done, _ = env.step(game_action)
                total_reward += reward
                
                # Update agent
                agent.update(state, action, reward, next_state, done)
                state = next_state
                
                # Track wins/losses
                if done:
                    if reward == 1.0:
                        wins += 1
                    else:
                        losses += 1
            
            # Print progress
            if episode % 1000 == 0:
                print(f"Episode {episode}:")
                print(f"  Exploration rate: {agent.exploration_rate:.3f}")
                print(f"  Win rate: {wins / 1000:.2f}")
                print(f"  Loss rate: {losses / 1000:.2f}")
                wins = 0
                losses = 0
        
        return agent

def play_against_agent(env, agent, num_games=10):
    wins = 0
    for _ in range(num_games):
        state = env.reset()
        done = False
        while not done:
            if state['current_player'] == 0:  # Agent's turn
                possible_actions = len(state['player_hand']) + 1
                q_values = agent.q_table[agent._state_to_key(state)][:possible_actions]
                action = np.argmax(q_values)
                if action == len(state['player_hand']):
                    game_action = 'draw'
                else:
                    game_action = action
            else:  # Opponent's turn
                # Use the same rule-based opponent as during training
                valid_moves = [i for i, card in enumerate(env.opponent_hand) 
                            if env._is_valid_move(card)]
                game_action = valid_moves[0] if valid_moves else 'draw'
            
            state, _, done, _ = env.step(game_action)
        
        if not env.player_hand:  # Agent won
            wins += 1
    
    print(f"Agent won {wins}/{num_games} games against rule-based opponent")

if __name__ == "__main__":
    # Initialize environment and agent
    env = UnoEnv(hand_size=5)
    agent = QLearningAgent(env)
    
    # Train the agent
    print("Starting training...")
    trained_agent = agent.train_agent(env, agent, episodes=10000)
    print("Training completed!")
    
    # Test the agent
    play_against_agent(env, trained_agent)