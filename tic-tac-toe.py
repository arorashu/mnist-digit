#!/usr/bin/env python
# coding: utf-8

# ## Train an agent to play tic tac toe
# 
# ### Strategies
# 1. Play at random
# 2. Ideal Player
# 3. Imitation learning
# 

# In[1]:


import itertools
import random


# In[2]:


COMPUTER = False
HUMAN = True

class Player():
    HUMAN = 0
    RANDOM = 1
    EXPERT = 2
    STUDENT = 3


class Entry():
    Empty = '-'
    X = 'X'
    O = 'O'
    

class TicTacToe():
    """
    define the game class
    COMPUTER always plays an 'O'
    HUMAN always plays a 'X'
    """
    
    MNMX_MOVES = 0
    
    def __init__(self):
        """
        turn = False -> computer's turn, True-> human turn
        """
        self.state = ['-'] * 9
        self.turn = random.choice([COMPUTER, HUMAN])
        self.game_ended = False
        self.winner = Entry.Empty
        self.computer_player = Player.EXPERT
     
    def __str__(self):
        x = str(self.state[0:3]) + '\n' + str(self.state[3:6]) + '\n' \
          + str(self.state[6:9])
        return (f'board state: \n{x}\n' +
                f'player turn: {self.turn}\n' +
                f'game ended: {self.game_ended}\n' +
                f'winner: {self.winner}\n')
    
    def pretty_state(self, state):
        x = str(self.state[0:3]) + '\n' + str(self.state[3:6]) + '\n' \
          + str(self.state[6:9])
        return x
    
    
    def play(self):
        print('play a turn')
        if self.game_ended:
            print('Game Over')
            print(self)
            return
        avail_positions = []
        for i, x in enumerate(self.state):
            if x == Entry.Empty: avail_positions.append(i)
        if len(avail_positions) == 0:
            self.game_ended = True
            self.winner = 'DRAW'
            print('board is full')
            return
        if self.turn == COMPUTER:
            print('COMPUTER to play')
            print(f'available positions: {avail_positions}')
            if self.computer_player == Player.RANDOM:
                play_id = random.choice(avail_positions) 
                print(play_id)
                self.state[play_id] = Entry.O
            elif self.computer_player == Player.EXPERT:
                play_id = self.play_pro()
                self.state[play_id] = Entry.O
        elif self.turn == HUMAN:
            print('HUMAN to play')
            self.user_input_prompt()
            valid_input = False
            while not valid_input:
                inp = input('where do you wanna play [0-9]?')
                if str.isdigit(inp): valid_input = True
                if valid_input:
                    pos = int(inp)
                    if pos not in avail_positions:
                        valid_input = False
                if not valid_input:
                    print('invalid input')
                    print(f'please enter a number from the list: {avail_positions}')
            # got a valid position to play
            self.state[pos] = Entry.X
        
        self.evaluate()
        self.turn = not self.turn
        print(self)
        
    def play_pro(self):
        """
        play as an expert(pro)
        using minimax
        """
        state_copy = self.state.copy()
        self.MNMX_MOVES = 0
        best_move, best_score = self._minimax(state_copy, COMPUTER)
        print(f'minimax moves taken: {self.MNMX_MOVES}')
        return best_move
        
    def _evaluate(self, state):
        """
        evaluate state, returns game_ended
        """
        rows = [self.state[k:k+3] for k in range(0, 9, 3)]
        cols = [[self.state[k], self.state[k+3], self.state[k+6]]
                 for k in range(0, 3, 1)]
        diags = [[self.state[0], self.state[4], self.state[8]],
                 [self.state[2], self.state[4], self.state[6]]]
        arrs = [rows, cols, diags]
        for arr in itertools.chain(*arrs):
            if (arr[0] != Entry.Empty
                    and arr[0] == arr[1]
                    and arr[0] == arr[2]):
                return True
        return False
        
    def _minimax(self, state, player):
        self.MNMX_MOVES += 1
#         print(f'enter mnmx with state:\n{self.pretty_state(state)}')
        empty_pos = self.get_available_pos(state)
        if len(empty_pos) == 0:
            print('no moves available. exiting!')
            print(f'player: {player}')
        new_state = self.state
        best_score = -100
        best_move = -1
        for pos in empty_pos:
#             print(f'make move: {pos}')
            if player == COMPUTER: new_state[pos] = Entry.O
            else: new_state[pos] = Entry.X
            if self._evaluate(new_state): # played the winning move
#                 print('winning minimax move')
#                 print(f'player: {player}, state:\n{state}')
#                 return pos, 10
                cur_score = 10
            else:
                cur_score = -100
                if len(empty_pos) == 1: # draw state, last move
                    cur_score = 0
                else:
                    # play more
                    _, opp_score = self._minimax(new_state, not player)
                    cur_score = -opp_score
            if cur_score > best_score:
                best_score = cur_score
                best_move = pos
            # reset state
            new_state[pos] = Entry.Empty
#             print(f'UNDO move: {pos}')
#         print(f'player: {player}, best_move = {pos}, best_score = {best_score}')
#         print(f'exit mnmx with state:\n{self.pretty_state(state)}')
        return best_move, best_score
        
    def evaluate(self):
        """
        evaluate if there is a winner
        if game ended, update `game_ended` and `winner`
        """
        win = False
        # check rows
        rows = [self.state[k:k+3] for k in range(0, 9, 3)]
        cols = [[self.state[k], self.state[k+3], self.state[k+6]]
                 for k in range(0, 3, 1)]
        diags = [[self.state[0], self.state[4], self.state[8]],
                 [self.state[2], self.state[4], self.state[6]]]
        arrs = [rows, cols, diags]
        for arr in itertools.chain(*arrs):
            if (arr[0] != Entry.Empty
                    and arr[0] == arr[1]
                    and arr[0] == arr[2]):
                win = True
                print(f'winning row: {arr}')
                break
        if win:
            print('we have a winner')
            if self.turn: self.winner = "HUMAN"
            else: self.winner = "COMPUTER"
            self.game_ended = True
            
    def get_available_pos(self, state):
        avail_positions = []
        for i, x in enumerate(state):
            if x == Entry.Empty: avail_positions.append(i)
        return avail_positions
    
    def get_state(self):
        state = 0
        for i in range(9):
            s = self.state[i]
            val = 0
            if s == Entry.X:
                val = 0x3
            elif s == Entry.O:
                val = 0x2
            state |= val << (i*2)
        return state
        
    def user_input_prompt(self):
        """
        shows prompt human user to get position to play
        """
        prompt = ''
        for i, x in enumerate(self.state):
            prompt += f'[{i}| {x}]'
            if (i+1) % 3 == 0: prompt += '\n'
        
        print(f'board state: \n{prompt}\n')
        
    def reset(self):
        self.state = ['-'] * 9
        self.turn = random.choice([COMPUTER, HUMAN])
        self.game_ended = False
        self.winner = Entry.Empty
    


# In[3]:


game = TicTacToe()
def play_new_game(game):
    print(f'old game state: {game}')
    game.reset()
    while not game.game_ended:
        game.play()
        
    print('done.')

play_new_game(game)


