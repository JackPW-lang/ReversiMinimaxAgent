
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

class TranspositionTable:
    def __init__(self):
        self.table = {}

    def store(self, board_hash, value, depth, flag):
        # Store the evaluation, depth, and a flag (e.g., exact, upper bound, lower bound)
        self.table[board_hash] = {"value": value, "depth": depth, "flag": flag}

    def lookup(self, board_hash):
        # Return the stored value if it exists
        return self.table.get(board_hash, None)
@register_agent("student_agent")
class StudentAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(StudentAgent, self).__init__()
    self.name = "StudentAgent"
    self.transposition_table = TranspositionTable()

  def step(self, chess_board, player, opponent):
    """
    Implement the step function of your agent here.
    You can use the following variables to access the chess board:
    - chess_board: a numpy array of shape (board_size, board_size)
      where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
      and 2 represents Player 2's discs (Brown).
    - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
    - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).

    You should return a tuple (r,c), where (r,c) is the position where your agent
    wants to place the next disc. Use functions in helpers to determine valid moves
    and more helpful tools.

    Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
    """
    start_time = time.time()
    time_limit = 1.9 #1.95  # Set the time limit to 1.95 seconds
    valid_moves=get_valid_moves(chess_board,player)
    best_move = order_moves(chess_board,valid_moves,player, opponent)[0]
    depth = 1  # Start with a shallow depth

    while time.time() - start_time < time_limit:
        _, move = self.minimax(chess_board, depth, True, player, opponent, -float('inf'), float('inf'), start_time, time_limit)
        if move:  # If a valid move is found, update best_move
            best_move = move
        if time.time() - start_time >= time_limit:
            break
        depth += 1  # Increment depth for the next iteration

    time_taken = time.time() - start_time
    print(f"My AI's turn took {time_taken:.6f} seconds.")
    return best_move

  def minimax(self, board, depth, maximizing_player, player, opponent, alpha, beta, start_time, time_limit):
      board_hash = hash(board.tostring())  # Generate a unique hash for the board state

      # Check transposition table
      cached = self.transposition_table.lookup(board_hash)
      if cached and cached['depth'] >= depth:
          if cached['flag'] == "exact":
              return cached['value'], None
          elif cached['flag'] == "lower_bound":
              alpha = max(alpha, cached['value'])
          elif cached['flag'] == "upper_bound":
              beta = min(beta, cached['value'])
          if alpha >= beta:
              return cached['value'], None

      # Time check
      if time.time() - start_time >= time_limit:
          return None, None

      # End condition check
      is_endgame, _, _ = check_endgame(board, player, opponent)
      if depth == 0 or is_endgame or time.time() - start_time >= time_limit:
          value = evaluate_board(board, player, opponent)
          self.transposition_table.store(board_hash, value, depth, "exact")
          return value, None

      valid_moves = get_valid_moves(board, player if maximizing_player else opponent)
      if not valid_moves:  # Pass turn if no valid moves
          if maximizing_player:
              return self.minimax(board, depth, False, player, opponent, alpha, beta, start_time, time_limit)
          else:
              return self.minimax(board, depth, True, player, opponent, alpha, beta, start_time, time_limit)

      if maximizing_player:
          max_eval = -float("inf")
          best_move = None
          for move in order_moves(board, valid_moves, player, opponent):
              if time.time() - start_time >= time_limit:
                  return None, None

              new_board = deepcopy(board)
              execute_move(new_board, move, player)
              eval, _ = self.minimax(new_board, depth - 1, False, player, opponent, alpha, beta, start_time, time_limit)
              if eval is None:
                  return None, None

              if eval > max_eval:
                  max_eval = eval
                  best_move = move
              alpha = max(alpha, eval)
              if beta <= alpha:
                  break
          self.transposition_table.store(board_hash, max_eval, depth, "exact" if alpha >= beta else "lower_bound")
          return max_eval, best_move
      else:
          min_eval = float("inf")
          best_move = None
          for move in order_moves(board, valid_moves, opponent, player):
              if time.time() - start_time >= time_limit:
                  return None, None

              new_board = deepcopy(board)
              execute_move(new_board, move, opponent)
              eval, _ = self.minimax(new_board, depth - 1, True, player, opponent, alpha, beta, start_time, time_limit)
              if eval is None:
                  return None, None

              if eval < min_eval:
                  min_eval = eval
                  best_move = move
              beta = min(beta, eval)
              if beta <= alpha:
                  break
          self.transposition_table.store(board_hash, min_eval, depth, "exact" if beta <= alpha else "upper_bound")
          return min_eval, best_move


'''

def maximize(chess_board, depth, player, opponent, alpha, beta, start_time, time_limit):
    """
    Maximizing player logic for the minimax algorithm.
    Returns:
        Tuple (max_eval, best_move)
    """
    if time.time() - start_time >= time_limit:
        return None, None

    is_endgame, player_score, opponent_score = check_endgame(chess_board, player, opponent)
    if depth == 0 or is_endgame or time.time() - start_time >= time_limit:
        return evaluate_board(chess_board, player, opponent), None

    valid_moves = get_valid_moves(chess_board, player)
    if not valid_moves:
        if time.time() - start_time >= time_limit:
            return None, None
        else:
            return minimize(chess_board, depth, opponent, player, alpha, beta, start_time, time_limit)

    max_eval = -float('inf')
    best_move = None

    for move in order_moves(chess_board, valid_moves, player, opponent):
        if time.time() - start_time >= time_limit:
            return None, None

        new_board = deepcopy(chess_board)
        execute_move(new_board, move, player)

        eval_score, _ = minimize(new_board, depth - 1, opponent, player, alpha, beta, start_time, time_limit)

        if eval_score is None:
            return None, None

        if eval_score > max_eval:
            max_eval = eval_score
            best_move = move

        alpha = max(alpha, max_eval)
        if beta <= alpha:
            break

    return max_eval, best_move


def minimize(chess_board, depth, player, opponent, alpha, beta, start_time, time_limit):
    """
    Minimizing player logic for the minimax algorithm.
    Returns:
        Tuple (min_eval, best_move)
    """
    if time.time() - start_time >= time_limit:
        return None, None

    is_endgame, player_score, opponent_score = check_endgame(chess_board, player, opponent)
    if depth == 0 or is_endgame or (time.time() - start_time >= time_limit):
        return evaluate_board(chess_board, player, opponent), None

    valid_moves = get_valid_moves(chess_board, player)
    if not valid_moves:
        if time.time() - start_time >= time_limit:
            return None, None
        else:
            return maximize(chess_board, depth, opponent, player, alpha, beta, start_time, time_limit)

    min_eval = float('inf')
    best_move = None

    for move in order_moves(chess_board, valid_moves, player, opponent):
        if time.time() - start_time >= time_limit:
            return None, None

        # Simulate the move
        new_board = deepcopy(chess_board)
        execute_move(new_board, move, player)

        # Recursive call to maximize
        eval_score, _ = maximize(new_board, depth - 1, opponent, player, alpha, beta, start_time, time_limit)

        if eval_score is None:
            return None, None

        if eval_score < min_eval:
            min_eval = eval_score
            best_move = move

        beta = min(beta, min_eval)
        if beta <= alpha:
            break

    return min_eval, best_move


def minimax(chess_board, depth, maximizing_player, player, opponent, alpha, beta, start_time, time_limit=2):
    """
    Minimax algorithm with Alpha-Beta Pruning and time constraints.
    Returns:
        Tuple (best_score, best_move)
    """
    if time.time() - start_time >= time_limit:
        return None, None

    if maximizing_player:
        return maximize(chess_board, depth, player, opponent, alpha, beta, start_time, time_limit)
    else:
        return minimize(chess_board, depth, player, opponent, alpha, beta, start_time, time_limit)
'''

def order_moves(chess_board, valid_moves, player, opponent):

    corners = [(0, 0), (0, len(chess_board) - 1),
               (len(chess_board) - 1, 0), (len(chess_board) - 1, len(chess_board) - 1)]

    x_squares = [(1, 1), (0,1), (1,0), (1, len(chess_board) - 2), (0, len(chess_board) - 2), (1,len(chess_board)-1),
                 (len(chess_board) - 2, 0),(len(chess_board) - 2, 1),(len(chess_board)-1,1),
                 (len(chess_board) - 2, len(chess_board) - 2), (len(chess_board)-2, len(chess_board)-1),
                 (len(chess_board)-1, len(chess_board) - 2)]

    def heuristic(move):
        board_size = chess_board.shape[0]
        corner_bonus = 0
        edge_bonus=0

        if board_size==1:
            if move in corners:
                return 50000

            top_edge = [(0, i) for i in range(1, board_size - 1)]
            bottom_edge = [(board_size - 1, i) for i in range(1, board_size - 1)]
            left_edge = [(i, 0) for i in range(1, board_size - 1)]
            right_edge = [(i, board_size - 1) for i in range(1, board_size - 1)]
            all_edges = top_edge + bottom_edge + left_edge + right_edge

            if move in x_squares:
                return -200

            if move in all_edges:
                return 150

        if board_size==6:

            if move in corners:
                corner_bonus= 500

            if move in x_squares and not corner_is_controlled(move, corners, chess_board, player):
                return -50

            new_board = deepcopy(chess_board)
            execute_move(new_board, move, player)
            player_moves = len(get_valid_moves(new_board, player))
            opponent_moves = len(get_valid_moves(new_board, opponent))
            mobility_score = player_moves - opponent_moves
            flip_score = count_capture(chess_board, move, player)

            if get_game_phase(chess_board) == 'late':
                flip_score = flip_score * 15

        else:
            top_edge = [(0, i) for i in range(1, board_size - 1)]
            bottom_edge = [(board_size - 1, i) for i in range(1, board_size - 1)]
            left_edge = [(i, 0) for i in range(1, board_size - 1)]
            right_edge = [(i, board_size - 1) for i in range(1, board_size - 1)]
            all_edges = top_edge + bottom_edge + left_edge + right_edge

            edge_bonus = 200 if move in all_edges else 0

            if move in corners:
                corner_bonus= 100000

            if move in x_squares and not corner_is_controlled(move, corners, chess_board, player):
                return -3000

            new_board = deepcopy(chess_board)
            execute_move(new_board, move, player)
            player_moves = len(get_valid_moves(new_board, player))
            opponent_moves = len(get_valid_moves(new_board, opponent))
            mobility_score = (player_moves - opponent_moves)

            flip_score = count_capture(chess_board, move, player)

            if get_game_phase(chess_board)=='late':
                flip_score=flip_score*15

        return 10*mobility_score + 7*flip_score +corner_bonus +edge_bonus

    return sorted(valid_moves, key=heuristic, reverse=True)


def count_stable_discs(board, player):
    stable = np.zeros(board.shape, dtype=bool)
    corners = [(0, 0), (0, board.shape[0] - 1), (board.shape[0] - 1, 0), (board.shape[0] - 1, board.shape[0] - 1)]


    for corner in corners:
        if board[corner] == player:
            stable[corner] = True


    for direction in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
        for r in range(board.shape[0]):
            for c in range(board.shape[1]):
                if board[r, c] == player and is_adjacent_to_stable((r, c), stable):
                    stable[r, c] = True

    return np.sum(stable)

def is_adjacent_to_stable(cell, stable):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
    r, c = cell
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < stable.shape[0] and 0 <= nc < stable.shape[1] and stable[nr, nc]:
            return True
    return False


def potential_mobility(board, opponent):
    """
    Calculate the potential mobility for the player.
    Potential mobility is the number of empty squares adjacent to opponent discs.
    """
    board_size = len(board)
    potential_mobility = 0
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for r in range(board_size):
        for c in range(board_size):
            if board[r, c] == 0:  # Empty square
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < board_size and 0 <= nc < board_size and board[nr, nc] == opponent:
                        potential_mobility += 1
                        break  # Count each square only once

    return potential_mobility  # Normalize

def get_game_phase(chess_board): #this function makes sense
    total_squares = chess_board.size  # Total squares on the board (36 for 6x6)
    #print(total_squares)
    filled_squares = np.count_nonzero(chess_board)  #count filled squares

    if filled_squares <= total_squares * (1 / 3):  # Less than 1/3 full
        return 'early'
    elif filled_squares <= total_squares * (2 / 3):  # Between 1/3 and 2/3 full
        return 'mid'
    else:  # More than 2/3 full
        return 'late'

def corner_is_controlled(x_square, corners, chess_board, player):   #this function makes sense too
    for corner in corners:
        if abs(x_square[0] - corner[0]) <=1 and abs(x_square[1] - corner[1]) <= 1:
            if chess_board[corner] == player:
                return True
    return False

def count_frontier_tiles(chess_board, player):
    board_size = chess_board.shape[0]
    frontier_count = 0
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for r in range(board_size):
        for c in range(board_size):
            if chess_board[r, c] == player:  # Player's disc
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    # Check if adjacent square is empty
                    if 0 <= nr < board_size and 0 <= nc < board_size and chess_board[nr, nc] == 0:
                        frontier_count += 1
                        break  # Count each tile only once as a frontier tile
    return frontier_count

def evaluate_board(chess_board, player, opponent):
    board_size= chess_board.shape[0]
    disc_score=0
    player_discs = np.sum(chess_board == player)
    opponent_discs = np.sum(chess_board == opponent)
    disc_score = 100*(player_discs - opponent_discs)/(player_discs + opponent_discs)
    #disc score


    player_moves = len(get_valid_moves(chess_board, player))
    opponent_moves = len(get_valid_moves(chess_board, opponent))
    if(player_moves+opponent_moves)!=0:
            mobility_score = 100*(player_moves - opponent_moves)/ (player_moves +opponent_moves)
    else:
            mobility_score=0

    corners = [(0, 0), (0, len(chess_board) - 1), (len(chess_board) - 1, 0), (len(chess_board) - 1, len(chess_board) - 1)]
    player_corners = sum(1 for corner in corners if chess_board[corner] == player)
    opponent_corners = sum(1 for corner in corners if chess_board[corner] == opponent)
    if(player_corners+opponent_corners !=0):
        corner_score = 100* (player_corners - opponent_corners)/ (player_corners+opponent_corners)
    else:
        corner_score=0

    player_stable = count_stable_discs(chess_board, player)
    opponent_stable = count_stable_discs(chess_board, opponent)
    stability_score = 0
    if player_stable + opponent_stable > 0:
        stability_score = 100 * (player_stable - opponent_stable) / (player_stable + opponent_stable)


    potential_mobility_score = potential_mobility(chess_board, opponent)

    x_squares = [
        (0, 1), (1, 0), (1, 1),  # Top-left
        (0, board_size - 2), (1, board_size - 2), (1, board_size - 1),  # Top-right
        (board_size - 2, 0), (board_size - 2, 1), (board_size - 1, 1),  # Bottom-left
        (board_size - 2, board_size - 1), (board_size - 1, board_size - 2), (board_size - 2, board_size - 2)
        # Bottom-right
    ]
    #print(x_squares)

    x_square_bonus= sum(
    10 if chess_board[x]==opponent and not corner_is_controlled(x, corners, chess_board, player)
    else 0
    for x in x_squares)


    x_square_penalty = sum(
        -50 if chess_board[x] == player and not corner_is_controlled(x, corners, chess_board, player)
        else 0
        for x in x_squares
        )

    top_edge = [(0, i) for i in range(1, board_size - 1)]
    bottom_edge = [(board_size - 1, i) for i in range(1, board_size - 1)]
    left_edge = [(i, 0) for i in range(1, board_size - 1)]
    right_edge = [(i, board_size - 1) for i in range(1, board_size - 1)]

    # Combine all edge positions
    all_edges = top_edge + bottom_edge + left_edge + right_edge

    # Remove X-squares from the edges
    all_edges = [pos for pos in all_edges if pos not in x_squares]

    # Count edge control
    player_edges = sum(1 for pos in all_edges if chess_board[pos] == player)
    opponent_edges = sum(1 for pos in all_edges if chess_board[pos] == opponent)

    # Calculate edge control score
    if player_edges + opponent_edges > 0:
        edge_control_score = 100 * (player_edges - opponent_edges) / (player_edges + opponent_edges)
    else:
        edge_control_score = 0


    frontier_score=0
    if chess_board.shape[0]!=6:
        player_frontiers = count_frontier_tiles(chess_board, player)
        opponent_frontiers = count_frontier_tiles(chess_board, opponent)
        if player_frontiers + opponent_frontiers != 0:
            frontier_score = -100 * (player_frontiers - opponent_frontiers) / (player_frontiers + opponent_frontiers)

    game_phase = get_game_phase(chess_board)

    if(board_size==6):  #won 49/50 games on random, 8/10 gpt greedy
        if game_phase == 'early':
            weights = {'disc': 2, 'mobility': 5, 'corner': 30, 'stability': 4, 'potential_mobility': 3,
                       'x_square_penalty': 30, 'frontier':0}
        elif game_phase == 'mid':
            weights = {'disc': 5, 'mobility': 4, 'corner': 50, 'stability': 8, 'potential_mobility': 3,
                       'x_square_penalty': 30, 'frontier':0}
        else:  # Late game
            weights = {'disc': 15, 'mobility': 2, 'corner': 20, 'stability': 15, 'potential_mobility': 0,
                       'x_square_penalty': 10, 'frontier':0}

    elif (board_size == 12):  #12x12 board wins 50/50 games on random
        if game_phase == 'early':
            weights = {'disc': 5, 'mobility': 15, 'corner': 50, 'stability': 8, 'potential_mobility': 20,
                       'x_square_penalty': 300, 'frontier': 0}

        elif game_phase == 'mid':
            weights = {'disc': 10, 'mobility': 10, 'corner': 2000, 'stability': 15, 'potential_mobility': 10,
                       'x_square_penalty': 300, 'frontier': 0}
        else:  # Late game, changed corner from 30 to 2000
            weights = {'disc': 50, 'mobility': 2, 'corner': 2000, 'stability': 25, 'potential_mobility': 0,
                       'x_square_penalty': 300, 'frontier': 0}

    elif (board_size == 10):
        if game_phase == 'early':
            weights = {'disc': 10, 'mobility': 20, 'corner': 1000, 'stability': 10, 'potential_mobility': 20,
                       'x_square_penalty': 3000, 'frontier': 10}

        elif game_phase == 'mid':
            weights = {'disc': 15, 'mobility': 10, 'corner': 2000, 'stability': 25, 'potential_mobility': 15,
                       'x_square_penalty': 2000, 'frontier': 30}

        else:  # Late game
            weights = {'disc': 100, 'mobility': 5, 'corner': 2000, 'stability': 40, 'potential_mobility': 0,
                       'x_square_penalty': 3000, 'frontier': 0}

    elif (board_size == 8):  # For 8x8 board - has won 79/80 tested against random
        if game_phase == 'early':
            weights = {
                'disc': 5,
                'mobility': 15,
                'corner': 50,
                'stability': 8,
                'potential_mobility': 15,
                'x_square_penalty': 200,
                'frontier': 0
            }

        elif game_phase == 'mid':
            weights = {
                'disc': 10,
                'mobility': 10,
                'corner': 2000,
                'stability': 20,
                'potential_mobility': 10,
                'x_square_penalty': 500,
                'frontier': 10
            }

        else:  # Late game
            weights = {
                'disc': 30,
                'mobility': 5,
                'corner': 3000,
                'stability': 50,
                'potential_mobility': 0,
                'x_square_penalty': 2000,
                'frontier': 0
            }
        '''
        if game_phase == 'early':
            weights = {'disc': 0, 'mobility': 5, 'corner': 100, 'stability': 0, 'potential_mobility': 0,
                       'x_square_penalty': 350, 'frontier': 0, 'edge': 300} #maybe increase edge

        elif game_phase == 'mid':
            weights = {'disc': 0, 'mobility': 0, 'corner': 500, 'stability': 0, 'potential_mobility': 0,
                       'x_square_penalty': 500, 'frontier': 0, 'edge': 100}
            # increase x penalty
        else:  # Late game, changed corner from 30 to 2000
            #corner needs to be favored every time
            weights = {'disc': 10, 'mobility': 0, 'corner': 7000, 'stability': 0, 'potential_mobility': 0,
                       'x_square_penalty': 350, 'frontier': 0, 'edge': 100}
            #increase edge weight
        '''

        '''
        if game_phase == 'early':
            weights = {'disc': 0, 'mobility': 0, 'corner': 10, 'stability': 0, 'potential_mobility': 0,
                       'x_square_penalty': 100, 'frontier': 0, 'edge': 20}

        elif game_phase == 'mid':
            weights = {'disc': 0, 'mobility': 0, 'corner': 15, 'stability': 0, 'potential_mobility': 0,
                       'x_square_penalty': 100, 'frontier': 0, 'edge': 10}
            # switched corner and edge
        else:  # Late game, changed corner from 30 to 2000
            weights = {'disc': 0, 'mobility': 0, 'corner': 150, 'stability': 0, 'potential_mobility': 0,
                       'x_square_penalty': 300, 'frontier': 0, 'edge': 10}

        '''

        '''
        if game_phase == 'early':
            weights = {'disc': 0, 'mobility': 0, 'corner': 0, 'stability': 0, 'potential_mobility': 0,
                       'x_square_penalty': 350, 'frontier': 0, 'edge': 100}

        elif game_phase == 'mid':
            weights = {'disc': 0, 'mobility': 0, 'corner': 200, 'stability': 0, 'potential_mobility': 0,
                       'x_square_penalty': 500, 'frontier': 0, 'edge': 100}
        else:  # Late game, changed corner from 30 to 2000
            weights = {'disc': 10, 'mobility': 0, 'corner': 100, 'stability': 0, 'potential_mobility': 0,
                       'x_square_penalty': 150, 'frontier': 0, 'edge': 50}

            if game_phase == 'early':
            weights = {'disc': 0, 'mobility': 5, 'corner': 0, 'stability': 0, 'potential_mobility': 0,
                       'x_square_penalty': 350, 'frontier': 0, 'edge':150}
    '''
    '''

        elif game_phase == 'mid':
            weights = {'disc': 0, 'mobility': 0, 'corner': 500, 'stability': 0, 'potential_mobility': 0,
                       'x_square_penalty': 150, 'frontier': 0, 'edge':100}
            #switched corner and edge
        else:  # Late game, changed corner from 30 to 2000
            weights = {'disc': 10, 'mobility': 0, 'corner': 700, 'stability': 0, 'potential_mobility': 0,
                       'x_square_penalty': 150, 'frontier': 0, 'edge': 50}'''

    '''
        if game_phase == 'early':
            weights = {'disc': 5, 'mobility': 15, 'corner': 50, 'stability': 8, 'potential_mobility': 20,
                       'x_square_penalty': 100, 'frontier': 0}

        elif game_phase == 'mid':
            weights = {'disc': 10, 'mobility': 10, 'corner': 2000, 'stability': 15, 'potential_mobility': 10,
                       'x_square_penalty': 300, 'frontier':0}

        else:  # Late game
            weights = {'disc': 50, 'mobility': 2, 'corner': 30, 'stability': 25, 'potential_mobility': 0,
                       'x_square_penalty': 200, 'frontier': 0}
    '''

    return (
            weights['disc'] * disc_score +
            weights['mobility'] * mobility_score +
            weights['corner'] * corner_score +
            weights['stability'] * stability_score +
            weights['potential_mobility'] * potential_mobility_score
            + weights['x_square_penalty'] *x_square_penalty
            +weights['frontier'] *frontier_score
            #+x_square_bonus
            #+edge_control_score
    )
