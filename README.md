# Reversi Minimax Agent
Here, we write a Reversi-playing minimax agent using Python. The agent is capable of playing games using board sizes 6x6, 8x8, 10x10, 12x12, and beats random agents 99.8 percent of the time! Performs admirably against human players, with performace for each board size being tailored using weights/penalties for certain moves. Below is the documentation on this project, but if you'd like to view it in latex, you can download the file:
```
#Some code
```

# Collaboration
This Minimax agent was a joint effort between I, Jack Parry-Wingfield and Mira Kandlikar-Bloch. Neither of us claim any credit for the other files, which were provided as starter code to start and run the Reversi game, load the GUI, and create the human and random agent opponents. This starter code cand be found and accessed at 
https://github.com/dmeger/COMP424-Fall2024.

# COMP 424 Final Project: Reversi Agent
**Mira Kandlikar-Bloch, Jack Parry-Wingfield**  
*November 2024*  

## Introduction and Executive Summary  
Our Reversi agent uses an implementation of the minimax algorithm, which we have enhanced with alpha-beta pruning and iterative deepening. We chose this combination of algorithmic techniques because it effectively balances decision-making with computational efficiency.  

Reversi is a zero-sum game, meaning one player's gain directly corresponds to the other's loss. Such a game is well suited for the minimax algorithm, which is designed to handle game strategy by choosing moves that maximize the player's advantage while minimizing the opponent's. Minimax enables our agent to consider the immediate impact of a move and its long-term consequences by simulating the opponent's best possible responses. This foresight is especially critical in Reversi, where early moves often dictate endgame results.  

We chose to implement alpha-beta pruning to improve computational efficiency. Alpha-beta pruning enhances the minimax algorithm by "cutting off" branches of the game tree that do not affect the outcome, reducing the number of nodes evaluated by our agent. This is particularly advantageous for our algorithm due to the 2-second time limit for each move. Through alpha-beta pruning, we ensured that our agent does not perform unnecessary computation. Its available time is used optimally to find the best move. This, combined with move ordering and a transposition table, was a large factor in our agent's success.  

To ensure that our agent always makes an informed move, we implemented iterative deepening. We incrementally searched the game tree, increasing the depth until we hit either the time limit, the endgame, or the required depth. We then returned the best move found so far. This mitigated situations where the agent would not have enough time to find a good move in a traditional minimax search.  

Using an iterative design process and research of relevant game heuristics, we built up our agent's algorithm and evaluation function. Based on the testing we did across boards and the `random` and `greedy_gpt_corners` agents, we achieved a strong play quality for our agent. We also found that it was relatively difficult to beat our agent when playing against it.  

The following sections detail the algorithm specifics and implementation, an analysis of our agent's performance, and possible further improvements.

## Agent Design and Implementation

### Minimax with Alpha-Beta Pruning

Our Reversi Agent uses a minimax algorithm with alpha-beta pruning and iterative deepening for gameplay. This section outlines the implementation and further theoretical justification for our agent design.

In the `step` function of `student_agent.py`, we initialize the start time, a time limit of 1.9 seconds, the best move as decided by our `order_moves` function (detailed in section 2.5), and `depth = 1`. While our agent is within the time limit, we repeatedly call minimax while incrementing the depth by 1.

Minimax is implemented as a class method for the `StudentAgent` class. It takes as input the following parameters:

- **Board**: the current board state.  
- **Depth**: an integer specifying how deep the agent should explore, initialized as 1.  
- **Maximizing player**: `True` if the current player is the maximizing player, `False` otherwise. Initialized as `True`.  
- **Player**: An identifier (1 or 2) for the agent executing the algorithm.  
- **Opponent**: The identifier for the opposing player. Used to simulate the opponent's potential moves and responses.  
- **Alpha**: a value representing the current lower bound of the search, initialized to `-infinity`.  
- **Beta**: a value representing the current upper bound of the search, initialized to `+infinity`.  
- **Start Time**: a timestamp indicating when the turn started.  
- **Time limit**: a value representing the amount of time the agent has to make a move, in this case, 1.9 seconds.  

The initial call to the `minimax` function is made in the `step` function. In `minimax`, the current board state is turned into a hash value to serve as a unique identifier for the transposition table. The function then checks the transposition table to determine if the current board state was previously evaluated. If a matching entry is found and is suitable for the current search, the stored evaluation is used directly. Further details on the transposition table and its integration into minimax are outlined in section 2.3.

If no suitable entry is found, the function proceeds with the minimax search. It checks whether the current depth is zero, if the game has reached an end state, or if the time limit has been exceeded. In such cases, the board is evaluated using the evaluation function, and the result is stored in the transposition table with an "exact" flag before being returned.

Otherwise, the function determines valid moves using `get_valid_moves` and prioritizes them using a heuristic-based `order_moves` function (outlined in section 2.5). For each valid move, the function makes a recursive call, swapping between maximizing and minimizing logic, with the depth decremented each time. During these recursive calls, alpha and beta values are continuously updated, and branches are pruned when no further exploration is beneficial. After evaluating all moves (or pruning as needed), the evaluation score is stored in the transposition table. The best move and evaluation score are then returned to the previous minimax call.

### Iterative Deepening

After each iteration of minimax, if a move is returned, we assign it as the best move. The assumption is that after each iteration with an increased depth, the algorithm has a better idea of the best move, as it can "see" further down the game tree—whatever it returns will be at least as good as the previous move. In this fashion, we use iterative deepening to store the best move so far while still exploring the minimax tree. Then, if we run out of time during a call to minimax, we can be sure to return the best move found from the previous depth.

While iterative deepening is a powerful tool for ensuring the best move within the given time limit, it does come with a trade-off: repeated computations. This redundancy occurs because the algorithm recalculates evaluations at shallower depths multiple times as it incrementally deepens its search. To solve this, we implemented the transposition table, which effectively allows our agent to "skip" over computations done in previous iterations, allowing it to reach deeper game states more quickly [8]. The following section outlines the implementation of the transposition table.

### The Transposition Table

We implemented our transposition table as a separate class, `TranspositionTable`, within `student_agent.py`. Each instance of a `TranspositionTable` object has a "table," or dictionary, that stores previously computed evaluations of board states. The dictionary uses a unique hash of a board state's string representation as its key, ensuring each specific configuration of the board state corresponds to a single, distinct entry in the table [9]. The value associated with each key is a dictionary containing the following information about a board state:

1. **The Evaluation Score**: A numerical score that represents the utility of the given state of the board, based on our evaluation function. Storing this value avoids recalculating the score for the same board state multiple times during the minimax algorithm.

2. **The Search Depth**: Indicates the depth in the game tree where the board evaluation was performed. Since deeper evaluations generally provide a more accurate assessment of a state's utility, this information allows our agent to decide whether the stored evaluation is still reliable.  
   - If the current search is performed at a depth less than or equal to the depth associated with a stored board state, the agent can safely reuse the evaluation.  
   - If the stored evaluation score was performed at a shallower depth, the agent ignores the stored value and performs a deeper search. Once the deeper evaluation is done, the agent overwrites the old evaluation score with a new, more accurate score for the board state.

3. **The Bound Flag**: A critical component of the transposition table that allows integration with alpha-beta pruning. It indicates to the agent how the evaluation score of a board state should be interpreted. The bound flag takes on one of three possible values:
   - **Exact**: The evaluation score represents the exact utility of the board state.
   - **Lower bound**: Indicates that the stored evaluation is the minimum score the maximizing player can achieve (used when a beta cut-off occurs).
   - **Upper bound**: Indicates that the stored value is the maximum score the minimizing player can achieve (used when an alpha cut-off occurs).

   These flags enable the agent to efficiently prune branches and refine alpha-beta thresholds. If a lower bound exceeds the beta threshold or an upper bound is below the alpha threshold, the branch is pruned. Additionally, lower bounds can increase alpha, and upper bounds can decrease beta, tightening the search window and enabling earlier pruning. This integration with iterative deepening and alpha-beta pruning enhances the efficiency and accuracy of the minimax algorithm.

## The Evaluation Function

A major and arguably the most important aspect of our agent is the evaluation function. The evaluation function is responsible for assigning a numerical score to each board state at the base case of minimax, representing its utility for the agent. The larger the evaluation score, the better the board state. This score allows the agent to make informed decisions about which moves to prioritize at each stage of the game.

Our evaluation function incorporates the following heuristics tailored to Reversi.

### Disc Count

In Reversi, the Disc Count can be an important indicator of a player's dominance on the board. It represents the difference between the number of discs controlled by the agent and those controlled by the opponent. A higher disc count may signify an advantageous position, as the player with more discs is closer to winning. However, this heuristic must be used with caution. In early and mid-game scenarios, maximizing the disc count can lead to an unstable position, making it easier for the opponent to flip many discs. Later in the game, the Disc Count becomes more significant as it directly determines the winner. Because of this, disc count is generally weighted very little in the beginning and mid-game stages, and later at the end.

We calculated the Disc Count as follows:
```
$$
100 \times \frac{\text{player}_{\text{discs}} - \text{opponent}_{\text{discs}}}{\text{player}_{\text{discs}} + \text{opponent}_{\text{discs}}}
$$
```

This formula normalizes the Disc Count relative to the total number of discs on the board. A positive Disc Count reflects a larger number of discs for the player, while a negative Disc Count reflects a larger number of discs for the opponent. The motivation for this heuristic and the formula are adapted from source (1).

### Mobility

Mobility is the relative number of valid moves a player can make given the current board state. A higher number of moves ensures flexibility, allowing our agent to avoid bad plays that could lead to a loss. Mobility is particularly important in the early and mid-game phases, where controlling the board and restricting the opponent’s options lead to an advantage and avoid plays that could allow the opponent to take control (4).

We calculate Mobility using the formula:

$$
100 \times \frac{\text{player\_moves} - \text{opponent\_moves}}{\text{player\_moves} + \text{opponent\_moves}}
$$

A positive mobility score indicates a larger number of moves for the player, while a negative score indicates a larger number of moves for the opponent. The motivation for this heuristic and the formula are directly adapted from source (1).

### Corner Control

Corner Control, or the relative number of corners occupied by a player, is one of the most important heuristics in Reversi. Corners are stable, meaning they cannot be flipped once occupied. They also serve as anchors, providing stability to adjacent discs and reducing the opponent's ability to control the board. We calculate Corner Control by summing the number of corners occupied by a player on the current board. Then we use the formula:

$$
100 \times \frac{\text{player\_corners} - \text{opponent\_corners}}{\text{player\_corners} + \text{opponent\_corners}}
$$

The evaluation function assigns a very high weight to this heuristic, particularly in the mid-to-late game when corners become critical for securing victory. Earlier in the game, corners are given less weight as they are often still inaccessible by both players due to the smaller number of tiles on the board.

### Corner-Adjacent Penalization

A heuristic closely related to Corner Control, and thus highly significant in the game of Reversi, is the penalization of corner-adjacent tiles. These tiles are disadvantageous because occupying them allows the opponent to gain control of the adjacent corners.

Therefore, our evaluation function assigns a high penalty to boards where our agent occupies these squares.

The penalty is calculated based on the control status of the adjacent corner. If the adjacent corner is controlled by the player, no penalty is assigned, since these tiles no longer pose a risk. If the corner is controlled by the opponent or unoccupied, the penalty is high.

We calculate the Corner Adjacent Penalty as:

$$
\text{Corner Penalty} = -50 \times \text{Number of player's discs adjacent to non-player occupied corners.}
$$

This heuristic is particularly important in the early and mid-game phases. Unlike other heuristics, we chose to penalize the raw count of the player's discs in these positions due to their severe drawbacks. While it can be advantageous if the opponent occupies these tiles, the risk of placing our own discs there is so significant that the penalty is designed to strongly deter the agent from making such moves.

### Stability

Stability is defined as the number of a player's discs that cannot be flipped, regardless of subsequent moves by the opponent. Stable discs are advantageous because they provide a secure foundation for future plays and ensure a consistent presence on the board. Stability is particularly critical in the late game, where securing stable discs often determines a game's outcome. 

We calculate Stability as:

$$
100 \times \frac{\text{player\_stable\_discs} - \text{opponent\_stable\_discs}}{\text{player\_stable\_discs} + \text{opponent\_stable\_discs}}
$$

### Potential Mobility

Potential Mobility is the number of empty tiles adjacent to the opponent's discs. This heuristic evaluates the agent's ability to increase its valid moves in future turns, which expands future mobility and restricts the opponent's options.

$$
\text{Potential Mobility} = \text{Number of empty squares adjacent to opponent discs.}
$$

### Frontier Discs

Frontier discs are discs adjacent to at least one empty square. These discs are vulnerable because they are more likely to be flipped by the opponent in subsequent moves. 

$$
\text{Frontier Score} = -100 \times \frac{\text{player\_frontiers} - \text{opponent\_frontiers}}{\text{player\_frontiers} + \text{opponent\_frontiers}}
$$

### Game Phase and Board Size Adjustments

The importance of various heuristics changes as the game progresses. We implemented a function that classifies the game phase as early, mid-game, or late based on the proportion of filled tiles. Along with heuristic weight changing with the game phase, it also varies depending on board size. 

By tailoring heuristic weights based on both board size and game phase, the evaluation function ensures the agent adapts its strategy to the changing dynamics of the game, optimizing performance across different scenarios and improving overall gameplay in Reversi.



