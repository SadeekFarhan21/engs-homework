# Lab Assignment 1 Write-Up: Rock, Paper, Scissors

## Algorithm Description

The algorithm models the CPU opponent as a **first-order Markov chain**: it assumes the CPU's next move depends only on its most recent move. During training data collection, I recorded each round as a tuple of `(cpu_move, player_move, outcome)` using integers `0=Rock, 1=Paper, 2=Scissors`.

### Design

From the recorded game history, I build a 3×3 **transition matrix** where `transition[i][j]` counts how many times the CPU played move `j` immediately after move `i`. To predict the CPU's next move, I look at the row corresponding to its last move and pick the column with the highest count. I then return the move that beats that prediction.

```
Transition matrix learned from training data (row = last CPU move, col = next CPU move):

              → Rock  Paper  Scissors
  Rock      [   1     15      11   ]
  Paper     [  10      8      18   ]
  Scissors  [  16     13       7   ]

Example: if CPU just played Rock (row 0), it most likely plays Paper next (15 times).
         So we play Scissors to beat Paper.
```

If no prior transitions exist for the CPU's last move (e.g., early in the game), the algorithm falls back to the CPU's overall most frequent move. If fewer than 2 rounds have been played, it picks randomly.

### Pseudocode

```
function predict(history):
    if len(history) < 2: return random move

    build transition[3][3] from cpu_moves in history
    last_cpu = cpu_moves[-1]

    if transition[last_cpu] has any counts:
        predicted_next = argmax(transition[last_cpu])
    else:
        predicted_next = most frequent cpu move overall

    return the move that beats predicted_next
```

---

## Model Evaluation

The algorithm was evaluated using leave-one-out simulation on the 100-round training dataset: for each round `i`, the model was trained on rounds `0..i-1` and asked to predict round `i`.

| Outcome | Count | Percentage |
|---------|-------|------------|
| Wins    | 43    | 43.4%      |
| Losses  | 22    | 22.2%      |
| Ties    | 34    | 34.3%      |
| **Total** | **99** | — |

**Win rate (excluding ties): 66.2%** vs. a random baseline of 50%.

The model outperforms random play by a meaningful margin, suggesting the CPU's move sequence is not truly random and the Markov assumption captures some of its patterns.

---

## Reflection

This lab was a good reminder that even simple statistical models can be surprisingly effective when the underlying system has exploitable structure. I went in expecting to need something more complex, but a first-order Markov chain — essentially just counting transitions in a 3×3 table — was enough to achieve a 66% win rate on decisive rounds. The bigger takeaway for me was the value of designing on paper first: by thinking through the transition matrix idea before writing any code, the implementation itself was straightforward and I could catch edge cases (e.g., no prior transitions for a given move) cleanly in the design phase rather than mid-debug.