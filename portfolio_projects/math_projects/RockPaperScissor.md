![Holberton School Logo](https://cdn.prod.website-files.com/6105315644a26f77912a1ada/63eea844ae4e3022154e2878_Holberton.png)

## Rock-Paper-Scissors Simulator

#### **Project Description**  
This project involves simulating a classic game of Rock-Paper-Scissors between a human player and an AI opponent. The AI makes random choices for each round using **Numpy** for random selection. The objective is to analyze the game's outcomes—wins, losses, and ties—and visualize these results to understand randomness, probabilities, and game mechanics.

#### **Learning Objectives**  
- Use **Numpy** to implement random choice selection for the AI.  
- Apply conditional logic to determine game outcomes (win, lose, tie).  
- Track and visualize results using **Matplotlib**.  
- Understand probabilities in a structured game context.

---

### **Project Requirements**

1. **Game Setup**  
   - Define the three possible moves: `Rock`, `Paper`, and `Scissors`.  
   - Allow the player to input their choice for each round (e.g., `Rock`, `Paper`, or `Scissors`).  
   - Generate a random choice for the AI opponent using **Numpy's random functions**.  

2. **Game Logic**  
   - Implement rules to determine the winner of each round:  
     - Rock beats Scissors.  
     - Scissors beat Paper.  
     - Paper beats Rock.  
     - The same choice results in a tie.  
   - Use conditional statements to compare the player’s choice with the AI’s choice and determine the outcome.

3. **Tracking Results**  
   - Track the results of multiple rounds:  
     - **Wins**: When the player beats the AI.  
     - **Losses**: When the AI beats the player.  
     - **Ties**: When both choose the same move.  
   - Store these results for visualization and analysis.

4. **Visualization**  
   - After a predefined number of rounds (e.g., `50` rounds), visualize the results:  
     - Create a bar chart to show the total number of wins, losses, and ties.  
     - Highlight the outcome that occurred most frequently.  

5. **Statistics**  
   - Calculate the probabilities of each outcome (win, loss, tie) across all rounds.  
   - Compare these probabilities to theoretical expectations assuming a random AI.  

6. **Stretch Goals (Optional)**  
   - Implement a **strategy-based AI** that learns and adapts to the player’s choices (e.g., countering the most frequent player move).  
   - Simulate multiple trials (e.g., `100 games of 50 rounds each`) and visualize the distribution of outcomes across trials.  
   - Add a score-tracking mechanism where the player earns points for wins and loses points for losses.  

---


