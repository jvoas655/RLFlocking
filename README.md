# RLFlocking
A Reinforcement Learning base Flocking Project

This is the discrete policy branch of our project.

To run the Sarsa baseline:
```
python3.8 discrete.py --num_agents 40 --num_episodes 1000 --algorithm qlearning --gamma 0.95 --epsilon 0.05 --hidden 32 --comment official_baseline --checkpoint_frequency 100
```

To run the Q-learning baseline:
```
python3.8 discrete.py --num_agents 40 --num_episodes 1000 --algorithm qlearning --gamma 0.95 --epsilon 0.05 --hidden 32 --comment official_baseline --checkpoint_frequency 100
```
