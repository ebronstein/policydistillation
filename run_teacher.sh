set -eux

# default of 5000000 iterations
# python natureqn_atari.py PongDeterministic-v4 PongDeterministic-v4_greyscale_huber_teacher

# 20000000 iterations
python natureqn_atari.py PongDeterministic-v4 PongDeterministic-v4_greyscale_huber_teacher -nt 20000000