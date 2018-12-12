set -eux

# top half
# python natureqn_atari.py PongNoFrameskip-v4 PongNoFrameskip-v4_greyscale_huber_top_half_teacher -nt 10000000 -qns large -ss ball_top_half

# bottom half
python natureqn_atari.py PongNoFrameskip-v4 PongNoFrameskip-v4_greyscale_huber_bottom_half_teacher -nt 10000000 -qns large -ss ball_bottom_half