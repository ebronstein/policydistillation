set -eux

TEACHER_CHECKPOINT_DIR="~/deeprl/policydistillation/results/Pong-v0_greyscale_mse_2018-11-06-13-43-39/teacher/model.weights"

# MSE with softmax sharpening
python distilledqn_atari.py Pong-v0 Pong-v0_greyscale_mse_softmax_0.01 $TEACHER_CHECKPOINT_DIR mse softmax -stqt 0.01

# NLL with softmax sharpening
python distilledqn_atari.py Pong-v0 Pong-v0_greyscale_nll_softmax_0.01 $TEACHER_CHECKPOINT_DIR nll softmax -stqt 0.01

# KL without softmax sharpening/softening
python distilledqn_atari.py Pong-v0 Pong-v0_greyscale_kl $TEACHER_CHECKPOINT_DIR kl none

# KL with softmax softening
python distilledqn_atari.py Pong-v0 Pong-v0_greyscale_kl_softmax_2 $TEACHER_CHECKPOINT_DIR kl softmax -stqt 2

# MSE with softmax softening
python distilledqn_atari.py Pong-v0 Pong-v0_greyscale_mse_softmax_2 $TEACHER_CHECKPOINT_DIR mse softmax -stqt 2