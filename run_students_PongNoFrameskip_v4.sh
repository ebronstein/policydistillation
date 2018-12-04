set -eux

TEACHER_CHECKPOINT_DIR="results/PongNoFrameskip-v4_greyscale_huber_teacher/teacher_PongNoFrameskip-v4_greyscale_huber_teacher/model.weights/"
TEACHER_NAME="PongNoFrameskip_v4_greyscale_huber"

## Softmax sharpening

# MSE of action probabilities with softmax sharpening
# DONE: max evaluated reward: 11.32 +/- 0.50
# python distilledqn_atari.py PongNoFrameskip-v4 PongNoFrameskip-v4_greyscale_mse_prob_softmax_0.01 $TEACHER_CHECKPOINT_DIR mse_prob softmax_tau -stqt 0.01
# debugging new multi-teacher format
python distilledqn_atari.py PongNoFrameskip-v4 PongNoFrameskip-v4_greyscale_mse_prob_softmax_0.01_multiteacher_test mse_prob softmax_tau none -stqt 0.01 -tcd $TEACHER_CHECKPOINT_DIR -tcn $TEACHER_NAME

# MSE of action probabilities with softmax sharpening + NLL
# python distilledqn_atari.py PongNoFrameskip-v4 PongNoFrameskip-v4_greyscale_mse_prob_nll_softmax_0.01 $TEACHER_CHECKPOINT_DIR mse_prob_nll softmax_tau -stqt 0.01

# KL of action probabilities with softmax sharpening
# DONE: max evaluated reward: 11.88 +/- 0.71
python distilledqn_atari.py PongNoFrameskip-v4 PongNoFrameskip-v4_greyscale_kl_softmax_0.01_multiteacher_test kl softmax_tau none -stqt 0.01 -tcd $TEACHER_CHECKPOINT_DIR -tcn $TEACHER_NAME

## No softmax

# KL without softmax sharpening/softening
# DONE: max evaluated reward: 11.18 +/- 0.58
# python distilledqn_atari.py PongNoFrameskip-v4 PongNoFrameskip-v4_greyscale_kl $TEACHER_CHECKPOINT_DIR kl none

# MSE without softmax sharpening/softening
# filename: PongNoFrameskip-v4_greyscale_mse_student_2018-11-07-10-16-56
# DONE: max evaluated reward: 8.40 +/- 0.82

# NLL without softmax sharpening/softening
# DONE: max evaluated reward: 10.86 +/- 0.57
python distilledqn_atari.py PongNoFrameskip-v4 PongNoFrameskip-v4_greyscale_nll_softmax_0.01_multiteacher_test nll none none -tcd $TEACHER_CHECKPOINT_DIR -tcn $TEACHER_NAME

## Softmax softening
# KL with softmax softening
# DONE: max evaluated reward: 10.96
# python distilledqn_atari.py PongNoFrameskip-v4 PongNoFrameskip-v4_greyscale_kl_softmax_2 $TEACHER_CHECKPOINT_DIR kl softmax_tau -stqt 2

# MSE of action probabilities with softmax softening
# DONE: max evaluated reward: 10.82 +/- 0.81
# python distilledqn_atari.py PongNoFrameskip-v4 PongNoFrameskip-v4_greyscale_mse_prob_softmax_2 $TEACHER_CHECKPOINT_DIR mse_prob softmax_tau -stqt 2


## DON'T DO THESE

# MSE of Q values with softmax sharpening
# This will do the MSE between the teacher Q values divided by tau and the student
# Q values, which is kinda the same as just doing the MSE
# DONE: max evaluated reward: 4.52 +/- 0.79
# python distilledqn_atari.py PongNoFrameskip-v4 PongNoFrameskip-v4_greyscale_mse_qval_softmax_0.01 $TEACHER_CHECKPOINT_DIR mse_qval softmax_tau -stqt 0.01

# MSE of action probabilities with or without softmax sharpening/softening
# This will just make the student learn through the softmax function, which isn't
# necessary unless we want the student's Q values to be of the same scale
# relative to teach other but of a different absolute scale (since only the relative
# scale affects the action probabilities due to the softmax).
# python distilledqn_atari.py PongNoFrameskip-v4 PongNoFrameskip-v4_greyscale_mse_prob $TEACHER_CHECKPOINT_DIR mse_prob