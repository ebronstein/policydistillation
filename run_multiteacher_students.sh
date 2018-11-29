set -eux

### Mean of teacher Q values

# Learn from student_Pong-v0_greyscale_mse_prob_softmax_0.01 and student_Pong-v0_greyscale_mse_qval_softmax_0.01
# DONE: doesn't work at all
# python distilledqn_atari.py Pong-v0 Pong_v0_multiteacher_mean_mse_1 kl softmax_tau mean -tcd results/Pong-v0_greyscale_mse_teacher_2018-11-06-13-43-39/student_Pong-v0_greyscale_mse_prob_softmax_0.01/model.weights/ results/Pong-v0_greyscale_mse_teacher_2018-11-06-13-43-39/student_Pong-v0_greyscale_mse_qval_softmax_0.01/model.weights/ -tcn Pong-v0_greyscale_mse_prob_softmax_0.01 Pong-v0_greyscale_mse_qval_softmax_0.01

# Learn from student_Pong-v0_greyscale_mse_prob_softmax_0.01 and student_Pong-v0_greyscale_kl
# DONE: doesn't work at all
python distilledqn_atari.py Pong-v0 Pong_v0_multiteacher_mean_mse_1 kl softmax_tau mean -tcd results/Pong-v0_greyscale_mse_teacher_2018-11-06-13-43-39/student_Pong-v0_greyscale_mse_prob_softmax_0.01/model.weights/ results/Pong-v0_greyscale_mse_teacher_2018-11-06-13-43-39/student_Pong-v0_greyscale_kl/model.weights/ -tcn Pong-v0_greyscale_mse_prob_softmax_0.01 Pong-v0_greyscale_mse_qval_softmax_0.01


### Random teacher

# Learn from student_Pong-v0_greyscale_mse_prob_softmax_0.01 and student_Pong-v0_greyscale_mse_qval_softmax_0.01
# DONE: doesn't work at all; max average reward: -20.88 +/- 0.05
# python distilledqn_atari.py Pong-v0 Pong_v0_multiteacher_random_mse_1 kl softmax_tau random -tcd results/Pong-v0_greyscale_mse_teacher_2018-11-06-13-43-39/student_Pong-v0_greyscale_mse_prob_softmax_0.01/model.weights/ results/Pong-v0_greyscale_mse_teacher_2018-11-06-13-43-39/student_Pong-v0_greyscale_mse_qval_softmax_0.01/model.weights/ -tcn Pong-v0_greyscale_mse_prob_softmax_0.01 Pong-v0_greyscale_mse_qval_softmax_0.01

# Learn from student_Pong-v0_greyscale_mse_prob_softmax_0.01 and student_Pong-v0_greyscale_kl
python distilledqn_atari.py Pong-v0 Pong_v0_multiteacher_random_mse_1 kl softmax_tau random -tcd results/Pong-v0_greyscale_mse_teacher_2018-11-06-13-43-39/student_Pong-v0_greyscale_mse_prob_softmax_0.01/model.weights/ results/Pong-v0_greyscale_mse_teacher_2018-11-06-13-43-39/student_Pong-v0_greyscale_kl/model.weights/ -tcn Pong-v0_greyscale_mse_prob_softmax_0.01 Pong-v0_greyscale_mse_qval_softmax_0.01