set -eux

# checkpoint directories
STUDENT_KL_DIR="results/PongNoFrameskip-v4_greyscale_huber_teacher/student_PongNoFrameskip-v4_greyscale_kl_softmax_0.01_multiteacher_test/model.weights/"
STUDENT_MSE_PROB_DIR="results/PongNoFrameskip-v4_greyscale_huber_teacher/student_PongNoFrameskip-v4_greyscale_mse_prob_softmax_0.01_multiteacher_test/model.weights/"
STUDENT_NLL_DIR="results/PongNoFrameskip-v4_greyscale_huber_teacher/student_PongNoFrameskip-v4_greyscale_nll_softmax_0.01_multiteacher_test/model.weights/"
TEACHER_HUBER_DIR="results/PongNoFrameskip-v4_greyscale_huber_teacher/teacher_PongNoFrameskip-v4_greyscale_huber_teacher/model.weights/"

# checkpoint names
STUDENT_KL_NAME="student_PongNoFrameskip-v4_greyscale_kl_softmax_0.01_multiteacher_test"
STUDENT_MSE_PROB_NAME="student_PongNoFrameskip-v4_greyscale_mse_prob_softmax_0.01_multiteacher_test"
STUDENT_NLL_NAME="student_PongNoFrameskip-v4_greyscale_nll_softmax_0.01_multiteacher_test"
TEACHER_HUBER_NAME="teacher_PongNoFrameskip-v4_greyscale_huber_teacher"

### RandomBandit

# python distilledqn_atari.py PongNoFrameskip-v4 PongNoFrameskip_v4_multiteacher_kl_softmax_0.01_randombandit kl softmax_tau random_bandit -stqt 0.01 -nt 5000000 -tqns small small small large -tcd $STUDENT_KL_DIR $STUDENT_MSE_PROB_DIR $STUDENT_NLL_DIR $TEACHER_HUBER_DIR  -tcn $STUDENT_KL_NAME $STUDENT_MSE_PROB_NAME $STUDENT_NLL_NAME $TEACHER_HUBER_NAME




### EpsilonGreedyBandit

# python distilledqn_atari.py PongNoFrameskip-v4 PongNoFrameskip_v4_multiteacher_kl_softmax_0.01_eps_greedy_bandit kl softmax_tau eps_greedy_bandit -brm rolling_avg -stqt 0.01 -nt 5000000 -tqns small small small large -tcd $STUDENT_KL_DIR $STUDENT_MSE_PROB_DIR $STUDENT_NLL_DIR $TEACHER_HUBER_DIR  -tcn $STUDENT_KL_NAME $STUDENT_MSE_PROB_NAME $STUDENT_NLL_NAME $TEACHER_HUBER_NAME



### UCB1Bandit

python distilledqn_atari.py PongNoFrameskip-v4 PongNoFrameskip_v4_multiteacher_kl_softmax_0.01_ucb1_bandit kl softmax_tau ucb1_bandut -brm rolling_avg -stqt 0.01 -nt 5000000 -tqns small small small large -tcd $STUDENT_KL_DIR $STUDENT_MSE_PROB_DIR $STUDENT_NLL_DIR $TEACHER_HUBER_DIR  -tcn $STUDENT_KL_NAME $STUDENT_MSE_PROB_NAME $STUDENT_NLL_NAME $TEACHER_HUBER_NAME