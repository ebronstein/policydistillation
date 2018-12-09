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

# Learn from student_Pong-v0_greyscale_mse_prob_softmax_0.01 and student_Pong-v0_greyscale_kl
python distilledqn_atari.py PongNoFrameskip-v4 PongNoFrameskip_v4_multiteacher_kl_softmax_0.01_randombandit kl softmax_tau random_bandit -tcd $STUDENT_KL_DIR $STUDENT_MSE_PROB_DIR $STUDENT_NLL_DIR $TEACHER_HUBER_DIR  -tcn $STUDENT_KL_NAME $STUDENT_MSE_PROB_NAME $STUDENT_NLL_NAME $TEACHER_HUBER_NAME -tqns small small small large -stqt 0.01 -nt 5000000
# python distilledqn_atari.py PongNoFrameskip-v4 PongNoFrameskip_v4_multiteacher_kl_softmax_0.01_randombandit kl softmax_tau random_bandit -tcd $STUDENT_KL_DIR $STUDENT_MSE_PROB_DIR $STUDENT_NLL_DIR  -tcn $STUDENT_KL_NAME $STUDENT_MSE_PROB_NAME $STUDENT_NLL_NAME -stqt 0.01 -nt 5000000
