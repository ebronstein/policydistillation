class BaseConfig():
    def __init__(self, env_name, exp_name, output_path):
        # output config
        self.output_path = output_path
        self.model_output = self.output_path + "model.weights/"
        self.log_path     = self.output_path + "log.txt"
        self.plot_output  = self.output_path + "scores.png"
        self.record_path  = self.output_path + "monitor/"

        # experiment config
        self.env_name = env_name
        self.exp_name = exp_name

    def get_config(self):
        params = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        config_dict = {}
        for p in params:
            config_dict[p] = eval('self.' + p)
        return config_dict

class TeacherBaseConfig(BaseConfig):
    # student/teacher config
    student = False


class StudentBaseConfig(BaseConfig):
    def __init__(self, env_name, exp_name, output_path, teacher_checkpoint_dirs, 
                teacher_checkpoint_names):
        BaseConfig.__init__(self, env_name, exp_name, output_path)
        self.teacher_checkpoint_dirs = teacher_checkpoint_dirs
        self.teacher_checkpoint_names = teacher_checkpoint_names

    # student/teacher config
    student = True


class CartPole_v0_config_teacher(TeacherBaseConfig):
    # env config
    render_train     = False
    render_test      = False
    env_name         = "CartPole-v0"
    overwrite_render = True
    record           = False
    high             = 1. # high=1 ==> state/high=state in preprocess_state function

    # model and training config
    num_episodes_test = 50
    grad_clip         = False
    saving_freq       = 250000
    log_freq          = 50
    eval_freq         = 250000
    record_freq       = 250000
    soft_epsilon      = 0. # original: 0.05. Note: this is only used for taking 
                           # random action when evaluating the policy.

    # hyper params
    q_values_model     = 'feedforward_nn'
    double_q           = False
    n_layers           = 2 # original: 2
    nn_size            = 24 # original: 24
    nn_activation      = 'relu'
    nsteps_train       = 5000000
    batch_size         = 32 # original: 20
    buffer_size        = 1000000
    target_update_freq = 10000
    gamma              = 0.95
    learning_freq      = 1 # original: 4
    state_history      = 1
    lr_begin           = 0.001
    lr_end             = 0.0001
    lr_nsteps          = nsteps_train/2
    eps_begin          = 1
    eps_end            = 0.01
    eps_nsteps         = 1000000
    learning_start     = 50000

class Pong_v0_config_teacher(TeacherBaseConfig):
    # env config
    render_train     = False
    render_test      = False
    env_name         = "Pong-v0"
    overwrite_render = True
    record           = False
    high             = 255.

    # processing
    preprocess_state = 'greyscale'

    # model and training config
    num_episodes_test = 50
    grad_clip         = True
    clip_val          = 10
    saving_freq       = 250000
    log_freq          = 50
    eval_freq         = 250000
    record_freq       = 250000
    soft_epsilon      = 0.05

    # nature paper hyper params
    q_values_model = 'nature_cnn'
    double_q           = False
    nsteps_train       = 5000000
    batch_size         = 32
    buffer_size        = 1000000
    target_update_freq = 10000
    gamma              = 0.99
    learning_freq      = 4
    state_history      = 4
    skip_frame         = 4
    lr_begin           = 0.0001 # original implementation: 0.00025, Berkeley implementation: 1e-4
    lr_end             = 0.00005
    lr_nsteps          = nsteps_train/2
    eps_begin          = 1
    eps_end            = 0.1
    eps_nsteps         = 1000000
    learning_start     = 50000

class Pong_v0_config_student(StudentBaseConfig):
    # env config
    render_train     = False
    render_test      = False
    env_name         = "Pong-v0"
    overwrite_render = True
    record           = False
    high             = 255.

    # processing
    preprocess_state = 'greyscale'
    # process teacher Q values with the given method
    process_teacher_q = 'softmax'
    # choose the teacher Q values at each iteration with the given method
    choose_teacher_q = 'mean'
    # tau value for softmax_teacher_q to sharpen (tau < 1) or soften (tau > 1)
    # the teacher Q values
    softmax_teacher_q_tau = 0.01

    # student training
    student_loss = 'kl' # student loss when training on teacher Q values
    mse_prob_loss_weight = 1. # weight associated with the MSE over action probabilities loss
    nll_loss_weight = 1. # weight associated with the NLL over Q values/action probabilities loss

    # model and training config
    num_episodes_test = 50
    grad_clip         = True
    clip_val          = 10
    saving_freq       = 250000
    log_freq          = 50
    eval_freq         = 250000
    record_freq       = 250000
    soft_epsilon      = 0.05

    # nature paper hyper params
    q_values_model = 'nature_cnn'
    double_q           = False
    nsteps_train       = 5000000
    batch_size         = 32
    buffer_size        = 1000000
    target_update_freq = 10000
    gamma              = 0.99
    learning_freq      = 4
    state_history      = 4
    skip_frame         = 4
    lr_begin           = 0.00025 # original implementation: 0.00025, Berkeley implementation: 1e-4
    lr_end             = 0.00005
    lr_nsteps          = nsteps_train/2
    eps_begin          = 1
    eps_end            = 0.1
    eps_nsteps         = 1000000
    learning_start     = 50000

class PongDeterministic_v4_config_teacher(TeacherBaseConfig):
    # env config
    render_train     = False
    render_test      = False
    env_name         = "Pong-v0"
    overwrite_render = True
    record           = False
    high             = 255.

    # processing
    preprocess_state = 'greyscale'

    # model and training config
    num_episodes_test = 50
    grad_clip         = True
    clip_val          = 10
    saving_freq       = 250000
    log_freq          = 50
    eval_freq         = 250000
    record_freq       = 250000
    soft_epsilon      = 0.05

    # nature paper hyper params
    q_values_model = 'nature_cnn'
    double_q           = True
    nsteps_train       = 5000000
    batch_size         = 32
    buffer_size        = 1000000
    target_update_freq = 10000
    gamma              = 0.99
    learning_freq      = 4
    state_history      = 4
    skip_frame         = 4
    lr_begin           = 0.0001 # original implementation: 0.00025, Berkeley implementation: 1e-4
    lr_end             = 0.00005
    lr_nsteps          = nsteps_train/2
    eps_begin          = 1
    eps_end            = 0.1
    eps_nsteps         = 1000000
    learning_start     = 50000

# class atariconfig_student():
#     # env config
#     render_train     = False
#     render_test      = False
#     env_name         = "Pong-v0"
#     overwrite_render = True
#     record           = False
#     high             = 255.

#     # output config
#     output_path  = "results/atari/student/"
#     model_output = output_path + "model.weights/"
#     log_path     = output_path + "log.txt"
#     plot_output  = output_path + "scores.png"
#     record_path  = output_path + "monitor/"

#     # student/teacher config
#     student = True
#     teacher_checkpoint_dir = "results/atari/teacher/model.weights"

#     # model and training config
#     num_episodes_test = 50
#     grad_clip         = True
#     clip_val          = 10
#     saving_freq       = 250000
#     log_freq          = 50
#     eval_freq         = 250000
#     record_freq       = 250000
#     soft_epsilon      = 0.05

#     # nature paper hyper params
#     q_values_model     = 'nature_cnn'
#     double_q           = False
#     nsteps_train       = 5000000
#     batch_size         = 32
#     buffer_size        = 1000000
#     target_update_freq = 10000
#     gamma              = 0.99
#     learning_freq      = 4
#     state_history      = 4
#     skip_frame         = 4
#     lr_begin           = 0.00025
#     lr_end             = 0.00005
#     lr_nsteps          = nsteps_train/2
#     eps_begin          = 1
#     eps_end            = 0.1
#     eps_nsteps         = 1000000
#     learning_start     = 50000


# class testconfig_teacher():
#     # env config
#     render_train     = False
#     render_test      = False
#     overwrite_render = True
#     record           = False
#     high             = 255.

#     # output config
#     output_path  = "results/test/teacher/"
#     model_output = output_path + "model.weights/"
#     log_path     = output_path + "log.txt"
#     plot_output  = output_path + "scores.png"

#     # model and training config
#     num_episodes_test = 20
#     grad_clip         = True
#     clip_val          = 10
#     saving_freq       = 5000
#     log_freq          = 50
#     eval_freq         = 100
#     soft_epsilon      = 0

#     # hyper params
#     nsteps_train       = 1000
#     batch_size         = 32
#     buffer_size        = 500
#     target_update_freq = 500
#     gamma              = 0.99
#     learning_freq      = 4
#     state_history      = 4
#     lr_begin           = 0.00025
#     lr_end             = 0.0001
#     lr_nsteps          = nsteps_train/2
#     eps_begin          = 1
#     eps_end            = 0.01
#     eps_nsteps         = nsteps_train/2
#     learning_start     = 200

# class testconfig_student():
#     # env config
#     render_train     = False
#     render_test      = False
#     overwrite_render = True
#     record           = False
#     high             = 255.

#     # output config
#     output_path  = "results/test/student/"
#     model_output = output_path + "model.weights/"
#     log_path     = output_path + "log.txt"
#     plot_output  = output_path + "scores.png"

#     # model and training config
#     num_episodes_test = 20
#     grad_clip         = True
#     clip_val          = 10
#     saving_freq       = 5000
#     log_freq          = 50
#     eval_freq         = 100
#     soft_epsilon      = 0

#     # hyper params
#     nsteps_train       = 1000
#     batch_size         = 32
#     buffer_size        = 500
#     target_update_freq = 500
#     gamma              = 0.99
#     learning_freq      = 4
#     state_history      = 4
#     lr_begin           = 0.00025
#     lr_end             = 0.0001
#     lr_nsteps          = nsteps_train/2
#     eps_begin          = 1
#     eps_end            = 0.01
#     eps_nsteps         = nsteps_train/2
#     learning_start     = 200