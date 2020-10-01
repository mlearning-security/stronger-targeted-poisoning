import argparse


def setup_argparse():
    parser = argparse.ArgumentParser(description='handle poisoning and test')

    # dataset
    parser.add_argument("-d", "--dataset", default='',
                        help='dataset')
    parser.add_argument("-id", "--mal_0_id", default=1, type=int,
                        help='id of malware0')
    parser.add_argument("-param", "--parameter", default=-1, type=int,
                        help='parameter for feel like to use')
    parser.add_argument("-d_seed", "--random_state", default=0, type=int,
                        help='seed for sampling form dataset')
    parser.add_argument("-selection", "--feature_selection", default=-1, type=int,
                        help='feature selection by Mutual Information')
    parser.add_argument("-scaler", "--scaler", default=-1, type=int,
                        help='scaler, -1:no_scaling, 0:MinMaxScaler[0:1]')

    # params for poisoning
    parser.add_argument("-p", "--poisoning_num", default=10, type=int,
                        help='number of poisoning points, -1:eval model')
    parser.add_argument("-eps", "--epsilon", default=1e-10, type=float,
                        help='termination condition epsilon')
    parser.add_argument("-eta", "--eta", default=0.1, type=float,
                        help='learning rate for poisoning')
    parser.add_argument("-s", "--steps", default=10, type=int,
                        help='learning steps')
    parser.add_argument("-t", "--T", default=10, type=int,
                        help='T for poisoning')
    parser.add_argument("-spmode", "--specific_mode", default=0, type=int,
                        help='specific mode')
    parser.add_argument("-mulmode", "--multi_mode", default=0, type=int,
                        help='multi mode')
    parser.add_argument("-constraint", "--constraint", default=0, type=int,
                        help='type of constraint (0: nothing, 1:l2-norm square)')
    parser.add_argument("-beta", "--beta", default=0.1, type=float,
                        help='parameter of constraint')
    parser.add_argument("-max", "--step_max", default=-1, type=int,
                        help='maximum of steps')
    parser.add_argument("-i_seed", "--init_seed", default=0, type=int,
                        help='seed for selecting initial data')
    parser.add_argument("-elim", "--eliminate", default=0.15, type=float,
                        help='eliminate rate for sphere defense')

    parser.add_argument("-term", "--termination", action="store_true",
                        help='if use termination condition or not')
    parser.add_argument("-phi", "--phi_map", action="store_true",
                        help='if apply phi_map or not')
    parser.add_argument("-sp", "--specific", action="store_true",
                        help='error-specific')
    parser.add_argument("-multi", "--multi", action="store_true",
                        help='multi attack')
    parser.add_argument("-decay", "--decay", action="store_true",
                        help='decay on')
    parser.add_argument("-sphere", "--sphere_defense", action="store_true",
                        help='sphere defense on')
    parser.add_argument("-flip", "--label_flip", action="store_true",
                        help='label flip attack')
    parser.add_argument("-solver", "--solver", action="store_true",
                        help='attack using solver')

    # params for test
    parser.add_argument("-epo", "--epochs", default=1000, type=int,
                        help='epochs in test phase')
    parser.add_argument("-lr", "--learning_rate", default=0.01, type=float,
                        help='learning rate for test')
    parser.add_argument("-shuffle", "--shuffle", action="store_true",
                        help='shuffle on')
    parser.add_argument("-peval", "--eval_poisoning", action="store_true",
                        help='evaluate poisoning attack by reading poisoning data')

    # gpu
    parser.add_argument("-gpu", "--use_cuda", default=-1, type=int,
                        help='use_cuda')

    # save
    parser.add_argument("-save", "--save_directory", default='.',
                        help='name of save directory')
    # read poisoning data
    parser.add_argument("-read", "--pdata_path", default='',
                        help='path of poisoning data')

    return parser
