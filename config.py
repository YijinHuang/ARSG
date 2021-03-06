BASIC_CONFIG = {
    'save_path': './result/large_att',
    'record_path': './result/log/large_att',
    'num_classes': 41,  # number of categories
    'random_seed': 0,  # random seed for reproducibilty
    'device': 'cuda'  # 'cuda' / 'cpu'
}

DATA_CONFIG = {
    'train_data': './timit_data2/train.npy',
    'train_label': './timit_data2/train_label.npy',
    'dev_data': './timit_data2/test.npy',
    'dev_label': './timit_data2/test_label.npy',
    'test_data': './data/test.npy',
    'dimension': 123,
    'mean': 'auto',  # 'auto' or a list of three numbers for RGB
    'std': 'auto',  # 'auto' or a list of three numbers for RGB
}

CRITERION_CONFIG = {
    'cross_entropy': {},
    'mean_square_root': {},
    'L1': {},
    'smooth_L1': {},
    'kappa_loss': {
        'num_classes': BASIC_CONFIG['num_classes']
    },
    'focal_loss': {
        'alpha': 5,
        'reduction': 'mean'
    }
}

SCHEDULER_CONFIG = {
    'exponential': {
        'gamma': 0.9  # Multiplicative factor of learning rate decay
    },
    'multiple_steps': {
        'milestones': [50, 75, 90],  # List of epoch indices. Must be increasing
        'gamma': 0.5,  # Multiplicative factor of learning rate decay
    },
    'cosine': {
        'T_max': 100,  # Maximum number of iterations.
        'eta_min': 0  # Minimum learning rate.
    },
    'reduce_on_plateau': {
        'mode': 'min',  # In min mode, lr will be reduced when the quantity monitored has stopped decreasing
        'factor': 0.1,  # Factor by which the learning rate will be reduced
        'patience': 5,  # Number of epochs with no improvement after which learning rate will be reduced.
        'threshold': 1e-4,  # Threshold for measuring the new optimum
        'eps': 1e-5,  # Minimal decay applied to lr
    },
    'clipped_cosine': {
        'T_max': 25,
        'min_lr': 1e-4  # lr will stay as min_lr when achieve it
    }
}

TRAIN_CONFIG = {
    'epochs': 100,  # total training epochs
    'batch_size': 64,  # training batch size
    'optimizer': 'ADADELTA',  # SGD / ADAM / ADADELTA
    'criterion': 'cross_entropy',  # one str name in CRITERION_CONFIG above.
    'criterion_config': CRITERION_CONFIG['cross_entropy'],  # loss function configurations, the key should the be same with criterion
    'learning_rate': 0.01,  # initial learning rate
    'lr_scheduler': 'multiple_steps',  # one str name in SCHEDULER_CONFIG above
    'lr_scheduler_config': SCHEDULER_CONFIG['multiple_steps'],  # scheduler configurations, the key should the be same with lr_cheduler
    'momentum': 0.9,  # momentum for SGD optimizer
    'nesterov': False,  # nesterov for SGD optimizer
    'weight_decay': 0.000005,  # weight decay for SGD and ADAM
    'warmup_epochs': 0,  # warmup epochs
    'num_workers': 8,  # number of cpus used to load data at each step
    'save_interval': 10,  # the steps interval of saving model
    'eval_interval': 5,
    'pin_memory': True,  # enables fast data transfer to CUDA-enabled GPUs
}
