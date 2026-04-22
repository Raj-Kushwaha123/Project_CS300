# Configuration for Fine-Tuning Experiment

CONFIG = {
    # Dataset Configuration
    'data_percentages': [0.2, 0.5, 1.0],  # Percentages of data to use for training
    'dataset_name': 'CIFAR10',
    'num_classes': 10,
    
    # Model Configuration
    'model_name': 'resnet18',
    'pretrained': True,
    'input_size': 224,
    
    # Training Configuration
    'epochs': 5,
    'batch_size': 64,
    'learning_rate': 0.001,
    'optimizer': 'Adam',
    
    # Data Split
    'train_split': 0.8,  # 80% train, 20% validation
    
    # Device Configuration
    'device': 'cuda',  # 'cuda' or 'cpu'
    
    # Paths
    'data_dir': './data',
    'model_dir': './models',
    'plot_dir': './plots',
    'results_dir': './results',
    
    # Fine-tuning Strategies
    'strategies': {
        'fc': 'Train only classifier (FC layer)',
        'last': 'Train last block + classifier',
        'full': 'Train all layers (full fine-tuning)'
    }
}

# Fine-tuning strategy definitions
STRATEGIES = {
    'fc': {
        'name': 'FC Only',
        'description': 'Train only the fully connected classifier layer',
        'freeze_all': True,
        'unfreeze_fc': True,
    },
    'last': {
        'name': 'Last Block + FC',
        'description': 'Train the last residual block and classifier',
        'freeze_all': True,
        'unfreeze_last_block': True,
        'unfreeze_fc': True,
    },
    'full': {
        'name': 'Full Fine-Tuning',
        'description': 'Train all layers of the network',
        'freeze_all': False,
        'unfreeze_all': True,
    }
}
