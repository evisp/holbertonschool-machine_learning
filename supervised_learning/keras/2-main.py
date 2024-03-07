#!/usr/bin/env python3

import tensorflow as tf

build_model = __import__('1-input').build_model
optimize_model = __import__('2-optimize').optimize_model

if __name__ == '__main__':
    model = build_model(784, [256, 256, 10], ['tanh', 'tanh', 'softmax'], 0.001, 0.95)
    optimize_model(model, 0.01, 0.99, 0.9)
    print(model.loss)
    opt_config = model.optimizer.get_config()
    print(type(model.optimizer))
    print(opt_config['learning_rate'], opt_config['beta_1'], opt_config['beta_2'])
