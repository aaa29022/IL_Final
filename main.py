import numpy as np
import os
import argparse
from environment import Environment
from driver import Driver


def dispatcher(env, reweight, ensemble):
    print('Reweight:', reweight)
    print('Ensemble:', ensemble)

    driver = Driver(env, reweight, ensemble)

    while driver.itr < env.n_train_iters:

        # Train
        if env.train_mode:
            driver.train_step()

        # Test
        if driver.itr % env.test_interval == 0:

            # measure performance
            R = []
            for n in range(env.n_episodes_test):
                R.append(driver.collect_experience(record=True, vis=env.vis_flag, noise_flag=False, n_steps=1000))

            # update stats
            driver.reward_mean = sum(R) / len(R)
            driver.reward_std = np.std(R)

            # print info line
            driver.print_info_line('full')

            # save snapshot
            if env.train_mode and env.save_models:
                driver.save_model(dir_name=env.config_dir)

            # test prediction (for development)
            # driver.predict_forward_model('./model_results/1/1')

        driver.itr += 1


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reweight', action='store_true',
                        help='Reweight the training examples or not (default: False)')
    parser.add_argument('-e', '--ensemble', type=int, default=1,
                        help='Number of ensemble models (default: 1)')
    args = parser.parse_args()

    # print('Reweight:', args.reweight, 'Ensemble:', args.ensemble)

    # load environment
    env = Environment(os.path.curdir, 'Hopper-v1')

    # start training
    dispatcher(env=env, reweight=args.reweight, ensemble=args.ensemble)
