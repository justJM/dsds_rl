from __future__ import print_function
import argparse

from dqn.pong_agent import PongAgent


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Deep Q Learning')
    parser.add_argument('--replay_buffer_fill_len', type=int, default=100,
                        help='how many elements should replay buffer contain before training start')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training')
    parser.add_argument('--episodes', type=int, default=100,
                        help='how many episodes to iterate')
    parser.add_argument('--max_epsilon_steps', type=int, default=10000,
                        help='maximum number of epsilon steps')
    parser.add_argument('--epsilon_start', type=float, default=1.0,
                        help='start epsilon value')
    parser.add_argument('--epsilon_final', type=float, default=0.2,
                        help='final epsilon value')
    parser.add_argument('--sync_target_net_freq', type=int, default=1000,
                        help='how often to sync estimation and target networks')
    args = parser.parse_args()


    agent = PongAgent()
    agent.train(replay_buffer_fill_len=args.replay_buffer_fill_len,
                batch_size=args.batch_size,
                episodes=args.episodes,
                max_epsilon_steps=args.max_epsilon_steps,
                epsilon_start=args.epsilon_start,
                epsilon_final=args.epsilon_final,
                sync_target_net_freq=args.sync_target_net_freq)

    agent.close_env()



if __name__ == '__main__':
    main()
