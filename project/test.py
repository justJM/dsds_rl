from __future__ import print_function
import sys

from dqn.pong_agent import PongAgent

model = sys.argv[1]

test_agent = PongAgent('test')

test_agent.load_model(model)
test_agent.play(1)

test_agent.close_env()
