import unittest
from env import MillEnv_Descrete


class TestMillEnv_Descrete(unittest.TestCase):
    env = MillEnv_Descrete.MillEnviroment(True)

    def move_generator(self, testMethode):
      for i in range(624):
        testMethode(self.env.discret_mapper(i), i)

    def checkPlacingPhase(self, tup: tuple, action: int):
      env = MillEnv_Descrete.MillEnviroment(True)
      if tup[0] == 24:
        info = env.step(action)[3]
        print('PLACING PHASE CHECK',info)
        self.assertIs(info['propermove'], True)

    def checkMovingPhase(self, tup: tuple, action: int):
      env = MillEnv_Descrete.MillEnviroment(False)
      for i in range(17):
        self.env.step(600 + i)
      if tup[0] < 24:
        info = env.step(action)[3]
        print('MOVING PHASE CHECK',info)
        self.assertIs(info['propermove'], True)

    def test_placing_phase(self):
      self.move_generator(self.checkPlacingPhase)

    def test_moving_phase(self):
      self.move_generator(self.checkMovingPhase)

if __name__ == '__main__':
    unittest.main()