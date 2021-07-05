import json

class Player(object):
  picesOnHand: int
  picesOnBoard: int
  ableToFly: bool
  idx: int
  won: bool
  def __init__(self, idx: int, testmode = False):
    if testmode:
      self.picesOnHand = 1000
    else:
      self.picesOnHand = 9
    self.picesOnBoard = 0
    self.idx = idx
    self.ableToFly = False
    self.won = False
  def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)
