from .GamePhase import GamePhase
from .Player import Player
import numpy as np
from gym import spaces
import gym as gym
from termcolor import colored
class MillEnviroment(gym.Env):
    nonPlayerIdx: int = 2
    #s_in
    killCtrParam: int = 24
    #s_i
    placingCtrParam: int = 24

    cFields: str = "correctFields"
    incFields: str = "incorrectFields"
    cPlanedMove: str = "correctMovePlaned"
    incPlanedMove: str = "incorrectMovePlaned"
    cMove: str = "correctMove"
    incMove: str = "incorrectMove"
    cKillCtrParam: str = "correctKillCtrParam"
    incKillCtrParam: str = "incorrectKillCtrParam"
    cKillField: str = "correctKillField"
    incKillField: str = "incorrectKillField"
    cPlacingCtrParam: str = "correctPlacingCtrParam"
    incPlacingCtrParam: str = "incorrectPlacingCtrParam"
    cPlacingParam: str = "correctPlacingField"
    incPlacingParam: str = "incorrectPlacingField"
    mill: str = "mill"
    star: str = "star"
    twoInRow: str = "twoInRow"
    noReward: str = "noReward"
    won: str = "won"

    # game_board:
    # A0(0)                A3(1)                  A6(2)
    #       B1(3)          B3(4)           B5(5)
    #              C2(6)   C3(7)   C4(8)
    # D0(9) D1(10) D2(11)          D4(12)  D5(13) D6(14)
    #              E2(15)  E3(16)  E4(17)
    #       F1(18)         F3(19)          F5(20)
    # G0(21)               G3(22)                 G6(23)
    reward_range = (-np.inf, np.inf)

    # 25 * 25 Matrix but - 1 -> count from 0
    action_space = spaces.Discrete(624)
    observation_space = spaces.MultiDiscrete([ 
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3,
    ])

    def __init__(self, inRow, notInRow, testmode = False):
        self.testmode = testmode
        self.phase: GamePhase = GamePhase.Placing
        self.inRow: int = inRow
        self.notInRow: int = notInRow
        self.killMode: bool = False
        self.done: bool = False
        self.lastgameStatus: dict[int, int]
        self.gameBoard: dict[int, int] = {
            0: self.nonPlayerIdx,
            1: self.nonPlayerIdx,
            2: self.nonPlayerIdx,
            3: self.nonPlayerIdx,
            4: self.nonPlayerIdx,
            5: self.nonPlayerIdx,
            6: self.nonPlayerIdx,
            7: self.nonPlayerIdx,
            8: self.nonPlayerIdx,
            9: self.nonPlayerIdx,
            10: self.nonPlayerIdx,
            11: self.nonPlayerIdx,
            12: self.nonPlayerIdx,
            13: self.nonPlayerIdx,
            14: self.nonPlayerIdx,
            15: self.nonPlayerIdx,
            16: self.nonPlayerIdx,
            17: self.nonPlayerIdx,
            18: self.nonPlayerIdx,
            19: self.nonPlayerIdx,
            20: self.nonPlayerIdx,
            21: self.nonPlayerIdx,
            22: self.nonPlayerIdx,
            23: self.nonPlayerIdx,
        }

        self.edges: dict = {
            0: [1, 9],
            1: [0, 4, 2],
            2: [1, 14],
            3: [10, 4],
            4: [3, 1, 7, 5],
            5: [4, 13],
            6: [11, 7],
            7: [6, 4, 8],
            8: [7, 12],
            9: [0, 10, 20],
            10: [9, 3, 11, 18],
            11: [6, 10, 15],
            12: [8, 13, 17],
            13: [12, 5, 19, 14],
            14: [2, 13, 22],
            15: [11, 16],
            16: [15, 18, 17],
            17: [12, 16],
            18: [10, 19],
            19: [18, 16, 22, 20],
            20: [19, 13],
            21: [9, 22],
            22: [21, 19, 23],
            23: [22, 14],
        }

        self.mills: list = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10, 11],
            [12, 13, 14],
            [15, 16, 17],
            [18, 19, 20],
            [21, 22, 23],
            [0, 9, 21],
            [3, 10, 18],
            [6, 11, 15],
            [1, 4, 7],
            [16, 19, 22],
            [8, 12, 17],
            [5, 13, 20],
            [2, 14, 23],
        ]

        self.gameBoardMapping: list[list[tuple[int, str]]]  = [
            [[0,"O"],   [-1,"--"], [-1,"--"], [-1,"--"],  [1, "O"], [-1,"--"], [-1,"--"], [-1,"--"],  [2, "O"]],
            [[-1,"| "],  [3, "O"], [-1,"--"], [-1,"--"],  [4, "O"], [-1,"--"], [-1,"--"],  [5, "O"], [-1," | "]],
            [[-1,"| "], [-1,"| "],   [6, "O"], [-1,"--"], [7, "O"], [-1,"--"], [8, "O"], [-1," | "], [-1,"| "], ],
            [[-1,"| "], [-1,"| "],   [-1,"| "],  [-1,"   "], [-1," | "], [-1,"| "], [-1,"| "],],
            [[9, "O"], [-1,"-"], [10, "O"], [-1,"-"], [11, "O"], [-1,"\t  "], [12, "O"], [-1,"-"], [13, "O"], [-1,"-"], [14, "O"]],
            [[-1,"| "], [-1,"| "],   [-1,"| "],  [-1,"   "], [-1," | "], [-1,"| "], [-1,"| "],],
            [[-1,"| "], [-1,"| "],  [15, "O"], [-1,"--"], [16, "O"], [-1,"--"], [17, "O"], [-1," | "], [-1,"| "]],
            [[-1,"| "], [18, "O"], [-1,"--"], [-1,"--"], [19, "O"], [-1,"--"], [-1,"--"],  [20, "O"], [-1," | "]],
            [[21, "O"], [-1,"--"], [-1,"--"], [-1,"--"],  [22, "O"], [-1,"--"], [-1,"--"], [-1,"--"],  [23, "O"]]
        ]

    ##### Helper
    def board_to_list(self):
        return list(self.gameBoard.values())

    def set_gameboard(self, newObsList):
        self.phase = newObsList[len(newObsList)]
        self.inRow = newObsList[len(newObsList)-1]
        if self.inRow == 0:
            self.notInRow = 1
        else: 
            self.notInRow = 0
        newGameBoard = newObsList[:len(newObsList)-2]
        for idx in range(len(newGameBoard)):
            self.gameBoard[idx] = newGameBoard[idx]

    def get_observation_space(self):
        board = self.board_to_list()
        board.append(self.inRow)
        board.append(self.phase.value)
        return np.array(board)

    def discret_mapper(self, action):
        return (action // 25, action % 25 )
    #####

    ##### MOVEMET
    # Check if fields are on gameboard
    def validate_selected_fields_correctness(self, s_i, s_in):
        if (
            s_i in self.gameBoard
            and self.gameBoard[s_i] == self.inRow
            and s_in in self.gameBoard
        ):
            return  True
        return False

    # Check if planed move is correct (the way)
    def validate_planed_move_action(self, s_i, s_in):
        if self.phase == 2:
          if s_i in self.edges and s_in in self.edges[s_i]:
              return True
          return False
        else:
          return True 

    # Check if planed move is correct (moving own pice on empty space)
    def validate_move_correctenes(self, s_i, s_in):
        if (
            s_i in self.gameBoard
            and self.gameBoard[s_i] == self.inRow
            and s_in in self.gameBoard
            and self.gameBoard[s_in] == self.nonPlayerIdx
        ):
            return True
        return False

    ######

    #### REMOVE
    # Check control parameters for remove action
    def validate_control_param_remove_action(self, ctrParam):
        if ctrParam == self.killCtrParam:
            return True
        return False

    # Check if opponent pice selected
    def validate_remove_action_correctenes(self, selectedPice):
        if (
            selectedPice in self.gameBoard
            and self.gameBoard[selectedPice] == self.notInRow
        ):
            return True
        return False

    #####

    ##### PLACING
    # Check control parameters for placing action
    def validate_placing_action_ctr_param(self, ctrParam):
        if ctrParam == self.placingCtrParam:
            return True
        return False

    # Check if empty space selected
    def validate_placing_action(self, selectedSpot):
        if (
            selectedSpot in self.gameBoard
            and self.gameBoard[selectedSpot] == self.nonPlayerIdx
        ):
            return True
        return False

    #####

    ##### PATTERNS
    def check_on_mill(self, newPice):
        for mill in self.mills:
            if newPice in mill:
                for spot in mill:
                    if self.gameBoard[spot] != self.inRow.idx:
                        return False
                self.killMode = True
                return True
            else: 
                return False


    ##### ENV ACTIONS
    # action place piece on position
    def place_pice(self, s_i, s_in):
        changeState: bool = False
        validations = {
            "ctr": self.validate_placing_action_ctr_param(s_i),
            "action": self.validate_placing_action(s_in),
        }
        if validations["ctr"] and validations["action"]:
            self.gameBoard[s_in] = self.inRow
            changeState = True
        return changeState

    # Env action move piece to position
    def move_pice(self, s_i, s_in):
        changeState: bool = False
        validations = {
            "fields": self.validate_selected_fields_correctness(s_i, s_in),
            "move": self.validate_planed_move_action(s_i, s_in),
            "correct": self.validate_move_correctenes(s_i, s_in),
        }
        if (
            validations["fields"]
            and validations["move"]
            and validations["correct"]
        ):
            self.gameBoard[s_i] = self.nonPlayerIdx
            self.gameBoard[s_in] = self.inRow
            changeState = True
        return changeState

    # Env action remove piece on position
    def remove_pice(self, s_i, s_in):
        changeState: bool = True
        validations = {
            "ctr": self.validate_control_param_remove_action(s_in),
            "action": self.validate_remove_action_correctenes(s_i),
        }
        if validations["ctr"] and validations["action"]:
            self.gameBoard[s_in] = self.nonPlayerIdx
            self.killMode = False
        return changeState

    #####

    def step(self, action):
        self.lastgameStatus = self.gameBoard.copy()
        action_tuple = self.discret_mapper(action)
        self.action_tuple = action_tuple
        if self.phase == GamePhase.Placing:
          result: bool = self.place_pice(action_tuple[0], action_tuple[1])
        elif self.phase == GamePhase.Moving and self.killMode: 
          result: bool = self.remove_pice(action_tuple[0], action_tuple[1])
        else: 
          result: bool = self.move_pice(action_tuple[0], action_tuple[1])

        observation = self.get_observation_space()
        info = {'propermove': result, 'gamestate': self.phase, 'action': action_tuple, 'kilmode': self.killMode}
        
        return (observation, result, self.done, info)

    def reset(self):
        self.phase: GamePhase = GamePhase.Placing
        self.inRow: int = 0
        self.notInRow: int = 1
        self.killMode: bool = False
        self.done: bool = False
        self.gameBoard: dict[int, int] = {
            0: self.nonPlayerIdx,
            1: self.nonPlayerIdx,
            2: self.nonPlayerIdx,
            3: self.nonPlayerIdx,
            4: self.nonPlayerIdx,
            5: self.nonPlayerIdx,
            6: self.nonPlayerIdx,
            7: self.nonPlayerIdx,
            8: self.nonPlayerIdx,
            9: self.nonPlayerIdx,
            10: self.nonPlayerIdx,
            11: self.nonPlayerIdx,
            12: self.nonPlayerIdx,
            13: self.nonPlayerIdx,
            14: self.nonPlayerIdx,
            15: self.nonPlayerIdx,
            16: self.nonPlayerIdx,
            17: self.nonPlayerIdx,
            18: self.nonPlayerIdx,
            19: self.nonPlayerIdx,
            20: self.nonPlayerIdx,
            21: self.nonPlayerIdx,
            22: self.nonPlayerIdx,
            23: self.nonPlayerIdx,
        }
        return self.get_observation_space()

    def render(self, mode="human", close=False):
        self.set_gameboard_mapping()
        printBoard = self.search_gameboard_changes()
        gameBoardStr: list = []
        for row in self.gameBoardMapping:
            gameBoardStr.append("\n")
            for entry in row:
                if entry[0] != -1:
                    item = entry
                    item[1] = self.player_to_str(printBoard[item[0]])
                    gameBoardStr.append(item[1])
                else:
                    gameBoardStr.append(entry[1])
        gameBoardStr.append("\n")
        gameBoardStr.append(str(self.action_tuple) )
        visual = ''.join(gameBoardStr)
        print(visual)

    def player_to_str(self, player_num):
        if player_num == 0:
            return "W"
        elif player_num == 1:
            return "B"
        elif player_num == 10:
            return colored("W", "yellow")
        elif player_num == 11:
            return colored("B", "blue")
        elif player_num == 12:
            return colored("O", "white")
        else: 
            return "O"

    def search_gameboard_changes(self):
        printBorad = self.gameBoard.copy()
        print(self.lastgameStatus)
        print(printBorad)
        for idx in range(printBorad.__len__()):
            if printBorad[idx] != self.lastgameStatus[idx]:
                if printBorad[idx] == 0:
                    printBorad[idx] = 10
                elif printBorad[idx] == 1:
                    printBorad[idx] = 11
        return printBorad
        
    def close(self):
        self.__init__()

    def set_gameboard_mapping(self):
        self.gameBoardMapping = [
            [[0,"O"],   [-1,"--"], [-1,"--"], [-1,"--"],  [1, "O"], [-1,"--"], [-1,"--"], [-1,"--"],  [2, "O"]],
            [[-1,"| "],  [3, "O"], [-1,"--"], [-1,"--"],  [4, "O"], [-1,"--"], [-1,"--"],  [5, "O"], [-1," | "]],
            [[-1,"| "], [-1,"| "],   [6, "O"], [-1,"--"], [7, "O"], [-1,"--"], [8, "O"], [-1," | "], [-1,"| "], ],
            [[-1,"| "], [-1,"| "],   [-1,"| "],  [-1,"   "], [-1," | "], [-1,"| "], [-1,"| "],],
            [[9, "O"], [-1,"-"], [10, "O"], [-1,"-"], [11, "O"], [-1,"\t  "], [12, "O"], [-1,"-"], [13, "O"], [-1,"-"], [14, "O"]],
            [[-1,"| "], [-1,"| "],   [-1,"| "],  [-1,"   "], [-1," | "], [-1,"| "], [-1,"| "],],
            [[-1,"| "], [-1,"| "],  [15, "O"], [-1,"--"], [16, "O"], [-1,"--"], [17, "O"], [-1," | "], [-1,"| "]],
            [[-1,"| "], [18, "O"], [-1,"--"], [-1,"--"], [19, "O"], [-1,"--"], [-1,"--"],  [20, "O"], [-1," | "]],
            [[21, "O"], [-1,"--"], [-1,"--"], [-1,"--"],  [22, "O"], [-1,"--"], [-1,"--"], [-1,"--"],  [23, "O"]]
        ]

