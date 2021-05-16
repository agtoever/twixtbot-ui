# -*- coding: utf-8 -*-
"""Neural Network/Monte Carlo Twixt Bot Player

This module implements a Twixt playing bot based on Jordan Lampe's
twixtbot (https://github.com/BonyJordan/twixtbot), which is based on
AlphaZero (https://en.wikipedia.org/wiki/AlphaZero) neural network /
Monte Carlo simulation techniques.

The module was adjusted to fit into the twixtbot-gui. Alterations include:
    - creating callbacks for threaded Monte Carlo evaluation, so that
      the progress of the evaluation can be shown in the GUI.
    - replacing the pick_move method with and eval_game method, which
      allows the user to adjust the bot's strength.

"""

# Module imports of standard Python libraries
import numpy
import random

# Module imports of Joran Lampe's twixtbot libraries
import logging
import constants as ct
import backend.naf as naf
import backend.nneval as nneval
import backend.nnmcts as nnmcts
import backend.swapmodel as swapmodel
import backend.twixt as twixt


def _select_move(moves, P, bot_strength):
    """ Returns a move based on moves, P and bot_strength

    Note:
        A bot_strength < 1.0 results in non-deterministic behaviour

    Args:
        moves (List(twixt.Point)): sorted list of moves; best first
        P (List(float)): P-score for each move [0,1] (higher=better)
        bot_strength ([float]): bot strengt: 0 = weak; 1 = best

    Returns:
        twixt.Point: selected move
    """

    # Normalize bot strength
    w = min(0.99, max(bot_strength, 0.0001))

    # Normalize P
    Parr = numpy.array(P)
    Pnorm = (Parr/sum(Parr))**(w/(1-w))

    # Return P-weighted choice from moves
    sel_move = random.choices(moves, weights=Pnorm)[0]
    print(f'selected move: {sel_move} ({moves.index(sel_move)}-best choice)')
    print(f'Top 10 moves + P: {dict(zip(moves[:10], P[:10]))}')
    return sel_move

def _P1k(P):
    """Scale P list from [0.0, 1.0] to [0, 1000]"""
    return [int(round(1000 * p)) for p in P]

def transform_bot_strength(w):
    if w < 0.0001:
        return 0.0001
    if w > 0.99:
        return 0.99
    return w


class Player:
    """Implements the bot player.

    Attributes (type, default values between brackets):

        *** attributes related to the AlphaZero model ***
        model (str, None): path to saved tenslorflow model ('saved_model.pb')
        num_trials (int, 100): number of Monte Carlo simulations to run.
        temperature (int, 0): controls the policy which move is taken after
            the MCTS. Valid values are one of: 0.0, 0.5 or 1.0
        random_rotation (int, 0): random rotate/mirror the board (0 or 1)
        smart_root (int, 0): smart-select the MCTS starting moves (0 or 1)
        allow_swap (int, 1): allow swapping in second move (0 or 1)
        add_noise (int, 0): add dirichlet noise to P (0 or 1)
        cpuct (int, 1): MCTS constant that balances exploitation (higher
            value) versus exploration (lower value).
        verbosity (int, 0): prints more status info (0, 1, 2 or 3)

        *** other attributes ***
        cached_game_eval (dict): cached versions of (score, P, moves) tuple
            with hash(game) as dict key.
        nm (nnmcts.NeuralMCTS): the actual Neural MCTS model

        # TODO: implement send_message in this class and make nm.eval_game
        # respond to the GUI using a callback. This untangles nnmplayer
        # and nnmcts a bit more...
    """

    def __init__(self, **kwargs):
        """ Creates the bot player.

        Args:
            **kwargs (dict): bot parameters (see class attributes)

        Raises:
            ValueError if an invalid bot parameter is given.
        """
        self.logger = logging.getLogger(ct.LOGGER)
        self.model = kwargs.get('model', None)
        self.trials = int(kwargs.get('trials', 100))
        self.temperature = float(kwargs.get('temperature', 0))
        self.random_rotation = int(kwargs.get('random_rotation', 0))

        self.smart_root = int(kwargs.get('smart_root', 0))
        self.allow_swap = int(kwargs.get('allow_swap', 1))
        self.add_noise = float(kwargs.get('add_noise', 0))
        self.cpuct = float(kwargs.get('cpuct', 1))
        self.board = kwargs.get('board', None)
        self.evaluator = kwargs.get('evaluator', None)

        self.cached_game_eval = {}

        if self.temperature not in (0.0, 0.5, 1.0):
            raise ValueError("Unsupported temperature")

        if self.model:
            # assert not self.socket
            if self.evaluator is None:
                self.evaluator = nneval.NNEvaluater(self.model)

            nneval_ = self.evaluator

            def nnfunc(game):
                nips = naf.NetInputs(game)
                if self.random_rotation:
                    rot = random.randint(0, 3)
                    nips.rotate(rot)
                else:
                    rot = 0
                pw, ml = nneval_.eval_one(nips)
                if len(pw) == 3:
                    pw = naf.three_to_one(pw)
                if len(pw) == 1 and len(pw[0] == 3):
                    pw = naf.three_to_one(pw[0])
                ml = naf.rotate_policy_array(ml, rot)
                if len(ml) == 1:
                    ml = ml[0]
                return pw, ml
        else:
            raise ValueError("Specify model or resource")

        self.nm = nnmcts.NeuralMCTS(
            nnfunc,
            add_noise=self.add_noise,
            smart_root=self.smart_root,
            cpuct=self.cpuct,
            board=self.board,
            visualize_mcts=False
        )

    def pick_move(self,
                  game: twixt.Game,
                  window=None,
                  event=None):
        raise NotImplementedError

    def eval_game(self, game: twixt.Game,
                  bot_strength=1.0, use_mcts=False,
                  window=None, event=None):
        """ Let the robot evaluate the game and return game statistics.

        Notes:
            The underlying nm.eval_game are cached by this function, so there
            is no performance penalty if eval_game is called multiple times
            for the same game state.

            A bot_strength < 1.0 results in non-deterministic behaviour

        Args:
            game (twixt.Game): Twixt game for which the bot evaluates.
            bot_strength (float): bot strengt: 0 = weak; 1 = best. Default 1.
            use_mcts (Bool): use Monte Carlo Simulator. Default False.
            window (PySimpleGUI.Window, optional): windows used for sending
                status updates. Defaults to None.
            event (threading.Event): thread stops if get_state() is True
                MCTS thread status. Defaults to None.

        Returns:
            (score, moves, P, botmove) (Tuple):
                score (float): value head [-1,1] for [black,white]'s strength
                moves (List(twixt.Point)): sorted list of moves; best first
                P (List(float)): P-score for each move [0,1] (higher=better)
                botmove (twixt.Point): move chosen based on bot_strength
        """

        # Evaluate game; uses cached data if possible.
        score, moves, P = self.nm.eval_game(game)

        # Handle first en second move special cases
        if self.allow_swap and len(game.history) < 2:
            if len(game.history) == 0:
                self.report = "swapmodel"
                m = [swapmodel.choose_first_move()] + moves
                P = [1000] + P
                selected_move = _select_move(m, P, bot_strength)
                self.nm.send_message(window, game, "done", moves=m, P=P,
                                     selected_move=selected_move)
                return (score, m, P, selected_move)

            elif swapmodel.want_swap(game.history[0]):
                self.report = "swapmodel"
                m = [twixt.SWAP] + moves
                P = [1000] + P
                selected_move = _select_move(m, P, bot_strength)
                self.nm.send_message(window, game, "done", moves=m, P=P,
                                     selected_move=selected_move)
                return (score, m, P, selected_move)
            # else:
            #   didn't want to swap => compute move

<<<<<<< HEAD
<<<<<<< HEAD
        if self.num_trials == 0:
            # don't use MCTS but just evaluate and return best move
            _, moves, P = self.nm.eval_game(game)
            return self.nm.create_response(game, "done", 0,
                                           0, moves=moves,
                                           P=[int(round(p * 1000)) for p in P])

        N = self.nm.mcts(game, self.num_trials, window, event)
=======
=======
>>>>>>> 45893d1 (Merge of master info adjust bot level (again))
        if self.trials == 0 or not use_mcts:
            # don't use MCTS
            selected_move = _select_move(moves, P, bot_strength)
            self.nm.send_message(window, game, "done", 0, 0, moves=moves,
                                 P=P, selected_move=selected_move)
            return (score, moves, P, selected_move)
>>>>>>> be2d40f (Merged master into code for adjusting bot level)

        N = self.nm.mcts(game, self.trials, window, event)
=======
        if self.num_trials == 0:
            # don't use MCTS but just evaluate and return best move
            _, moves, P = self.nm.eval_game(game)
            self.nm.send_message(window, game, "done", 0,
                                 0, moves=moves, P=[int(round(p * 1000)) for p in P])

            # TODO: implement bot weakness (using parameter w)
            # Idea:
            # - w has a value from 0 (no weakness) to 1 (max. weakness)
            # - w = 0 means strongest play;
            #   w = 1 means weakest play = uniform random choice of move
            #   from any of the moves considered. 
            # *** FOR NN PLAY (WITHOUT MCTS) ***
            # - Transform P to probability weights w = P ^ (1/(1-w))
            #   this gives weights [1,0,0,0...] for w = 0
            #   and weights [1/n,1/n, ..., 1/n] for w = 1
            import random
            w = transform_bot_strength(bot_strength)
            p = normalize(P)**(w/(1-w))
            move = random.choices(moves,weights=p)[0]
            print(f'moves: {moves}\nP: {P}\nw: {w}\np: {p}\nmove: {move}')
            return move
            # return moves[0]

        N = self.nm.mcts(game, self.num_trials, window, event)
>>>>>>> 43a1dcb (Committing temporary state of work on adjusting bot strength)
        self.report = self.nm.report

        # When a forcing win or forcing draw move is found, there's no policy
        # array returned
        if isinstance(N, (str, twixt.Point)):
<<<<<<< HEAD
            return self.nm.create_response(game, "done", self.num_trials,
                                           self.num_trials, True,
                                           P=[1000, 0, 0])
=======
            P = [1000, 0, 0]
            self.nm.send_message(window, game, "done", self.trials,
                                 self.trials, True, P=[1000], moves=[N],
                                 selected_move=N)
            return (score, [N], [1000], N)
>>>>>>> be2d40f (Merged master into code for adjusting bot level)

        """
        if self.temperature == 0.0:
            mx = N.max()
            weights = numpy.where(N == mx, 1.0, 0.0)
        elif self.temperature == 1.0:
            weights = N
        elif self.temperature == 0.5:
            weights = N ** 2
<<<<<<< HEAD
        self.logger.debug("weights=%s", weights)

        # flake8 said index isn't used. Not sure why this code is here.
        # Just to be sure: commenting out the code instead of deleting:
        # index = numpy.random.choice(numpy.arange(
        #     len(weights)), p=weights / weights.sum())

        return self.nm.create_response(game, "done", self.num_trials,
                                       self.num_trials, False)
=======
        self.logger.info("weights=%s", weights)
        index = numpy.random.choice(numpy.arange(
            len(weights)), p=weights / weights.sum())

        self.nm.send_message(window, game, "done", self.trials,
                             self.trials, False)
        """

        moves = [naf.policy_index_point(game, idx) for idx in range(len(N))]
        selected_move = _select_move(moves, N, bot_strength)
        self.nm.send_message(window, game, "done", 0, 0, moves=moves,
                             P=_P1k(P), selected_move=selected_move)
        return (score, moves, _P1k(P), selected_move)
>>>>>>> be2d40f (Merged master into code for adjusting bot level)
