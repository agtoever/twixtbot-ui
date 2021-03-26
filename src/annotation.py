# -*- coding: utf-8 -*-

import backend.twixt as twixt


class Annotation:
    def __init__(self, moves_history, botA, botB, allow_self_crossing_lines):
        self.scl = allow_self_crossing_lines
        self.game = twixt.Game(allow_self_crossing_lines)
        self.bots = [botA, botB]
        self.analysis = {'moves': list(moves_history)}
        self.annotation = ''

    def annotate(self):        
        self.perform_moves_analysis()
        
        for func in [self.intro,
                     self.statistics,
                     self.policy_compare,
                     self.weakest_move
                     ]:
            self.annotation += func()
        return self.annotation

    def _analyze_current_move(self):
        # Let the bot evaluate the game
        score, pmoves, P = self.bots[self.game.turn].nm.eval_game(
            self.game, self.game.SIZE**2)
        score = (2 * self.game.turn - 1) * score
        
        # Prune bot output to P > 0
        P0 = P.index(0)
        pmoves, P = pmoves[:P0], P[:P0]
        
        if len(self.game.history) > 0:
            move_in_policy = self.game.history[-1] in self.analysis['policy_moves'][-1]
        else:
            move_in_policy = None
        
        if move_in_policy:
            rel_rank = 1 - (self.analysis['policy_moves'][-1].index(self.game.history[-1]) / len(self.analysis['policy_moves'][-1]))
        else:
            rel_rank = None

        if len(self.game.history) > 1:
            score_incr1 = score > self.analysis['score'][-1]
            score_incr1_amnt = score - self.analysis['score'][-1]
            score_incr2 = score > self.analysis['score'][-2]
            score_incr2_amnt = score - self.analysis['score'][-2]
        else:
            score_incr1 = score_incr1_amnt = 0
            score_incr2 = score_incr2_amnt = 0

        for key, value in {
                'score': score,
                'policy_moves': pmoves,
                'policy_P': P,
                'score_increased1': score_incr1,
                'score_increased1_amount': score_incr1_amnt,
                'score_increased2': score_incr2,
                'score_increased2_amount': score_incr2_amnt,
                'move_in_policy': move_in_policy,
                'relative_rank': rel_rank}.items():
            if key in self.analysis:
                self.analysis[key].append(value)
            elif value is not None:
                self.analysis[key] = [value]

    def perform_moves_analysis(self):
        # Replay moves, while capturing relevant data
        self._analyze_current_move()  # TODO: deze kan misschien weg...
        for move in self.analysis['moves']:
            self.game.play(move)
            self._analyze_current_move()
        print(self.analysis)

    def intro(self):
        text = ('TwixT bot gui analyzes your game and gives hints on '
                'on the pivotal moments in the game. The analysis is based '
                'on the evaluation of the game by the AlphaZero bot. '
                'This means that the anaysis itself is biased by the ' 
                'perspective of that bot.\n\n')
        return text

    def statistics(self):
        text = ''
        for player in [0, 1]:
            turns = len(self.analysis['moves'][player::2])
            winprob = ((-2 * player + 1) * self.analysis['score'][-1] + 1) * 50
            incr1_moves = sum(self.analysis['score_increased1'][player+1::2])
            incr2_moves = sum(self.analysis['score_increased2'][player+1::2])
            
            text += (f'*** GAME STATISTICS FOR PLAYER {player + 1} ***\n'
                     f'- Turns played: {turns}\n'
                     f'- Current probability to win: {winprob:.0f}%\n'
                     f'- Move count that increased score relative to the '
                     f'last move of the other player: {incr1_moves} '
                     f'({incr1_moves * 100 // turns}% of the moves)\n'
                     f'- Move count that increased score relative to the last'
                     f' own move: {incr2_moves} ({incr2_moves * 100 // turns}%'
                     f' of the moves)\n'
                     '\n\n'
                     )
        return text

    def policy_compare(self):
        text = ''
        for player in [0, 1]:
            turns = len(self.analysis['moves'][player::2])
            pol_moves = sum(self.analysis['move_in_policy'][player::2])
            non_pol_moves = [(f'{i + 1}.{str(move).upper()} - a better move '
                              f"is: {str(pmoves[0]).upper()}")
                             for i, (move, pmoves) in enumerate(zip(
                                self.analysis['moves'],
                                self.analysis['policy_moves']))
                             if (not self.analysis['move_in_policy'][i])
                             and (i % 2 == player)]

            rel_rank = list(filter(None, self.analysis['relative_rank']))
            rel_rank = 100 * sum(rel_rank) // len(rel_rank)

            text += (f'*** POLICY COMPARISON FOR PLAYER {player + 1} ***\n'
                     f'- Move count within policy: {pol_moves} ('
                     f'{pol_moves * 100 // turns}% of the moves)\n'
                     f'- Moves outside policy (potential bad moves): '
                     f" {', '.join(non_pol_moves)}\n"
                     f'- Relative rank of the policy moves (0 = worst; 100'
                     f' = best): {rel_rank:.0f}% best hit.\n'
                     '\n\n'
                     )
        return text
    
    def weakest_move(self):
        text = ''
        for player in [0, 1]:
            wm_sc = [(-2 * player + 1) * sc for sc in
                     self.analysis['score_increased2_amount'][player::2]]
            worst_move_idx = wm_sc.index(min(wm_sc)) * 2 - player
            worst_move_str = (f'{worst_move_idx + 1}. ' +
                str(self.analysis['moves'][worst_move_idx]).upper())
            alt_move = str(self.analysis['policy_moves'][worst_move_idx][0]).upper()

            text += (f'*** WEAKEST MOVE ANALYSIS FOR PLAYER {player + 1} ***\n'
                     f'- Worst move by score evaluation: {worst_move_str}; '
                     f'best alternative: {alt_move}'
                     '\n\n'
                     )
        return text
