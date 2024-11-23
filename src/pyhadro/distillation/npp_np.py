import numpy as np
import jax.numpy as jp
import lqcdpy
from lqcdpy.distillation import combined_contraction, cross_contraction, sequential, loop, corr_fill, PerambBaryon

class PerambNucleonPion2NucleonPionPion(PerambBaryon):
    
    def __init__(self, eps_src, eps_snk, perambs, mins_src, mins_snk, t0, P=None, Gamma_src=None, Gamma_snk=None, 
                 Gamma_pi_src=None, Gamma_pi_snk=None, tval=None):
        self.t0 = t0

        self.eps_src = np.conjugate(eps_src[self.t0])
        self.eps_snk = eps_snk

        self.perambs = perambs
        self.Nt = perambs[0].shape[0]
        self.tval = range(self.Nt) if tval is None else tval

        self.mins_src = mins_src
        self.mins_snk = mins_snk

        self.P = np.eye(4) if P is None else P


        Cg5 = lqcdpy.distillation.mat.nucleon['Cg5']
        self.Gamma_snk = Cg5 if Gamma_snk is None else Gamma_snk
        self.Gamma_src = Cg5 if Gamma_src is None else Gamma_src

        g5 = lqcdpy.distillation.mat.pion['g5']
        self.Gamma_pi_src = g5 if Gamma_pi_src is None else Gamma_pi_src
        self.Gamma_pi_snk = g5 if Gamma_pi_snk is None else Gamma_pi_snk

        self.pions = {
            'src/pi+': lambda t: (self.mins_src[t], self.Gamma_pi_src),
            'snk/pi+': lambda t: (self.mins_snk[t], self.Gamma_pi_snk),
            'snk/pi-': lambda t: (self.mins_snk[t], self.Gamma_pi_snk),
        }

    def seq0(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi+"](t) ])

    def seq1(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi-"](t) ])

    def seq2(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi+"](self.t0) ])

    def seq3(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi+"](t), self.pions["src/pi+"](self.t0) ])

    def seq4(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi+"](t), self.pions["snk/pi-"](t) ])

    def seq5(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi-"](t), self.pions["snk/pi+"](t) ])

    def seq6(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi-"](t), self.pions["snk/pi+"](t), self.pions["src/pi+"](self.t0) ])

    def seq7(self, t): 
        return loop([ self.perambs[t][t], self.perambs[t][t] ], [self.pions["snk/pi+"](t), self.pions["snk/pi-"](t) ])

    def seq8(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi+"](self.t0), self.pions["snk/pi+"](t) ])

    def seq9(self, t): 
        return loop([ self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi+"](t), self.pions["src/pi+"](self.t0) ])

    def seq10(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["src/pi+"](self.t0), self.pions["snk/pi+"](t), self.pions["snk/pi-"](t) ])

    def diagrams(self, t):
        S0 = self.seq0(t)
        S1 = self.seq1(t)
        S2 = self.seq2(t)
        S3 = self.seq3(t)
        S4 = self.seq4(t)
        S5 = self.seq5(t)
        S6 = self.seq6(t)
        S7 = self.seq7(t)
        S8 = self.seq8(t)
        S9 = self.seq9(t)
        S10 = self.seq10(t)


        def contraction(func, ops, factor):
            d0, d1 = func(self.eps_src, self.eps_snk[t], ops, self.P)
            d0 *= factor
            d1 *= factor
            return d0, d1


        # shortcuts

        L = self.perambs[self.t0][t]
        # GX
        GS3 = self.Gamma_snk @ S3
        GL = self.Gamma_snk @ L
        GS4 = self.Gamma_snk @ S4
        GS0 = self.Gamma_snk @ S0

        # XG 
        S6G = S6 @ self.Gamma_src
        S10G = S10 @ self.Gamma_src
        S2G = S2 @ self.Gamma_src
        S1G = S1 @ self.Gamma_src

        # GXG 
        GS3G = self.Gamma_snk @ S3 @ self.Gamma_src
        GLG = self.Gamma_snk @ L @ self.Gamma_src
        GS4G = self.Gamma_snk @ S4 @ self.Gamma_src


        diags = []

        # combined contractions
        for pre, ops in [(1, [S1G, GS0, S2 ]), (1, [L, GS4G, S2 ]), (1, [S2G, GS0, S1 ]), (1, [L, GS3G, S1 ]), (1, [S5, GLG, S2 ]), (1, [L, GLG, S6 ]), (1, [S8, GLG, S1 ]), (1, [L, GLG, S10 ])]:
            di, dj = contraction(combined_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)


        # cross contractions
        for pre, ops in [(1, [S1G, L, GS3 ]), (1, [S2G, L, GS4 ]), (1, [S5, S2G, GL ]), (1, [S6G, L, GL ]), (1, [S1G, S8, GL ]), (1, [S10G, L, GL ])]:
            di, dj = contraction(cross_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)


        # LOOP CONTRACTIONS
        # combined contractions
        for pre, ops, loops in [(-1, [L, GLG, S2 ], [S7]), (-1, [L, GLG, S1 ], [S9])]:
            factor = pre * np.prod(loops)
            di, dj = contraction(combined_contraction, ops, factor=factor)
            diags.append(di)
            diags.append(dj)


        # cross contractions
        for pre, ops, loops in [(-1, [S2G, L, GL ], [S7]), (-1, [S1G, L, GL ], [S9])]:
            factor = pre * np.prod(loops)
            di, dj = contraction(cross_contraction, ops, factor=factor)
            diags.append(di)
            diags.append(dj)


        return diags


    def corr(self, t):
        return - sum(self.diagrams(t))
    


class PerambNucleonPion2NucleonPion0Pion0(PerambBaryon):
    
    def __init__(self, eps_src, eps_snk, perambs, mins_src, mins_snk, t0, P=None, Gamma_src=None, Gamma_snk=None, 
                 Gamma_pi_src=None, Gamma_pi_snk=None, tval=None):
        self.t0 = t0

        self.eps_src = np.conjugate(eps_src[self.t0])
        self.eps_snk = eps_snk

        self.perambs = perambs
        self.Nt = perambs[0].shape[0]
        self.tval = range(self.Nt) if tval is None else tval

        self.mins_src = mins_src
        self.mins_snk = mins_snk

        self.P = np.eye(4) if P is None else P


        Cg5 = lqcdpy.distillation.mat.nucleon['Cg5']
        self.Gamma_snk = Cg5 if Gamma_snk is None else Gamma_snk
        self.Gamma_src = Cg5 if Gamma_src is None else Gamma_src

        g5 = lqcdpy.distillation.mat.pion['g5']
        self.Gamma_pi_src = g5 if Gamma_pi_src is None else Gamma_pi_src
        self.Gamma_pi_snk = g5 if Gamma_pi_snk is None else Gamma_pi_snk

        self.pions = {
            'src/pi+': lambda t: (self.mins_src[t], self.Gamma_pi_src),
            'snk/pi0/0': lambda t: (self.mins_snk[t], self.Gamma_pi_snk),
            'snk/pi0/1': lambda t: (self.mins_snk[t], self.Gamma_pi_snk),
        }
    
    def seq0(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/0"](t), self.pions["src/pi+"](self.t0) ])

    def seq1(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi0/1"](t) ])

    def seq2(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi0/0"](t) ])

    def seq3(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/1"](t), self.pions["src/pi+"](self.t0) ])

    def seq4(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi0/0"](t), self.pions["snk/pi0/1"](t) ])

    def seq5(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi+"](self.t0) ])

    def seq7(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/0"](t), self.pions["snk/pi0/1"](t), self.pions["src/pi+"](self.t0) ])

    def seq9(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi0/1"](t), self.pions["snk/pi0/0"](t) ])

    def seq10(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/1"](t), self.pions["snk/pi0/0"](t), self.pions["src/pi+"](self.t0) ])

    def seq11(self, t): 
        return loop([ self.perambs[t][t], self.perambs[t][t] ], [self.pions["snk/pi0/0"](t), self.pions["snk/pi0/1"](t) ])

    def seq12(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi+"](self.t0), self.pions["snk/pi0/1"](t) ])

    def seq13(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi+"](self.t0), self.pions["snk/pi0/0"](t) ])

    def seq14(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["src/pi+"](self.t0), self.pions["snk/pi0/0"](t), self.pions["snk/pi0/1"](t) ])

    def seq15(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["src/pi+"](self.t0), self.pions["snk/pi0/1"](t), self.pions["snk/pi0/0"](t) ])

    def seq16(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi0/0"](t), self.pions["src/pi+"](self.t0), self.pions["snk/pi0/1"](t) ])

    def seq17(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi0/1"](t), self.pions["src/pi+"](self.t0), self.pions["snk/pi0/0"](t) ])

    def diagrams(self, t):
        S0 = self.seq0(t)
        S1 = self.seq1(t)
        S2 = self.seq2(t)
        S3 = self.seq3(t)
        S4 = self.seq4(t)
        S5 = self.seq5(t)
        S7 = self.seq7(t)
        S9 = self.seq9(t)
        S10 = self.seq10(t)
        S11 = self.seq11(t)
        S12 = self.seq12(t)
        S13 = self.seq13(t)
        S14 = self.seq14(t)
        S15 = self.seq15(t)
        S16 = self.seq16(t)
        S17 = self.seq17(t)


        def contraction(func, ops, factor):
            d0, d1 = func(self.eps_src, self.eps_snk[t], ops, self.P)
            d0 *= factor
            d1 *= factor
            return d0, d1


        # shortcuts

        L = self.perambs[self.t0][t]
        # GX
        GS2 = self.Gamma_snk @ S2
        GS9 = self.Gamma_snk @ S9
        GL = self.Gamma_snk @ L
        GS1 = self.Gamma_snk @ S1
        GS4 = self.Gamma_snk @ S4

        # XG 
        S16G = S16 @ self.Gamma_src
        S0G = S0 @ self.Gamma_src
        S12G = S12 @ self.Gamma_src
        S14G = S14 @ self.Gamma_src
        S7G = S7 @ self.Gamma_src
        S5G = S5 @ self.Gamma_src
        S3G = S3 @ self.Gamma_src
        S15G = S15 @ self.Gamma_src
        S13G = S13 @ self.Gamma_src
        S17G = S17 @ self.Gamma_src
        S10G = S10 @ self.Gamma_src

        # GXG 
        GLG = self.Gamma_snk @ L @ self.Gamma_src
        GS9G = self.Gamma_snk @ S9 @ self.Gamma_src
        GS2G = self.Gamma_snk @ S2 @ self.Gamma_src
        GS1G = self.Gamma_snk @ S1 @ self.Gamma_src
        GS4G = self.Gamma_snk @ S4 @ self.Gamma_src


        diags = []

        # combined contractions
        for pre, ops in [(1, [L, GLG, S7 ]), (1, [S1, GLG, S0 ]), (1, [L, GLG, S10 ]), (1, [S4, GLG, S5 ]), (1, [S9, GLG, S5 ]), 
                         (-1, [L, GLG, S17 ]), (-1, [L, GLG, S16 ]), (1, [L, GLG, S15 ]), (1, [L, GS2G, S12 ]), (1, [L, GS9G, S5 ]), 
                         (-1, [S2, GS1G, S5 ]), (-1, [S1, GS2G, S5 ]), (1, [S2, GLG, S3 ]), (-1, [L, GS1G, S0 ]), (1, [L, GS1G, S13 ]), 
                         (-1, [L, GS2G, S3 ]), (-1, [S2, GLG, S12 ]), (1, [L, GS4G, S5 ]), (1, [L, GLG, S14 ]), (-1, [S1, GLG, S13 ])]:
            di, dj = contraction(combined_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)


        # cross contractions
        for pre, ops in [(1, [S3G, S2, GL ]), (-1, [L, S16G, GL ]), (1, [S7G, L, GL ]), (-1, [L, S0G, GS1 ]), (1, [S5G, L, GS4 ]), 
                         (1, [S13G, L, GS1 ]), (-1, [S12G, S2, GL ]), (1, [S12G, L, GS2 ]), (1, [L, S14G, GL ]), (-1, [S3G, L, GS2 ]), 
                         (1, [S5G, L, GS9 ]), (-1, [S2, S5G, GS1 ]), (-1, [S5G, S1, GS2 ]), (-1, [S17G, L, GL ]), (1, [L, S10G, GL ]), 
                         (1, [S4, S5G, GL ]), (1, [S15G, L, GL ]), (-1, [S13G, S1, GL ]), (1, [S9, S5G, GL ]), (1, [S1, S0G, GL ])]:
            di, dj = contraction(cross_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)


        # LOOP CONTRACTIONS
        # combined contractions
        for pre, ops, loops in [(-2, [L, GLG, S5 ], [S11])]:
            factor = pre * np.prod(loops)
            di, dj = contraction(combined_contraction, ops, factor=factor)
            diags.append(di)
            diags.append(dj)


        # cross contractions
        for pre, ops, loops in [(-2, [S5G, L, GL ], [S11])]:
            factor = pre * np.prod(loops)
            di, dj = contraction(cross_contraction, ops, factor=factor)
            diags.append(di)
            diags.append(dj)


        return diags


    def corr(self, t):
        return - 0.5 * sum(self.diagrams(t))
    


class PerambNucleonPion02NucleonPionPion(PerambBaryon):
    
    def __init__(self, eps_src, eps_snk, perambs, mins_src, mins_snk, t0, P=None, Gamma_src=None, Gamma_snk=None, 
                 Gamma_pi_src=None, Gamma_pi_snk=None, tval=None):
        self.t0 = t0

        self.eps_src = np.conjugate(eps_src[self.t0])
        self.eps_snk = eps_snk

        self.perambs = perambs
        self.Nt = perambs[0].shape[0]
        self.tval = range(self.Nt) if tval is None else tval

        self.mins_src = mins_src
        self.mins_snk = mins_snk

        self.P = np.eye(4) if P is None else P


        Cg5 = lqcdpy.distillation.mat.nucleon['Cg5']
        self.Gamma_snk = Cg5 if Gamma_snk is None else Gamma_snk
        self.Gamma_src = Cg5 if Gamma_src is None else Gamma_src

        g5 = lqcdpy.distillation.mat.pion['g5']
        self.Gamma_pi_src = g5 if Gamma_pi_src is None else Gamma_pi_src
        self.Gamma_pi_snk = g5 if Gamma_pi_snk is None else Gamma_pi_snk

        self.pions = {
            'src/pi0': lambda t: (self.mins_src[t], self.Gamma_pi_src),
            'snk/pi+': lambda t: (self.mins_snk[t], self.Gamma_pi_snk),
            'snk/pi-': lambda t: (self.mins_snk[t], self.Gamma_pi_snk),
        }
    
    def seq0(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi+"](t) ])

    def seq1(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi-"](t) ])

    def seq2(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0"](self.t0) ])

    def seq3(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi+"](t), self.pions["src/pi0"](self.t0) ])

    def seq5(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi+"](t), self.pions["snk/pi-"](t) ])

    def seq6(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi0"](self.t0), self.pions["snk/pi-"](t) ])

    def seq7(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi+"](t), self.pions["src/pi0"](self.t0), self.pions["snk/pi-"](t) ])

    def seq8(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi-"](t), self.pions["snk/pi+"](t) ])

    def seq9(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi-"](t), self.pions["snk/pi+"](t), self.pions["src/pi0"](self.t0) ])

    def seq10(self, t): 
        return loop([ self.perambs[t][t], self.perambs[t][t] ], [self.pions["snk/pi+"](t), self.pions["snk/pi-"](t) ])

    def seq11(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["src/pi0"](self.t0), self.pions["snk/pi-"](t), self.pions["snk/pi+"](t) ])

    def seq12(self, t): 
        return loop([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[t][self.t0] ], [self.pions["snk/pi+"](t), self.pions["src/pi0"](self.t0), self.pions["snk/pi-"](t) ])

    def seq13(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi-"](t), self.pions["src/pi0"](self.t0) ])

    def seq14(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi+"](t), self.pions["snk/pi-"](t), self.pions["src/pi0"](self.t0) ])

    def seq15(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi0"](self.t0), self.pions["snk/pi+"](t) ])

    def seq16(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["src/pi0"](self.t0), self.pions["snk/pi+"](t), self.pions["snk/pi-"](t) ])

    def seq17(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi-"](t), self.pions["src/pi0"](self.t0), self.pions["snk/pi+"](t) ])

    def seq18(self, t): 
        return loop([ self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi+"](t), self.pions["snk/pi-"](t), self.pions["src/pi0"](self.t0) ])

    def diagrams(self, t):
        S0 = self.seq0(t)
        S1 = self.seq1(t)
        S2 = self.seq2(t)
        S3 = self.seq3(t)
        S5 = self.seq5(t)
        S6 = self.seq6(t)
        S7 = self.seq7(t)
        S8 = self.seq8(t)
        S9 = self.seq9(t)
        S10 = self.seq10(t)
        S11 = self.seq11(t)
        S12 = self.seq12(t)
        S13 = self.seq13(t)
        S14 = self.seq14(t)
        S15 = self.seq15(t)
        S16 = self.seq16(t)
        S17 = self.seq17(t)
        S18 = self.seq18(t)


        def contraction(func, ops, factor):
            d0, d1 = func(self.eps_src, self.eps_snk[t], ops, self.P)
            d0 *= factor
            d1 *= factor
            return d0, d1


        # shortcuts

        L = self.perambs[self.t0][t]
        # GX
        GS2 = self.Gamma_snk @ S2
        GS16 = self.Gamma_snk @ S16
        GS5 = self.Gamma_snk @ S5
        GS7 = self.Gamma_snk @ S7
        GS0 = self.Gamma_snk @ S0
        GS15 = self.Gamma_snk @ S15
        GL = self.Gamma_snk @ L
        GS3 = self.Gamma_snk @ S3
        GS14 = self.Gamma_snk @ S14

        # XG 
        LG = L @ self.Gamma_src
        S2G = S2 @ self.Gamma_src
        S8G = S8 @ self.Gamma_src
        S9G = S9 @ self.Gamma_src
        S17G = S17 @ self.Gamma_src
        S11G = S11 @ self.Gamma_src

        # GXG 
        GS15G = self.Gamma_snk @ S15 @ self.Gamma_src
        GS0G = self.Gamma_snk @ S0 @ self.Gamma_src
        GS3G = self.Gamma_snk @ S3 @ self.Gamma_src


        diags = []

        # combined contractions
        for pre, ops in [(1, [S2G, GL, S8 ]), (-1, [S13, GS0G, L ]), (-1, [LG, GL, S17 ]), (1, [S9G, GL, L ]), (-1, [S1, GS15G, L ]), (-1, [LG, GS14, L ]), 
                         (1, [S2G, GS5, L ]), (1, [S8G, GL, S2 ]), (-1, [LG, GS16, L ]), (1, [S6, GS0G, L ]), (1, [LG, GS7, L ]), (-1, [S17G, GL, L ]), 
                         (1, [S1, GS0G, S2 ]), (1, [LG, GS5, S2 ]), (1, [LG, GL, S9 ]), (1, [S11G, GL, L ]), (-1, [LG, GS2, S8 ]), (-1, [S8G, GS2, L ]), 
                         (1, [S1, GS3G, L ]), (1, [LG, GL, S11 ])]:
            di, dj = contraction(combined_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)


        # cross contractions
        for pre, ops in [(-1, [S1, LG, GS15 ]), (-1, [LG, S13, GS0 ]), (1, [S6, LG, GS0 ]), (1, [S1, LG, GS3 ]), (1, [S2G, S1, GS0 ])]:
            di, dj = contraction(cross_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)


        # LOOP CONTRACTIONS
        # combined contractions
        for pre, ops, loops in [(-1, [LG, GL, S2 ], [S10]), (-1, [S2G, GL, L ], [S10]), (1, [LG, GL, L ], [S18]), (-1, [LG, GL, L ], [S12]), (1, [LG, GS2, L ], [S10])]:
            factor = pre * np.prod(loops)
            di, dj = contraction(combined_contraction, ops, factor=factor)
            diags.append(di)
            diags.append(dj)


        return diags


    def corr(self, t):
        return - 1/np.sqrt(2) * sum(self.diagrams(t))
    


class PerambNucleonPion02NucleonPion0Pion0(PerambBaryon):
    
    def __init__(self, eps_src, eps_snk, perambs, mins_src, mins_snk, t0, P=None, Gamma_src=None, Gamma_snk=None, Gamma_pi_src=None, 
                 Gamma_pi_snk=None, tval=None):
        self.t0 = t0

        self.eps_src = np.conjugate(eps_src[self.t0])
        self.eps_snk = eps_snk

        self.perambs = perambs
        self.Nt = perambs[0].shape[0]
        self.tval = range(self.Nt) if tval is None else tval

        self.mins_src = mins_src
        self.mins_snk = mins_snk

        self.P = np.eye(4) if P is None else P


        Cg5 = lqcdpy.distillation.mat.nucleon['Cg5']
        self.Gamma_snk = Cg5 if Gamma_snk is None else Gamma_snk
        self.Gamma_src = Cg5 if Gamma_src is None else Gamma_src

        g5 = lqcdpy.distillation.mat.pion['g5']
        self.Gamma_pi_src = g5 if Gamma_pi_src is None else Gamma_pi_src
        self.Gamma_pi_snk = g5 if Gamma_pi_snk is None else Gamma_pi_snk

        self.pions = {
            'src/pi0': lambda t: (self.mins_src[t], self.Gamma_pi_src),
            'snk/pi0/0': lambda t: (self.mins_snk[t], self.Gamma_pi_snk),
            'snk/pi0/1': lambda t: (self.mins_snk[t], self.Gamma_pi_snk),
        }

    def seq0(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/0"](t), self.pions["src/pi0"](self.t0) ])

    def seq1(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi0/1"](t) ])

    def seq2(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi0/0"](t) ])

    def seq3(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/1"](t), self.pions["src/pi0"](self.t0) ])

    def seq5(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi0/0"](t), self.pions["snk/pi0/1"](t) ])

    def seq6(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0"](self.t0) ])

    def seq8(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi0"](self.t0), self.pions["snk/pi0/1"](t) ])

    def seq9(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/0"](t), self.pions["snk/pi0/1"](t), self.pions["src/pi0"](self.t0) ])

    def seq10(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi0/0"](t), self.pions["src/pi0"](self.t0), self.pions["snk/pi0/1"](t) ])

    def seq11(self, t): 
        return loop([ self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi0/1"](t), self.pions["src/pi0"](self.t0) ])

    def seq13(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi0/1"](t), self.pions["snk/pi0/0"](t) ])

    def seq14(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi0"](self.t0), self.pions["snk/pi0/0"](t) ])

    def seq15(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/1"](t), self.pions["snk/pi0/0"](t), self.pions["src/pi0"](self.t0) ])

    def seq16(self, t): 
        return loop([ self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi0/0"](t), self.pions["src/pi0"](self.t0) ])

    def seq17(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi0/1"](t), self.pions["src/pi0"](self.t0), self.pions["snk/pi0/0"](t) ])

    def seq18(self, t): 
        return loop([ self.perambs[t][t], self.perambs[t][t] ], [self.pions["snk/pi0/0"](t), self.pions["snk/pi0/1"](t) ])

    def seq19(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["src/pi0"](self.t0), self.pions["snk/pi0/0"](t), self.pions["snk/pi0/1"](t) ])

    def seq20(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["src/pi0"](self.t0), self.pions["snk/pi0/1"](t), self.pions["snk/pi0/0"](t) ])

    def diagrams(self, t):
        S0 = self.seq0(t)
        S1 = self.seq1(t)
        S2 = self.seq2(t)
        S3 = self.seq3(t)
        S5 = self.seq5(t)
        S6 = self.seq6(t)
        S8 = self.seq8(t)
        S9 = self.seq9(t)
        S10 = self.seq10(t)
        S11 = self.seq11(t)
        S13 = self.seq13(t)
        S14 = self.seq14(t)
        S15 = self.seq15(t)
        S16 = self.seq16(t)
        S17 = self.seq17(t)
        S18 = self.seq18(t)
        S19 = self.seq19(t)
        S20 = self.seq20(t)


        def contraction(func, ops, factor):
            d0, d1 = func(self.eps_src, self.eps_snk[t], ops, self.P)
            d0 *= factor
            d1 *= factor
            return d0, d1


        # shortcuts

        L = self.perambs[self.t0][t]
        # GX
        GS2 = self.Gamma_snk @ S2
        GS19 = self.Gamma_snk @ S19
        GS9 = self.Gamma_snk @ S9
        GS5 = self.Gamma_snk @ S5
        GS0 = self.Gamma_snk @ S0
        GS10 = self.Gamma_snk @ S10
        GS15 = self.Gamma_snk @ S15
        GS20 = self.Gamma_snk @ S20
        GS6 = self.Gamma_snk @ S6
        GS8 = self.Gamma_snk @ S8
        GL = self.Gamma_snk @ L
        GS1 = self.Gamma_snk @ S1
        GS3 = self.Gamma_snk @ S3
        GS17 = self.Gamma_snk @ S17
        GS14 = self.Gamma_snk @ S14
        GS13 = self.Gamma_snk @ S13

        # XG 
        LG = L @ self.Gamma_src
        S0G = S0 @ self.Gamma_src
        S14G = S14 @ self.Gamma_src
        S20G = S20 @ self.Gamma_src
        S5G = S5 @ self.Gamma_src
        S3G = S3 @ self.Gamma_src
        S2G = S2 @ self.Gamma_src
        S15G = S15 @ self.Gamma_src
        S8G = S8 @ self.Gamma_src
        S19G = S19 @ self.Gamma_src
        S9G = S9 @ self.Gamma_src
        S13G = S13 @ self.Gamma_src
        S6G = S6 @ self.Gamma_src
        S10G = S10 @ self.Gamma_src
        S17G = S17 @ self.Gamma_src
        S1G = S1 @ self.Gamma_src

        # GXG 


        diags = []

        # combined contractions
        for pre, ops in [(1, [S2G, GL, S8 ]), (-1, [LG, GS17, L ]), (-1, [S1G, GS6, S2 ]), (-1, [S0G, GS1, L ]), (1, [S13G, GL, S6 ]), 
                         (1, [LG, GS8, S2 ]), (1, [S10G, GL, L ]), (1, [S6G, GL, S5 ]), (1, [LG, GL, S9 ]), (1, [S2G, GS3, L ]), 
                         (1, [S5G, GL, S6 ]), (1, [S15G, GL, L ]), (-1, [LG, GS2, S8 ]), (1, [LG, GL, S20 ]), (-1, [S6G, GS2, S1 ]), 
                         (-1, [S13G, GS6, L ]), (-1, [LG, GS6, S5 ]), (1, [LG, GS0, S1 ]), (1, [S1G, GL, S14 ]), (1, [S0G, GL, S1 ]), 
                         (-1, [LG, GS9, L ]), (1, [S3G, GL, S2 ]), (1, [S6G, GS5, L ]), (1, [S2G, GS8, L ]), (1, [LG, GS14, S1 ]), 
                         (-1, [LG, GS2, S3 ]), (1, [LG, GL, S10 ]), (1, [S6G, GS13, L ]), (1, [LG, GL, S17 ]), (1, [S19G, GL, L ]), 
                         (-1, [S6G, GS1, S2 ]), (1, [LG, GS3, S2 ]), (-1, [S1G, GS2, S6 ]), (1, [LG, GS13, S6 ]), (1, [S8G, GL, S2 ]), 
                         (-1, [LG, GS1, S14 ]), (-1, [S3G, GS2, L ]), (1, [S6G, GL, S13 ]), (1, [S17G, GL, L ]), (1, [S9G, GL, L ]), 
                         (-1, [S5G, GS6, L ]), (1, [S2G, GL, S3 ]), (-1, [LG, GS6, S13 ]), (1, [LG, GL, S19 ]), (1, [S1G, GS0, L ]), 
                         (-1, [LG, GS19, L ]), (-1, [LG, GS1, S0 ]), (1, [S1G, GS14, L ]), (-1, [LG, GS10, L ]), (1, [S1G, GL, S0 ]), 
                         (-1, [S14G, GS1, L ]), (-1, [S2G, GS1, S6 ]), (1, [LG, GL, S15 ]), (-1, [S8G, GS2, L ]), (-1, [S2G, GS6, S1 ]), 
                         (1, [S14G, GL, S1 ]), (-1, [LG, GS15, L ]), (1, [LG, GS5, S6 ]), (1, [S20G, GL, L ]), (-1, [LG, GS20, L ])]:
            di, dj = contraction(combined_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)


        # combined contractions
        for pre, ops, loops in [(-2, [LG, GL, S1 ], [S16]), (-2, [S1G, GL, L ], [S16]), (-2, [S6G, GL, L ], [S18]), (-2, [LG, GL, S2 ], [S11]), 
                                (2, [LG, GS2, L ], [S11]), (2, [LG, GS1, L ], [S16]), (2, [LG, GS6, L ], [S18]), (-2, [S2G, GL, L ], [S11]), (-2, [LG, GL, S6 ], [S18])]:
            factor = pre * np.prod(loops)
            di, dj = contraction(combined_contraction, ops, factor=factor)
            diags.append(di)
            diags.append(dj)


        return diags


    def corr(self, t):
        return - 1/(2*np.sqrt(2)) * sum(self.diagrams(t))
    


class PerambNucleonPionPion2NucleonPion(PerambBaryon):
    
    def __init__(self, eps_src, eps_snk, perambs, mins_src, mins_snk, t0, P=None, Gamma_src=None, Gamma_snk=None, 
                 Gamma_pi_src=None, Gamma_pi_snk=None, tval=None):
        self.t0 = t0

        self.eps_src = np.conjugate(eps_src[self.t0])
        self.eps_snk = eps_snk

        self.perambs = perambs
        self.Nt = perambs[0].shape[0]
        self.tval = range(self.Nt) if tval is None else tval

        self.mins_src = mins_src
        self.mins_snk = mins_snk

        self.P = np.eye(4) if P is None else P


        Cg5 = lqcdpy.distillation.mat.nucleon['Cg5']
        self.Gamma_snk = Cg5 if Gamma_snk is None else Gamma_snk
        self.Gamma_src = Cg5 if Gamma_src is None else Gamma_src

        g5 = lqcdpy.distillation.mat.pion['g5']
        self.Gamma_pi_src = g5 if Gamma_pi_src is None else Gamma_pi_src
        self.Gamma_pi_snk = g5 if Gamma_pi_snk is None else Gamma_pi_snk

        self.pions = {
            'snk/pi+': lambda t: (self.mins_snk[t], self.Gamma_pi_snk),
            'src/pi+': lambda t: (self.mins_src[t], self.Gamma_pi_src),
            'src/pi-': lambda t: (self.mins_src[t], self.Gamma_pi_src),
        }

    def seq0(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi+"](t) ])

    def seq1(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi-"](self.t0) ])

    def seq2(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi+"](self.t0) ])

    def seq3(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi+"](t), self.pions["src/pi+"](self.t0) ])

    def seq4(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["src/pi-"](self.t0), self.pions["src/pi+"](self.t0) ])

    def seq5(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["src/pi+"](self.t0), self.pions["src/pi-"](self.t0) ])

    def seq6(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi+"](t), self.pions["src/pi+"](self.t0), self.pions["src/pi-"](self.t0) ])

    def seq7(self, t): 
        return loop([ self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["src/pi-"](self.t0), self.pions["src/pi+"](self.t0) ])

    def seq8(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi+"](self.t0), self.pions["snk/pi+"](t) ])

    def seq9(self, t): 
        return loop([ self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi+"](t), self.pions["src/pi+"](self.t0) ])

    def seq10(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi-"](self.t0), self.pions["src/pi+"](self.t0), self.pions["snk/pi+"](t) ])

    def diagrams(self, t):
        S0 = self.seq0(t)
        S1 = self.seq1(t)
        S2 = self.seq2(t)
        S3 = self.seq3(t)
        S4 = self.seq4(t)
        S5 = self.seq5(t)
        S6 = self.seq6(t)
        S7 = self.seq7(t)
        S8 = self.seq8(t)
        S9 = self.seq9(t)
        S10 = self.seq10(t)


        def contraction(func, ops, factor):
            d0, d1 = func(self.eps_src, self.eps_snk[t], ops, self.P)
            d0 *= factor
            d1 *= factor
            return d0, d1


        # shortcuts

        L = self.perambs[self.t0][t]
        # GX
        GS2 = self.Gamma_snk @ S2
        GS8 = self.Gamma_snk @ S8
        GS5 = self.Gamma_snk @ S5
        GL = self.Gamma_snk @ L

        # XG 
        S6G = S6 @ self.Gamma_src
        S0G = S0 @ self.Gamma_src
        S10G = S10 @ self.Gamma_src
        S1G = S1 @ self.Gamma_src

        # GXG 
        GS5G = self.Gamma_snk @ S5 @ self.Gamma_src
        GS8G = self.Gamma_snk @ S8 @ self.Gamma_src
        GLG = self.Gamma_snk @ L @ self.Gamma_src


        diags = []

        # combined contractions
        for pre, ops in [(1, [S0G, GS2, S1 ]), (1, [S1G, GS2, S0 ]), (1, [S3, GLG, S1 ]), (1, [S4, GLG, S0 ]), (1, [L, GS5G, S0 ]), (1, [L, GLG, S6 ]), (1, [L, GS8G, S1 ]), (1, [L, GLG, S10 ])]:
            di, dj = contraction(combined_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)


        # cross contractions
        for pre, ops in [(1, [S3, S1G, GL ]), (1, [S0G, S4, GL ]), (1, [S0G, L, GS5 ]), (1, [S6G, L, GL ]), (1, [S1G, L, GS8 ]), (1, [S10G, L, GL ])]:
            di, dj = contraction(cross_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)


        # LOOP CONTRACTIONS
        # combined contractions
        for pre, ops, loops in [(-1, [L, GLG, S0 ], [S7]), (-1, [L, GLG, S1 ], [S9])]:
            factor = pre * np.prod(loops)
            di, dj = contraction(combined_contraction, ops, factor=factor)
            diags.append(di)
            diags.append(dj)


        # cross contractions
        for pre, ops, loops in [(-1, [S0G, L, GL ], [S7]), (-1, [S1G, L, GL ], [S9])]:
            factor = pre * np.prod(loops)
            di, dj = contraction(cross_contraction, ops, factor=factor)
            diags.append(di)
            diags.append(dj)


        return diags


    def corr(self, t):
        return sum(self.diagrams(t))
    



class PerambNucleonPionPion2NucleonPion0(PerambBaryon):
    
    def __init__(self, eps_src, eps_snk, perambs, mins_src, mins_snk, t0, P=None, Gamma_src=None, Gamma_snk=None, 
                 Gamma_pi_src=None, Gamma_pi_snk=None, tval=None):
        self.t0 = t0

        self.eps_src = np.conjugate(eps_src[self.t0])
        self.eps_snk = eps_snk

        self.perambs = perambs
        self.Nt = perambs[0].shape[0]
        self.tval = range(self.Nt) if tval is None else tval

        self.mins_src = mins_src
        self.mins_snk = mins_snk

        self.P = np.eye(4) if P is None else P


        Cg5 = lqcdpy.distillation.mat.nucleon['Cg5']
        self.Gamma_snk = Cg5 if Gamma_snk is None else Gamma_snk
        self.Gamma_src = Cg5 if Gamma_src is None else Gamma_src

        g5 = lqcdpy.distillation.mat.pion['g5']
        self.Gamma_pi_src = g5 if Gamma_pi_src is None else Gamma_pi_src
        self.Gamma_pi_snk = g5 if Gamma_pi_snk is None else Gamma_pi_snk

        self.pions = {
            'snk/pi0': lambda t: (self.mins_snk[t], self.Gamma_pi_snk),
            'src/pi+': lambda t: (self.mins_src[t], self.Gamma_pi_src),
            'src/pi-': lambda t: (self.mins_src[t], self.Gamma_pi_src),
        }

    def seq0(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi0"](t) ])

    def seq1(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi-"](self.t0) ])

    def seq2(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi+"](self.t0) ])

    def seq3(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0"](t), self.pions["src/pi+"](self.t0) ])

    def seq4(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["src/pi-"](self.t0), self.pions["src/pi+"](self.t0) ])

    def seq6(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi-"](self.t0), self.pions["snk/pi0"](t) ])

    def seq7(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi-"](self.t0), self.pions["snk/pi0"](t), self.pions["src/pi+"](self.t0) ])

    def seq8(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["src/pi+"](self.t0), self.pions["src/pi-"](self.t0) ])

    def seq9(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0"](t), self.pions["src/pi+"](self.t0), self.pions["src/pi-"](self.t0) ])

    def seq10(self, t): 
        return loop([ self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["src/pi-"](self.t0), self.pions["src/pi+"](self.t0) ])

    def seq11(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi+"](self.t0), self.pions["src/pi-"](self.t0), self.pions["snk/pi0"](t) ])

    def seq12(self, t): 
        return loop([ self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0"](t), self.pions["src/pi+"](self.t0), self.pions["src/pi-"](self.t0) ])

    def seq13(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0"](t), self.pions["src/pi-"](self.t0) ])

    def seq14(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0"](t), self.pions["src/pi-"](self.t0), self.pions["src/pi+"](self.t0) ])

    def seq15(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi+"](self.t0), self.pions["snk/pi0"](t) ])

    def seq16(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi-"](self.t0), self.pions["src/pi+"](self.t0), self.pions["snk/pi0"](t) ])

    def seq17(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi+"](self.t0), self.pions["snk/pi0"](t), self.pions["src/pi-"](self.t0) ])

    def seq18(self, t): 
        return loop([ self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0"](t), self.pions["src/pi-"](self.t0), self.pions["src/pi+"](self.t0) ])

    def diagrams(self, t):
        S0 = self.seq0(t)
        S1 = self.seq1(t)
        S2 = self.seq2(t)
        S3 = self.seq3(t)
        S4 = self.seq4(t)
        S6 = self.seq6(t)
        S7 = self.seq7(t)
        S8 = self.seq8(t)
        S9 = self.seq9(t)
        S10 = self.seq10(t)
        S11 = self.seq11(t)
        S12 = self.seq12(t)
        S13 = self.seq13(t)
        S14 = self.seq14(t)
        S15 = self.seq15(t)
        S16 = self.seq16(t)
        S17 = self.seq17(t)
        S18 = self.seq18(t)


        def contraction(func, ops, factor):
            d0, d1 = func(self.eps_src, self.eps_snk[t], ops, self.P)
            d0 *= factor
            d1 *= factor
            return d0, d1


        # shortcuts

        L = self.perambs[self.t0][t]
        # GX
        GS16 = self.Gamma_snk @ S16
        GS7 = self.Gamma_snk @ S7
        GS0 = self.Gamma_snk @ S0
        GS6 = self.Gamma_snk @ S6
        GL = self.Gamma_snk @ L
        GS1 = self.Gamma_snk @ S1
        GS14 = self.Gamma_snk @ S14
        GS13 = self.Gamma_snk @ S13
        GS4 = self.Gamma_snk @ S4

        # XG 
        LG = L @ self.Gamma_src
        S0G = S0 @ self.Gamma_src
        S8G = S8 @ self.Gamma_src
        S9G = S9 @ self.Gamma_src
        S17G = S17 @ self.Gamma_src
        S11G = S11 @ self.Gamma_src

        # GXG 
        GS13G = self.Gamma_snk @ S13 @ self.Gamma_src
        GS6G = self.Gamma_snk @ S6 @ self.Gamma_src
        GS1G = self.Gamma_snk @ S1 @ self.Gamma_src


        diags = []

        # combined contractions
        for pre, ops in [(-1, [LG, GL, S17 ]), (1, [S8G, GL, S0 ]), (1, [S9G, GL, L ]), (-1, [LG, GS14, L ]), (-1, [S15, GS1G, L ]), (1, [LG, GS4, S0 ]), (-1, [S8G, GS0, L ]), (-1, [S2, GS13G, L ]), (1, [S0G, GL, S8 ]), (1, [S0G, GS4, L ]), (-1, [LG, GS16, L ]), (1, [LG, GS7, L ]), (1, [S3, GS1G, L ]), (-1, [S17G, GL, L ]), (1, [LG, GL, S9 ]), (1, [S11G, GL, L ]), (1, [S2, GS6G, L ]), (1, [S2, GS1G, S0 ]), (-1, [LG, GS0, S8 ]), (1, [LG, GL, S11 ])]:
            di, dj = contraction(combined_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)


        # cross contractions
        for pre, ops in [(1, [S2, LG, GS6 ]), (1, [LG, S3, GS1 ]), (-1, [LG, S15, GS1 ]), (1, [S0G, S2, GS1 ]), (-1, [S2, LG, GS13 ])]:
            di, dj = contraction(cross_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)


        # LOOP CONTRACTIONS
        # combined contractions
        for pre, ops, loops in [(-1, [S0G, GL, L ], [S10]), (1, [LG, GL, L ], [S18]), (-1, [LG, GL, L ], [S12]), (1, [LG, GS0, L ], [S10]), (-1, [LG, GL, S0 ], [S10])]:
            factor = pre * np.prod(loops)
            di, dj = contraction(combined_contraction, ops, factor=factor)
            diags.append(di)
            diags.append(dj)


        return diags



    def corr(self, t):
        return 1/np.sqrt(2) * sum(self.diagrams(t))
    


class PerambNucleonPion0Pion02NucleonPion(PerambBaryon):
    
    def __init__(self, eps_src, eps_snk, perambs, mins_src, mins_snk, t0, P=None, Gamma_src=None, Gamma_snk=None, Gamma_pi_src=None, Gamma_pi_snk=None, tval=None):
        self.t0 = t0

        self.eps_src = np.conjugate(eps_src[self.t0])
        self.eps_snk = eps_snk

        self.perambs = perambs
        self.Nt = perambs[0].shape[0]
        self.tval = range(self.Nt) if tval is None else tval

        self.mins_src = mins_src
        self.mins_snk = mins_snk

        self.P = np.eye(4) if P is None else P


        Cg5 = lqcdpy.distillation.mat.nucleon['Cg5']
        self.Gamma_snk = Cg5 if Gamma_snk is None else Gamma_snk
        self.Gamma_src = Cg5 if Gamma_src is None else Gamma_src

        g5 = lqcdpy.distillation.mat.pion['g5']
        self.Gamma_pi_src = g5 if Gamma_pi_src is None else Gamma_pi_src
        self.Gamma_pi_snk = g5 if Gamma_pi_snk is None else Gamma_pi_snk

        self.pions = {
            'snk/pi+': lambda t: (self.mins_snk[t], self.Gamma_pi_snk),
            'src/pi0/0': lambda t: (self.mins_src[t], self.Gamma_pi_src),
            'src/pi0/1': lambda t: (self.mins_src[t], self.Gamma_pi_src),
        }
    
    def seq0(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi+"](t), self.pions["src/pi0/1"](self.t0) ])

    def seq1(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/0"](self.t0) ])

    def seq2(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi+"](t) ])

    def seq3(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/1"](self.t0) ])

    def seq5(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi+"](t), self.pions["src/pi0/0"](self.t0) ])

    def seq6(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/1"](self.t0) ])

    def seq8(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/1"](self.t0), self.pions["src/pi0/0"](self.t0) ])

    def seq9(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi+"](t), self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/1"](self.t0) ])

    def seq10(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi+"](t), self.pions["src/pi0/1"](self.t0), self.pions["src/pi0/0"](self.t0) ])

    def seq11(self, t): 
        return loop([ self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/1"](self.t0) ])

    def seq12(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi0/0"](self.t0), self.pions["snk/pi+"](t) ])

    def seq13(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi0/1"](self.t0), self.pions["snk/pi+"](t) ])

    def seq14(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/1"](self.t0), self.pions["snk/pi+"](t) ])

    def seq15(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi0/1"](self.t0), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi+"](t) ])

    def seq16(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/1"](self.t0), self.pions["snk/pi+"](t), self.pions["src/pi0/0"](self.t0) ])

    def seq17(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/0"](self.t0), self.pions["snk/pi+"](t), self.pions["src/pi0/1"](self.t0) ])

    def diagrams(self, t):
        S0 = self.seq0(t)
        S1 = self.seq1(t)
        S2 = self.seq2(t)
        S3 = self.seq3(t)
        S5 = self.seq5(t)
        S6 = self.seq6(t)
        S8 = self.seq8(t)
        S9 = self.seq9(t)
        S10 = self.seq10(t)
        S11 = self.seq11(t)
        S12 = self.seq12(t)
        S13 = self.seq13(t)
        S14 = self.seq14(t)
        S15 = self.seq15(t)
        S16 = self.seq16(t)
        S17 = self.seq17(t)


        def contraction(func, ops, factor):
            d0, d1 = func(self.eps_src, self.eps_snk[t], ops, self.P)
            d0 *= factor
            d1 *= factor
            return d0, d1


        # shortcuts

        L = self.perambs[self.t0][t]
        # GX
        GS6 = self.Gamma_snk @ S6
        GS8 = self.Gamma_snk @ S8
        GL = self.Gamma_snk @ L
        GS1 = self.Gamma_snk @ S1
        GS3 = self.Gamma_snk @ S3

        # XG 
        S16G = S16 @ self.Gamma_src
        S0G = S0 @ self.Gamma_src
        S12G = S12 @ self.Gamma_src
        S14G = S14 @ self.Gamma_src
        S5G = S5 @ self.Gamma_src
        S2G = S2 @ self.Gamma_src
        S15G = S15 @ self.Gamma_src
        S9G = S9 @ self.Gamma_src
        S13G = S13 @ self.Gamma_src
        S17G = S17 @ self.Gamma_src
        S10G = S10 @ self.Gamma_src

        # GXG 
        GLG = self.Gamma_snk @ L @ self.Gamma_src
        GS8G = self.Gamma_snk @ S8 @ self.Gamma_src
        GS6G = self.Gamma_snk @ S6 @ self.Gamma_src
        GS3G = self.Gamma_snk @ S3 @ self.Gamma_src
        GS1G = self.Gamma_snk @ S1 @ self.Gamma_src


        diags = []

        # combined contractions
        for pre, ops in [(-1, [S1, GLG, S0 ]), (1, [L, GLG, S10 ]), (-1, [L, GS6G, S12 ]), (1, [L, GLG, S9 ]), (1, [S6, GLG, S12 ]), (-1, [L, GLG, S17 ]), 
                         (-1, [L, GLG, S16 ]), (1, [L, GS6G, S5 ]), (1, [L, GLG, S15 ]), (-1, [S1, GS6G, S2 ]), (-1, [S6, GS1G, S2 ]), (-1, [S6, GLG, S5 ]), 
                         (1, [S8, GLG, S2 ]), (1, [S3, GLG, S2 ]), (1, [L, GS3G, S2 ]), (1, [L, GS1G, S0 ]), (-1, [L, GS1G, S13 ]), (1, [L, GS8G, S2 ]), 
                         (1, [L, GLG, S14 ]), (1, [S1, GLG, S13 ])]:
            di, dj = contraction(combined_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)


        # cross contractions
        for pre, ops in [(-1, [L, S16G, GL ]), (1, [L, S0G, GS1 ]), (-1, [S2G, S6, GS1 ]), (-1, [S1, S2G, GS6 ]), (1, [L, S9G, GL ]), (-1, [S12G, L, GS6 ]), 
                         (-1, [S13G, L, GS1 ]), (1, [S6, S12G, GL ]), (1, [L, S14G, GL ]), (-1, [S5G, S6, GL ]), (1, [S2G, S8, GL ]), (1, [S2G, L, GS3 ]), 
                         (1, [L, S5G, GS6 ]), (-1, [S17G, L, GL ]), (1, [S2G, S3, GL ]), (1, [S2G, L, GS8 ]), (1, [L, S10G, GL ]), (1, [S15G, L, GL ]), 
                         (1, [S13G, S1, GL ]), (-1, [S1, S0G, GL ])]:
            di, dj = contraction(cross_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)


        # LOOP CONTRACTIONS
        # combined contractions
        for pre, ops, loops in [(-2, [L, GLG, S2 ], [S11])]:
            factor = pre * np.prod(loops)
            di, dj = contraction(combined_contraction, ops, factor=factor)
            diags.append(di)
            diags.append(dj)


        # cross contractions
        for pre, ops, loops in [(-2, [S2G, L, GL ], [S11])]:
            factor = pre * np.prod(loops)
            di, dj = contraction(cross_contraction, ops, factor=factor)
            diags.append(di)
            diags.append(dj)


        return diags

    def corr(self, t):
        return 1/2 * sum(self.diagrams(t))
    



class PerambNucleonPion0Pion02NucleonPion0(PerambBaryon):
    
    def __init__(self, eps_src, eps_snk, perambs, mins_src, mins_snk, t0, P=None, Gamma_src=None, Gamma_snk=None, Gamma_pi_src=None, Gamma_pi_snk=None, tval=None):
        self.t0 = t0

        self.eps_src = np.conjugate(eps_src[self.t0])
        self.eps_snk = eps_snk

        self.perambs = perambs
        self.Nt = perambs[0].shape[0]
        self.tval = range(self.Nt) if tval is None else tval


        self.mins_src = mins_src
        self.mins_snk = mins_snk

        self.P = np.eye(4) if P is None else P


        Cg5 = lqcdpy.distillation.mat.nucleon['Cg5']
        self.Gamma_snk = Cg5 if Gamma_snk is None else Gamma_snk
        self.Gamma_src = Cg5 if Gamma_src is None else Gamma_src

        g5 = lqcdpy.distillation.mat.pion['g5']
        self.Gamma_pi_src = g5 if Gamma_pi_src is None else Gamma_pi_src
        self.Gamma_pi_snk = g5 if Gamma_pi_snk is None else Gamma_pi_snk

        self.pions = {
            'snk/pi0': lambda t: (self.mins_snk[t], self.Gamma_pi_snk),
            'src/pi0/0': lambda t: (self.mins_src[t], self.Gamma_pi_src),
            'src/pi0/1': lambda t: (self.mins_src[t], self.Gamma_pi_src),
        }

    def seq0(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0"](t), self.pions["src/pi0/1"](self.t0) ])

    def seq1(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/0"](self.t0) ])

    def seq2(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi0"](t) ])

    def seq3(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/1"](self.t0) ])

    def seq5(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0"](t), self.pions["src/pi0/0"](self.t0) ])

    def seq6(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/1"](self.t0) ])

    def seq8(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/1"](self.t0), self.pions["src/pi0/0"](self.t0) ])

    def seq9(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0"](t), self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/1"](self.t0) ])

    def seq10(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0"](t), self.pions["src/pi0/1"](self.t0), self.pions["src/pi0/0"](self.t0) ])

    def seq11(self, t): 
        return loop([ self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/1"](self.t0) ])

    def seq13(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi0/0"](self.t0), self.pions["snk/pi0"](t) ])

    def seq14(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi0/1"](self.t0), self.pions["snk/pi0"](t) ])

    def seq15(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/0"](self.t0), self.pions["snk/pi0"](t), self.pions["src/pi0/1"](self.t0) ])

    def seq16(self, t): 
        return loop([ self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi0"](t), self.pions["src/pi0/1"](self.t0) ])

    def seq17(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/1"](self.t0), self.pions["snk/pi0"](t) ])

    def seq18(self, t): 
        return loop([ self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi0"](t), self.pions["src/pi0/0"](self.t0) ])

    def seq19(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/1"](self.t0), self.pions["snk/pi0"](t), self.pions["src/pi0/0"](self.t0) ])

    def seq20(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi0/1"](self.t0), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi0"](t) ])

    def diagrams(self, t):
        S0 = self.seq0(t)
        S1 = self.seq1(t)
        S2 = self.seq2(t)
        S3 = self.seq3(t)
        S5 = self.seq5(t)
        S6 = self.seq6(t)
        S8 = self.seq8(t)
        S9 = self.seq9(t)
        S10 = self.seq10(t)
        S11 = self.seq11(t)
        S13 = self.seq13(t)
        S14 = self.seq14(t)
        S15 = self.seq15(t)
        S16 = self.seq16(t)
        S17 = self.seq17(t)
        S18 = self.seq18(t)
        S19 = self.seq19(t)
        S20 = self.seq20(t)


        def contraction(func, ops, factor):
            d0, d1 = func(self.eps_src, self.eps_snk[t], ops, self.P)
            d0 *= factor
            d1 *= factor
            return d0, d1


        # shortcuts

        L = self.perambs[self.t0][t]
        # GX
        GS2 = self.Gamma_snk @ S2
        GS19 = self.Gamma_snk @ S19
        GS9 = self.Gamma_snk @ S9
        GS5 = self.Gamma_snk @ S5
        GS0 = self.Gamma_snk @ S0
        GS10 = self.Gamma_snk @ S10
        GS15 = self.Gamma_snk @ S15
        GS20 = self.Gamma_snk @ S20
        GS6 = self.Gamma_snk @ S6
        GS8 = self.Gamma_snk @ S8
        GL = self.Gamma_snk @ L
        GS1 = self.Gamma_snk @ S1
        GS3 = self.Gamma_snk @ S3
        GS17 = self.Gamma_snk @ S17
        GS14 = self.Gamma_snk @ S14
        GS13 = self.Gamma_snk @ S13

        # XG 
        LG = L @ self.Gamma_src
        S0G = S0 @ self.Gamma_src
        S14G = S14 @ self.Gamma_src
        S20G = S20 @ self.Gamma_src
        S5G = S5 @ self.Gamma_src
        S3G = S3 @ self.Gamma_src
        S2G = S2 @ self.Gamma_src
        S15G = S15 @ self.Gamma_src
        S8G = S8 @ self.Gamma_src
        S19G = S19 @ self.Gamma_src
        S9G = S9 @ self.Gamma_src
        S13G = S13 @ self.Gamma_src
        S6G = S6 @ self.Gamma_src
        S10G = S10 @ self.Gamma_src
        S17G = S17 @ self.Gamma_src
        S1G = S1 @ self.Gamma_src

        # GXG 


        diags = []

        # combined contractions
        for pre, ops in [(1, [S2G, GL, S8 ]), (-1, [LG, GS17, L ]), (-1, [S1G, GS6, S2 ]), (-1, [S0G, GS1, L ]), (1, [S13G, GL, S6 ]), (1, [LG, GS8, S2 ]), (1, [S10G, GL, L ]), (1, [S6G, GL, S5 ]), (1, [LG, GL, S9 ]), (1, [S2G, GS3, L ]), (1, [S5G, GL, S6 ]), (1, [S15G, GL, L ]), (-1, [LG, GS2, S8 ]), (1, [LG, GL, S20 ]), (-1, [S6G, GS2, S1 ]), (-1, [S13G, GS6, L ]), (-1, [LG, GS6, S5 ]), (1, [LG, GS0, S1 ]), (1, [S1G, GL, S14 ]), (1, [S0G, GL, S1 ]), (-1, [LG, GS9, L ]), (1, [S3G, GL, S2 ]), (1, [S6G, GS5, L ]), (1, [S2G, GS8, L ]), (1, [LG, GS14, S1 ]), (-1, [LG, GS2, S3 ]), (1, [LG, GL, S10 ]), (1, [S6G, GS13, L ]), (1, [LG, GL, S17 ]), (1, [S19G, GL, L ]), (-1, [S6G, GS1, S2 ]), (1, [LG, GS3, S2 ]), (-1, [S1G, GS2, S6 ]), (1, [LG, GS13, S6 ]), (1, [S8G, GL, S2 ]), (-1, [LG, GS1, S14 ]), (-1, [S3G, GS2, L ]), (1, [S6G, GL, S13 ]), (1, [S17G, GL, L ]), (1, [S9G, GL, L ]), (-1, [S5G, GS6, L ]), (1, [S2G, GL, S3 ]), (-1, [LG, GS6, S13 ]), (1, [LG, GL, S19 ]), (1, [S1G, GS0, L ]), (-1, [LG, GS19, L ]), (-1, [LG, GS1, S0 ]), (1, [S1G, GS14, L ]), (-1, [LG, GS10, L ]), (1, [S1G, GL, S0 ]), (-1, [S14G, GS1, L ]), (-1, [S2G, GS1, S6 ]), (1, [LG, GL, S15 ]), (-1, [S8G, GS2, L ]), (-1, [S2G, GS6, S1 ]), (1, [S14G, GL, S1 ]), (-1, [LG, GS15, L ]), (1, [LG, GS5, S6 ]), (1, [S20G, GL, L ]), (-1, [LG, GS20, L ])]:
            di, dj = contraction(combined_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)


        # combined contractions
        for pre, ops, loops in [(-2, [LG, GL, S1 ], [S16]), (-2, [S1G, GL, L ], [S16]), (-2, [S6G, GL, L ], [S18]), (-2, [LG, GL, S2 ], [S11]), (2, [LG, GS2, L ], [S11]), (2, [LG, GS1, L ], [S16]), (2, [LG, GS6, L ], [S18]), (-2, [S2G, GL, L ], [S11]), (-2, [LG, GL, S6 ], [S18])]:
            factor = pre * np.prod(loops)
            di, dj = contraction(combined_contraction, ops, factor=factor)
            diags.append(di)
            diags.append(dj)
            
        return diags


    def corr(self, t):
        return 1/(2*np.sqrt(2)) * sum(self.diagrams(t))
    


