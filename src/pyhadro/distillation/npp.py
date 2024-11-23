import numpy as np
import jax.numpy as jp
import lqcdpy
from lqcdpy.distillation import combined_contraction, cross_contraction, sequential, loop, corr_fill, PerambBaryon
 



class PerambNucleonPionPion2NucleonPionPion(PerambBaryon):
    
    def __init__(self, eps_src, perambs, mins_src_pi_plus, t0, P=None, eps_snk=None, 
                 mins_src_pi_minus=None, mins_snk_pi_plus=None, mins_snk_pi_minus=None,
                 Gamma_src_pi_plus=None, Gamma_src_pi_minus=None, Gamma_snk_pi_plus=None, Gamma_snk_pi_minus=None,
                 Gamma_src=None, Gamma_snk=None, tval=None):
        
        ## < O_p(x)  O_pi+(z)  O_pi-(q)  ~O_pi-(s) ~O_pi+(w) ~O_p(y) >

        self.t0 = t0

        self.eps_src = np.conjugate(eps_src[self.t0])
        if eps_snk is None:
            self.eps_snk = eps_src
        else:
            self.eps_snk = eps_snk

        
        self.perambs = perambs
        self.Nt = perambs[0].shape[0]

        self.tval = range(self.Nt) if tval is None else tval


        self.mins_src_pi_plus  = mins_src_pi_plus
        self.mins_src_pi_minus = mins_src_pi_plus if mins_src_pi_minus is None else mins_src_pi_minus
        self.mins_snk_pi_plus  = mins_src_pi_plus if mins_snk_pi_plus is None else mins_snk_pi_plus
        self.mins_snk_pi_minus = mins_src_pi_plus if mins_snk_pi_minus is None else mins_snk_pi_minus

        g_default = lqcdpy.distillation.mat.pion['g5']

        self.Gamma_src_pi_plus  = g_default if Gamma_src_pi_plus is None else Gamma_src_pi_plus
        self.Gamma_src_pi_minus = g_default if Gamma_src_pi_minus is None else Gamma_src_pi_minus
        self.Gamma_snk_pi_plus  = g_default if Gamma_snk_pi_plus is None else Gamma_snk_pi_plus
        self.Gamma_snk_pi_minus = g_default if Gamma_snk_pi_minus is None else Gamma_snk_pi_minus


        self.pions = {
            'src/pi+': lambda t: (self.mins_src_pi_plus[t], self.Gamma_src_pi_plus),
            'src/pi-': lambda t: (self.mins_src_pi_minus[t], self.Gamma_src_pi_minus),
            'snk/pi+': lambda t: (self.mins_snk_pi_plus[t], self.Gamma_snk_pi_plus),
            'snk/pi-': lambda t: (self.mins_snk_pi_minus[t], self.Gamma_snk_pi_minus),
        }

        cg_default = lqcdpy.distillation.mat.nucleon['Cg5']

        self.Gamma_src = cg_default if Gamma_src is None else Gamma_src
        self.Gamma_snk = cg_default if Gamma_snk is None else Gamma_snk


        self.P = np.eye(4) if P is None else P

    def seq0(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi+"](t) ])

    def seq1(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi-"](t), self.pions["src/pi-"](self.t0) ])

    def seq2(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi+"](self.t0) ])

    def seq3(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi+"](t), self.pions["src/pi+"](self.t0) ])

    def seq4(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi-"](t), self.pions["src/pi-"](self.t0), self.pions["src/pi+"](self.t0) ])

    def seq5(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi+"](t), self.pions["snk/pi-"](t), self.pions["src/pi-"](self.t0) ])

    def seq6(self, t): 
        return loop([ self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi-"](t), self.pions["src/pi-"](self.t0) ])

    def seq7(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi+"](t), self.pions["snk/pi-"](t), self.pions["src/pi-"](self.t0), self.pions["src/pi+"](self.t0) ])

    def seq8(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi-"](t) ])

    def seq9(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["src/pi+"](self.t0), self.pions["src/pi-"](self.t0) ])

    def seq10(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi+"](t), self.pions["src/pi+"](self.t0), self.pions["src/pi-"](self.t0) ])

    def seq11(self, t): 
        return loop([ self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["src/pi-"](self.t0), self.pions["src/pi+"](self.t0) ])

    def seq12(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi+"](t), self.pions["snk/pi-"](t) ])

    def seq13(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi+"](self.t0), self.pions["src/pi-"](self.t0), self.pions["snk/pi-"](t) ])

    def seq14(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi+"](t), self.pions["src/pi+"](self.t0), self.pions["src/pi-"](self.t0), self.pions["snk/pi-"](t) ])

    def seq15(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi-"](t), self.pions["snk/pi+"](t) ])

    def seq16(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi-"](self.t0) ])

    def seq17(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi-"](t), self.pions["snk/pi+"](t), self.pions["src/pi+"](self.t0) ])

    def seq18(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["src/pi-"](self.t0), self.pions["src/pi+"](self.t0) ])

    def seq19(self, t): 
        return loop([ self.perambs[t][t], self.perambs[t][t] ], [self.pions["snk/pi+"](t), self.pions["snk/pi-"](t) ])

    def seq20(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["src/pi-"](self.t0), self.pions["snk/pi-"](t), self.pions["snk/pi+"](t) ])

    def seq21(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi-"](self.t0), self.pions["snk/pi-"](t), self.pions["snk/pi+"](t), self.pions["src/pi+"](self.t0) ])

    def seq22(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi+"](self.t0), self.pions["snk/pi+"](t) ])

    def seq23(self, t): 
        return loop([ self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi+"](t), self.pions["src/pi+"](self.t0) ])

    def seq24(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi-"](self.t0), self.pions["src/pi+"](self.t0), self.pions["snk/pi+"](t) ])

    def seq25(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["src/pi+"](self.t0), self.pions["snk/pi+"](t), self.pions["snk/pi-"](t) ])

    def seq26(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi-"](self.t0), self.pions["snk/pi-"](t) ])

    def seq27(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["src/pi-"](self.t0), self.pions["src/pi+"](self.t0), self.pions["snk/pi+"](t), self.pions["snk/pi-"](t) ])

    def seq28(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi-"](t), self.pions["snk/pi+"](t), self.pions["src/pi+"](self.t0), self.pions["src/pi-"](self.t0) ])

    def seq29(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["src/pi+"](self.t0), self.pions["src/pi-"](self.t0), self.pions["snk/pi-"](t), self.pions["snk/pi+"](t) ])

    def seq30(self, t): 
        return loop([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0] ], [self.pions["snk/pi+"](t), self.pions["src/pi+"](self.t0), self.pions["src/pi-"](self.t0), self.pions["snk/pi-"](t) ])

    def seq31(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi-"](t), self.pions["src/pi-"](self.t0), self.pions["src/pi+"](self.t0), self.pions["snk/pi+"](t) ])

    def seq32(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi+"](self.t0), self.pions["snk/pi+"](t), self.pions["snk/pi-"](t), self.pions["src/pi-"](self.t0) ])

    def seq33(self, t): 
        return loop([ self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi+"](t), self.pions["snk/pi-"](t), self.pions["src/pi-"](self.t0), self.pions["src/pi+"](self.t0) ])

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
        S11 = self.seq11(t)
        S12 = self.seq12(t)
        S13 = self.seq13(t)
        S14 = self.seq14(t)
        S15 = self.seq15(t)
        S16 = self.seq16(t)
        S17 = self.seq17(t)
        S18 = self.seq18(t)
        S19 = self.seq19(t)
        S20 = self.seq20(t)
        S21 = self.seq21(t)
        S22 = self.seq22(t)
        S23 = self.seq23(t)
        S24 = self.seq24(t)
        S25 = self.seq25(t)
        S26 = self.seq26(t)
        S27 = self.seq27(t)
        S28 = self.seq28(t)
        S29 = self.seq29(t)
        S30 = self.seq30(t)
        S31 = self.seq31(t)
        S32 = self.seq32(t)
        S33 = self.seq33(t)


        def contraction(func, ops, factor):
            d0, d1 = func(self.eps_src, self.eps_snk[t], ops, self.P)
            d0 *= factor
            d1 *= factor
            return d0, d1


        # shortcuts

        L = self.perambs[self.t0][t]
        # GX
        GS27 = self.Gamma_snk @ S27
        GS16 = self.Gamma_snk @ S16
        GS5 = self.Gamma_snk @ S5
        GS7 = self.Gamma_snk @ S7
        GS26 = self.Gamma_snk @ S26
        GS0 = self.Gamma_snk @ S0
        GS10 = self.Gamma_snk @ S10
        GS18 = self.Gamma_snk @ S18
        GS20 = self.Gamma_snk @ S20
        GL = self.Gamma_snk @ L
        GS3 = self.Gamma_snk @ S3
        GS14 = self.Gamma_snk @ S14
        GS12 = self.Gamma_snk @ S12
        GS21 = self.Gamma_snk @ S21
        GS24 = self.Gamma_snk @ S24

        # XG 
        LG = L @ self.Gamma_src
        S31G = S31 @ self.Gamma_src
        S32G = S32 @ self.Gamma_src
        S28G = S28 @ self.Gamma_src
        S29G = S29 @ self.Gamma_src
        S15G = S15 @ self.Gamma_src
        S9G = S9 @ self.Gamma_src
        S22G = S22 @ self.Gamma_src
        S1G = S1 @ self.Gamma_src

        # GXG 
        GS24G = self.Gamma_snk @ S24 @ self.Gamma_src
        GS16G = self.Gamma_snk @ S16 @ self.Gamma_src
        GS0G = self.Gamma_snk @ S0 @ self.Gamma_src
        GS5G = self.Gamma_snk @ S5 @ self.Gamma_src
        GS10G = self.Gamma_snk @ S10 @ self.Gamma_src
        GS20G = self.Gamma_snk @ S20 @ self.Gamma_src


        diags = []

        # combined contractions
        for pre, ops in [(-1, [S2, GS0G, S1 ]), (-1, [LG, GS3, S1 ]), (-1, [S1G, GS3, L ]), (-1, [S4, GS0G, L ]), (-1, [S2, GS5G, L ]), (-1, [LG, GS7, L ]), 
                         (-1, [S8, GS0G, S9 ]), (-1, [S8, GS10G, L ]), (-1, [LG, GS12, S9 ]), (-1, [S9G, GS12, L ]), (-1, [S13, GS0G, L ]), (-1, [LG, GS14, L ]), 
                         (-1, [S2, GS16G, S15 ]), (-1, [LG, GS18, S15 ]), (-1, [S17, GS16G, L ]), (-1, [S15G, GS18, L ]), (-1, [S2, GS20G, L ]), (-1, [LG, GS21, L ]), 
                         (-1, [S8, GS16G, S22 ]), (-1, [S8, GS24G, L ]), (-1, [LG, GS26, S22 ]), (-1, [S25, GS16G, L ]), (-1, [S22G, GS26, L ]), (-1, [LG, GS27, L ]), 
                         (-1, [S15G, GL, S9 ]), (-1, [S9G, GL, S15 ]), (-1, [LG, GL, S28 ]), (-1, [S28G, GL, L ]), (-1, [LG, GL, S29 ]), (-1, [S29G, GL, L ]), 
                         (-1, [S22G, GL, S1 ]), (-1, [S1G, GL, S22 ]), (-1, [LG, GL, S31 ]), (-1, [S31G, GL, L ]), (-1, [LG, GL, S32 ]), (-1, [S32G, GL, L ])]:
            di, dj = contraction(combined_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)


        # cross contractions
        for pre, ops in [(-1, [S1G, S2, GS0 ]), (-1, [S4, LG, GS0 ]), (-1, [S2, LG, GS5 ]), (-1, [S8, S9G, GS0 ]), (-1, [S8, LG, GS10 ]), (-1, [S13, LG, GS0 ]), 
                         (-1, [S15G, S2, GS16 ]), (-1, [S17, LG, GS16 ]), (-1, [S2, LG, GS20 ]), (-1, [S8, S22G, GS16 ]), (-1, [S8, LG, GS24 ]), (-1, [S25, LG, GS16 ])]:
            di, dj = contraction(cross_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)


        # LOOP CONTRACTIONS
        # combined contractions
        for pre, ops, loops in [(1, [S2, GS0G, L ], [S6]), (1, [LG, GS3, L ], [S6]), (1, [S8, GS0G, L ], [S11]), (1, [LG, GS12, L ], [S11]), (1, [S2, GS16G, L ], [S19]), 
                                (1, [LG, GS18, L ], [S19]), (1, [S8, GS16G, L ], [S23]), (1, [LG, GS26, L ], [S23]), (1, [LG, GL, S15 ], [S11]), (1, [S15G, GL, L ], [S11]), 
                                (1, [LG, GL, S9 ], [S19]), (1, [S9G, GL, L ], [S19]), (-1, [LG, GL, L ], [S19, S11]), (1, [LG, GL, L ], [S30]), (1, [LG, GL, S1 ], [S23]), 
                                (1, [S1G, GL, L ], [S23]), (1, [LG, GL, S22 ], [S6]), (1, [S22G, GL, L ], [S6]), (1, [LG, GL, L ], [S33]), (-1, [LG, GL, L ], [S6, S23])]:
            factor = pre * np.prod(loops)
            di, dj = contraction(combined_contraction, ops, factor=factor)
            diags.append(di)
            diags.append(dj)


        # cross contractions
        for pre, ops, loops in [(1, [S2, LG, GS0 ], [S6]), (1, [S8, LG, GS0 ], [S11]), (1, [S2, LG, GS16 ], [S19]), (1, [S8, LG, GS16 ], [S23])]:
            factor = pre * np.prod(loops)
            di, dj = contraction(cross_contraction, ops, factor=factor)
            diags.append(di)
            diags.append(dj)


        return diags


    
    def corr(self, t):
        return sum(self.diagrams(t))
    



class PerambNucleon2NucleonPionPion(PerambBaryon):

    def __init__(self, eps_src, perambs, mins_snk_pi_plus, t0, P_src=None, P_snk=None, eps_snk=None, mins_snk_pi_minus=None,
                 Gamma_snk_pi_plus=None, Gamma_snk_pi_minus=None, Gamma_src=None, Gamma_snk=None, tval=None):
        
        ## < O_p(x)  O_pi+(z)  O_pi-(q)  ~O_p(y) >

        self.t0 = t0

        self.eps_src = np.conjugate(eps_src[self.t0])
        if eps_snk is None:
            self.eps_snk = eps_src
        else:
            self.eps_snk = eps_snk

        
        self.perambs = perambs
        self.Nt = perambs[0].shape[0]
        self.tval = range(self.Nt) if tval is None else tval

        self.mins_snk_pi_plus  = mins_snk_pi_plus if mins_snk_pi_plus is None else mins_snk_pi_plus
        self.mins_snk_pi_minus = mins_snk_pi_plus if mins_snk_pi_minus is None else mins_snk_pi_minus

        g_default = lqcdpy.distillation.mat.pion['g5']

        self.Gamma_snk_pi_plus  = g_default if Gamma_snk_pi_plus is None else Gamma_snk_pi_plus
        self.Gamma_snk_pi_minus = g_default if Gamma_snk_pi_minus is None else Gamma_snk_pi_minus


        self.pions = {
            'snk/pi+': lambda t: (self.mins_snk_pi_plus[t], self.Gamma_snk_pi_plus),
            'snk/pi-': lambda t: (self.mins_snk_pi_minus[t], self.Gamma_snk_pi_minus),
        }

        cg_default = lqcdpy.distillation.mat.nucleon['Cg5']

        self.Gamma_src = cg_default if Gamma_src is None else Gamma_src
        self.Gamma_snk = cg_default if Gamma_snk is None else Gamma_snk

        self.P_src = np.eye(4) if P_src is None else P_src
        self.P_snk = np.eye(4) if P_snk is None else P_snk

    def seq0(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi+"](t) ])

    def seq1(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi-"](t) ])

    def seq2(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi+"](t), self.pions["snk/pi-"](t) ])

    def seq3(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi-"](t), self.pions["snk/pi+"](t) ])

    def seq4(self, t): 
        return loop([ self.perambs[t][t], self.perambs[t][t] ], [self.pions["snk/pi+"](t), self.pions["snk/pi-"](t) ])

    def diagrams(self, t):
        S0 = self.seq0(t)
        S1 = self.seq1(t)
        S2 = self.seq2(t)
        S3 = self.seq3(t)
        S4 = self.seq4(t)


        def contraction(func, ops, factor):
            d0, d1 = func(self.eps_src, self.eps_snk[t], ops, self.P_snk, self.P_src)
            d0 *= factor
            d1 *= factor
            return d0, d1


        # shortcuts

        L = self.perambs[self.t0][t]
        # GX
        GS2 = self.Gamma_snk @ S2
        GL = self.Gamma_snk @ L
        GS0 = self.Gamma_snk @ S0

        # XG 
        S3G = S3 @ self.Gamma_src
        LG = L @ self.Gamma_src

        # GXG 
        GS0G = self.Gamma_snk @ S0 @ self.Gamma_src


        diags = []

        # combined contractions
        for pre, ops in [(-1, [S1, GS0G, L ]), (-1, [LG, GS2, L ]), (-1, [LG, GL, S3 ]), (-1, [S3G, GL, L ])]:
            di, dj = contraction(combined_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)


        # cross contractions
        for pre, ops in [(-1, [S1, LG, GS0 ])]:
            di, dj = contraction(cross_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)


        # LOOP CONTRACTIONS
        # combined contractions
        for pre, ops, loops in [(1, [LG, GL, L ], [S4])]:
            factor = pre * np.prod(loops)
            di, dj = contraction(combined_contraction, ops, factor=factor)
            diags.append(di)
            diags.append(dj)


        return diags

    def corr(self, t):
        return sum(self.diagrams(t))
    


class PerambNucleonPionPion2Nucleon(PerambBaryon):

    def __init__(self, eps_src, perambs, mins_src_pi_plus, t0, P_snk=None, P_src=None, eps_snk=None, mins_src_pi_minus=None,
                 Gamma_src_pi_plus=None, Gamma_src_pi_minus=None, Gamma_src=None, Gamma_snk=None, tval=None):
        
        ## < O_p(x)  O_pi+(z)  O_pi-(q)  ~O_p(y) >

        self.t0 = t0

        self.eps_src = np.conjugate(eps_src[self.t0])
        if eps_snk is None:
            self.eps_snk = eps_src
        else:
            self.eps_snk = eps_snk

        
        self.perambs = perambs
        self.Nt = perambs[0].shape[0]
        self.tval = range(self.Nt) if tval is None else tval

        self.mins_src_pi_plus  = mins_src_pi_plus if mins_src_pi_plus is None else mins_src_pi_plus
        self.mins_src_pi_minus = mins_src_pi_plus if mins_src_pi_minus is None else mins_src_pi_minus

        g_default = lqcdpy.distillation.mat.pion['g5']

        self.Gamma_src_pi_plus  = g_default if Gamma_src_pi_plus is None else Gamma_src_pi_plus
        self.Gamma_src_pi_minus = g_default if Gamma_src_pi_minus is None else Gamma_src_pi_minus


        self.pions = {
            'src/pi+': lambda t: (self.mins_src_pi_plus[t], self.Gamma_src_pi_plus),
            'src/pi-': lambda t: (self.mins_src_pi_minus[t], self.Gamma_src_pi_minus),
        }

        cg_default = lqcdpy.distillation.mat.nucleon['Cg5']

        self.Gamma_src = cg_default if Gamma_src is None else Gamma_src
        self.Gamma_snk = cg_default if Gamma_snk is None else Gamma_snk

        self.P_snk = np.eye(4) if P_snk is None else P_snk
        self.P_src = np.eye(4) if P_src is None else P_src

    def seq0(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi-"](self.t0) ])

    def seq1(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi+"](self.t0) ])

    def seq2(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["src/pi-"](self.t0), self.pions["src/pi+"](self.t0) ])

    def seq3(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["src/pi+"](self.t0), self.pions["src/pi-"](self.t0) ])

    def seq4(self, t): 
        return loop([ self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["src/pi-"](self.t0), self.pions["src/pi+"](self.t0) ])

    def diagrams(self, t):
        S0 = self.seq0(t)
        S1 = self.seq1(t)
        S2 = self.seq2(t)
        S3 = self.seq3(t)
        S4 = self.seq4(t)


        def contraction(func, ops, factor):
            d0, d1 = func(self.eps_src, self.eps_snk[t], ops, self.P_snk, self.P_src)
            d0 *= factor
            d1 *= factor
            return d0, d1


        # shortcuts

        L = self.perambs[self.t0][t]
        # GX
        GS2 = self.Gamma_snk @ S2
        GL = self.Gamma_snk @ L
        GS0 = self.Gamma_snk @ S0

        # XG 
        S3G = S3 @ self.Gamma_src
        LG = L @ self.Gamma_src

        # GXG 
        GS0G = self.Gamma_snk @ S0 @ self.Gamma_src


        diags = []

        # combined contractions
        for pre, ops in [(-1, [S1, GS0G, L ]), (-1, [LG, GS2, L ]), (-1, [LG, GL, S3 ]), (-1, [S3G, GL, L ])]:
            di, dj = contraction(combined_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)


        # cross contractions
        for pre, ops in [(-1, [S1, LG, GS0 ])]:
            di, dj = contraction(cross_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)


        # LOOP CONTRACTIONS
        # combined contractions
        for pre, ops, loops in [(1, [LG, GL, L ], [S4])]:
            factor = pre * np.prod(loops)
            di, dj = contraction(combined_contraction, ops, factor=factor)
            diags.append(di)
            diags.append(dj)


        return diags


    
    def corr(self, t):
        return sum(self.diagrams(t))
    


class PerambNucleonPionPion2NucleonPion0Pion0(PerambBaryon):
    
    def __init__(self, eps_src, perambs, mins_src_pi_plus, t0, P=None, eps_snk=None, 
                 mins_src_pi_minus=None, mins_snk_pi_plus=None, mins_snk_pi_minus=None,
                 Gamma_src_pi_plus=None, Gamma_src_pi_minus=None, Gamma_snk_pi_plus=None, Gamma_snk_pi_minus=None,
                 Gamma_src=None, Gamma_snk=None, tval=None):
        
        ## < O_p(x)  O_pi+(z)  O_pi-(q)  ~O_pi-(s) ~O_pi+(w) ~O_p(y) >

        self.t0 = t0

        self.eps_src = np.conjugate(eps_src[self.t0])
        if eps_snk is None:
            self.eps_snk = eps_src
        else:
            self.eps_snk = eps_snk

        
        self.perambs = perambs
        self.Nt = perambs[0].shape[0]
        self.tval = range(self.Nt) if tval is None else tval


        self.mins_src_pi_plus  = mins_src_pi_plus
        self.mins_src_pi_minus = mins_src_pi_plus if mins_src_pi_minus is None else mins_src_pi_minus
        self.mins_snk_pi_plus  = mins_src_pi_plus if mins_snk_pi_plus is None else mins_snk_pi_plus
        self.mins_snk_pi_minus = mins_src_pi_plus if mins_snk_pi_minus is None else mins_snk_pi_minus

        g_default = lqcdpy.distillation.mat.pion['g5']

        self.Gamma_src_pi_plus  = g_default if Gamma_src_pi_plus is None else Gamma_src_pi_plus
        self.Gamma_src_pi_minus = g_default if Gamma_src_pi_minus is None else Gamma_src_pi_minus
        self.Gamma_snk_pi_plus  = g_default if Gamma_snk_pi_plus is None else Gamma_snk_pi_plus
        self.Gamma_snk_pi_minus = g_default if Gamma_snk_pi_minus is None else Gamma_snk_pi_minus


        self.pions = {
            'src/pi+': lambda t: (self.mins_src_pi_plus[t], self.Gamma_src_pi_plus),
            'src/pi-': lambda t: (self.mins_src_pi_minus[t], self.Gamma_src_pi_minus),
            'snk/pi0/0': lambda t: (self.mins_snk_pi_plus[t], self.Gamma_snk_pi_plus),
            'snk/pi0/1': lambda t: (self.mins_snk_pi_minus[t], self.Gamma_snk_pi_minus),
        }

        cg_default = lqcdpy.distillation.mat.nucleon['Cg5']

        self.Gamma_src = cg_default if Gamma_src is None else Gamma_src
        self.Gamma_snk = cg_default if Gamma_snk is None else Gamma_snk


        self.P = np.eye(4) if P is None else P

    def seq0(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/0"](t), self.pions["src/pi+"](self.t0) ])

    def seq1(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi0/1"](t) ])

    def seq2(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi-"](self.t0) ])

    def seq3(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi0/0"](t) ])

    def seq4(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/1"](t), self.pions["src/pi+"](self.t0) ])

    def seq5(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["src/pi-"](self.t0), self.pions["src/pi+"](self.t0) ])

    def seq6(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi0/0"](t), self.pions["snk/pi0/1"](t) ])

    def seq7(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi+"](self.t0) ])

    def seq9(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi-"](self.t0), self.pions["snk/pi0/1"](t) ])

    def seq10(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/0"](t), self.pions["snk/pi0/1"](t), self.pions["src/pi+"](self.t0) ])

    def seq11(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi-"](self.t0), self.pions["snk/pi0/1"](t), self.pions["src/pi+"](self.t0) ])

    def seq13(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi0/1"](t), self.pions["snk/pi0/0"](t) ])

    def seq14(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi-"](self.t0), self.pions["snk/pi0/0"](t) ])

    def seq15(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/1"](t), self.pions["snk/pi0/0"](t), self.pions["src/pi+"](self.t0) ])

    def seq16(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi-"](self.t0), self.pions["snk/pi0/0"](t), self.pions["src/pi+"](self.t0) ])

    def seq17(self, t): 
        return loop([ self.perambs[t][t], self.perambs[t][t] ], [self.pions["snk/pi0/0"](t), self.pions["snk/pi0/1"](t) ])

    def seq18(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["src/pi-"](self.t0), self.pions["snk/pi0/0"](t), self.pions["snk/pi0/1"](t) ])

    def seq19(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["src/pi-"](self.t0), self.pions["snk/pi0/1"](t), self.pions["snk/pi0/0"](t) ])

    def seq20(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi-"](self.t0), self.pions["snk/pi0/0"](t), self.pions["snk/pi0/1"](t), self.pions["src/pi+"](self.t0) ])

    def seq21(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi-"](self.t0), self.pions["snk/pi0/1"](t), self.pions["snk/pi0/0"](t), self.pions["src/pi+"](self.t0) ])

    def seq22(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/0"](t), self.pions["src/pi+"](self.t0), self.pions["src/pi-"](self.t0) ])

    def seq23(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/1"](t), self.pions["src/pi+"](self.t0), self.pions["src/pi-"](self.t0) ])

    def seq24(self, t): 
        return loop([ self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["src/pi-"](self.t0), self.pions["src/pi+"](self.t0) ])

    def seq25(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["src/pi+"](self.t0), self.pions["src/pi-"](self.t0) ])

    def seq26(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi+"](self.t0), self.pions["src/pi-"](self.t0), self.pions["snk/pi0/1"](t) ])

    def seq27(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/0"](t), self.pions["snk/pi0/1"](t), self.pions["src/pi+"](self.t0), self.pions["src/pi-"](self.t0) ])

    def seq28(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi0/0"](t), self.pions["src/pi+"](self.t0), self.pions["src/pi-"](self.t0), self.pions["snk/pi0/1"](t) ])

    def seq29(self, t): 
        return loop([ self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/1"](t), self.pions["src/pi+"](self.t0), self.pions["src/pi-"](self.t0) ])

    def seq30(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi+"](self.t0), self.pions["src/pi-"](self.t0), self.pions["snk/pi0/0"](t) ])

    def seq31(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/1"](t), self.pions["snk/pi0/0"](t), self.pions["src/pi+"](self.t0), self.pions["src/pi-"](self.t0) ])

    def seq32(self, t): 
        return loop([ self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/0"](t), self.pions["src/pi+"](self.t0), self.pions["src/pi-"](self.t0) ])

    def seq33(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi0/1"](t), self.pions["src/pi+"](self.t0), self.pions["src/pi-"](self.t0), self.pions["snk/pi0/0"](t) ])

    def seq34(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["src/pi+"](self.t0), self.pions["src/pi-"](self.t0), self.pions["snk/pi0/0"](t), self.pions["snk/pi0/1"](t) ])

    def seq35(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["src/pi+"](self.t0), self.pions["src/pi-"](self.t0), self.pions["snk/pi0/1"](t), self.pions["snk/pi0/0"](t) ])

    def seq36(self, t): 
        return loop([ self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/0"](t), self.pions["snk/pi0/1"](t), self.pions["src/pi+"](self.t0), self.pions["src/pi-"](self.t0) ])

    def seq37(self, t): 
        return loop([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0] ], [self.pions["snk/pi0/0"](t), self.pions["src/pi+"](self.t0), self.pions["src/pi-"](self.t0), self.pions["snk/pi0/1"](t) ])

    def seq38(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/0"](t), self.pions["snk/pi0/1"](t), self.pions["src/pi-"](self.t0) ])

    def seq39(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/0"](t), self.pions["snk/pi0/1"](t), self.pions["src/pi-"](self.t0), self.pions["src/pi+"](self.t0) ])

    def seq40(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/0"](t), self.pions["src/pi-"](self.t0) ])

    def seq41(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/0"](t), self.pions["src/pi-"](self.t0), self.pions["src/pi+"](self.t0) ])

    def seq42(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi+"](self.t0), self.pions["snk/pi0/1"](t) ])

    def seq43(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi0/0"](t), self.pions["src/pi-"](self.t0), self.pions["src/pi+"](self.t0), self.pions["snk/pi0/1"](t) ])

    def seq44(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi+"](self.t0), self.pions["snk/pi0/1"](t), self.pions["src/pi-"](self.t0) ])

    def seq45(self, t): 
        return loop([ self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/1"](t), self.pions["src/pi-"](self.t0), self.pions["src/pi+"](self.t0) ])

    def seq46(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/1"](t), self.pions["src/pi-"](self.t0) ])

    def seq47(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/1"](t), self.pions["src/pi-"](self.t0), self.pions["src/pi+"](self.t0) ])

    def seq48(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/1"](t), self.pions["snk/pi0/0"](t), self.pions["src/pi-"](self.t0) ])

    def seq49(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/1"](t), self.pions["snk/pi0/0"](t), self.pions["src/pi-"](self.t0), self.pions["src/pi+"](self.t0) ])

    def seq50(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi+"](self.t0), self.pions["snk/pi0/0"](t), self.pions["src/pi-"](self.t0) ])

    def seq51(self, t): 
        return loop([ self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/0"](t), self.pions["src/pi-"](self.t0), self.pions["src/pi+"](self.t0) ])

    def seq52(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi+"](self.t0), self.pions["snk/pi0/0"](t) ])

    def seq53(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi0/1"](t), self.pions["src/pi-"](self.t0), self.pions["src/pi+"](self.t0), self.pions["snk/pi0/0"](t) ])

    def seq54(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi-"](self.t0), self.pions["src/pi+"](self.t0), self.pions["snk/pi0/1"](t) ])

    def seq55(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["src/pi+"](self.t0), self.pions["snk/pi0/0"](t), self.pions["snk/pi0/1"](t) ])

    def seq56(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["src/pi-"](self.t0), self.pions["src/pi+"](self.t0), self.pions["snk/pi0/0"](t), self.pions["snk/pi0/1"](t) ])

    def seq57(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["src/pi+"](self.t0), self.pions["snk/pi0/1"](t), self.pions["snk/pi0/0"](t) ])

    def seq58(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["src/pi-"](self.t0), self.pions["src/pi+"](self.t0), self.pions["snk/pi0/1"](t), self.pions["snk/pi0/0"](t) ])

    def seq59(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi-"](self.t0), self.pions["src/pi+"](self.t0), self.pions["snk/pi0/0"](t) ])

    def seq60(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi+"](self.t0), self.pions["snk/pi0/0"](t), self.pions["snk/pi0/1"](t), self.pions["src/pi-"](self.t0) ])

    def seq61(self, t): 
        return loop([ self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/0"](t), self.pions["snk/pi0/1"](t), self.pions["src/pi-"](self.t0), self.pions["src/pi+"](self.t0) ])

    def seq62(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi+"](self.t0), self.pions["snk/pi0/1"](t), self.pions["snk/pi0/0"](t), self.pions["src/pi-"](self.t0) ])

    def seq63(self, t): 
        return loop([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0] ], [self.pions["snk/pi0/0"](t), self.pions["src/pi-"](self.t0), self.pions["src/pi+"](self.t0), self.pions["snk/pi0/1"](t) ])

    def seq64(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi0/1"](t), self.pions["src/pi-"](self.t0), self.pions["snk/pi0/0"](t) ])

    def seq65(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/1"](t), self.pions["src/pi-"](self.t0), self.pions["snk/pi0/0"](t), self.pions["src/pi+"](self.t0) ])

    def seq66(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi0/0"](t), self.pions["src/pi+"](self.t0), self.pions["snk/pi0/1"](t) ])

    def seq67(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi-"](self.t0), self.pions["snk/pi0/0"](t), self.pions["src/pi+"](self.t0), self.pions["snk/pi0/1"](t) ])

    def seq68(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/0"](t), self.pions["src/pi+"](self.t0), self.pions["snk/pi0/1"](t), self.pions["src/pi-"](self.t0) ])

    def seq69(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi+"](self.t0), self.pions["snk/pi0/1"](t), self.pions["src/pi-"](self.t0), self.pions["snk/pi0/0"](t) ])

    def seq70(self, t): 
        return loop([ self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi0/0"](t), self.pions["src/pi+"](self.t0), self.pions["snk/pi0/1"](t), self.pions["src/pi-"](self.t0) ])

    def seq71(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi0/0"](t), self.pions["src/pi-"](self.t0), self.pions["snk/pi0/1"](t) ])

    def seq72(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/0"](t), self.pions["src/pi-"](self.t0), self.pions["snk/pi0/1"](t), self.pions["src/pi+"](self.t0) ])

    def seq73(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi0/1"](t), self.pions["src/pi+"](self.t0), self.pions["snk/pi0/0"](t) ])

    def seq74(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi-"](self.t0), self.pions["snk/pi0/1"](t), self.pions["src/pi+"](self.t0), self.pions["snk/pi0/0"](t) ])

    def seq75(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/1"](t), self.pions["src/pi+"](self.t0), self.pions["snk/pi0/0"](t), self.pions["src/pi-"](self.t0) ])

    def seq76(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi+"](self.t0), self.pions["snk/pi0/0"](t), self.pions["src/pi-"](self.t0), self.pions["snk/pi0/1"](t) ])

    def seq77(self, t): 
        return loop([ self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi0/0"](t), self.pions["src/pi-"](self.t0), self.pions["snk/pi0/1"](t), self.pions["src/pi+"](self.t0) ])

    def diagrams(self, t):
        S0 = self.seq0(t)
        S1 = self.seq1(t)
        S2 = self.seq2(t)
        S3 = self.seq3(t)
        S4 = self.seq4(t)
        S5 = self.seq5(t)
        S6 = self.seq6(t)
        S7 = self.seq7(t)
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
        S21 = self.seq21(t)
        S22 = self.seq22(t)
        S23 = self.seq23(t)
        S24 = self.seq24(t)
        S25 = self.seq25(t)
        S26 = self.seq26(t)
        S27 = self.seq27(t)
        S28 = self.seq28(t)
        S29 = self.seq29(t)
        S30 = self.seq30(t)
        S31 = self.seq31(t)
        S32 = self.seq32(t)
        S33 = self.seq33(t)
        S34 = self.seq34(t)
        S35 = self.seq35(t)
        S36 = self.seq36(t)
        S37 = self.seq37(t)
        S38 = self.seq38(t)
        S39 = self.seq39(t)
        S40 = self.seq40(t)
        S41 = self.seq41(t)
        S42 = self.seq42(t)
        S43 = self.seq43(t)
        S44 = self.seq44(t)
        S45 = self.seq45(t)
        S46 = self.seq46(t)
        S47 = self.seq47(t)
        S48 = self.seq48(t)
        S49 = self.seq49(t)
        S50 = self.seq50(t)
        S51 = self.seq51(t)
        S52 = self.seq52(t)
        S53 = self.seq53(t)
        S54 = self.seq54(t)
        S55 = self.seq55(t)
        S56 = self.seq56(t)
        S57 = self.seq57(t)
        S58 = self.seq58(t)
        S59 = self.seq59(t)
        S60 = self.seq60(t)
        S61 = self.seq61(t)
        S62 = self.seq62(t)
        S63 = self.seq63(t)
        S64 = self.seq64(t)
        S65 = self.seq65(t)
        S66 = self.seq66(t)
        S67 = self.seq67(t)
        S68 = self.seq68(t)
        S69 = self.seq69(t)
        S70 = self.seq70(t)
        S71 = self.seq71(t)
        S72 = self.seq72(t)
        S73 = self.seq73(t)
        S74 = self.seq74(t)
        S75 = self.seq75(t)
        S76 = self.seq76(t)
        S77 = self.seq77(t)


        def contraction(func, ops, factor):
            d0, d1 = func(self.eps_src, self.eps_snk[t], ops, self.P)
            d0 *= factor
            d1 *= factor
            return d0, d1


        # shortcuts

        L = self.perambs[self.t0][t]
        # GX
        GS71 = self.Gamma_snk @ S71
        GS5 = self.Gamma_snk @ S5
        GS38 = self.Gamma_snk @ S38
        GS53 = self.Gamma_snk @ S53
        GS41 = self.Gamma_snk @ S41
        GS40 = self.Gamma_snk @ S40
        GS3 = self.Gamma_snk @ S3
        GS56 = self.Gamma_snk @ S56
        GS21 = self.Gamma_snk @ S21
        GS54 = self.Gamma_snk @ S54
        GS9 = self.Gamma_snk @ S9
        GS16 = self.Gamma_snk @ S16
        GS58 = self.Gamma_snk @ S58
        GS46 = self.Gamma_snk @ S46
        GS20 = self.Gamma_snk @ S20
        GS65 = self.Gamma_snk @ S65
        GL = self.Gamma_snk @ L
        GS59 = self.Gamma_snk @ S59
        GS48 = self.Gamma_snk @ S48
        GS39 = self.Gamma_snk @ S39
        GS2 = self.Gamma_snk @ S2
        GS19 = self.Gamma_snk @ S19
        GS74 = self.Gamma_snk @ S74
        GS49 = self.Gamma_snk @ S49
        GS11 = self.Gamma_snk @ S11
        GS1 = self.Gamma_snk @ S1
        GS14 = self.Gamma_snk @ S14
        GS43 = self.Gamma_snk @ S43
        GS64 = self.Gamma_snk @ S64
        GS18 = self.Gamma_snk @ S18
        GS6 = self.Gamma_snk @ S6
        GS67 = self.Gamma_snk @ S67
        GS72 = self.Gamma_snk @ S72
        GS13 = self.Gamma_snk @ S13
        GS47 = self.Gamma_snk @ S47

        # XG 
        S30G = S30 @ self.Gamma_src
        S62G = S62 @ self.Gamma_src
        S6G = S6 @ self.Gamma_src
        S23G = S23 @ self.Gamma_src
        S28G = S28 @ self.Gamma_src
        S69G = S69 @ self.Gamma_src
        S68G = S68 @ self.Gamma_src
        S44G = S44 @ self.Gamma_src
        S76G = S76 @ self.Gamma_src
        S75G = S75 @ self.Gamma_src
        S26G = S26 @ self.Gamma_src
        S13G = S13 @ self.Gamma_src
        S1G = S1 @ self.Gamma_src
        S35G = S35 @ self.Gamma_src
        LG = L @ self.Gamma_src
        S27G = S27 @ self.Gamma_src
        S25G = S25 @ self.Gamma_src
        S33G = S33 @ self.Gamma_src
        S31G = S31 @ self.Gamma_src
        S3G = S3 @ self.Gamma_src
        S34G = S34 @ self.Gamma_src
        S60G = S60 @ self.Gamma_src
        S50G = S50 @ self.Gamma_src
        S22G = S22 @ self.Gamma_src

        # GXG 
        GS9G = self.Gamma_snk @ S9 @ self.Gamma_src
        GS71G = self.Gamma_snk @ S71 @ self.Gamma_src
        GS64G = self.Gamma_snk @ S64 @ self.Gamma_src
        GS38G = self.Gamma_snk @ S38 @ self.Gamma_src
        GS18G = self.Gamma_snk @ S18 @ self.Gamma_src
        GS48G = self.Gamma_snk @ S48 @ self.Gamma_src
        GS40G = self.Gamma_snk @ S40 @ self.Gamma_src
        GS46G = self.Gamma_snk @ S46 @ self.Gamma_src
        GS19G = self.Gamma_snk @ S19 @ self.Gamma_src
        GS14G = self.Gamma_snk @ S14 @ self.Gamma_src
        GS2G = self.Gamma_snk @ S2 @ self.Gamma_src


        diags = []

        # combined contractions
        for pre, ops in [(1, [S7, GS71G, L ]), (1, [S7, GS46G, S3 ]), (1, [S42, GS14G, L ]), (-1, [S3G, GL, S26 ]), (-1, [S50G, GS1, L ]), (-1, [S7, GS2G, S6 ]), 
                         (-1, [S3G, GL, S23 ]), (-1, [LG, GL, S33 ]), (-1, [S25G, GL, S6 ]), (1, [S66, GS2G, L ]), (1, [S75G, GL, L ]), (-1, [S7, GS18G, L ]), 
                         (-1, [S31G, GL, L ]), (-1, [S55, GS2G, L ]), (-1, [S25G, GS13, L ]), (1, [LG, GS3, S23 ]), (-1, [S1G, GL, S22 ]), (-1, [LG, GS43, L ]), 
                         (1, [LG, GS59, S1 ]), (-1, [S6G, GS5, L ]), (1, [S0, GS46G, L ]), (-1, [S23G, GL, S3 ]), (-1, [LG, GS5, S13 ]), (1, [S68G, GL, L ]), 
                         (-1, [S1G, GL, S30 ]), (1, [LG, GS47, S3 ]), (-1, [S4, GS14G, L ]), (-1, [LG, GL, S28 ]), (1, [LG, GS41, S1 ]), (1, [LG, GS67, L ]), 
                         (-1, [LG, GS16, S1 ]), (1, [S7, GS40G, S1 ]), (1, [S1G, GL, S50 ]), (1, [S50G, GL, S1 ]), (-1, [S26G, GL, S3 ]), (1, [S52, GS9G, L ]), 
                         (1, [S4, GS40G, L ]), (-1, [S3G, GS5, S1 ]), (-1, [LG, GL, S35 ]), (1, [LG, GS1, S22 ]), (1, [S1G, GS59, L ]), (1, [LG, GS74, L ]), 
                         (-1, [S7, GS14G, S1 ]), (-1, [LG, GS6, S25 ]), (-1, [S27G, GL, L ]), (1, [S44G, GL, S3 ]), (1, [S3G, GS47, L ]), (-1, [S1G, GS5, S3 ]), 
                         (1, [S1G, GS3, S25 ]), (1, [S25G, GS3, S1 ]), (1, [LG, GL, S68 ]), (-1, [LG, GL, S34 ]), (-1, [LG, GS5, S6 ]), (-1, [S34G, GL, L ]), 
                         (-1, [LG, GS1, S50 ]), (-1, [LG, GS49, L ]), (-1, [S13G, GS5, L ]), (1, [LG, GL, S75 ]), (1, [LG, GS1, S30 ]), (1, [S52, GS2G, S1 ]), 
                         (-1, [S7, GS48G, L ]), (1, [S3G, GS54, L ]), (-1, [S4, GS2G, S3 ]), (1, [S76G, GL, L ]), (1, [S73, GS2G, L ]), (-1, [S42, GS40G, L ]), 
                         (-1, [LG, GL, S31 ]), (-1, [S25G, GL, S13 ]), (-1, [LG, GS3, S44 ]), (-1, [S3G, GS11, L ]), (-1, [S1G, GS16, L ]), (-1, [S15, GS2G, L ]), 
                         (-1, [LG, GL, S60 ]), (-1, [S10, GS2G, L ]), (-1, [S25G, GS6, L ]), (1, [S3G, GL, S44 ]), (1, [S25G, GS1, S3 ]), (-1, [LG, GS13, S25 ]), 
                         (-1, [S30G, GL, S1 ]), (1, [S69G, GL, L ]), (1, [S3G, GS1, S25 ]), (-1, [S0, GS2G, S1 ]), (-1, [S13G, GL, S25 ]), (1, [LG, GS54, S3 ]), 
                         (-1, [S35G, GL, L ]), (-1, [S33G, GL, L ]), (-1, [LG, GS20, L ]), (1, [S22G, GS1, L ]), (-1, [S7, GS2G, S13 ]), (-1, [LG, GS56, L ]), 
                         (-1, [LG, GS21, L ]), (-1, [LG, GS53, L ]), (-1, [S7, GS19G, L ]), (-1, [LG, GS39, L ]), (-1, [LG, GS58, L ]), (-1, [S57, GS2G, L ]), 
                         (1, [S26G, GS3, L ]), (1, [S1G, GS41, L ]), (-1, [S44G, GS3, L ]), (1, [LG, GS65, L ]), (1, [LG, GL, S69 ]), (-1, [S62G, GL, L ]), 
                         (1, [LG, GL, S76 ]), (-1, [S6G, GL, S25 ]), (-1, [S0, GS9G, L ]), (1, [S30G, GS1, L ]), (-1, [S28G, GL, L ]), (-1, [S7, GS9G, S3 ]), 
                         (1, [S42, GS2G, S3 ]), (-1, [S7, GS38G, L ]), (1, [S7, GS64G, L ]), (-1, [S22G, GL, S1 ]), (-1, [S60G, GL, L ]), (1, [LG, GS3, S26 ]), 
                         (-1, [LG, GL, S27 ]), (-1, [LG, GS11, S3 ]), (-1, [LG, GL, S62 ]), (1, [LG, GS72, L ]), (-1, [S52, GS46G, L ]), (1, [S23G, GS3, L ])]:
            di, dj = contraction(combined_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)


        # cross contractions
        for pre, ops in [(-1, [S6G, S7, GS2 ]), (-1, [S3G, S7, GS9 ]), (1, [S73, LG, GS2 ]), (-1, [LG, S55, GS2 ]), (-1, [LG, S7, GS48 ]), (1, [S7, S3G, GS46 ]), 
                         (-1, [S4, S3G, GS2 ]), (-1, [S7, LG, GS18 ]), (1, [LG, S4, GS40 ]), (-1, [S57, LG, GS2 ]), (-1, [LG, S4, GS14 ]), (-1, [S7, S1G, GS14 ]), 
                         (-1, [LG, S15, GS2 ]), (1, [S42, S3G, GS2 ]), (-1, [S0, LG, GS9 ]), (-1, [S10, LG, GS2 ]), (-1, [LG, S7, GS19 ]), (1, [S42, LG, GS14 ]), 
                         (1, [LG, S0, GS46 ]), (1, [S7, LG, GS71 ]), (-1, [LG, S42, GS40 ]), (1, [LG, S66, GS2 ]), (-1, [S13G, S7, GS2 ]), (1, [S7, S1G, GS40 ]), 
                         (1, [S7, LG, GS64 ]), (-1, [LG, S7, GS38 ]), (1, [S52, LG, GS9 ]), (-1, [S1G, S0, GS2 ]), (-1, [LG, S52, GS46 ]), (1, [S1G, S52, GS2 ])]:
            di, dj = contraction(cross_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)


        # LOOP CONTRACTIONS
        # combined contractions
        for pre, ops, loops in [(1, [LG, GL, S3 ], [S29]), (-2, [LG, GL, L ], [S17, S24]), (1, [LG, GL, L ], [S63]), (1, [LG, GL, S1 ], [S32]), (-1, [LG, GL, L ], [S77]), 
                                (-1, [S3G, GL, L ], [S45]), (1, [S3G, GL, S1 ], [S24]), (1, [LG, GL, L ], [S37]), (1, [LG, GL, L ], [S61]), (1, [S3G, GL, L ], [S29]), 
                                (-1, [S3G, GS1, L ], [S24]), (-1, [S1G, GS3, L ], [S24]), (-1, [LG, GL, L ], [S70]), (1, [LG, GS1, L ], [S51]), (1, [S6G, GL, L ], [S24]), 
                                (2, [LG, GS5, L ], [S17]), (-1, [LG, GL, S3 ], [S45]), (1, [LG, GS13, L ], [S24]), (1, [LG, GS3, L ], [S45]), (1, [S1G, GL, L ], [S32]), 
                                (1, [LG, GS6, L ], [S24]), (-1, [LG, GS3, L ], [S29]), (-1, [LG, GS1, S3 ], [S24]), (1, [LG, GL, L ], [S36]), (-1, [LG, GL, S1 ], [S51]), 
                                (1, [S1G, GL, S3 ], [S24]), (2, [S7, GS2G, L ], [S17]), (1, [LG, GL, S13 ], [S24]), (2, [S25G, GL, L ], [S17]), (1, [S13G, GL, L ], [S24]), 
                                (-1, [LG, GS1, L ], [S32]), (1, [LG, GL, S6 ], [S24]), (-1, [S1G, GL, L ], [S51]), (-1, [LG, GS3, S1 ], [S24]), (2, [LG, GL, S25 ], [S17])]:
            factor = pre * np.prod(loops)
            di, dj = contraction(combined_contraction, ops, factor=factor)
            diags.append(di)
            diags.append(dj)


        # cross contractions
        for pre, ops, loops in [(2, [S7, LG, GS2 ], [S17])]:
            factor = pre * np.prod(loops)
            di, dj = contraction(cross_contraction, ops, factor=factor)
            diags.append(di)
            diags.append(dj)


        return diags


    
    def corr(self, t):
        return 0.5 * sum(self.diagrams(t))
    


class PerambNucleonPion0Pion02NucleonPionPion(PerambBaryon):
    
    def __init__(self, eps_src, perambs, mins_src_pi_plus, t0, P=None, eps_snk=None, 
                 mins_src_pi_minus=None, mins_snk_pi_plus=None, mins_snk_pi_minus=None,
                 Gamma_src_pi_plus=None, Gamma_src_pi_minus=None, Gamma_snk_pi_plus=None, Gamma_snk_pi_minus=None,
                 Gamma_src=None, Gamma_snk=None, tval=None):
        
        ## < O_p(x)  O_pi+(z)  O_pi-(q)  ~O_pi-(s) ~O_pi+(w) ~O_p(y) >

        self.t0 = t0

        self.eps_src = np.conjugate(eps_src[self.t0])
        if eps_snk is None:
            self.eps_snk = eps_src
        else:
            self.eps_snk = eps_snk

        
        self.perambs = perambs
        self.Nt = perambs[0].shape[0]
        self.tval = range(self.Nt) if tval is None else tval


        self.mins_src_pi_plus  = mins_src_pi_plus
        self.mins_src_pi_minus = mins_src_pi_plus if mins_src_pi_minus is None else mins_src_pi_minus
        self.mins_snk_pi_plus  = mins_src_pi_plus if mins_snk_pi_plus is None else mins_snk_pi_plus
        self.mins_snk_pi_minus = mins_src_pi_plus if mins_snk_pi_minus is None else mins_snk_pi_minus

        g_default = lqcdpy.distillation.mat.pion['g5']

        self.Gamma_src_pi_plus  = g_default if Gamma_src_pi_plus is None else Gamma_src_pi_plus
        self.Gamma_src_pi_minus = g_default if Gamma_src_pi_minus is None else Gamma_src_pi_minus
        self.Gamma_snk_pi_plus  = g_default if Gamma_snk_pi_plus is None else Gamma_snk_pi_plus
        self.Gamma_snk_pi_minus = g_default if Gamma_snk_pi_minus is None else Gamma_snk_pi_minus


        self.pions = {
            'src/pi0/0': lambda t: (self.mins_src_pi_plus[t], self.Gamma_src_pi_plus),
            'src/pi0/1': lambda t: (self.mins_src_pi_minus[t], self.Gamma_src_pi_minus),
            'snk/pi+': lambda t: (self.mins_snk_pi_plus[t], self.Gamma_snk_pi_plus),
            'snk/pi-': lambda t: (self.mins_snk_pi_minus[t], self.Gamma_snk_pi_minus),
        }

        cg_default = lqcdpy.distillation.mat.nucleon['Cg5']

        self.Gamma_src = cg_default if Gamma_src is None else Gamma_src
        self.Gamma_snk = cg_default if Gamma_snk is None else Gamma_snk


        self.P = np.eye(4) if P is None else P

    def seq0(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi+"](t), self.pions["src/pi0/0"](self.t0) ])

    def seq1(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi-"](t) ])

    def seq2(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/0"](self.t0) ])

    def seq3(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi+"](t) ])

    def seq4(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/0"](self.t0) ])

    def seq6(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi+"](t), self.pions["src/pi0/0"](self.t0) ])

    def seq7(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/0"](self.t0) ])

    def seq9(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/0"](self.t0) ])

    def seq10(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi+"](t), self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/0"](self.t0) ])

    def seq11(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi+"](t), self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/0"](self.t0) ])

    def seq12(self, t): 
        return loop([ self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/0"](self.t0) ])

    def seq13(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi+"](t), self.pions["snk/pi-"](t) ])

    def seq14(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi0/0"](self.t0), self.pions["snk/pi-"](t) ])

    def seq15(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi0/0"](self.t0), self.pions["snk/pi-"](t) ])

    def seq16(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi+"](t), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi-"](t) ])

    def seq17(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi-"](t) ])

    def seq18(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi+"](t), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi-"](t) ])

    def seq19(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi-"](t) ])

    def seq20(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi+"](t), self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi-"](t) ])

    def seq21(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi+"](t), self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi-"](t) ])

    def seq22(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi-"](t), self.pions["snk/pi+"](t), self.pions["src/pi0/0"](self.t0) ])

    def seq23(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi-"](t), self.pions["snk/pi+"](t) ])

    def seq24(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi-"](t), self.pions["snk/pi+"](t), self.pions["src/pi0/0"](self.t0) ])

    def seq25(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi-"](t), self.pions["snk/pi+"](t), self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/0"](self.t0) ])

    def seq26(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi-"](t), self.pions["snk/pi+"](t), self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/0"](self.t0) ])

    def seq27(self, t): 
        return loop([ self.perambs[t][t], self.perambs[t][t] ], [self.pions["snk/pi+"](t), self.pions["snk/pi-"](t) ])

    def seq28(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["src/pi0/0"](self.t0), self.pions["snk/pi-"](t), self.pions["snk/pi+"](t) ])

    def seq29(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["src/pi0/0"](self.t0), self.pions["snk/pi-"](t), self.pions["snk/pi+"](t) ])

    def seq30(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/0"](self.t0), self.pions["snk/pi-"](t), self.pions["snk/pi+"](t), self.pions["src/pi0/0"](self.t0) ])

    def seq31(self, t): 
        return loop([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[t][self.t0] ], [self.pions["snk/pi+"](t), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi-"](t) ])

    def seq32(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi-"](t), self.pions["snk/pi+"](t) ])

    def seq33(self, t): 
        return loop([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[t][self.t0] ], [self.pions["snk/pi+"](t), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi-"](t) ])

    def seq34(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/0"](self.t0), self.pions["snk/pi-"](t), self.pions["snk/pi+"](t), self.pions["src/pi0/0"](self.t0) ])

    def seq35(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi-"](t), self.pions["snk/pi+"](t) ])

    def seq36(self, t): 
        return loop([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0] ], [self.pions["snk/pi+"](t), self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi-"](t) ])

    def seq37(self, t): 
        return loop([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0] ], [self.pions["snk/pi+"](t), self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi-"](t) ])

    def seq38(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi-"](t), self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/0"](self.t0) ])

    def seq39(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi+"](t), self.pions["snk/pi-"](t), self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/0"](self.t0) ])

    def seq40(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi-"](t), self.pions["src/pi0/0"](self.t0) ])

    def seq41(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi+"](t), self.pions["snk/pi-"](t), self.pions["src/pi0/0"](self.t0) ])

    def seq42(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi-"](t), self.pions["src/pi0/0"](self.t0) ])

    def seq43(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi+"](t), self.pions["snk/pi-"](t), self.pions["src/pi0/0"](self.t0) ])

    def seq44(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi-"](t), self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/0"](self.t0) ])

    def seq45(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi+"](t), self.pions["snk/pi-"](t), self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/0"](self.t0) ])

    def seq46(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi0/0"](self.t0), self.pions["snk/pi+"](t) ])

    def seq47(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/0"](self.t0), self.pions["snk/pi+"](t), self.pions["snk/pi-"](t), self.pions["src/pi0/0"](self.t0) ])

    def seq48(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi-"](t), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi+"](t) ])

    def seq49(self, t): 
        return loop([ self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi+"](t), self.pions["snk/pi-"](t), self.pions["src/pi0/0"](self.t0) ])

    def seq50(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["src/pi0/0"](self.t0), self.pions["snk/pi+"](t), self.pions["snk/pi-"](t) ])

    def seq51(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi+"](t) ])

    def seq52(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi+"](t), self.pions["snk/pi-"](t) ])

    def seq53(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi-"](t), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi+"](t) ])

    def seq54(self, t): 
        return loop([ self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi+"](t), self.pions["snk/pi-"](t), self.pions["src/pi0/0"](self.t0) ])

    def seq55(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi0/0"](self.t0), self.pions["snk/pi+"](t) ])

    def seq56(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/0"](self.t0), self.pions["snk/pi+"](t), self.pions["snk/pi-"](t), self.pions["src/pi0/0"](self.t0) ])

    def seq57(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi+"](t) ])

    def seq58(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi+"](t), self.pions["snk/pi-"](t) ])

    def seq59(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["src/pi0/0"](self.t0), self.pions["snk/pi+"](t), self.pions["snk/pi-"](t) ])

    def seq60(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi-"](t), self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi+"](t) ])

    def seq61(self, t): 
        return loop([ self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi+"](t), self.pions["snk/pi-"](t), self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/0"](self.t0) ])

    def seq62(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi-"](t), self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi+"](t) ])

    def seq63(self, t): 
        return loop([ self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi+"](t), self.pions["snk/pi-"](t), self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/0"](self.t0) ])

    def seq64(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/0"](self.t0), self.pions["snk/pi-"](t), self.pions["src/pi0/0"](self.t0) ])

    def seq65(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi+"](t), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi-"](t), self.pions["src/pi0/0"](self.t0) ])

    def seq66(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/0"](self.t0), self.pions["snk/pi+"](t), self.pions["src/pi0/0"](self.t0) ])

    def seq67(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi0/0"](self.t0), self.pions["snk/pi+"](t), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi-"](t) ])

    def seq68(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi-"](t), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi+"](t), self.pions["src/pi0/0"](self.t0) ])

    def seq69(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi0/0"](self.t0), self.pions["snk/pi-"](t), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi+"](t) ])

    def seq70(self, t): 
        return loop([ self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi+"](t), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi-"](t), self.pions["src/pi0/0"](self.t0) ])

    def seq71(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/0"](self.t0), self.pions["snk/pi-"](t), self.pions["src/pi0/0"](self.t0) ])

    def seq72(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi+"](t), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi-"](t), self.pions["src/pi0/0"](self.t0) ])

    def seq73(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/0"](self.t0), self.pions["snk/pi+"](t), self.pions["src/pi0/0"](self.t0) ])

    def seq74(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi0/0"](self.t0), self.pions["snk/pi+"](t), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi-"](t) ])

    def seq75(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi-"](t), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi+"](t), self.pions["src/pi0/0"](self.t0) ])

    def seq76(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi0/0"](self.t0), self.pions["snk/pi-"](t), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi+"](t) ])

    def seq77(self, t): 
        return loop([ self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi+"](t), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi-"](t), self.pions["src/pi0/0"](self.t0) ])

    def diagrams(self, t):
        S0 = self.seq0(t)
        S1 = self.seq1(t)
        S2 = self.seq2(t)
        S3 = self.seq3(t)
        S4 = self.seq4(t)
        S6 = self.seq6(t)
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
        S18 = self.seq18(t)
        S19 = self.seq19(t)
        S20 = self.seq20(t)
        S21 = self.seq21(t)
        S22 = self.seq22(t)
        S23 = self.seq23(t)
        S24 = self.seq24(t)
        S25 = self.seq25(t)
        S26 = self.seq26(t)
        S27 = self.seq27(t)
        S28 = self.seq28(t)
        S29 = self.seq29(t)
        S30 = self.seq30(t)
        S31 = self.seq31(t)
        S32 = self.seq32(t)
        S33 = self.seq33(t)
        S34 = self.seq34(t)
        S35 = self.seq35(t)
        S36 = self.seq36(t)
        S37 = self.seq37(t)
        S38 = self.seq38(t)
        S39 = self.seq39(t)
        S40 = self.seq40(t)
        S41 = self.seq41(t)
        S42 = self.seq42(t)
        S43 = self.seq43(t)
        S44 = self.seq44(t)
        S45 = self.seq45(t)
        S46 = self.seq46(t)
        S47 = self.seq47(t)
        S48 = self.seq48(t)
        S49 = self.seq49(t)
        S50 = self.seq50(t)
        S51 = self.seq51(t)
        S52 = self.seq52(t)
        S53 = self.seq53(t)
        S54 = self.seq54(t)
        S55 = self.seq55(t)
        S56 = self.seq56(t)
        S57 = self.seq57(t)
        S58 = self.seq58(t)
        S59 = self.seq59(t)
        S60 = self.seq60(t)
        S61 = self.seq61(t)
        S62 = self.seq62(t)
        S63 = self.seq63(t)
        S64 = self.seq64(t)
        S65 = self.seq65(t)
        S66 = self.seq66(t)
        S67 = self.seq67(t)
        S68 = self.seq68(t)
        S69 = self.seq69(t)
        S70 = self.seq70(t)
        S71 = self.seq71(t)
        S72 = self.seq72(t)
        S73 = self.seq73(t)
        S74 = self.seq74(t)
        S75 = self.seq75(t)
        S76 = self.seq76(t)
        S77 = self.seq77(t)


        def contraction(func, ops, factor):
            d0, d1 = func(self.eps_src, self.eps_snk[t], ops, self.P)
            d0 *= factor
            d1 *= factor
            return d0, d1


        # shortcuts

        L = self.perambs[self.t0][t]
        # GX
        GS55 = self.Gamma_snk @ S55
        GS41 = self.Gamma_snk @ S41
        GS3 = self.Gamma_snk @ S3
        GS56 = self.Gamma_snk @ S56
        GS21 = self.Gamma_snk @ S21
        GS4 = self.Gamma_snk @ S4
        GS9 = self.Gamma_snk @ S9
        GS58 = self.Gamma_snk @ S58
        GS16 = self.Gamma_snk @ S16
        GS0 = self.Gamma_snk @ S0
        GS46 = self.Gamma_snk @ S46
        GS20 = self.Gamma_snk @ S20
        GS65 = self.Gamma_snk @ S65
        GL = self.Gamma_snk @ L
        GS59 = self.Gamma_snk @ S59
        GS39 = self.Gamma_snk @ S39
        GS57 = self.Gamma_snk @ S57
        GS50 = self.Gamma_snk @ S50
        GS2 = self.Gamma_snk @ S2
        GS52 = self.Gamma_snk @ S52
        GS74 = self.Gamma_snk @ S74
        GS7 = self.Gamma_snk @ S7
        GS66 = self.Gamma_snk @ S66
        GS51 = self.Gamma_snk @ S51
        GS11 = self.Gamma_snk @ S11
        GS73 = self.Gamma_snk @ S73
        GS45 = self.Gamma_snk @ S45
        GS43 = self.Gamma_snk @ S43
        GS10 = self.Gamma_snk @ S10
        GS18 = self.Gamma_snk @ S18
        GS6 = self.Gamma_snk @ S6
        GS67 = self.Gamma_snk @ S67
        GS72 = self.Gamma_snk @ S72
        GS13 = self.Gamma_snk @ S13
        GS47 = self.Gamma_snk @ S47

        # XG 
        S30G = S30 @ self.Gamma_src
        S62G = S62 @ self.Gamma_src
        S48G = S48 @ self.Gamma_src
        S23G = S23 @ self.Gamma_src
        S28G = S28 @ self.Gamma_src
        S69G = S69 @ self.Gamma_src
        S2G = S2 @ self.Gamma_src
        S53G = S53 @ self.Gamma_src
        S68G = S68 @ self.Gamma_src
        S76G = S76 @ self.Gamma_src
        S75G = S75 @ self.Gamma_src
        S26G = S26 @ self.Gamma_src
        S29G = S29 @ self.Gamma_src
        S4G = S4 @ self.Gamma_src
        S24G = S24 @ self.Gamma_src
        S35G = S35 @ self.Gamma_src
        LG = L @ self.Gamma_src
        S25G = S25 @ self.Gamma_src
        S32G = S32 @ self.Gamma_src
        S7G = S7 @ self.Gamma_src
        S34G = S34 @ self.Gamma_src
        S60G = S60 @ self.Gamma_src
        S9G = S9 @ self.Gamma_src
        S22G = S22 @ self.Gamma_src

        # GXG 
        GS73G = self.Gamma_snk @ S73 @ self.Gamma_src
        GS10G = self.Gamma_snk @ S10 @ self.Gamma_src
        GS3G = self.Gamma_snk @ S3 @ self.Gamma_src
        GS66G = self.Gamma_snk @ S66 @ self.Gamma_src
        GS51G = self.Gamma_snk @ S51 @ self.Gamma_src
        GS0G = self.Gamma_snk @ S0 @ self.Gamma_src
        GS11G = self.Gamma_snk @ S11 @ self.Gamma_src
        GS55G = self.Gamma_snk @ S55 @ self.Gamma_src
        GS6G = self.Gamma_snk @ S6 @ self.Gamma_src
        GS46G = self.Gamma_snk @ S46 @ self.Gamma_src
        GS57G = self.Gamma_snk @ S57 @ self.Gamma_src


        diags = []

        # combined contractions
        for pre, ops in [(-1, [S4G, GL, S23 ]), (-1, [LG, GS7, S53 ]), (1, [S1, GS55G, S2 ]), (1, [LG, GS41, S7 ]), (-1, [S1, GS10G, L ]), (-1, [S7G, GL, S24 ]), 
                         (-1, [S1, GS3G, S4 ]), (1, [S15, GS46G, L ]), (-1, [S14, GS0G, L ]), (1, [S7G, GS2, S23 ]), (1, [S40, GS0G, L ]), (-1, [S15, GS6G, L ]), 
                         (1, [S75G, GL, L ]), (-1, [S4G, GS13, L ]), (-1, [S22G, GL, S2 ]), (-1, [S28G, GL, S7 ]), (-1, [LG, GS52, L ]), (1, [S1, GS73G, L ]), 
                         (-1, [S1, GS3G, S9 ]), (1, [S68G, GL, L ]), (1, [LG, GS67, L ]), (1, [LG, GS59, S2 ]), (1, [S53G, GL, S7 ]), (-1, [LG, GS13, S4 ]), 
                         (-1, [S1, GS57G, L ]), (-1, [S14, GS3G, S7 ]), (-1, [S40, GS55G, L ]), (-1, [LG, GL, S35 ]), (1, [LG, GS43, S2 ]), (-1, [S30G, GL, L ]), 
                         (1, [LG, GS74, L ]), (-1, [LG, GL, S25 ]), (-1, [S48G, GS2, L ]), (1, [S7G, GS50, L ]), (-1, [LG, GS47, L ]), (1, [S2G, GS43, L ]), 
                         (-1, [S53G, GS7, L ]), (-1, [LG, GL, S30 ]), (-1, [S38, GS3G, L ]), (1, [S23G, GS7, S2 ]), (1, [LG, GL, S68 ]), (-1, [LG, GS45, L ]), 
                         (-1, [LG, GL, S34 ]), (-1, [S34G, GL, L ]), (-1, [S42, GS46G, L ]), (-1, [S25G, GL, L ]), (-1, [S29G, GL, S2 ]), (1, [S71, GS3G, L ]), 
                         (1, [LG, GL, S75 ]), (-1, [S44, GS3G, L ]), (-1, [S23G, GL, S4 ]), (1, [S29G, GS2, L ]), (-1, [LG, GL, S32 ]), (1, [LG, GS50, S7 ]), 
                         (1, [S42, GS3G, S2 ]), (1, [S2G, GS59, L ]), (1, [S76G, GL, L ]), (-1, [S23G, GL, S9 ]), (-1, [S26G, GL, L ]), (-1, [S9G, GL, S23 ]), 
                         (1, [S40, GS3G, S7 ]), (-1, [LG, GL, S60 ]), (-1, [LG, GS9, S23 ]), (1, [S1, GS66G, L ]), (1, [S1, GS46G, S7 ]), (-1, [S2G, GL, S29 ]), 
                         (1, [LG, GS2, S29 ]), (-1, [S1, GS6G, S7 ]), (1, [S69G, GL, L ]), (1, [S14, GS55G, L ]), (1, [S2G, GS7, S23 ]), (1, [LG, GS7, S28 ]), 
                         (-1, [S35G, GL, L ]), (-1, [LG, GL, S26 ]), (-1, [S17, GS3G, L ]), (1, [S24G, GS7, L ]), (1, [LG, GS2, S22 ]), (1, [LG, GS7, S24 ]), 
                         (-1, [S15, GS3G, S2 ]), (-1, [LG, GS20, L ]), (-1, [S19, GS3G, L ]), (-1, [LG, GS18, S7 ]), (1, [S48G, GL, S2 ]), (-1, [S1, GS11G, L ]), 
                         (-1, [LG, GS56, L ]), (-1, [LG, GS21, L ]), (-1, [LG, GS39, L ]), (-1, [LG, GS58, L ]), (1, [S42, GS6G, L ]), (-1, [LG, GS13, S9 ]), 
                         (-1, [S32G, GL, L ]), (-1, [S7G, GS18, L ]), (1, [LG, GS65, L ]), (1, [LG, GL, S69 ]), (1, [S7G, GL, S53 ]), (1, [S28G, GS7, L ]), 
                         (-1, [S62G, GL, L ]), (1, [LG, GL, S76 ]), (-1, [LG, GS2, S48 ]), (-1, [S9G, GS13, L ]), (-1, [S23G, GS4, L ]), (1, [S2G, GL, S48 ]), 
                         (1, [S7G, GS41, L ]), (-1, [S1, GS51G, L ]), (-1, [S7G, GS13, S2 ]), (-1, [LG, GS16, S2 ]), (-1, [S1, GS0G, S2 ]), (-1, [LG, GS4, S23 ]), 
                         (-1, [S60G, GL, L ]), (-1, [S23G, GS9, L ]), (1, [S22G, GS2, L ]), (1, [S64, GS3G, L ]), (-1, [S24G, GL, S7 ]), (-1, [S7G, GL, S28 ]), 
                         (-1, [LG, GL, S62 ]), (1, [LG, GS72, L ]), (-1, [S2G, GS16, L ]), (1, [S23G, GS2, S7 ]), (-1, [S2G, GL, S22 ]), (-1, [S2G, GS13, S7 ])]:
            di, dj = contraction(combined_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)


        # cross contractions
        for pre, ops in [(1, [LG, S1, GS73 ]), (-1, [S40, LG, GS55 ]), (-1, [LG, S15, GS6 ]), (-1, [S15, S2G, GS3 ]), (1, [LG, S15, GS46 ]), (-1, [S44, LG, GS3 ]), 
                         (-1, [LG, S14, GS0 ]), (-1, [S1, LG, GS51 ]), (-1, [S1, LG, GS10 ]), (-1, [S19, LG, GS3 ]), (1, [LG, S40, GS0 ]), (-1, [S1, S7G, GS6 ]), 
                         (1, [S71, LG, GS3 ]), (-1, [LG, S17, GS3 ]), (-1, [LG, S1, GS57 ]), (-1, [S42, LG, GS46 ]), (1, [S7G, S40, GS3 ]), (1, [LG, S14, GS55 ]), 
                         (1, [S1, S2G, GS55 ]), (-1, [S1, S4G, GS3 ]), (1, [S42, S2G, GS3 ]), (1, [LG, S64, GS3 ]), (1, [LG, S1, GS66 ]), (1, [LG, S42, GS6 ]), 
                         (-1, [S38, LG, GS3 ]), (-1, [S1, S9G, GS3 ]), (-1, [S7G, S14, GS3 ]), (1, [S7G, S1, GS46 ]), (-1, [S1, LG, GS11 ]), (-1, [S2G, S1, GS0 ])]:
            di, dj = contraction(cross_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)


        # LOOP CONTRACTIONS
        # combined contractions
        for pre, ops, loops in [(-1, [S2G, GL, L ], [S49]), (1, [LG, GL, L ], [S63]), (2, [LG, GS13, L ], [S12]), (-1, [S7G, GL, L ], [S54]), (2, [S23G, GL, L ], [S12]), 
                                (1, [LG, GS9, L ], [S27]), (-1, [LG, GL, L ], [S77]), (1, [LG, GL, S2 ], [S31]), (1, [LG, GL, L ], [S37]), (1, [LG, GL, L ], [S61]), 
                                (1, [S7G, GL, L ], [S33]), (1, [LG, GS4, L ], [S27]), (1, [LG, GS7, L ], [S54]), (-2, [LG, GL, L ], [S27, S12]), (-1, [LG, GL, S7 ], [S54]), 
                                (-1, [LG, GL, L ], [S70]), (-1, [LG, GS2, S7 ], [S27]), (2, [S1, GS3G, L ], [S12]), (1, [LG, GL, S4 ], [S27]), (2, [LG, GL, S23 ], [S12]), 
                                (1, [S9G, GL, L ], [S27]), (-1, [LG, GS2, L ], [S31]), (1, [LG, GS2, L ], [S49]), (1, [S4G, GL, L ], [S27]), (1, [LG, GL, L ], [S36]), 
                                (-1, [S2G, GS7, L ], [S27]), (-1, [LG, GL, S2 ], [S49]), (-1, [LG, GS7, L ], [S33]), (1, [LG, GL, S7 ], [S33]), (1, [S7G, GL, S2 ], [S27]), 
                                (-1, [LG, GS7, S2 ], [S27]), (1, [S2G, GL, L ], [S31]), (1, [S2G, GL, S7 ], [S27]), (-1, [S7G, GS2, L ], [S27]), (1, [LG, GL, S9 ], [S27])]:
            factor = pre * np.prod(loops)
            di, dj = contraction(combined_contraction, ops, factor=factor)
            diags.append(di)
            diags.append(dj)


        # cross contractions
        for pre, ops, loops in [(2, [LG, S1, GS3 ], [S12])]:
            factor = pre * np.prod(loops)
            di, dj = contraction(cross_contraction, ops, factor=factor)
            diags.append(di)
            diags.append(dj)


        return diags


    
    def corr(self, t):
        return 0.5 * sum(self.diagrams(t))
    
