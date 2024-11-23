import numpy as np
import jax.numpy as jp
import lqcdpy
from lqcdpy.distillation import combined_contraction, cross_contraction, sequential, loop, corr_fill, run, PerambBaryon, PerambMeson


class PerambNucleon2Nucleon(PerambBaryon):
    def __init__(self, eps_src, peramb, t0, P=None, Gamma_src=None, Gamma_snk=None, P_snk=None, P_src=None, eps_snk=None, tval=None):
        self.t0 = t0
        self.eps_src = np.conjugate(eps_src[self.t0])
        self.eps_snk = eps_src if eps_snk is None else eps_snk

        self.peramb = peramb
        self.Nt = peramb.shape[0]
        self.tval = range(self.Nt) if tval is None else tval


        self.P_src = np.eye(4) if P_src is None else P_src
        self.P_snk = np.eye(4) if P_snk is None else P_snk
        
        if Gamma_snk is None:
            self.Gamma_snk = lqcdpy.distillation.mat.nucleon['Cg5']
        else:
            self.Gamma_snk = Gamma_snk
        if Gamma_src is None:
            self.Gamma_src = lqcdpy.distillation.mat.nucleon['Cg5']
        else:
            self.Gamma_src = Gamma_src

    def diagrams(self, t):


        def contraction(func, ops, factor):
            d0, d1 = func(self.eps_src, self.eps_snk[t], ops, self.P_snk, self.P_src)
            d0 *= factor
            d1 *= factor
            return d0, d1


        # shortcuts

        L = self.peramb[t]
        # GX
        GL = self.Gamma_snk @ L

        # XG 
        LG = L @ self.Gamma_src

        # GXG 


        diags = []

        # combined contractions
        for pre, ops in [(-1, [LG, GL, L ])]:
            di, dj = contraction(combined_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)


        return diags


    def corr(self, t):
        return sum(self.diagrams(t))
    



class PerambNucleon2NucleonPion(PerambBaryon):

    def __init__(self, eps_src, eps_snk, perambs, mins, t0, P_snk=None, P_src=None, Gamma_src=None, Gamma_snk=None, Gamma_pi=None, tval=None):
        self.t0 = t0
        self.perambs = perambs
        self.eps_src = np.conjugate(eps_src[self.t0])
        self.eps_snk = eps_snk

        self.Nt = perambs[0].shape[0]
        self.tval = range(self.Nt) if tval is None else tval

        self.mins = mins
        self.P_src = np.eye(4) if P_src is None else P_src
        self.P_snk = np.eye(4) if P_snk is None else P_snk

        if Gamma_src is None:
            self.Gamma_src = lqcdpy.distillation.mat.nucleon['Cg5']
        else:
            self.Gamma_src = Gamma_src

        if Gamma_snk is None:
            self.Gamma_snk = lqcdpy.distillation.mat.nucleon['Cg5']
        else:
            self.Gamma_snk = Gamma_snk

        if Gamma_pi is None:
            self.Gamma_pi = lqcdpy.distillation.mat.pion['g5']
        else:
            self.Gamma_pi = Gamma_pi

        self.pions = {
            "snk/pi+": lambda t: (self.mins[t], self.Gamma_pi)
        }

    def seq0(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi+"](t) ])

    def diagrams(self, t):
        S0 = self.seq0(t)


        def contraction(func, ops, factor):
            d0, d1 = func(self.eps_src, self.eps_snk[t], ops, self.P_snk, self.P_src)
            d0 *= factor
            d1 *= factor
            return d0, d1


        # shortcuts

        L = self.perambs[self.t0][t]
        # GX
        GL = self.Gamma_snk @ L

        # XG 
        S0G = S0 @ self.Gamma_src

        # GXG 
        GLG = self.Gamma_snk @ L @ self.Gamma_src


        diags = []

        # combined contractions
        for pre, ops in [(1, [L, GLG, S0 ])]:
            di, dj = contraction(combined_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)


        # cross contractions
        for pre, ops in [(1, [S0G, L, GL ])]:
            di, dj = contraction(cross_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)


        # LOOP CONTRACTIONS
        return diags

    def corr(self, t):
        return - sum(self.diagrams(t))


class PerambNucleonPion2Nucleon(PerambBaryon):

    def __init__(self, eps_src, eps_snk, perambs, mins, t0, Gamma_src=None, Gamma_snk=None, Gamma_pi=None, P_src=None, P_snk=None, tval=None):
        self.t0 = t0
        self.eps_src = np.conjugate(eps_src[self.t0])
        self.eps_snk = eps_snk
        self.perambs = perambs
        self.Nt = perambs[0].shape[0]

        self.tval = range(self.Nt) if tval is None else tval

        self.mins = mins
        self.P_src = np.eye(4) if P_src is None else P_src
        self.P_snk = np.eye(4) if P_snk is None else P_snk


        if Gamma_src is None:
            self.Gamma_src = lqcdpy.distillation.mat.nucleon['Cg5']
        else:
            self.Gamma_src = Gamma_src

        if Gamma_snk is None:
            self.Gamma_snk = lqcdpy.distillation.mat.nucleon['Cg5']
        else:
            self.Gamma_snk = Gamma_snk

        if Gamma_pi is None:
            self.Gamma_pi = lqcdpy.distillation.mat.pion['g5']
        else:
            self.Gamma_pi = Gamma_pi

        self.pions = {
            "src/pi+": lambda t: (self.mins[t], self.Gamma_pi)
        }
        

    def seq0(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi+"](self.t0) ])

    def diagrams(self, t):
        S0 = self.seq0(t)


        def contraction(func, ops, factor):
            d0, d1 = func(self.eps_src, self.eps_snk[t], ops, self.P_snk, self.P_src)
            d0 *= factor
            d1 *= factor
            return d0, d1


        # shortcuts

        L = self.perambs[self.t0][t]
        # GX
        GL = self.Gamma_snk @ L

        # XG 
        S0G = S0 @ self.Gamma_src

        # GXG 
        GLG = self.Gamma_snk @ L @ self.Gamma_src


        diags = []

        # combined contractions
        for pre, ops in [(1, [L, GLG, S0 ])]:
            di, dj = contraction(combined_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)


        # cross contractions
        for pre, ops in [(1, [S0G, L, GL ])]:
            di, dj = contraction(cross_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)


        # LOOP CONTRACTIONS
        return diags

    def corr(self, t):
        return sum(self.diagrams(t))


class PerambNucleonPion2NucleonPion(PerambBaryon):

    def __init__(self, eps_src, eps_snk, perambs, mins_src, mins_snk, t0, Gamma_src=None, Gamma_snk=None, 
                 Gamma_pi_src=None, Gamma_pi_snk=None, P_src=None, P_snk=None, tval=None):
        self.t0 = t0
        self.eps_src = np.conjugate(eps_src[self.t0])
        self.eps_snk = eps_snk
        self.perambs = perambs
        self.Nt = perambs[0].shape[0]
        self.tval = range(self.Nt) if tval is None else tval

        self.mins_src = mins_src
        self.mins_snk = mins_snk
        
        self.P_src = np.eye(4) if P_src is None else P_src
        self.P_snk = np.eye(4) if P_snk is None else P_snk
        
        if Gamma_src is None:
            self.Gamma_src = lqcdpy.distillation.mat.nucleon['Cg5']
        else:
            self.Gamma_src = Gamma_src

        if Gamma_snk is None:
            self.Gamma_snk = lqcdpy.distillation.mat.nucleon['Cg5']
        else:
            self.Gamma_snk = Gamma_snk
        
        if Gamma_pi_src is None:
            self.Gamma_pi_src = lqcdpy.distillation.mat.pion['g5']
        else:
            self.Gamma_pi_src = Gamma_pi_src

        if Gamma_pi_snk is None:
            self.Gamma_pi_snk = lqcdpy.distillation.mat.pion['g5']
        else:
            self.Gamma_pi_src = lqcdpy.distillation.mat.pion['g5']

        self.pions = {
            "src/pi+": lambda t: (self.mins_src[t], self.Gamma_pi_src),
            "snk/pi+": lambda t: (self.mins_snk[t], self.Gamma_pi_snk)
        }

    def seq0(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi+"](t) ])

    def seq1(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi+"](self.t0) ])

    def seq2(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi+"](t), self.pions["src/pi+"](self.t0) ])

    def seq3(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi+"](self.t0), self.pions["snk/pi+"](t) ])

    def seq4(self, t): 
        return loop([ self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi+"](t), self.pions["src/pi+"](self.t0) ])

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
        GS3 = self.Gamma_snk @ S3
        GL = self.Gamma_snk @ L
        GS1 = self.Gamma_snk @ S1

        # XG 
        LG = L @ self.Gamma_src
        S2G = S2 @ self.Gamma_src

        # GXG 
        GS1G = self.Gamma_snk @ S1 @ self.Gamma_src


        diags = []

        # combined contractions
        for pre, ops in [(-1, [LG, GL, S2 ]), (-1, [S0, GS1G, L ]), (-1, [S2G, GL, L ]), (-1, [LG, GS3, L ])]:
            di, dj = contraction(combined_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)


        # cross contractions
        for pre, ops in [(-1, [S0, LG, GS1 ])]:
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
        return - sum(self.diagrams(t))
   


    
