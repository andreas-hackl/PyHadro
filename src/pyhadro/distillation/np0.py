import numpy as np
import lqcdpy
from lqcdpy.distillation import combined_contraction, cross_contraction, sequential, loop, PerambBaryon


class PerambNucleonPion02NucleonPion0(PerambBaryon):

    def __init__(self, eps_src, eps_snk, perambs, mins_src, mins_snk, t0, P_src=None, P_snk=None, Gamma_src=None, Gamma_snk=None, Gamma_pi_src=None, Gamma_pi_snk=None, tval=None):
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


        Cg5 = lqcdpy.distillation.mat.nucleon['Cg5']
        self.Gamma_snk = Cg5 if Gamma_snk is None else Gamma_snk
        self.Gamma_src = Cg5 if Gamma_src is None else Gamma_src

        g5 = lqcdpy.distillation.mat.pion['g5']
        self.Gamma_pi_src = g5 if Gamma_pi_src is None else Gamma_pi_src
        self.Gamma_pi_snk = g5 if Gamma_pi_snk is None else Gamma_pi_snk

        self.pions = {
            'src/pi0': lambda t: (self.mins_src[t], self.Gamma_pi_src),
            'snk/pi0': lambda t: (self.mins_snk[t], self.Gamma_pi_snk),
        }

    def seq0(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi0"](t) ])

    def seq1(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0"](self.t0) ])

    def seq2(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0"](t), self.pions["src/pi0"](self.t0) ])

    def seq5(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi0"](self.t0), self.pions["snk/pi0"](t) ])

    def seq6(self, t): 
        return loop([ self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi0"](t), self.pions["src/pi0"](self.t0) ])
    
    def diagrams(self, t):
        S0 = self.seq0(t)
        S1 = self.seq1(t)
        S2 = self.seq2(t)
        S5 = self.seq5(t)
        S6 = self.seq6(t)

        def contraction(func, ops, factor):
            d0, d1 = func(self.eps_src, self.eps_snk[t], ops, self.P_snk, self.P_src)
            d0 *= factor
            d1 *= factor
            return d0, d1


        # shortcuts

        D = self.perambs[self.t0][t]
        U = self.perambs[self.t0][t]
        # GX
        GD = self.Gamma_snk @ D
        GS0 = self.Gamma_snk @ S0
        GS1 = self.Gamma_snk @ S1 
        GS2 = self.Gamma_snk @ S2
        GS5 = self.Gamma_snk @ S5


        # XG 
        S2G = S2 @ self.Gamma_src
        UG = U @ self.Gamma_src
        S5G = S5 @ self.Gamma_src
        S1G = S1 @ self.Gamma_src
        S0G = S0 @ self.Gamma_src

        # GXG 


        diags = []

        # combined contractions UU
        for pre, ops in [(-1, [S0G, GD, S1 ]), (-1, [S1G, GD, S0 ]), (-1, [UG, GD, S2 ]), 
                         (-1, [S2G, GD, U ]), (-1, [UG, GD, S5 ]), (-1, [S5G, GD, U ])]:
            di, dj = contraction(combined_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)

        # combined_contraction UD
        for pre, ops in [(+1, [UG, GS1, S0 ]), (+1, [S0G, GS1, U ])]:
            di, dj = contraction(combined_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)

        # combined_contraction DU
        for pre, ops in [(+1, [UG, GS0, S1 ]), (+1, [S1G, GS0, U ])]:
            di, dj = contraction(combined_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)

        # combined_contraction DD
        for pre, ops in [(-1, [UG, GS2, U ]), (-1, [UG, GS5, U ])]:
            di, dj = contraction(combined_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)
        
        # combined_contraction LOOP   UU + DD 
        for pre, ops, loops in [(2, [UG, GD, U ], [S6])]:
            factor = pre * np.prod(loops)
            di, dj = contraction(combined_contraction, ops, factor=factor)
            diags.append(di)
            diags.append(dj)
    
        return diags
    
    def corr(self, t):
         return - 0.5 * sum(self.diagrams(t))               # Minus sign comes from O_{pi+}^\dagger(p) = - O_{pi-}(-p) 
    




class PerambNucleonPion02NucleonPion(PerambBaryon):

    def __init__(self, eps_src, eps_snk, perambs, mins_src, mins_snk, t0, P_src=None, P_snk=None, Gamma_src=None, Gamma_snk=None, Gamma_pi_src=None, Gamma_pi_snk=None, tval=None):

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


        Cg5 = lqcdpy.distillation.mat.nucleon['Cg5']
        self.Gamma_snk = Cg5 if Gamma_snk is None else Gamma_snk
        self.Gamma_src = Cg5 if Gamma_src is None else Gamma_src

        g5 = lqcdpy.distillation.mat.pion['g5']
        self.Gamma_pi_src = g5 if Gamma_pi_src is None else Gamma_pi_src
        self.Gamma_pi_snk = g5 if Gamma_pi_snk is None else Gamma_pi_snk

        self.pions = {
            'src/pi0': lambda t: (self.mins_src[t], self.Gamma_pi_src),
            'snk/pi+': lambda t: (self.mins_snk[t], self.Gamma_pi_snk),
        }

    def seq0(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi+"](t) ])

    def seq1(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0"](self.t0) ])

    def seq2(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi+"](t), self.pions["src/pi0"](self.t0) ])

    def seq4(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi0"](self.t0), self.pions["snk/pi+"](t) ])


    def diagrams(self, t):
        S0 = self.seq0(t)
        S1 = self.seq1(t)
        S2 = self.seq2(t)
        S4 = self.seq4(t)


        def contraction(func, ops, factor):
            d0, d1 = func(self.eps_src, self.eps_snk[t], ops, self.P_snk, self.P_src)
            d0 *= factor
            d1 *= factor
            return d0, d1


        # shortcuts

        D = self.perambs[self.t0][t]
        U = self.perambs[self.t0][t]
        # GX
        GS1 = self.Gamma_snk @ S1
        GU = self.Gamma_snk @ U

        # XG 
        S2G = S2 @ self.Gamma_src
        S0G = S0 @ self.Gamma_src
        S4G = S4 @ self.Gamma_src

        # GXG 
        GS1G = self.Gamma_snk @ S1 @ self.Gamma_src
        GUG = self.Gamma_snk @ U @ self.Gamma_src


        diags = []

        # combined contractions U
        for pre, ops in [(-1, [D, GS1G, S0 ]), (-1, [D, GUG, S2 ])]:
            di, dj = contraction(combined_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)


        # cross contractions U
        for pre, ops in [(-1, [S0G, D, GS1 ]), (-1, [S2G, D, GU ])]:
            di, dj = contraction(cross_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)

        # combined contractions D
        for pre, ops in [(+1, [S1, GUG, S0 ]), (+1, [D, GUG, S4 ])]:
            di, dj = contraction(combined_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)


        # cross contractions D
        for pre, ops in [(+1, [S0G, S1, GU ]), (+1, [S4G, D, GU ])]:
            di, dj = contraction(cross_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)

        return diags
    
    def corr(self, t):
         return - 1/np.sqrt(2) * sum(self.diagrams(t))               # Minus sign comes from O_{pi+}^\dagger(p) = - O_{pi-}(-p) 
    


class PerambNucleonPion2NucleonPion0(PerambBaryon):
    def __init__(self, eps_src, eps_snk, perambs, mins_src, mins_snk, t0, P_src=None, P_snk=None, Gamma_src=None, Gamma_snk=None, Gamma_pi_src=None, Gamma_pi_snk=None, tval=None):
        self.t0 = t0

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


        Cg5 = lqcdpy.distillation.mat.nucleon['Cg5']
        self.Gamma_snk = Cg5 if Gamma_snk is None else Gamma_snk
        self.Gamma_src = Cg5 if Gamma_src is None else Gamma_src

        g5 = lqcdpy.distillation.mat.pion['g5']
        self.Gamma_pi_src = g5 if Gamma_pi_src is None else Gamma_pi_src
        self.Gamma_pi_snk = g5 if Gamma_pi_snk is None else Gamma_pi_snk

        self.pions = {
            'src/pi+': lambda t: (self.mins_src[t], self.Gamma_pi_src),
            'snk/pi0': lambda t: (self.mins_snk[t], self.Gamma_pi_snk),
        }

    def seq0(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi0"](t) ])

    def seq1(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi+"](self.t0) ])

    def seq2(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0"](t), self.pions["src/pi+"](self.t0) ])

    def seq4(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi+"](self.t0), self.pions["snk/pi0"](t) ])

    def diagrams(self, t):
        S0 = self.seq0(t)
        S1 = self.seq1(t)
        S2 = self.seq2(t)
        S4 = self.seq4(t)


        def contraction(func, ops, factor):
            d0, d1 = func(self.eps_src, self.eps_snk[t], ops, self.P_snk, self.P_src)
            d0 *= factor
            d1 *= factor
            return d0, d1


        # shortcuts

        D = self.perambs[self.t0][t]
        U = self.perambs[self.t0][t]
        # GX
        GD = self.Gamma_snk @ D
        GS0 = self.Gamma_snk @ S0

        # XG 
        S1G = S1 @ self.Gamma_src
        S2G = S2 @ self.Gamma_src
        S4G = S4 @ self.Gamma_src 

        # GXG 
        GDG = self.Gamma_snk @ D @ self.Gamma_src
        GS0G = self.Gamma_snk @ S0 @ self.Gamma_src


        diags = []

        # combined contractions U
        for pre, ops in [(-1, [S0, GDG, S1 ]), (-1, [U, GDG, S2 ])]:
            di, dj = contraction(combined_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)


        # cross contractions U
        for pre, ops in [(-1, [S0, S1G, GD ]), (-1, [S2G, U, GD ])]:
            di, dj = contraction(cross_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)


        # combined contractions D
        for pre, ops in [(+1, [U, GS0G, S1]), (+1, [U, GDG, S4])]:
            di, dj = contraction(combined_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)

        # cross contraction D
        for pre, ops in [(+1, [S1G, U, GS0]), (+1, [S4G, U, GD])]:
            di, dj = contraction(cross_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)

        return diags
    
    def corr(self, t):
         return - 1/np.sqrt(2) * sum(self.diagrams(t))               # Minus sign comes from O_{pi+}^\dagger(p) = - O_{pi-}(-p) 
    



class PerambNucleon2NucleonPion0(PerambBaryon):

    def __init__(self, eps_src, eps_snk, perambs, mins, t0, P_src=None, P_snk=None, Gamma_src=None, Gamma_snk=None, Gamma_pi=None, tval=None):
        self.t0 = t0

        self.eps_src = np.conjugate(eps_src[self.t0])
        self.eps_snk = eps_snk

        self.perambs = perambs
        self.Nt = perambs[0].shape[0]
        self.mins = mins 

        self.tval = range(self.Nt) if tval is None else tval

        self.P_src = np.eye(4) if P_src is None else P_src
        self.P_snk = np.eye(4) if P_snk is None else P_snk
        
        Cg5 = lqcdpy.distillation.mat.nucleon['Cg5']

        self.Gamma_src = Cg5 if Gamma_src is None else Gamma_src
        self.Gamma_snk = Cg5 if Gamma_snk is None else Gamma_snk
        
        g5 = lqcdpy.distillation.mat.pion['g5']
        self.Gamma_pi_snk = g5 if Gamma_pi is None else Gamma_pi

        self.pions = {
            "snk/pi0": lambda t: (self.mins[t], self.Gamma_pi_snk)
        }

    def seq0(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi0"](t) ])

    def diagrams(self, t):
        S0 = self.seq0(t)


        def contraction(func, ops, factor):
            d0, d1 = func(self.eps_src, self.eps_snk[t], ops, self.P_snk, self.P_src)
            d0 *= factor
            d1 *= factor
            return d0, d1


        # shortcuts

        D = self.perambs[self.t0][t]
        U = self.perambs[self.t0][t]
        # GX
        GD = self.Gamma_snk @ D
        GS0 = self.Gamma_snk @ S0

        # XG 
        UG = U @ self.Gamma_src
        S0G = S0 @ self.Gamma_src

        # GXG 


        diags = []

        # combined contractions U
        for pre, ops in [(1, [UG, GD, S0 ]), (1, [S0G, GD, U ])]:
            di, dj = contraction(combined_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)


        # combined contractions
        for pre, ops in [(-1, [UG, GS0, U ])]:
            di, dj = contraction(combined_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)


        return diags
    
    def corr(self, t):
         return 1/np.sqrt(2) * sum(self.diagrams(t))
    



class PerambNucleonPion02Nucleon(PerambBaryon):

    def __init__(self, eps_src, eps_snk, perambs, mins, t0, P_src=None, P_snk=None, Gamma_src=None, Gamma_snk=None, Gamma_pi=None, tval=None):
        self.t0 = t0

        self.eps_src = np.conjugate(eps_src[self.t0])
        self.eps_snk = eps_snk

        self.perambs = perambs
        self.Nt = perambs[0].shape[0]
        self.mins = mins 

        self.tval = range(self.Nt) if tval is None else tval
        
        self.P_src = np.eye(4) if P_src is None else P_src
        self.P_snk = np.eye(4) if P_snk is None else P_snk
        
        Cg5 = lqcdpy.distillation.mat.nucleon['Cg5']

        self.Gamma_src = Cg5 if Gamma_src is None else Gamma_src
        self.Gamma_snk = Cg5 if Gamma_snk is None else Gamma_snk
        
        g5 = lqcdpy.distillation.mat.pion['g5']
        self.Gamma_pi_snk = g5 if Gamma_pi is None else Gamma_pi

        self.pions = {
            "src/pi0": lambda t: (self.mins[t], self.Gamma_pi_snk)
        }

    def seq0(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0"](self.t0) ])

    def diagrams(self, t):
        S0 = self.seq0(t)


        def contraction(func, ops, factor):
            d0, d1 = func(self.eps_src, self.eps_snk[t], ops, self.P_snk, self.P_src)
            d0 *= factor
            d1 *= factor
            return d0, d1


        # shortcuts

        D = self.perambs[self.t0][t]
        U = self.perambs[self.t0][t]
        # GX
        GD = self.Gamma_snk @ D
        GS0 = self.Gamma_snk @ S0

        # XG 
        UG = U @ self.Gamma_src
        S0G = S0 @ self.Gamma_src

        # GXG 


        diags = []

        # combined contractions U
        for pre, ops in [(1, [UG, GD, S0 ]), (1, [S0G, GD, U ])]:
            di, dj = contraction(combined_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)


        # combined contractions D
        for pre, ops in [(-1, [UG, GS0, U ])]:
            di, dj = contraction(combined_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)

        return diags
    
    def corr(self, t):
         return - 1/np.sqrt(2) * sum(self.diagrams(t))               # Minus sign comes from O_{pi0}^\dagger(p) = - O_{pi0}(-p) 
    



    


