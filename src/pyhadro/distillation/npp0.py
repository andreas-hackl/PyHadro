import numpy as np
import jax.numpy as jp
import lqcdpy
from lqcdpy.distillation import combined_contraction, cross_contraction, sequential, loop, corr_fill, PerambBaryon
 



class PerambNucleonPion0Pion02NucleonPion0Pion0(PerambBaryon):

    # In total 882 Diagrams
    
    def __init__(self, eps_src, perambs, mins_src_pi_plus, t0, P=None, eps_snk=None, 
                 mins_src_pi_minus=None, mins_snk_pi_plus=None, mins_snk_pi_minus=None,
                 Gamma_src_pi_plus=None, Gamma_src_pi_minus=None, Gamma_snk_pi_plus=None, Gamma_snk_pi_minus=None,
                 Gamma_src=None, Gamma_snk=None, tval=None):
        
        ## < O_p(x)  O_pi0(z)  O_pi0(q)  O_pi0(s) O_pi0(w) ~O_p(y) >

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
            'snk/pi0/0': lambda t: (self.mins_snk_pi_plus[t], self.Gamma_snk_pi_plus),
            'snk/pi0/1': lambda t: (self.mins_snk_pi_minus[t], self.Gamma_snk_pi_minus),
        }

        cg_default = lqcdpy.distillation.mat.nucleon['Cg5']

        self.Gamma_src = cg_default if Gamma_src is None else Gamma_src
        self.Gamma_snk = cg_default if Gamma_snk is None else Gamma_snk


        self.P = np.eye(4) if P is None else P

    def seq0(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/0"](t), self.pions["src/pi0/0"](self.t0) ])

    def seq1(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/1"](t), self.pions["src/pi0/1"](self.t0) ])

    def seq2(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/0"](t), self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/1"](self.t0) ])

    def seq3(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi0/1"](t) ])

    def seq5(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/0"](t), self.pions["src/pi0/1"](self.t0) ])

    def seq6(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/1"](t), self.pions["src/pi0/0"](self.t0) ])

    def seq8(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/0"](t), self.pions["src/pi0/1"](self.t0), self.pions["src/pi0/0"](self.t0) ])

    def seq9(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi0/0"](t) ])

    def seq10(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/1"](t), self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/1"](self.t0) ])

    def seq11(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/1"](t), self.pions["src/pi0/1"](self.t0), self.pions["src/pi0/0"](self.t0) ])

    def seq12(self, t): 
        return loop([ self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/1"](self.t0) ])

    def seq13(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/0"](t), self.pions["snk/pi0/1"](t), self.pions["src/pi0/1"](self.t0) ])

    def seq14(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/0"](self.t0) ])

    def seq15(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi0/0"](t), self.pions["snk/pi0/1"](t) ])

    def seq16(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/1"](self.t0) ])

    def seq18(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi0/0"](self.t0), self.pions["snk/pi0/1"](t) ])

    def seq19(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi0/0"](t), self.pions["src/pi0/1"](self.t0), self.pions["snk/pi0/1"](t) ])

    def seq20(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/0"](self.t0), self.pions["snk/pi0/1"](t), self.pions["src/pi0/1"](self.t0) ])

    def seq21(self, t): 
        return loop([ self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi0/1"](t), self.pions["src/pi0/1"](self.t0) ])

    def seq22(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/1"](self.t0), self.pions["snk/pi0/1"](t) ])

    def seq23(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/0"](t), self.pions["snk/pi0/1"](t), self.pions["src/pi0/0"](self.t0) ])

    def seq24(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/1"](self.t0) ])

    def seq25(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/1"](self.t0), self.pions["src/pi0/0"](self.t0) ])

    def seq26(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi0/0"](t), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi0/1"](t) ])

    def seq27(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi0/1"](self.t0), self.pions["snk/pi0/1"](t) ])

    def seq28(self, t): 
        return loop([ self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi0/1"](t), self.pions["src/pi0/0"](self.t0) ])

    def seq29(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/1"](self.t0), self.pions["snk/pi0/1"](t), self.pions["src/pi0/0"](self.t0) ])

    def seq30(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi0/1"](self.t0), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi0/1"](t) ])

    def seq31(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/0"](t), self.pions["snk/pi0/1"](t), self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/1"](self.t0) ])

    def seq32(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/0"](t), self.pions["snk/pi0/1"](t), self.pions["src/pi0/1"](self.t0), self.pions["src/pi0/0"](self.t0) ])

    def seq33(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/0"](t), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi0/1"](t), self.pions["src/pi0/1"](self.t0) ])

    def seq34(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi0/0"](t), self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/1"](self.t0), self.pions["snk/pi0/1"](t) ])

    def seq35(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/0"](t), self.pions["src/pi0/1"](self.t0), self.pions["snk/pi0/1"](t), self.pions["src/pi0/0"](self.t0) ])

    def seq36(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi0/0"](t), self.pions["src/pi0/1"](self.t0), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi0/1"](t) ])

    def seq40(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/1"](t), self.pions["snk/pi0/0"](t), self.pions["src/pi0/1"](self.t0) ])

    def seq41(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/0"](self.t0), self.pions["snk/pi0/0"](t), self.pions["src/pi0/1"](self.t0) ])

    def seq42(self, t): 
        return loop([ self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi0/0"](t), self.pions["src/pi0/1"](self.t0) ])

    def seq43(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi0/1"](t), self.pions["snk/pi0/0"](t) ])

    def seq44(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi0/0"](self.t0), self.pions["snk/pi0/0"](t) ])

    def seq45(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi0/1"](t), self.pions["src/pi0/1"](self.t0), self.pions["snk/pi0/0"](t) ])

    def seq46(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/1"](self.t0), self.pions["snk/pi0/0"](t) ])

    def seq47(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/1"](t), self.pions["snk/pi0/0"](t), self.pions["src/pi0/0"](self.t0) ])

    def seq48(self, t): 
        return loop([ self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi0/0"](t), self.pions["src/pi0/0"](self.t0) ])

    def seq49(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/1"](self.t0), self.pions["snk/pi0/0"](t), self.pions["src/pi0/0"](self.t0) ])

    def seq50(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi0/1"](t), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi0/0"](t) ])

    def seq51(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi0/1"](self.t0), self.pions["snk/pi0/0"](t) ])

    def seq52(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi0/1"](self.t0), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi0/0"](t) ])

    def seq53(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/1"](t), self.pions["snk/pi0/0"](t), self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/1"](self.t0) ])

    def seq54(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/1"](t), self.pions["src/pi0/1"](self.t0), self.pions["snk/pi0/0"](t), self.pions["src/pi0/0"](self.t0) ])

    def seq56(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/1"](t), self.pions["snk/pi0/0"](t), self.pions["src/pi0/1"](self.t0), self.pions["src/pi0/0"](self.t0) ])

    def seq57(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/1"](t), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi0/0"](t), self.pions["src/pi0/1"](self.t0) ])

    def seq59(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi0/1"](t), self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/1"](self.t0), self.pions["snk/pi0/0"](t) ])

    def seq60(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi0/1"](t), self.pions["src/pi0/1"](self.t0), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi0/0"](t) ])

    def seq61(self, t): 
        return loop([ self.perambs[t][t], self.perambs[t][t] ], [self.pions["snk/pi0/0"](t), self.pions["snk/pi0/1"](t) ])

    def seq62(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["src/pi0/0"](self.t0), self.pions["snk/pi0/0"](t), self.pions["snk/pi0/1"](t) ])

    def seq63(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["src/pi0/1"](self.t0), self.pions["snk/pi0/0"](t), self.pions["snk/pi0/1"](t) ])

    def seq64(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["src/pi0/0"](self.t0), self.pions["snk/pi0/1"](t), self.pions["snk/pi0/0"](t) ])

    def seq65(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["src/pi0/1"](self.t0), self.pions["snk/pi0/1"](t), self.pions["snk/pi0/0"](t) ])

    def seq66(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/0"](self.t0), self.pions["snk/pi0/0"](t), self.pions["snk/pi0/1"](t), self.pions["src/pi0/1"](self.t0) ])

    def seq68(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/1"](self.t0), self.pions["snk/pi0/0"](t), self.pions["snk/pi0/1"](t) ])

    def seq69(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/0"](self.t0), self.pions["snk/pi0/1"](t), self.pions["snk/pi0/0"](t), self.pions["src/pi0/1"](self.t0) ])

    def seq71(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi0/0"](self.t0), self.pions["snk/pi0/0"](t), self.pions["src/pi0/1"](self.t0), self.pions["snk/pi0/1"](t) ])

    def seq72(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/1"](self.t0), self.pions["snk/pi0/1"](t), self.pions["snk/pi0/0"](t) ])

    def seq73(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi0/0"](self.t0), self.pions["snk/pi0/1"](t), self.pions["src/pi0/1"](self.t0), self.pions["snk/pi0/0"](t) ])

    def seq75(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/1"](self.t0), self.pions["snk/pi0/0"](t), self.pions["snk/pi0/1"](t), self.pions["src/pi0/0"](self.t0) ])

    def seq76(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["src/pi0/1"](self.t0), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi0/0"](t), self.pions["snk/pi0/1"](t) ])

    def seq78(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/1"](self.t0), self.pions["snk/pi0/1"](t), self.pions["snk/pi0/0"](t), self.pions["src/pi0/0"](self.t0) ])

    def seq79(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi0/1"](self.t0), self.pions["snk/pi0/0"](t), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi0/1"](t) ])

    def seq80(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["src/pi0/1"](self.t0), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi0/1"](t), self.pions["snk/pi0/0"](t) ])

    def seq81(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["src/pi0/1"](self.t0), self.pions["snk/pi0/1"](t), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi0/0"](t) ])

    def seq82(self, t): 
        return loop([ self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/0"](t), self.pions["snk/pi0/1"](t), self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/1"](self.t0) ])

    def seq83(self, t): 
        return loop([ self.perambs[t][self.t0], self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["snk/pi0/0"](t), self.pions["snk/pi0/1"](t), self.pions["src/pi0/1"](self.t0), self.pions["src/pi0/0"](self.t0) ])

    def seq84(self, t): 
        return loop([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0] ], [self.pions["snk/pi0/0"](t), self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/1"](self.t0), self.pions["snk/pi0/1"](t) ])

    def seq85(self, t): 
        return loop([ self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi0/0"](t), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi0/1"](t), self.pions["src/pi0/1"](self.t0) ])

    def seq86(self, t): 
        return loop([ self.perambs[t][t], self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[t][self.t0] ], [self.pions["snk/pi0/0"](t), self.pions["src/pi0/1"](self.t0), self.pions["src/pi0/0"](self.t0), self.pions["snk/pi0/1"](t) ])

    def seq87(self, t): 
        return loop([ self.perambs[t][self.t0], self.perambs[self.t0][t], self.perambs[t][self.t0], self.perambs[self.t0][t] ], [self.pions["snk/pi0/0"](t), self.pions["src/pi0/1"](self.t0), self.pions["snk/pi0/1"](t), self.pions["src/pi0/0"](self.t0) ])

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
        S56 = self.seq56(t)
        S57 = self.seq57(t)
        S59 = self.seq59(t)
        S60 = self.seq60(t)
        S61 = self.seq61(t)
        S62 = self.seq62(t)
        S63 = self.seq63(t)
        S64 = self.seq64(t)
        S65 = self.seq65(t)
        S66 = self.seq66(t)
        S68 = self.seq68(t)
        S69 = self.seq69(t)
        S71 = self.seq71(t)
        S72 = self.seq72(t)
        S73 = self.seq73(t)
        S75 = self.seq75(t)
        S76 = self.seq76(t)
        S78 = self.seq78(t)
        S79 = self.seq79(t)
        S80 = self.seq80(t)
        S81 = self.seq81(t)
        S82 = self.seq82(t)
        S83 = self.seq83(t)
        S84 = self.seq84(t)
        S85 = self.seq85(t)
        S86 = self.seq86(t)
        S87 = self.seq87(t)


        def contraction(func, ops, factor):
            d0, d1 = func(self.eps_src, self.eps_snk[t], ops, self.P)
            d0 *= factor
            d1 *= factor
            return d0, d1


        # shortcuts

        L = self.perambs[self.t0][t]
        # GX
        GS6 = self.Gamma_snk @ S6
        GS68 = self.Gamma_snk @ S68
        GS31 = self.Gamma_snk @ S31
        GS16 = self.Gamma_snk @ S16
        GS73 = self.Gamma_snk @ S73
        GS1 = self.Gamma_snk @ S1
        GL = self.Gamma_snk @ L
        GS26 = self.Gamma_snk @ S26
        GS51 = self.Gamma_snk @ S51
        GS9 = self.Gamma_snk @ S9
        GS5 = self.Gamma_snk @ S5
        GS59 = self.Gamma_snk @ S59
        GS81 = self.Gamma_snk @ S81
        GS63 = self.Gamma_snk @ S63
        GS76 = self.Gamma_snk @ S76
        GS27 = self.Gamma_snk @ S27
        GS20 = self.Gamma_snk @ S20
        GS79 = self.Gamma_snk @ S79
        GS56 = self.Gamma_snk @ S56
        GS49 = self.Gamma_snk @ S49
        GS18 = self.Gamma_snk @ S18
        GS23 = self.Gamma_snk @ S23
        GS46 = self.Gamma_snk @ S46
        GS41 = self.Gamma_snk @ S41
        GS75 = self.Gamma_snk @ S75
        GS35 = self.Gamma_snk @ S35
        GS13 = self.Gamma_snk @ S13
        GS72 = self.Gamma_snk @ S72
        GS24 = self.Gamma_snk @ S24
        GS47 = self.Gamma_snk @ S47
        GS57 = self.Gamma_snk @ S57
        GS22 = self.Gamma_snk @ S22
        GS2 = self.Gamma_snk @ S2
        GS43 = self.Gamma_snk @ S43
        GS19 = self.Gamma_snk @ S19
        GS8 = self.Gamma_snk @ S8
        GS0 = self.Gamma_snk @ S0
        GS36 = self.Gamma_snk @ S36
        GS64 = self.Gamma_snk @ S64
        GS52 = self.Gamma_snk @ S52
        GS62 = self.Gamma_snk @ S62
        GS34 = self.Gamma_snk @ S34
        GS29 = self.Gamma_snk @ S29
        GS66 = self.Gamma_snk @ S66
        GS11 = self.Gamma_snk @ S11
        GS78 = self.Gamma_snk @ S78
        GS40 = self.Gamma_snk @ S40
        GS14 = self.Gamma_snk @ S14
        GS10 = self.Gamma_snk @ S10
        GS15 = self.Gamma_snk @ S15
        GS25 = self.Gamma_snk @ S25
        GS60 = self.Gamma_snk @ S60
        GS50 = self.Gamma_snk @ S50
        GS80 = self.Gamma_snk @ S80
        GS45 = self.Gamma_snk @ S45
        GS44 = self.Gamma_snk @ S44
        GS71 = self.Gamma_snk @ S71
        GS69 = self.Gamma_snk @ S69
        GS54 = self.Gamma_snk @ S54
        GS33 = self.Gamma_snk @ S33
        GS53 = self.Gamma_snk @ S53
        GS32 = self.Gamma_snk @ S32
        GS30 = self.Gamma_snk @ S30
        GS65 = self.Gamma_snk @ S65
        GS3 = self.Gamma_snk @ S3

        # XG 
        S3G = S3 @ self.Gamma_src
        S19G = S19 @ self.Gamma_src
        S75G = S75 @ self.Gamma_src
        S79G = S79 @ self.Gamma_src
        S29G = S29 @ self.Gamma_src
        S66G = S66 @ self.Gamma_src
        S46G = S46 @ self.Gamma_src
        S44G = S44 @ self.Gamma_src
        S33G = S33 @ self.Gamma_src
        S78G = S78 @ self.Gamma_src
        S68G = S68 @ self.Gamma_src
        S80G = S80 @ self.Gamma_src
        S20G = S20 @ self.Gamma_src
        S16G = S16 @ self.Gamma_src
        S62G = S62 @ self.Gamma_src
        S5G = S5 @ self.Gamma_src
        S40G = S40 @ self.Gamma_src
        S47G = S47 @ self.Gamma_src
        S8G = S8 @ self.Gamma_src
        S14G = S14 @ self.Gamma_src
        S69G = S69 @ self.Gamma_src
        S56G = S56 @ self.Gamma_src
        S60G = S60 @ self.Gamma_src
        S52G = S52 @ self.Gamma_src
        S81G = S81 @ self.Gamma_src
        S43G = S43 @ self.Gamma_src
        S18G = S18 @ self.Gamma_src
        S23G = S23 @ self.Gamma_src
        S57G = S57 @ self.Gamma_src
        S24G = S24 @ self.Gamma_src
        S54G = S54 @ self.Gamma_src
        S13G = S13 @ self.Gamma_src
        S35G = S35 @ self.Gamma_src
        LG = L @ self.Gamma_src
        S64G = S64 @ self.Gamma_src
        S50G = S50 @ self.Gamma_src
        S11G = S11 @ self.Gamma_src
        S65G = S65 @ self.Gamma_src
        S25G = S25 @ self.Gamma_src
        S71G = S71 @ self.Gamma_src
        S72G = S72 @ self.Gamma_src
        S53G = S53 @ self.Gamma_src
        S2G = S2 @ self.Gamma_src
        S27G = S27 @ self.Gamma_src
        S15G = S15 @ self.Gamma_src
        S9G = S9 @ self.Gamma_src
        S26G = S26 @ self.Gamma_src
        S32G = S32 @ self.Gamma_src
        S1G = S1 @ self.Gamma_src
        S76G = S76 @ self.Gamma_src
        S41G = S41 @ self.Gamma_src
        S0G = S0 @ self.Gamma_src
        S51G = S51 @ self.Gamma_src
        S36G = S36 @ self.Gamma_src
        S73G = S73 @ self.Gamma_src
        S63G = S63 @ self.Gamma_src
        S45G = S45 @ self.Gamma_src
        S34G = S34 @ self.Gamma_src
        S49G = S49 @ self.Gamma_src
        S31G = S31 @ self.Gamma_src
        S6G = S6 @ self.Gamma_src
        S30G = S30 @ self.Gamma_src
        S22G = S22 @ self.Gamma_src
        S59G = S59 @ self.Gamma_src
        S10G = S10 @ self.Gamma_src

        # GXG 


        diags = []

        # combined contractions
        for pre, ops in [(-1, [LG, GS60, L ]), (1, [S62G, GS24, L ]), (-1, [S44G, GS1, L ]), (1, [LG, GS65, S14 ]), (-1, [LG, GL, S80 ]), (1, [LG, GS23, S24 ]), (-1, [LG, GS73, L ]), 
                         (-1, [LG, GS51, S6 ]), (-1, [LG, GL, S72 ]), (-1, [S69G, GL, L ]), (-1, [S22G, GL, S9 ]), (-1, [S9G, GL, S30 ]), (1, [S16G, GS9, S3 ]), (1, [LG, GS24, S64 ]), 
                         (-1, [LG, GL, S56 ]), (1, [S65G, GS14, L ]), (-1, [S15G, GS16, L ]), (-1, [S25G, GL, S15 ]), (1, [S9G, GS3, S25 ]), (-1, [S3G, GS0, S24 ]), (-1, [S3G, GL, S2 ]), 
                         (-1, [S0G, GS1, L ]), (-1, [S23G, GL, S24 ]), (-1, [S68G, GL, L ]), (1, [S24G, GS14, S15 ]), (-1, [LG, GL, S33 ]), (1, [S9G, GS11, L ]), (1, [S14G, GS24, S43 ]), 
                         (-1, [S14G, GS5, S3 ]), (-1, [LG, GS80, L ]), (-1, [LG, GL, S32 ]), (1, [S44G, GS24, S3 ]), (1, [LG, GS14, S19 ]), (-1, [S64G, GL, S24 ]), (-1, [S24G, GS6, S9 ]), 
                         (-1, [LG, GS75, L ]), (1, [LG, GS9, S30 ]), (1, [LG, GS2, S3 ]), (-1, [S43G, GL, S16 ]), (-1, [LG, GS27, S0 ]), (-1, [S5G, GS6, L ]), (1, [S3G, GS49, L ]), 
                         (-1, [LG, GL, S78 ]), (-1, [S14G, GS27, S9 ]), (1, [S24G, GS9, S18 ]), (-1, [LG, GS43, S16 ]), (-1, [S73G, GL, L ]), (-1, [S9G, GL, S29 ]), (1, [S3G, GS24, S0 ]), 
                         (-1, [S24G, GL, S62 ]), (-1, [S18G, GL, S51 ]), (1, [LG, GS63, S14 ]), (-1, [S41G, GL, S3 ]), (1, [S14G, GS40, L ]), (-1, [S18G, GL, S5 ]), (1, [S24G, GS14, S43 ]), 
                         (-1, [LG, GS16, S43 ]), (1, [LG, GS9, S20 ]), (-1, [S14G, GL, S45 ]), (-1, [S0G, GL, S27 ]), (-1, [S3G, GS16, S9 ]), (-1, [LG, GL, S54 ]), (-1, [LG, GL, S31 ]), 
                         (1, [S24G, GS3, S44 ]), (-1, [LG, GS68, L ]), (-1, [S3G, GS44, S24 ]), (-1, [S24G, GS0, S3 ]), (-1, [LG, GL, S68 ]), (-1, [S56G, GL, L ]), (-1, [S57G, GL, L ]), 
                         (-1, [S14G, GL, S19 ]), (1, [S43G, GS24, S14 ]), (-1, [S29G, GL, S9 ]), (-1, [S5G, GS18, L ]), (1, [LG, GS3, S52 ]), (1, [S51G, GS14, S3 ]), (-1, [S36G, GL, L ]), 
                         (-1, [S46G, GL, S3 ]), (-1, [S6G, GS5, L ]), (1, [S3G, GS52, L ]), (-1, [LG, GS76, L ]), (1, [S24G, GS9, S6 ]), (-1, [LG, GL, S71 ]), (1, [S0G, GS3, S24 ]), 
                         (-1, [S71G, GL, L ]), (-1, [S9G, GS6, S24 ]), (1, [S14G, GS63, L ]), (1, [S18G, GS9, S24 ]), (1, [LG, GS47, S24 ]), (1, [S41G, GS3, L ]), (1, [S9G, GS10, L ]), 
                         (1, [S3G, GS2, L ]), (-1, [S50G, GL, S24 ]), (1, [S29G, GS9, L ]), (-1, [LG, GL, S60 ]), (-1, [LG, GS27, S44 ]), (-1, [LG, GS32, L ]), (-1, [LG, GS1, S0 ]), 
                         (-1, [S62G, GL, S24 ]), (-1, [S0G, GL, S1 ]), (1, [S9G, GS14, S1 ]), (1, [S1G, GS9, S14 ]), (-1, [S35G, GL, L ]), (-1, [S2G, GL, S3 ]), (1, [LG, GS24, S23 ]), 
                         (1, [LG, GS14, S45 ]), (-1, [S13G, GL, S14 ]), (1, [S11G, GS9, L ]), (-1, [S79G, GL, L ]), (1, [S15G, GS14, S24 ]), (1, [LG, GS8, S3 ]), (-1, [S51G, GS18, L ]), 
                         (1, [S9G, GS20, L ]), (-1, [S24G, GS15, S14 ]), (1, [LG, GS10, S9 ]), (1, [LG, GS49, S3 ]), (-1, [LG, GS25, S15 ]), (1, [S9G, GS24, S6 ]), (1, [S6G, GS24, S9 ]), 
                         (-1, [S51G, GS6, L ]), (1, [LG, GS30, S9 ]), (-1, [LG, GS34, L ]), (-1, [LG, GL, S34 ]), (-1, [S47G, GL, S24 ]), (-1, [S9G, GL, S10 ]), (-1, [LG, GS69, L ]), 
                         (1, [S24G, GS47, L ]), (1, [S43G, GS14, S24 ]), (-1, [S40G, GL, S14 ]), (-1, [LG, GS18, S5 ]), (1, [S19G, GS14, L ]), (-1, [S44G, GS27, L ]), (1, [S40G, GS14, L ]), 
                         (1, [LG, GS3, S41 ]), (1, [LG, GS3, S46 ]), (1, [S3G, GS14, S5 ]), (-1, [S16G, GL, S43 ]), (1, [S3G, GS41, L ]), (-1, [LG, GL, S73 ]), (-1, [S24G, GS43, S14 ]), 
                         (1, [S14G, GS9, S27 ]), (-1, [S9G, GS16, S3 ]), (1, [LG, GS24, S47 ]), (-1, [LG, GS66, L ]), (1, [S25G, GS3, S9 ]), (-1, [LG, GS71, L ]), (-1, [S24G, GL, S47 ]), 
                         (-1, [S3G, GL, S46 ]), (1, [S16G, GS3, S9 ]), (1, [S63G, GS14, L ]), (-1, [S14G, GL, S40 ]), (-1, [LG, GL, S66 ]), (-1, [LG, GS44, S27 ]), (-1, [S6G, GL, S5 ]), 
                         (-1, [S3G, GL, S8 ]), (1, [S3G, GS46, L ]), (1, [S1G, GS14, S9 ]), (-1, [S25G, GS15, L ]), (-1, [S9G, GL, S11 ]), (1, [LG, GS26, S24 ]), (-1, [S9G, GS1, S14 ]), 
                         (1, [S14G, GS45, L ]), (-1, [S3G, GS25, S9 ]), (-1, [S43G, GL, S25 ]), (1, [S23G, GS24, L ]), (1, [S30G, GS9, L ]), (-1, [S3G, GL, S41 ]), (1, [S14G, GS13, L ]), 
                         (-1, [S3G, GL, S49 ]), (-1, [LG, GS57, L ]), (-1, [S16G, GL, S15 ]), (-1, [S43G, GS16, L ]), (-1, [S27G, GL, S0 ]), (1, [S26G, GS24, L ]), (1, [S3G, GS8, L ]), 
                         (1, [S27G, GS14, S9 ]), (-1, [LG, GL, S35 ]), (-1, [S52G, GL, S3 ]), (1, [S9G, GS14, S27 ]), (1, [LG, GS45, S14 ]), (1, [LG, GS14, S40 ]), (1, [S3G, GS9, S25 ]), 
                         (-1, [LG, GL, S59 ]), (-1, [S60G, GL, L ]), (-1, [LG, GS36, L ]), (1, [S2G, GS3, L ]), (-1, [S27G, GS44, L ]), (-1, [S3G, GS5, S14 ]), (1, [S24G, GS50, L ]), 
                         (-1, [S5G, GL, S18 ]), (-1, [S15G, GS25, L ]), (-1, [S44G, GL, S1 ]), (1, [LG, GS19, S14 ]), (1, [LG, GS50, S24 ]), (-1, [LG, GS35, L ]), (-1, [LG, GS6, S51 ]), 
                         (-1, [S14G, GS1, S9 ]), (-1, [LG, GL, S76 ]), (-1, [S24G, GL, S64 ]), (-1, [S1G, GS44, L ]), (-1, [S81G, GL, L ]), (-1, [S8G, GL, S3 ]), (-1, [S11G, GL, S9 ]), 
                         (-1, [LG, GL, S36 ]), (1, [S24G, GS64, L ]), (1, [S14G, GS24, S15 ]), (-1, [S9G, GS18, S24 ]), (1, [S45G, GS14, L ]), (-1, [S75G, GL, L ]), (-1, [LG, GS31, L ]), 
                         (-1, [LG, GS15, S25 ]), (-1, [LG, GS5, S6 ]), (-1, [S9G, GL, S20 ]), (-1, [LG, GL, S79 ]), (-1, [LG, GS56, L ]), (-1, [LG, GL, S75 ]), (1, [S9G, GS3, S16 ]), 
                         (1, [S5G, GS14, S3 ]), (1, [LG, GS3, S49 ]), (-1, [S18G, GS51, L ]), (1, [S51G, GS3, S14 ]), (-1, [S24G, GL, S23 ]), (1, [LG, GS52, S3 ]), (1, [S50G, GS24, L ]), 
                         (1, [S20G, GS9, L ]), (1, [LG, GS13, S14 ]), (-1, [S44G, GL, S27 ]), (-1, [S3G, GL, S52 ]), (-1, [S24G, GS44, S3 ]), (-1, [S6G, GL, S51 ]), (1, [LG, GS14, S63 ]), 
                         (1, [S49G, GS3, L ]), (-1, [S5G, GL, S6 ]), (-1, [S30G, GL, S9 ]), (-1, [S9G, GS25, S3 ]), (-1, [S14G, GL, S65 ]), (1, [LG, GS62, S24 ]), (-1, [LG, GL, S81 ]), 
                         (-1, [LG, GS16, S15 ]), (-1, [S43G, GS25, L ]), (1, [LG, GS9, S22 ]), (1, [S44G, GS3, S24 ]), (1, [LG, GS9, S11 ]), (-1, [LG, GS44, S1 ]), (1, [S5G, GS3, S14 ]), 
                         (1, [S3G, GS14, S51 ]), (-1, [S25G, GL, S43 ]), (1, [LG, GS41, S3 ]), (1, [S14G, GS3, S5 ]), (1, [S27G, GS9, S14 ]), (-1, [LG, GS25, S43 ]), (1, [S24G, GS62, L ]), 
                         (1, [S24G, GS26, L ]), (-1, [S24G, GL, S26 ]), (-1, [S6G, GS51, L ]), (1, [S9G, GS29, L ]), (-1, [S59G, GL, L ]), (-1, [S54G, GL, L ]), (-1, [S9G, GS27, S14 ]), 
                         (-1, [LG, GS43, S25 ]), (1, [S8G, GS3, L ]), (-1, [S27G, GL, S44 ]), (-1, [S51G, GL, S18 ]), (-1, [S78G, GL, L ]), (1, [S14G, GS19, L ]), (-1, [S16G, GS43, L ]), 
                         (-1, [LG, GS15, S16 ]), (-1, [S1G, GL, S44 ]), (1, [S24G, GS23, L ]), (-1, [LG, GS59, L ]), (1, [S13G, GS14, L ]), (1, [LG, GS14, S13 ]), (-1, [LG, GL, S69 ]), 
                         (1, [LG, GS46, S3 ]), (-1, [LG, GS53, L ]), (-1, [S27G, GS0, L ]), (-1, [S18G, GS5, L ]), (1, [S46G, GS3, L ]), (-1, [S24G, GS18, S9 ]), (1, [S22G, GS9, L ]), 
                         (-1, [S24G, GL, S50 ]), (1, [S18G, GS24, S9 ]), (-1, [LG, GS79, L ]), (1, [S10G, GS9, L ]), (-1, [LG, GS18, S51 ]), (-1, [S14G, GS51, S3 ]), (-1, [LG, GL, S53 ]), 
                         (1, [LG, GS40, S14 ]), (-1, [S0G, GS27, L ]), (1, [LG, GS9, S29 ]), (-1, [S25G, GS43, L ]), (-1, [LG, GS81, L ]), (-1, [S49G, GL, S3 ]), (-1, [S51G, GL, S6 ]), 
                         (1, [S15G, GS24, S14 ]), (-1, [LG, GS51, S18 ]), (-1, [S33G, GL, L ]), (1, [S14G, GS3, S51 ]), (-1, [LG, GS78, L ]), (-1, [S76G, GL, L ]), (1, [S9G, GS22, L ]), 
                         (1, [S9G, GS30, L ]), (1, [S6G, GS9, S24 ]), (1, [LG, GS3, S8 ]), (-1, [S9G, GL, S22 ]), (1, [LG, GS24, S26 ]), (1, [S9G, GS24, S18 ]), (-1, [S1G, GL, S0 ]), 
                         (-1, [S20G, GL, S9 ]), (1, [LG, GS29, S9 ]), (-1, [S14G, GL, S13 ]), (-1, [S53G, GL, L ]), (-1, [S31G, GL, L ]), (1, [LG, GS64, S24 ]), (-1, [S16G, GS15, L ]), 
                         (-1, [LG, GS72, L ]), (-1, [S10G, GL, S9 ]), (1, [S24G, GS3, S0 ]), (-1, [S80G, GL, L ]), (1, [LG, GS24, S62 ]), (1, [LG, GS3, S2 ]), (-1, [S65G, GL, S14 ]), 
                         (1, [S64G, GS24, L ]), (-1, [LG, GS0, S27 ]), (1, [S14G, GS65, L ]), (-1, [S45G, GL, S14 ]), (-1, [LG, GS5, S18 ]), (1, [S52G, GS3, L ]), (-1, [LG, GS6, S5 ]), 
                         (-1, [S14G, GL, S63 ]), (-1, [LG, GS1, S44 ]), (-1, [LG, GS33, L ]), (1, [S25G, GS9, S3 ]), (1, [S3G, GS24, S44 ]), (-1, [S72G, GL, L ]), (1, [LG, GS9, S10 ]), 
                         (-1, [S15G, GL, S25 ]), (-1, [S63G, GL, S14 ]), (-1, [LG, GS0, S1 ]), (-1, [S14G, GS15, S24 ]), (1, [LG, GS24, S50 ]), (-1, [LG, GL, S57 ]), (-1, [S32G, GL, L ]), 
                         (-1, [S19G, GL, S14 ]), (-1, [S14G, GS43, S24 ]), (-1, [LG, GS54, L ]), (1, [LG, GS20, S9 ]), (1, [S3G, GS9, S16 ]), (1, [LG, GS22, S9 ]), (-1, [S34G, GL, L ]), 
                         (-1, [S66G, GL, L ]), (-1, [S15G, GL, S16 ]), (1, [LG, GS14, S65 ]), (-1, [S3G, GS51, S14 ]), (1, [S14G, GS9, S1 ]), (1, [S0G, GS24, S3 ]), (-1, [S26G, GL, S24 ]), 
                         (-1, [S1G, GS0, L ]), (1, [LG, GS11, S9 ]), (1, [S47G, GS24, L ])]:
            di, dj = contraction(combined_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)


        # combined contractions
        for pre, ops, loops in [(2, [LG, GL, S6 ], [S42]), (-2, [S3G, GS9, L ], [S12]), (2, [LG, GL, S1 ], [S48]), (-2, [S3G, GS14, L ], [S42]), (2, [S3G, GL, S24 ], [S48]), (2, [S24G, GL, S14 ], [S61]), 
                                (2, [S3G, GL, S9 ], [S12]), (-4, [LG, GL, L ], [S61, S12]), (2, [S24G, GL, S9 ], [S28]), (-2, [LG, GS3, S14 ], [S42]), (-4, [LG, GL, L ], [S28, S42]), 
                                (-2, [LG, GS24, S9 ], [S28]), (2, [S9G, GL, S14 ], [S21]), (2, [S0G, GL, L ], [S21]), (-2, [LG, GS9, S14 ], [S21]), (-2, [LG, GS9, S24 ], [S28]), 
                                (2, [S43G, GL, L ], [S12]), (2, [LG, GS0, L ], [S21]), (-2, [LG, GS3, S24 ], [S48]), (2, [S9G, GL, S3 ], [S12]), (2, [S9G, GL, S24 ], [S28]), (-2, [S9G, GS24, L ], [S28]), 
                                (2, [S1G, GL, L ], [S48]), (-2, [LG, GS14, S3 ], [S42]), (2, [LG, GL, L ], [S87]), (2, [LG, GS51, L ], [S28]), (-2, [S9G, GS14, L ], [S21]), (2, [LG, GL, S27 ], [S48]), 
                                (2, [S51G, GL, L ], [S28]), (-2, [LG, GS9, S3 ], [S12]), (2, [LG, GL, S18 ], [S42]), (2, [S44G, GL, L ], [S21]), (-2, [S24G, GS9, L ], [S28]), (2, [S25G, GL, L ], [S61]), 
                                (-2, [S3G, GS24, L ], [S48]), (2, [S14G, GL, S3 ], [S42]), (2, [LG, GL, S5 ], [S28]), (2, [S24G, GL, S3 ], [S48]), (2, [LG, GL, S25 ], [S61]), (2, [LG, GL, S44 ], [S21]), 
                                (2, [LG, GS6, L ], [S42]), (-2, [S14G, GS24, L ], [S61]), (2, [LG, GL, S43 ], [S12]), (2, [S16G, GL, L ], [S61]), (-2, [S14G, GS3, L ], [S42]), (2, [S15G, GL, L ], [S12]), 
                                (-2, [S24G, GS3, L ], [S48]), (-2, [LG, GS24, S14 ], [S61]), (2, [S14G, GL, S24 ], [S61]), (2, [LG, GS5, L ], [S28]), (2, [LG, GL, L ], [S82]), (2, [LG, GL, S0 ], [S21]), 
                                (2, [S27G, GL, L ], [S48]), (2, [LG, GL, L ], [S83]), (-2, [S9G, GS3, L ], [S12]), (2, [LG, GS1, L ], [S48]), (-4, [LG, GL, L ], [S48, S21]), (2, [S3G, GL, S14 ], [S42]), 
                                (2, [LG, GS43, L ], [S12]), (-2, [LG, GS14, S24 ], [S61]), (2, [LG, GL, L ], [S85]), (2, [LG, GL, S51 ], [S28]), (2, [LG, GL, L ], [S84]), (2, [LG, GS25, L ], [S61]), 
                                (2, [LG, GS27, L ], [S48]), (2, [LG, GS15, L ], [S12]), (2, [LG, GS18, L ], [S42]), (2, [LG, GL, S16 ], [S61]), (-2, [LG, GS24, S3 ], [S48]), (2, [S5G, GL, L ], [S28]), 
                                (-2, [LG, GS14, S9 ], [S21]), (-2, [LG, GS3, S9 ], [S12]), (2, [LG, GS44, L ], [S21]), (2, [S18G, GL, L ], [S42]), (-2, [S24G, GS14, L ], [S61]), (2, [LG, GL, S15 ], [S12]), 
                                (2, [S6G, GL, L ], [S42]), (-2, [S14G, GS9, L ], [S21]), (2, [S14G, GL, S9 ], [S21]), (2, [LG, GL, L ], [S86]), (2, [LG, GS16, L ], [S61])]:
            factor = pre * np.prod(loops)
            di, dj = contraction(combined_contraction, ops, factor=factor)
            diags.append(di)
            diags.append(dj)

        return diags

    def corr(self, t):
        return 0.25 * sum(self.diagrams(t))
    



class PerambNucleon2NucleonPion0Pion0(PerambBaryon):
    # 26 diagrams 
    def __init__(self, eps_src, perambs, mins_snk_pi_plus, t0, P=None, eps_snk=None, mins_snk_pi_minus=None,
                 Gamma_snk_pi_plus=None, Gamma_snk_pi_minus=None, Gamma_src=None, Gamma_snk=None, tval=None):
        
        ## < O_p(x)  O_pi+(z)  O_pi-(q)  ~O_p(y) >

        self.t0 = t0

        self.eps_src = np.conjugate(eps_src[self.t0])
        if eps_snk is None:
            self.eps_snk = eps_src
        else:
            self.eps_snk = eps_src

        
        self.perambs = perambs
        self.Nt = perambs[0].shape[0]
        self.tval = range(self.Nt) if tval is None else tval

        self.mins_snk_pi_plus  = mins_snk_pi_plus if mins_snk_pi_plus is None else mins_snk_pi_plus
        self.mins_snk_pi_minus = mins_snk_pi_plus if mins_snk_pi_minus is None else mins_snk_pi_minus

        g_default = lqcdpy.distillation.mat.pion['g5']

        self.Gamma_snk_pi_plus  = g_default if Gamma_snk_pi_plus is None else Gamma_snk_pi_plus
        self.Gamma_snk_pi_minus = g_default if Gamma_snk_pi_minus is None else Gamma_snk_pi_minus


        self.pions = {
            'snk/pi0/0': lambda t: (self.mins_snk_pi_plus[t], self.Gamma_snk_pi_plus),
            'snk/pi0/1': lambda t: (self.mins_snk_pi_minus[t], self.Gamma_snk_pi_minus),
        }

        cg_default = lqcdpy.distillation.mat.nucleon['Cg5']

        self.Gamma_src = cg_default if Gamma_src is None else Gamma_src
        self.Gamma_snk = cg_default if Gamma_snk is None else Gamma_snk

        self.P = np.eye(4) if P is None else P 

    def seq0(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi0/0"](t) ])

    def seq1(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi0/1"](t) ])

    def seq2(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi0/0"](t), self.pions["snk/pi0/1"](t) ])

    def seq5(self, t): 
        return sequential([ self.perambs[t][t], self.perambs[t][t], self.perambs[self.t0][t] ], [self.pions["snk/pi0/1"](t), self.pions["snk/pi0/0"](t) ])

    def seq6(self, t): 
        return loop([ self.perambs[t][t], self.perambs[t][t] ], [self.pions["snk/pi0/0"](t), self.pions["snk/pi0/1"](t) ])

    def diagrams(self, t):
        S0 = self.seq0(t)
        S1 = self.seq1(t)
        S2 = self.seq2(t)
        S5 = self.seq5(t)
        S6 = self.seq6(t)


        def contraction(func, ops, factor):
            d0, d1 = func(self.eps_src, self.eps_snk[t], ops, self.P)
            d0 *= factor
            d1 *= factor
            return d0, d1


        # shortcuts

        L = self.perambs[self.t0][t]
        # GX
        GS2 = self.Gamma_snk @ S2
        GS0 = self.Gamma_snk @ S0
        GS5 = self.Gamma_snk @ S5
        GS1 = self.Gamma_snk @ S1
        GL = self.Gamma_snk @ L

        # XG 
        S2G = S2 @ self.Gamma_src
        S0G = S0 @ self.Gamma_src
        LG = L @ self.Gamma_src
        S5G = S5 @ self.Gamma_src
        S1G = S1 @ self.Gamma_src

        # GXG 


        diags = []

        # combined contractions
        for pre, ops in [(1, [LG, GS0, S1 ]), (-1, [S1G, GL, S0 ]), (-1, [LG, GL, S5 ]), (-1, [S5G, GL, L ]), (-1, [LG, GL, S2 ]), (-1, [LG, GS2, L ]), 
                         (1, [LG, GS1, S0 ]), (-1, [S2G, GL, L ]), (-1, [S0G, GL, S1 ]), (-1, [LG, GS5, L ]), (1, [S1G, GS0, L ]), (1, [S0G, GS1, L ])]:
            di, dj = contraction(combined_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)


        # combined contractions
        for pre, ops, loops in [(2, [LG, GL, L ], [S6])]:
            factor = pre * np.prod(loops)
            di, dj = contraction(combined_contraction, ops, factor=factor)
            diags.append(di)
            diags.append(dj)


        return diags

    
    def corr(self, t):
        return 0.5 * sum(self.diagrams(t))
    


class PerambNucleonPion0Pion02Nucleon(PerambBaryon):
    # 26 diagrams 
    def __init__(self, eps_src, perambs, mins_src_pi_plus, t0, P=None, eps_snk=None, mins_src_pi_minus=None,
                 Gamma_src_pi_plus=None, Gamma_src_pi_minus=None, Gamma_src=None, Gamma_snk=None, tval=None):
        
        ## < O_p(x)  O_pi+(z)  O_pi-(q)  ~O_p(y) >

        self.t0 = t0

        self.eps_src = np.conjugate(eps_src[self.t0])
        if eps_snk is None:
            self.eps_snk = eps_src
        else:
            self.eps_snk = eps_src

        
        self.perambs = perambs
        self.Nt = perambs[0].shape[0]
        self.tval = range(self.Nt) if tval is None else tval

        self.mins_src_pi_plus  = mins_src_pi_plus if mins_src_pi_plus is None else mins_src_pi_plus
        self.mins_src_pi_minus = mins_src_pi_plus if mins_src_pi_minus is None else mins_src_pi_minus

        g_default = lqcdpy.distillation.mat.pion['g5']

        self.Gamma_src_pi_plus  = g_default if Gamma_src_pi_plus is None else Gamma_src_pi_plus
        self.Gamma_src_pi_minus = g_default if Gamma_src_pi_minus is None else Gamma_src_pi_minus


        self.pions = {
            'src/pi0/0': lambda t: (self.mins_src_pi_plus[t], self.Gamma_src_pi_plus),
            'src/pi0/1': lambda t: (self.mins_src_pi_minus[t], self.Gamma_src_pi_minus),
        }

        cg_default = lqcdpy.distillation.mat.nucleon['Cg5']

        self.Gamma_src = cg_default if Gamma_src is None else Gamma_src
        self.Gamma_snk = cg_default if Gamma_snk is None else Gamma_snk

        self.P = np.eye(4) if P is None else P 

    def seq0(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/0"](self.t0) ])

    def seq1(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/1"](self.t0) ])

    def seq2(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/1"](self.t0) ])

    def seq5(self, t): 
        return sequential([ self.perambs[self.t0][t], self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/1"](self.t0), self.pions["src/pi0/0"](self.t0) ])

    def seq6(self, t): 
        return loop([ self.perambs[self.t0][self.t0], self.perambs[self.t0][self.t0] ], [self.pions["src/pi0/0"](self.t0), self.pions["src/pi0/1"](self.t0) ])

    def diagrams(self, t):
        S0 = self.seq0(t)
        S1 = self.seq1(t)
        S2 = self.seq2(t)
        S5 = self.seq5(t)
        S6 = self.seq6(t)


        def contraction(func, ops, factor):
            d0, d1 = func(self.eps_src, self.eps_snk[t], ops, self.P)
            d0 *= factor
            d1 *= factor
            return d0, d1


        # shortcuts

        L = self.perambs[self.t0][t]
        # GX
        GS2 = self.Gamma_snk @ S2
        GS0 = self.Gamma_snk @ S0
        GS5 = self.Gamma_snk @ S5
        GS1 = self.Gamma_snk @ S1
        GL = self.Gamma_snk @ L

        # XG 
        S2G = S2 @ self.Gamma_src
        S0G = S0 @ self.Gamma_src
        LG = L @ self.Gamma_src
        S5G = S5 @ self.Gamma_src
        S1G = S1 @ self.Gamma_src

        # GXG 


        diags = []

        # combined contractions
        for pre, ops in [(1, [LG, GS0, S1 ]), (-1, [S1G, GL, S0 ]), (-1, [LG, GL, S5 ]), (-1, [S5G, GL, L ]), (-1, [LG, GL, S2 ]), (-1, [LG, GS2, L ]), 
                         (1, [LG, GS1, S0 ]), (-1, [S2G, GL, L ]), (-1, [S0G, GL, S1 ]), (-1, [LG, GS5, L ]), (1, [S1G, GS0, L ]), (1, [S0G, GS1, L ])]:
            di, dj = contraction(combined_contraction, ops, factor=pre)
            diags.append(di)
            diags.append(dj)


        # combined contractions
        for pre, ops, loops in [(2, [LG, GL, L ], [S6])]:
            factor = pre * np.prod(loops)
            di, dj = contraction(combined_contraction, ops, factor=factor)
            diags.append(di)
            diags.append(dj)


        return diags

    
    def corr(self, t):
        return 0.5 * sum(self.diagrams(t))
    
