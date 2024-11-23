import numpy as np
import jax
import jax.numpy as jp
import lqcdpy.distillation.mat as mat
from jax import config
config.update("jax_enable_x64", True)
import threading


g3 = mat.gamma[3]

@jax.jit
def dagger(A):
    return jp.transpose(jp.conjugate(A))

@jax.jit
def trace_contraction(eps_src, eps_snk, ops, P_snk, P_src=None):
    if len(ops) != 3:
        raise ValueError()

    if P_src is None:
        P_src = dagger(P_snk)
    
    tmp = jp.einsum('npbc,mqbc->nmpq', ops[0], ops[1], precision='highest')
    tmp = jp.einsum('lnm,nmpq->lpq', eps_snk, tmp, precision='highest')
    tmp = jp.einsum('lpq,load->opqad', tmp, ops[2], precision='highest')
    return jp.einsum('ma,opqad,opq,dn->mn', P_snk, tmp, eps_src, g3@P_src, precision='highest')

@jax.jit
def traceless_contraction(eps_src, eps_snk, ops, P_snk, P_src=None):
    if len(ops) != 3:
        raise ValueError()

    if P_src is None:
        P_src = dagger(P_snk)
    
    tmp = jp.einsum('lnm,npac->lpmac', eps_snk, ops[0], precision='highest')
    tmp = jp.einsum('lpmac,mqbc->lpqab', tmp, ops[1], precision='highest')
    tmp = jp.einsum('lpqab,lobd->opqad', tmp, ops[2], precision='highest')
    return jp.einsum('ma,opqad,opq,dn->mn', P_snk, tmp, eps_src, g3@P_src, precision='highest')

"""
def combined_contraction(eps_src, eps_snk, ops, P_snk, P_src=None):
    tr_c = trace_contraction(eps_src, eps_snk, ops, P_snk, P_src)
    tl_c = traceless_contraction(eps_src, eps_snk, ops, P_snk, P_src)
    return [tr_c, tl_c]

def cross_contraction(eps_src, eps_snk, ops, P_snk, P_src=None):
    tl_c_r = traceless_contraction(eps_src, eps_snk, ops, P_snk, P_src)
    tl_c_l = traceless_contraction(eps_src, eps_snk, [ops[1], ops[0], ops[2]], P_snk, P_src)
    return [tl_c_r, tl_c_l]
"""

@jax.jit
def combined_contraction(eps_src, eps_snk, ops, P_snk, P_src=None):
    if len(ops) != 3:
        raise ValueError()

    if P_src is None:
        P_src = dagger(P_snk)


    op3 = jp.einsum('loca,opq,am->lpqcm', ops[2], eps_src, g3@P_src, precision='highest')
    dq = jp.einsum('lnm,npac->lpmac', eps_snk, ops[0], precision='highest')
    dq = jp.einsum('lpmac,mqbc->lpqab', dq, ops[1], precision='highest')

    dq_t = jp.einsum('lpqaa->lpq', dq, precision='highest')

    tr_c = jp.einsum('ma,lpq,lpqan', P_snk, dq_t, op3, precision='highest')
    tr_l = jp.einsum('ma,lpqab,lpqbn', P_snk, dq, op3, precision='highest')
    return [tr_c, tr_l]

@jax.jit
def cross_contraction(eps_src, eps_snk, ops, P_snk, P_src=None):
    if len(ops) != 3:
        raise ValueError()

    if P_src is None:
        P_src = dagger(P_snk)


    op3 = jp.einsum('loca,am->locm', ops[2], g3@P_src, precision='highest')
    dq = jp.einsum('lnm,npac->lpmac', eps_snk, ops[0], precision='highest')
    dq = jp.einsum('lpmac,mqbc->lpqab', dq, ops[1], precision='highest')
    dq = jp.einsum('lpqab,opq->loab', dq, eps_src, precision='highest')

    dqT = jp.transpose(dq, axes=(0,1,3,2))   # using Q[A,B]^T = Q[B, A]

    cont_0 = jp.einsum('ma,loab,lobn->mn', P_snk, dq, op3, precision='highest')
    cont_1 = jp.einsum('ma,loab,lobn->mn', P_snk, dqT, op3, precision='highest')
    return [cont_0, cont_1]

@jax.jit
def sequential(perambs, pions):
    if len(perambs) != len(pions) + 1:
        raise ValueError()
    
    seq = perambs[0]
    for i in range(len(pions)):
        seq = jp.einsum('nmab,mp,bc->npac', seq, pions[i][0], pions[i][1], precision='highest')
        seq = jp.einsum('nmab,mpbc->npac', seq, perambs[i+1], precision='highest')
    return seq

@jax.jit
def loop(perambs, pions):
    if len(perambs) != len(pions):
        raise ValueError()
    
    seq = perambs[0]
    for i in range(len(pions)-1):
        seq = jp.einsum('nmab,mp,bc->npac', seq, pions[i][0], pions[i][1], precision='highest')
        seq = jp.einsum('nmab,mpbc->npac', seq, perambs[i+1], precision='highest')
    return jp.einsum('nmab,mn,ba->', seq, pions[-1][0], pions[-1][1], precision='highest')


def corr_fill(corrs, tval, Nt):
    Nspin = corrs[0].shape[0]
    c = np.zeros((Nt, Nspin, Nspin), dtype=np.cdouble)
    for t in range(Nt):
        if t in tval:
            i = tval.index(t)
            c[t,:,:] = corrs[i]
        else:
            c[t,:,:] = np.nan
    return c

def run(cont, Nspin=4):
    Nt = cont.Nt
    tval = cont.tval

    corr = np.ones((Nt, Nspin, Nspin), dtype=np.cdouble) * np.nan

    def runner(t):
        d = cont.corr(t)
        corr[t,:,:] = d

    threads = []
    for t in tval:
        t = threading.Thread(target=runner, args=(t,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    if Nspin == 1:
        return corr[:,0,0]
    return corr

class PerambBaryon:
    def __init__(self):
        self.perambs = []
        self.Nt = 1
        self.tval = [0]

    def corr(self, t):
        return None
    
    def __call__(self):
        return run(self, Nspin=4)
    
    def write(self, w, tag):
        c = self.__call__()
        w.write_spin(tag, c)
    
class PerambMeson:
    def __init__(self):
        self.perambs = []
        self.Nt = 1
        self.tval = [0]

    def corr(self, t):
        return None
    
    def __call__(self):
        return run(self, Nspin=1)
    
    def write(self, w, tag):
        c = self.__call__()
        w.write(tag, c)


class PerambPion2Pion(PerambMeson):
    
    def __init__(self, perambs, mins, t0, tval=None):
        self.t0 = t0
        self.perambs = perambs
        self.Nt = perambs[0].shape[0]
        self.tval = range(self.Nt) if tval is None else tval
        self.mins = mins
        
        self.Gamma_pi_src = mat.pion['g5']
        self.Gamma_pi_snk = mat.pion['g5']

    def corr(self, t):
        tmp = jp.einsum('nm,ab,mqbc->nqac', self.mins[t], self.Gamma_pi_snk, self.perambs[self.t0][t])
        tmp = jp.einsum('nqac,qp,cd->npad', tmp, self.mins[self.t0], self.Gamma_pi_src)
        return jp.einsum('npad,pnda->', tmp, self.perambs[t][self.t0])
    
    

class PerambEta2Eta(PerambMeson):
    
    def __init__(self, perambs, mins, t0, tval=None):
        self.t0 = t0
        self.perambs = perambs
        self.Nt = perambs[0].shape[0]
        self.tval = range(self.Nt) if tval is None else tval
        self.mins = mins
        
        self.Gamma_pi_src = mat.pion['g5']
        self.Gamma_pi_snk = mat.pion['g5']

    def corr(self, t):
        tmp = jp.einsum('nm,ab,mqbc->nqac', self.mins[t], self.Gamma_pi_snk, self.perambs[self.t0][t])
        tmp = jp.einsum('nqac,qp,cd->npad', tmp, self.mins[self.t0], self.Gamma_pi_src)
        c0 = jp.einsum('npad,pnda->', tmp, self.perambs[t][self.t0])

        l0 = jp.einsum('nm,ab,mnba->', self.mins[self.t0], self.Gamma_pi_src, self.perambs[self.t0][self.t0])
        l1 = jp.einsum('nm,ab,mnba->', self.mins[t], self.Gamma_pi_snk, self.perambs[t][t])
        c1 = l0 * l1

        return c0 - 2*c1 
    
    


    




