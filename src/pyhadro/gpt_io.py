import sys, os, struct, binascii
from fnmatch import fnmatch
import numpy as np
import itertools as it

# The following code is inspired by code from https://github.com/lehner/jks/blob/main/jks/corrIO.py


class gdict(dict):
    def glob(self, match):
        return dict([(k,v) for k, v in self.items() if fnmatch(k, match)])

    def glob_mean(self, match):
        data = self.glob(match)
        return 1/len(data) * sum([v for v in data.values()])

    def glob_sum(self, match):
        data = self.glob(match)
        return sum([v for v in data.values()])


def get_data(fn):
    tags = gdict()

    f = open(fn, "r+b")
    while True:
        rd = f.read(4)
        if len(rd) == 0:
            break
        ntag = struct.unpack("i", rd)[0]
        tag = f.read(ntag).decode("utf-8").strip()
        (crc32, ln) = struct.unpack("II", f.read(4 * 2))

        data = f.read(16 * ln)
        crc32comp = binascii.crc32(data) & 0xFFFFFFFF

        if crc32 != crc32comp:
            raise ValueError(f"Checksum error in {tag}")
        
        nt = tag[0:-1]
        tags[nt] = np.frombuffer(data, dtype=np.complex128, count=ln)
        
    f.close()
    return tags


def get_types(tags, idx):
    return sorted(list(set([tag.split('/')[idx] for tag in tags])))


def find_all(tag, c):
    idx = tag.find(c)
    res = [idx]
    while idx != -1:
        idx = tag.find(c, idx+1)
        res.append(idx)
    return res[:-1]

def find_spliter(tag):
    i0s = find_all(tag, '-')
    i1s = find_all(tag, '.')
    spliter_pos = []
    for i0 in i0s:
        if i0-1 not in i1s:
            spliter_pos.append(i0)
    if len(spliter_pos) != 1:
        raise IndexError(f'Multiple spliter in {tag}')
    return spliter_pos[0]

def get_operator(tag):
    sidx = find_spliter(tag)
    return tag[:sidx], tag[sidx+1:]



class writer:
    def __init__(self, fn):
        self.f = open(fn, "w+b")

    def write(self, t, cc):
        tag_data = (t + "\0").encode("utf-8")
        self.f.write(struct.pack("i", len(tag_data)))
        self.f.write(tag_data)

        ln = len(cc)
        ccr = [fff for sublist in ((c.real, c.imag) for c in cc) for fff in sublist]
        bindata = struct.pack("d" * 2 * ln, *ccr)
        crc32comp = binascii.crc32(bindata) & 0xFFFFFFFF
        self.f.write(struct.pack("II", crc32comp, ln))
        self.f.write(bindata)
        self.f.flush()

    def write_spin(self, t, c):
        for mu, nu in it.product(range(4), repeat=2):
            self.write(f"{t}/s_{mu}_{nu}", c[:,mu,nu])


    def close(self):
        self.f.close()
