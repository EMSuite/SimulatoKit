"""The occupancy draw of SimulatoKit's OccupancyDraw, ported to Python as an independent check of the Swift code.

SplitMix64 seeds xoshiro256** (GaussianGenerator), FrozenPhonons.configurationSeed, one uniform per shared site, the
first atom whose cumulative occupancy exceeds it. `python3 tools/pins/occupancy/draw.py` prints the masks that
OccupancyDrawTests.drawsAreReproducibleFromSeedAndConfiguration pins. Needs only the standard library.
"""
M=(1<<64)-1
def mix(v):
    z=(v+0x9E3779B97F4A7C15)&M
    z=((z^(z>>30))*0xBF58476D1CE4E5B9)&M
    z=((z^(z>>27))*0x94D049BB133111EB)&M
    return z^(z>>31)
def cseed(seed,c): return mix((mix(seed)+c)&M)
class G:
    def __init__(s,seed):
        x=[seed]
        def sm():
            x[0]=(x[0]+0x9E3779B97F4A7C15)&M
            z=x[0]
            z=((z^(z>>30))*0xBF58476D1CE4E5B9)&M
            z=((z^(z>>27))*0x94D049BB133111EB)&M
            return z^(z>>31)
        s.s=[sm(),sm(),sm(),sm()]
    def next(s):
        rotl=lambda v,k:((v<<k)|(v>>(64-k)))&M
        a=s.s
        r=(rotl((a[1]*5)&M,7)*9)&M
        t=(a[1]<<17)&M
        a[2]^=a[0];a[3]^=a[1];a[1]^=a[2];a[0]^=a[3];a[2]^=t;a[3]=rotl(a[3],45)
        return r
SALT=0x6F63637570616E63
def kept(sites,occ,n,seed,c):
    keep=[True]*n
    g=G(cseed(seed^SALT,c))
    for site,o in zip(sites,occ):
        u=(g.next()>>11)*2.0**-53
        cum=0;ch=None
        for i,oi in zip(site,o):
            cum+=oi
            if ch is None and u<cum: ch=i
            keep[i]=False
        if ch is not None: keep[ch]=True
    return keep


if __name__ == "__main__":
    # The test structure: O whole, Si 0.6 alone, Fe 0.3 and Ni 0.5 on one site.
    for c in range(6):
        print(c, [bool(k) for k in kept([[1], [2, 3]], [[0.6], [0.3, 0.5]], 4, 2026, c)])
