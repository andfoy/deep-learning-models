import os
import sys
import numpy as np

ORDER = 'C'

def flatten(A):
    A = np.ravel(A, order=ORDER)
    # A = A.flatten(1)
    return A.reshape(len(A), 1)

def init_params(n, d):
    r = np.sqrt(6)/np.sqrt(n+d+1)
    W = np.random.rand(n, d)*2*r - r
    return W

def init_weights(n, d, m):
    # weights = {}
    dim = 7*d**2 + 4*d*(n + 1) + m*(1 + d)
    W = np.zeros((dim, 1))
    start = 0
    end = d*n
    for c in ['i', 'f', 'c', 'o']:
        print "Wx%s: %d %d" % (c, start, end)
        # weights[c] = {}
        W[start:end] = flatten(init_params(d, n))
        start, end = end, end+d*d
        for p in ['h', 'c']:
            print "W%s%s: %d %d" % (p, c, start, end)
            add = True
            step = d*d
            if p == 'c':
                if c == 'c':
                    add = False
                    start, end = start-step, start 
                step = d*n
            if c == 'o':
                if p == 'c':
                    step = m*d
            if add:
                W[start:end] = flatten(init_params(d, d))
            start, end = end, end+step
    print "Why: %d %d" % (start, end)
    W[start:end] = flatten(init_params(m, d))
    return W

def unroll(V, d, n, partial=False):
    wx = V[0:d*n].reshape(d, n, order=ORDER)
    wh = V[d*n:d*n+d*d].reshape(d, d, order=ORDER)
    wc = None
    if not partial:
        wc = V[d*n+d*d:].reshape(d, d, order=ORDER)
    return wx, wh, wc

def logistic(z):
    return 1.0/(1+np.exp(-z))

def cost_function(W, xt, yt, h_prev, c_prev):
    n = xt.shape[0]
    m = yt.shape[0]
    d = h_prev.shape[0]
    fact = d*n + 2*d**2
    
    I = W[0:fact]
    F = W[fact:2*fact]
    C = W[2*fact:2*fact+d*n+d**2]
    O = W[2*fact+d*n+d**2:3*fact+d*n+d**2]
    why = W[3*fact+d*n+d**2:3*fact+d*n+d**2+m*d]
    B = W[3*fact+d*n+d**2+m*d:]

    wxi, whi, wci = unroll(I, d, n)
    wxf, whf, wcf = unroll(F, d, n)
    wxc, whc, _ = unroll(C, d, n, True)
    wxo, who, wco = unroll(O, d, n)
    why = why.reshape(m, d, order=ORDER)
    bi = B[0:d]
    bf = B[d:2*d]
    bc = B[2*d:3*d]
    bo = B[3*d:4*d]
    by = B[4*d:]

    it = logistic(np.dot(wxi, xt) + np.dot(whi, h_prev) + np.dot(wci, c_prev) + bi)
    ft = logistic(np.dot(wxf, xt) + np.dot(whf, h_prev) + np.dot(wcf, c_prev) + bf)
    ch = np.dot(wxc, xt) + np.dot(whc, h_prev) + bc
    ct = ft*c_prev + it*np.tanh(ch)
    ot = logistic(np.dot(wxo, xt) + np.dot(who, h_prev) + np.dot(wco, ct) + bo)
    ht = ot*np.tanh(ct)
    y_temp = np.dot(why, ht) + by
    exp_term = np.exp(y_temp)
    yh = exp_term/np.sum(exp_term)

    cost = -np.sum(yt*np.log(yh))

    d_yh = yh - yt # mx1
    d_why = d_yh*ht.T # mx1*1xd => mxd 
    d_by = d_yh # mx1
    d_ht = np.dot(why.T, d_yh) # dxm * mx1 => dx1
    d_ot = np.tanh(ct)*d_ht # dx1*dx1 => dx1
    d_ct = np.dot(wco.T, ((ot*(1 - ot))*np.tanh(ct)*d_ht)) + (ot*(1 - np.tanh(ct)**2))*d_ht # dx1
    d_wxo = np.dot((ot*(1 - ot)*d_ot), xt.T) # dxn 
    d_who = np.dot((ot*(1 - ot)*d_ot), h_prev.T) # d<xd 
    d_wco = np.dot((ot*(1 - ot)*d_ot), ct.T) # dxd
    d_bo = ot*(1 - ot)*d_ot # dx1
    d_wxc = np.dot((it*(1 - np.tanh(ch)**2)*d_ct), xt.T) # dxn 
    d_whc = np.dot((it*(1 - np.tanh(ch)**2)*d_ct), h_prev.T) # dxd
    d_bc = it*(1 - np.tanh(ch)**2)*d_ct # dx1
    d_ft = c_prev*d_ct # dx1
    d_it = np.tanh(ch)*d_ct # dx1
    d_wxf = np.dot((ft*(1 - ft)*d_ft), xt.T) # dxn 
    d_whf = np.dot((ft*(1 - ft)*d_ft), h_prev.T) # dxd
    d_wcf = np.dot((ft*(1 - ft)*d_ft), c_prev.T) # dxd
    d_bf = ft*(1 - ft)*d_ft # dx1
    d_wxi = np.dot((it*(1 - it)*d_it), xt.T) # dx1
    d_whi = np.dot((it*(1 - it)*d_it), h_prev.T) # dxd
    d_wci = np.dot((it*(1 - it)*d_it), c_prev.T) # dxd
    d_bi = it*(1 - it)*d_it

    params = [d_wxi, d_whi, d_wci, d_wxf, 
              d_whf, d_wcf, d_wxc, d_whc, 
              d_wxo, d_who, d_wco, d_why,
              d_bi, d_bf, d_bc, d_bo, d_by]
    rolled_params = map(flatten, params)
    # for v in params:
        # print v.shape

    grad = np.vstack(rolled_params)

    return cost, grad, ht, ct