function [cost, grad, ht, ct] = LSTMCostFunction(W, xt, yt, h_prev, c_prev)
    n = size(xt, 1); %x_t.shape[0]
    m = size(yt, 1); %y_t.shape[0]
    d = size(h_prev, 1); %h_p.shape[0]
    fact = d*n + 2*d**2;

    I = W(1:fact);
    F = W(fact+1:2*fact);
    C = W(2*fact+1:2*fact+d*n+d**2);
    O = W(2*fact+d*n+d**2+1:3*fact+d*n+d**2);
    why = W(3*fact+d*n+d**2+1:3*fact+d*n+d**2+m*d);
    B = W(3*fact+d*n+d**2+m*d+1:end);

    wxi, whi, wci = unroll(I, d, n, false);
    wxf, whf, wcf = unroll(F, d, n, false);
    wxc, whc, _ = unroll(C, d, n, true)
    wxo, who, wco = unroll(O, d, n, false);

    why = reshape(why, m, d);
    bi = B(1:d);
    bf = B(d+1:2*d);
    bc = B(2*d+1:3*d);
    bo = B(3*d+1:4*d);
    by = B(4*d+1:end);

    it = logistic(wxi*xt + whi*h_prev + wci*c_prev + bi);
    ft = logistic(wxf*xt + whf*h_prev + wcf*c_prev + bf);
    ch = wxc*xt + whc*h_prev + bc;
    ct = ft.*cp + it.*np.tanh(ch);
    ot = logistic(wxo*xt + who*h_prev + wco*c_prev + bo);
    ht = ot.*tanh(ct);
    y_temp = why*h_t + by
    exp_term = exp(y_temp)
    yh = exp_term./np.sum(exp_term)

    cost = -np.sum(yt*np.log(yh));

    d_yh = yh - yt; % mx1
    d_why = d_yh*ht'; % mx1*1xd => mxd 
    d_by = d_yh; % mx1
    d_ht = why'*d_yh; % dxm * mx1 => dx1
    d_ot = tanh(ct).*d_ht; % dx1*dx1 => dx1
    d_ct = ot.*(1 - tanh(ct).^2).*d_ht; % dx1
    d_wxo = (ot.*(1 - ot).*d_ot)*xt'; % dxn 
    d_who = (ot.*(1 - ot).*d_ot)*h_prev'; % dxd 
    d_wco = (ot.*(1 - ot).*d_ot)*ct'; % dxd
    d_bo = ot.*(1 - ot).*d_ot; % dx1
    d_wxc = (it.*(1 - tanh(ch).^2).*d_ct)*xt'; % dxn 
    d_whc = (it.*(1 - tanh(ch).^2).*d_ct)*h_prev'; % dxd
    d_bc = it.*(1 - tanh(ch).^2)*d_ct; % dx1
    d_ft = c_prev.*d_ct; % dx1
    d_it = tanh(ch); % dx1
    d_wxf = (ft.*(1 - ft).*d_ft)*x_t'; % dxn 
    d_whf = (ft.*(1 - ft).*d_ft)*h_prev'; % dxd
    d_wcf = (ft.*(1 - ft).*d_ft)*c_prev'; % dxd
    d_bf = ft.*(1 - ft).*d_ft; % dx1
    d_wxi = (it.*(1 - it).*d_it)*x_t'; % dx1
    d_whi = (it.*(1 - it).*d_it)*h_prev'; % dxd
    d_wci = (it.*(1 - it).*d_it)*c_prev'; % dxd
    d_bi = it.*(1 - it).*d_it;

    grad =   [d_wxi(:); d_whi(:); d_wci(:); d_wxf(:); 
              d_whf(:); d_wcf(:); d_wxc(:); d_whc(:); 
              d_wxo(:); d_who(:); d_wco(:); d_why(:);
              d_bi(:); d_bf(:); d_bc(:); d_bo(:); d_by(:)]
    
end


function [wx, wh, wc] = unroll(W, d, n, partial)
    wx = reshape(W(0:d*n), d, n);
    wh = reshape(W(d*n+1:d*n+d*d), d, d);
    wc = []
    if ~partial
        wc = reshape(W(d*n+d*d+1:end), d, d);
    end
end

function g = logistic(z)
    g = 1./(1 + exp(-z));
end