function [newsigsq] = em_bino(I, xnew, signewsq, A, startflag);
%em_bino is a helper function that computes sigma_eps squared (estimated 
%learning process variance).  
%
%variables:
%   xnew         x{k|K}, backward estimate of learning state
%   signewsq     SIG^2{k|K}, backward estimate of learning state variance  
%   A            A{k}
%   M            total number of backward estimates (K + 1)
%   covcalc      covariance estimate (equation A.13)*
%   term1        W{k|K}       (equation A.15)*
%   term2        W{k,k-1|K}   (equation A.14)*
%   term3        derived from W{1|K}     (applies equation A.15)* 
%   term4        W{K|K}     (applies equation A.15)*
%   newsigsq     SIG_EPSILON^2, estimate of learning state variance from EM (equation A.16)*

M           = size(xnew,2);  

xnewt      = xnew(3:M);
xnewtm1    = xnew(2:M-1);
signewsqt  = signewsq(3:M);
A          = A(2:end);



covcalc    = signewsqt.*A;

term1      = sum(xnewt.^2) + sum(signewsqt);
term2      = sum(covcalc) + sum(xnewt.*xnewtm1);

if startflag == 1
 term3      = 1.5*xnew(2)*xnew(2) + 2.0*signewsq(2); 
 term4      = xnew(end)^2 + signewsq(end);
elseif( startflag == 0)
 term3      = 2*xnew(2)*xnew(2) + 2*signewsq(2);
 term4      = xnew(end)^2 + signewsq(end);
elseif( startflag == 2)
 term3      = 1*xnew(2)*xnew(2) + 2*signewsq(2);
 term4      = xnew(end)^2 + signewsq(end);
 M = M-1;
end
newsigsq   = (2*(term1-term2)+term3-term4)/M;
