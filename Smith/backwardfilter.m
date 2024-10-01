function [xnew, signewsq, A] = backwardfilter(x, xold, sigsq, sigsqold);
%backwardfilter is a helper function that implements the backward filter
%smoothing algorithm to estimate the learning state at trial k, given all
%the data, as the Gaussian random variable with mean x{k|K} (xnew) and
%SIG^2{k|K} (signewsq).  
% 
%variables:
%   x            x{k|k}, posterior mode 
%   xold         x{k|k-1}, one-step prediction
%   sigsq        SIG^2{k|k}, posterior variance
%   sigsqold     SIG^2{k|k-1}, one-step prediction variance
%   A(i)         A{k}, (equation A.11)*
%   xnew         x{k|K}, backward estimate of learning state given all the data (equation A.10)*
%   signewsq     SIG^2{k|K}, backward estimate of learning state variance (equation A.12)*
%   T            total number of posterior mode estimates (K + 1)

T = size(x,2);

%Initial conditions: use values of posterior mode and posterior variance
xnew(T)     = x(T);
signewsq(T) = sigsq(T);

for i = T-1 :-1: 2
 %for each posterior mode prediction, compute new estimates given all of
 %the data from the experiment (estimates from ideal observer)
   A(i)        = sigsq(i)/sigsqold(i+1);
   xnew(i)     = x(i) + A(i)*(xnew(i+1) - xold(i+1));
   signewsq(i) = sigsq(i) + A(i)*A(i)*(signewsq(i+1)-sigsqold(i+1));
end


