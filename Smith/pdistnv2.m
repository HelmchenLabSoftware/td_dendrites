function [p05, p95, pmid, pmodnew, pcert] = pdistnv2(x, s, mu, background_prob);
%pdist is a helper function that calculates the confidence limits of the EM
%estimate of the learning state process.  For each trial, the function
%constructs the probability density for a correct response.  It then builds
%the cumulative density function from this and computes the p values of
%the confidence limits
%
%variables:
%   xx(ov)   EM estimate of learning state process
%   ss(ov)   EM estimate of learning state process variance
%   p05      the p value that gives the lower 5% confidence bound
%   p95      the p value that gives the upper 95% confidence bound
%   pmid     the p value that gives the 50% confidence bound
%   pmodnew    not computed


samps = [];
num_samps = 10000;
p05 = zeros(size(x));
p95 = zeros(size(x));
pmid = zeros(size(x));
pmodnew = zeros(size(x));
pcert  = zeros(size(x));

for ov = 1:size(x,2)
    
 xx = x(ov);
 ss = s(ov);
 samps = xx + sqrt(ss)*randn(num_samps ,1);
 pr_samps = exp(mu+samps)./(1+exp(mu+samps));
 
 order_pr_samps = sort(pr_samps);
 p05(ov) = order_pr_samps(fix(0.05*num_samps ));
 p95(ov) = order_pr_samps(fix(0.95*num_samps ));
 pmid(ov) = order_pr_samps(fix(0.5*num_samps ));
 pcert(ov) = length(find(pr_samps>background_prob))/num_samps;
end

