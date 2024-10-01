function  [p, xhat, sigsq, xhatold, sigsqold] ... 
		= forwardfilter(I, sigE, xguess, sigsqguess, mu);
%forwardfilter is a helper function that implements the forward recursive 
%filtering algorithm to estimate the learning state (hidden process) at 
%trial k as the Gaussian random variable with mean x{k|k} (xhat) and 
%SIG^2{k|k} (sigsq).  
% 
%variables:
%   xhatold      x{k|k-1}, one-step prediction (equation A.6)*
%   sigsqold     SIG^2{k|k-1}, one-step prediction variance (equation A.7)*
%   xhat         x{k|k}, posterior mode (equation A.8)* 
%   sigsq        SIG^2{k|k}, posterior variance (equation A.9)*
%   p            p{k|k}, observation model probability (equation 2.2)*
%   N            vector of number correct at each trial
%   Nmax         total number that could be correct at each trial
%   K            total number of trials
%   number_fail  saves the time steps if Newton's Method fails

K = size(I,2);
N = I(1,:);  
Nmax = I(2,:);

%Initial conditions: use values from previous iteration
xhat(1)   = xguess;    
sigsq(1)  = sigsqguess;
number_fail = [];

for k=2:K+1 
   %for each trial, compute estimates of the one-step prediction, the
   %posterior mode (using Newton's Method), and the posterior variance
   %(estimates from subject's POV)
   
   %Compute the one-step prediction estimate of mean and variance     
   xhatold(k)  = xhat(k-1);  
   sigsqold(k) = sigsq(k-1) + sigE^2;  

   %Use Newton's Method to compute the nonlinear posterior mode estimate                                    
   [xhat(k),flagfail] = newtonsolve(mu,  xhatold(k), sigsqold(k), N(k-1), Nmax(k-1));
   if flagfail>0 %if Newton's Method fails, number_fail saves the time step
      number_fail = [number_fail k];
   end
   
   %Compute the posterior variance estimate
   denom       = -1/sigsqold(k) - Nmax(k-1)*exp(mu)*exp(xhat(k))/(1+exp(mu)*exp(xhat(k)))^2;                                
   sigsq(k)    = -1/denom;

end

if isempty(number_fail)<1
   fprintf(2,'Newton convergence failed at times %d \n', number_fail)
end

%Compute the observation model probability estimate
p = exp(mu)*exp(xhat)./(1+exp(mu)*exp(xhat));

