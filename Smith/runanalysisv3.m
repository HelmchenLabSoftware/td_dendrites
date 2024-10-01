%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Script to run the analysis of the Individual subject learning curve using
% binomial Expectation Maximization
% Version 1.3
% 
% Anne Smith, April 28th, 2003
% 
% updated Anne Smith, August 10th, 2004  - adjusted initial variance for UpdaterFlag=2 case
% 
% updated Leo Walton, July 27th, 2006 - added comments, changed variable
% names

% updated Anne Smith, 2015 - improved Newtons and pdistn
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Given a set of behavioral experiment trial data, this program estimates an
% unobservable learning state process, defined as a random walk, by tracking
% the evolution of the observable trial data.  It uses a state-space model
% and Expectation-Maximization algorithm to estimate a learning curve that
% characterizes the dynamics of the learning process as a function of trial
% number.  For a thorough description of this method of analysis, see: 
%       Smith et al. (2004)  Dynamic Analysis of Learning in Behavioral
%           Experiments. The Journal of Neuroscience. 24(2):447-461.
% Throughout this code, references to equations from this journal article 
% will be indicated with a "*":  
% 
% Input: 
%        Responses      (1 x N vector) number of correct responses at
%                                                 each trial -- required
%
%        MaxResponse    (single value)  total number that could be correct 
%                                       at each trial.  1 for binary data.
%                                       -- required
%
%        BackgroundProb     (single value) probabilty of correct by chance
%                                       -- required
%
%        SigE       (single value) SIG_EPSILON, sqrt(variance) of learning
%                               state process -- optional. Default 0.005
% 
%        UpdaterFlag    (single value)
%                        0-to fix initial condition (more likely to give a result)
%                        1-to estimate initial condition 
%                        2-to remove xo from likelihood - this means that the latent
% 					     learning process is not started at 0 and allows for alot
% 					     of bias 
%                        -- optional. Default 2
                   
% functions variables
%        x, s   (vectors)         (hidden) learning state process and its variance (forward estimate)
%        xnew, signewsq (vectors) (hidden) learning state process and its variance (backward estimate)
%        newsigsq                 estimate of learning state process variance from EM 
%        A (vector)               A(k), (equation A.11)*                       
%        p (vectors)              mode of prob correct estimate 
%        p05,p95,pmid, pmode,pmatrix (vectors)     conf limits of prob correct estimate
%
% helper functions
%       forwardfilter   solves the forward recursive filtering algorithm to
%                       estimate p, x, s, xold, and sold
%       backwardfilter  solves the backward filter smoothing algorithm to 
%                       estimate xnew, signewsq, and A
%       embino         solves the maximization step of the EM algorithm to
%                       estimate newsigsq
%       pdistn          calculates the confidence limits (p05, p95, pmid, 
%                       pmode, pmatrix) of the EM estimate of the learning
%                       state process 
% 

function [t_learn, t_expert, pmid, p05] = runanalysisv3(Responses,MaxResponse, BackgroundProb, SigE, UpdaterFlag)

if nargin<4
    SigE = 0.005; %default variance of learning state process is sqrt(0.005)
end
if nargin<5
    UpdaterFlag = 2;  %default allows bias 
end
        
% check data format.  Reshape dataset if needed
[a,b] = size(Responses);
if a>b
    Responses = Responses';
end

I = [Responses; MaxResponse*ones(1,length(Responses))];

SigsqGuess  = SigE^2;

%set the value of mu from the chance of correct
 mu = log(BackgroundProb/(1-BackgroundProb)); 

%convergence criterion for SIG_EPSILON^2
 CvgceCrit = 1e-8;

%----------------------------------------------------------------------------------

xguess         = 0;  
NumberSteps  = 4000;

%loop through EM algorithm: forward filter, backward filter, then
%M-step
for i=1:NumberSteps
   
   %Compute the forward (filter algorithm) estimates of the learning state
   %and its variance: x{k|k} and sigsq{k|k}
   [p, x, s, xold, sold] = forwardfilter(I, SigE, xguess, SigsqGuess, mu);   
   
   %Compute the backward (smoothing algorithm) estimates of the learning 
   %state and its variance: x{k|K} and sigsq{k|K}
   [xnew, signewsq, A]   = backwardfilter(x, xold, s, sold); 

   if (UpdaterFlag == 1)
        xnew(1) = 0.5*xnew(2);   %updates the initial value of the latent process
        signewsq(1) = SigE^2;
   elseif(UpdaterFlag == 0)
        xnew(1) = 0;             %fixes initial value (no bias at all)
        signewsq(1) = SigE^2;
   elseif(UpdaterFlag == 2)
        xnew(1) = xnew(2);       %x(0) = x(1) means no prior chance probability
        signewsq(1) = signewsq(2);
   end
   
   %Compute the EM estimate of the learning state process variance
   [newsigsq(i)]         = em_bino(I, xnew, signewsq, A, UpdaterFlag);
   
   xnew1save(i) = xnew(1);
   
   %check for convergence
   if(i>1)
      a1 = abs(newsigsq(i) - newsigsq(i-1));
	  a2 = abs(xnew1save(i) -xnew1save(i-1));
      if( a1 < CvgceCrit & a2 < CvgceCrit & UpdaterFlag >= 1)
          fprintf(2, 'EM estimates of learning state process variance and start point converged after %d steps   \n',  i)
          break
      elseif ( a1 < CvgceCrit & UpdaterFlag == 0)
          fprintf(2, 'EM estimate of learning state process variance converged after %d steps   \n',  i)
          break
      end
   end
 
   SigE   = sqrt(newsigsq(i));
   xguess = xnew(1);
   SigsqGuess = signewsq(1);

end
   
if(i == NumberSteps)
     fprintf(2,'failed to converge after %d steps; convergence criterion was %f \n', i, CvgceCrit)
end

%-----------------------------------------------------------------------------------
%integrate and do change of variables to get confidence limits


[p05, p95, pmid, pmode, pmatrix] = pdistnv2(xnew, signewsq, mu, BackgroundProb);

pcert = pmatrix;
%-------------------------------------------------------------------------------------
%find the last point where the 90 interval crosses chance
%for the backward filter (cback)

 cback = find(p05 < BackgroundProb);
 if(~isempty(cback))
      if(cback(end) < size(I,2) )
           t_expert = cback(end);
      else
           t_expert = NaN;
      end
 else
      t_expert = NaN;
 end
 cback = find(p05(52:end) > BackgroundProb);
 if(~isempty(cback))
      if(cback(1) < size(I,2) )
           t_learn = 50 + cback(1);
      else
           t_learn = NaN;
      end
 else
      t_learn = NaN;
 end
