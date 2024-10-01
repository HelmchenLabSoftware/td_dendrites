function [x, timefail] = newtonsolve(mu,  xold, sigoldsq, N, Nmax);
%newtonsolve is a helper function that implements Newton's Method in order 
%to recursively estimate the posterior mode (x).  Once the subsequent estimates
%sufficiently converge, the function returns the last estimate.  If, having
%never met this convergence condition, the function goes through all of the
%recursions, then a special flag (timefail) - indicating the convergence 
%failure - is returned along with the last posterior mode estimate.
%
%variables: 
%   g(i)         derivative of the learning state process
%   gprime(i)    derivative of g
%   it(i)        estimate of posterior mode (A.8)*
%   x            x{k|k}, the posterior mode

it(1) = xold + sigoldsq*(N - Nmax*exp(mu)*exp(xold)/(1 ... 
                                  + exp(mu)*exp(xold)));
                 
for i = 1:100   
   g(i)     = xold + sigoldsq*(N - Nmax*exp(mu)*exp(it(i))/...
                              (1+exp(mu)*exp(it(i)))) - it(i);
   gprime(i)= -Nmax*sigoldsq*exp(mu)*exp(it(i))/(1+exp(mu)*exp(it(i)))^2 - 1;
   it(i+1)  = it(i) - g(i)/gprime(i);

   x        = it(i+1);
   %[i x it(i)]
   if abs(x-it(i))<1e-10  
      timefail = 0; 
      return
   end
end

%this tries a new initial condition if first Newtons doesn't work
if(i==100) 
   % fprintf(2, 'failed to converge trying an ic  of -1 %f %f %f \n',it(1),it(i), it(i+1));
    it(1) = -1;
    for i = 1:100   
       g(i)     = xold + sigoldsq*(N - Nmax*exp(mu)*exp(it(i))/...
                                  (1+exp(mu)*exp(it(i)))) - it(i);
       gprime(i)= -Nmax*sigoldsq*exp(mu)*exp(it(i))/(1+exp(mu)*exp(it(i)))^2 - 1;
       it(i+1)  = it(i) - g(i)/gprime(i);

       x        = it(i+1);
       %[i x it(i)]
       if abs(x-it(i))<1e-10    
          timefail = 0; 
          return
       end
     end
end


%this tries a new initial condition if second Newtons doesn't work
if(i==100) 
   % fprintf(2, 'failed to converge trying an ic of +1 %f %f %f \n',it(1),it(i), it(i+1));
    it(1) = 1;
    for i = 1:100   
       g(i)     = xold + sigoldsq*(N - Nmax*exp(mu)*exp(it(i))/...
                                  (1+exp(mu)*exp(it(i)))) - it(i);
       gprime(i)= -Nmax*sigoldsq*exp(mu)*exp(it(i))/(1+exp(mu)*exp(it(i)))^2 - 1;
       it(i+1)  = it(i) - g(i)/gprime(i);

       x        = it(i+1);
       %[i x it(i)]
       if abs(x-it(i))<1e-10    
          timefail = 0; 
          return
       end
    end
     
   fprintf(2, 'failed to converge after 3 attempts %f %f %f \n',it(1),it(i), it(i+1));
   timefail = 1;
   return 
end
