function [H, pValue, Lambda, Orders, CI] = chaostest(X, ActiveFN, maxOrders, alpha, flag)
%CHAOSTEST performs the test for Chaos to test the positivity of the
%   dominant Lyapunov Exponent LAMBDA.
%
%   The test hypothesis are:
%   Null hypothesis: LAMBDA >= 0 which indicates the presence of chaos.
%   Alternative hypothesis: LAMBDA < 0 indicates no chaos.
%   This is a one tailed test.
%
%   [H, pValue, LAMBDA, Orders, CI] = ...
%           CHAOSTEST(Series, ActiveFN, maxOrders, ALPHA, FLAG)
%
% Inputs:
%   Series - a vector of observation to test.
%
% Optional inputs:
%   ActiveFN - String containing the activation function to use in the
%     neural net estimation. ActiveFN can be the 'LOGISTIC' function
%     f(u) = 1 / (1 + exp(- u)), domain = [0, 1], or 'TANH' function
%     f(u) = tanh(u), domain = [-1, 1], or 'SIGMOID' function
%     f(u) = u * (1 + |u/2|) / (2 + |u| + u^2 / 2), domain = [-1, 1].
%     Default = 'TANH'.
%
%   maxOrders - the maximum orders that the chaos function defined
%     in CHAOSFN can take. This must be a vector containing 3 elements.
%     maxOrders = [L, m, q].
%     Increasing the model's orders can slow down calculations.
%     Default = [5, 6, 5].
%
%   ALPHA - The significance level for the test (default = 0.05)
%
%   FLAG  - String specifying the method to carry out the test, by
%     varying the triplet (L, m, q) {'VARY' or anything else} or by
%     fixing them {'FIX'}. Default = {'VARY'}.
%
% Outputs:
%   H = 0 => Do not reject the null hypothesis of Chaos at significance
%            level ALPHA.
%   H = 1 => Reject the null hypothesis of Chaos at significance level
%            ALPHA.
%
%   pValue - is the p-value, or the probability of observing the given
%     result by chance given that the null hypothesis is true. Small
%     values of pValue cast doubt on the validity of the null hypothesis
%     of Chaos.
%
%  LAMBDA - The dominant Lyapunov Exponent.
%     If LAMBDA is positive, this indicates the presence of Chaos.
%     If LAMBDA is negative, this indicates the absence of Chaos.
%
%   Orders - gives the triplet (L, m, q) that maximizes the Lyapunov
%     exponent computed from all L*m*q estimations.
%
%   CI - Confidence interval for LAMBDA at level ALPHA.
%
%   The algorithm uses the Jacobian method in contrast to the direct
%   method, it needs the optimazition and the statistics toolboxes.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                Copyright (c) 17 March 2009 by Ahmed BenSaïda           %
%                 Department of Finance, IHEC Sousse - Tunisia           %
%                       Email: ahmedbensaida@yahoo.com                   %
%                   $ Revision 5.0 $ Date: 2 April 2015 $                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% CHAOSTEST License Notice
% ----------------------------------------------------------
% 
% Copyright (c) 2009, Ahmed BenSaïda
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met:
% 
%     * Redistributions of source code must retain the above copyright
%       notice, this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above copyright
%       notice, this list of conditions and the following disclaimer in
%       the documentation and/or other materials provided with the distribution
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.
%

%
% References: 
%             BenSaïda, A. and Litimi, H. (2013), "High level chaos in the
%               exchange and index markets", Chaos, Solitons & Fractals 54:
%               90-95.
%             BenSaïda, A. (2014), "Noisy chaos in intraday financial
%               data: Evidence from the American index", Applied
%               Mathematics and Computation 226: 258-265.
%

% Check if the Optimization and Statistics toolboxes are installed.
% 1 if installed, 0 if not.
existToolbox = license('test','optimization_toolbox') && ...
    license('test','statistics_toolbox');

if ~existToolbox
    error('The ''CHAOSTEST'' function needs both Optimization and Statistics toolboxes.')
end

% Set initial conditions.

if (nargin >= 1) && (~isempty(X))

   if numel(X) == length(X)   % Check for a vector.
      X  =  X(:);             % Convert to a column vector.
   else
      error(' Observation ''Series'' must be a vector.');
   end

   % Remove any NaN (i.e. missing values) observation from 'Series'. 
   
    X(isnan(X)) =   [];
         
else
    error(' Must enter at least observation vector ''Series''.');
end

%
% Specify the activation function to use in CHAOSFN and ensure it to be a
% string. Set default if necessary.
%

if nargin >= 2 && ~isempty(ActiveFN)
    if ~ischar(ActiveFN)
        error(' Activation function ''ActiveFN'' must be a string.')
    end

    % Specify the activation function to use:
    % ActiveFN = {'logistic', 'tanh', 'funfit'}.
    
    if ~any(strcmpi(ActiveFN , {'tanh' , 'logistic' , 'sigmoid'}))
        error('Activaton function ''ActiveFN'' must be TANH, LOGISTIC, or SIGMOID');
    end
    
else
    ActiveFN    =   'tanh';
end

%
% Ensure the maximum orders maxL, maxM and maxQ are positive integers, and
% set default if necessary.
%

if (nargin >= 3) && ~isempty(maxOrders)
    if numel(maxOrders) ~= 3 || any((maxOrders - round(maxOrders) ~= 0)) || ...
            any((maxOrders <= 0))
      error(' Maximum orders ''maxOrders'' must a vector of 3 positive integers.');
    end
else
    maxOrders = [5, 6, 5];
end

maxL = maxOrders(1);
maxM = maxOrders(2);
maxQ = maxOrders(3);

% Check for a minimum size of vector X.
if length(X) <= maxL * maxM
    error(' Observation ''Series'' must have at least %d obeservations.',maxL*maxM+1)
end

%
% Ensure the significance level, ALPHA, is a 
% scalar, and set default if necessary.
%

if (nargin >= 4) && ~isempty(alpha)
   if numel(alpha) > 1
      error(' Significance level ''Alpha'' must be a scalar.');
   end
   if (alpha <= 0 || alpha >= 1)
      error(' Significance level ''Alpha'' must be between 0 and 1.'); 
   end
else
   alpha  =  0.05;
end

%
% Check the method to carry out the test, by varying (L, m, q) or by fixing
% them, and set default if necessary. The method must be a string.
%

if nargin >= 5 && ~isempty(flag)
    if ~ischar(flag)
        error(' Regressions type ''FLAG'' must be a string.')
    end    
else
    flag  =   'VARY';
end

if strcmpi(flag, 'FIX')
    StartL  =   maxL;
    StartM  =   maxM;
    StartQ  =   maxQ;
else
    StartL  =   1;
    StartM  =   1;
    StartQ  =   1;
end

% Initialize the Lyapunov Exponent to a small number.
Lambda =   - Inf;

%
% Create the structure OPTIONS to use for nonlinear least square function
% LSQNONLIN (included in the optimization toolbox).
%

Options =   optimset('lsqnonlin');
Options =   optimset(Options, 'Display', 'off');

% Use the user supplied analytical Jacobian in CHAOSFN.
Options =   optimset(Options, 'Jacobian', 'on');

% Initialize the starting value for the coefficient THETA.
StartValue  =   0.5;

for L = StartL:maxL
    for m = StartM:maxM
        for q = StartQ:maxQ
            
            A = StartValue * ones(q, m); % Contains the coefficients gamma(i,j).
            B = StartValue * ones(q, 1); % Contains the coefficients gamma(0,j).
            C = StartValue * ones(1, q); % Contains the coefficients beta(j).
            D = StartValue; % Constant.
            %E = StartValue * ones(1, m); % Linear coefficients for lagged X.
            
            Theta0 = [reshape(A', 1, q*m), B', C, D];
            %Theta0 = [reshape(A', 1, q*m), B', C, D, E];
            
            Theta  = lsqnonlin(@chaosfn, Theta0, [], [], Options, X, L, m, q, ActiveFN);
        
            T      =  length(X) - m*L; % New sample size.

%
% Use an inner function JACOBMATX to compute the Jacobian needed
% to compute the Lyapunov Exponent LAMBDA. This jacobian is relative
% to X and not to parameter THETA. N.B: the jacobian given by
% LSQNONLIN is relative to parameter THETA.
%

            JacobianMatrix   =   jacobmatx(Theta, X, L, m, q, ActiveFN);

%
% We distinguish between the "sample size" T used for estimating
% the Jacobian, and the "block length" M which is the number of
% evaluation points used for estimating LAMBDA (M <= T).
%

            h   =   round(T ^ (1/3)); % Number of blocks or step,
                                      % in this case M = T / h = T ^ (2/3).

% We can choose the full sample by setting h = 1. When h > 1,
% the obtained exponent is the Local Lyapunov Exponent. The block length M
% is T / h.

            newIndex    =   (h:h:T);

            M   =   length(newIndex); % Equally spaced subsample's size.

%
% Compute the dominant Lyapunov Exponent LAMBDA. The dominant exponent
% corresponds to the maximum eigenvalue of Tn(M)'*Tn(M). The following
% procedure uses a sample estimation of the Lyapunov exponent as
% described by Whang Y. and Linton O. (1999).
%

            Tn  =   JacobianMatrix(:, :, newIndex);
            for t = 2:M
                Tn(:, :, t)  =   Tn(:, :, t) * Tn(:,:,t-1);
            end

%
% Compute the "QR" estimate of LAMBDA as suggested by Abarbanel et al.
% (1991) by multiplying Tn by a unit vector U0 to reduce the systematic
% positive bias in the formal estimate of LAMBDA. U0 is chosen at random
% with respect to uniform measure on the unit sphere.
%

% In practice: U0 = [1; 0; 0; ...; 0].

            U0    =   [1; zeros(m-1, 1)];
            
% Or choose the U0 at random in the interval [0, 1] and ensure that the sum
% is unity. But this method gives differetn lambda each time we conduct the
% test on the same data, so we avoid it.

            %U0 = rand(m,1);
            %U0 = U0 / sum(U0);
            
            % To avoid 'Input to EIG must not contain NaN or Inf' replace
            % infinite values with REALMAX.
            
            v   =   eig(min(realmax, (Tn(:, :, M) * U0)' * (Tn(:, :, M) * U0)));
                        
            % LAMBDA is the largest Lyapunov exponent for the specified
            % orders. To avoid Log of zero, when max(v) is zero replace
            % it with REALMIN.
            
            Lambda1  =  1/(2*M) * log(max([v ; realmin])); 

%
% Choose the largest Lyapunov exponent from all regressions carried by
% varying L, m, and q. So we have L*m*q regressions in total, the chosen
% LAMBDA is the largest of all. The triplet (L, m, q) can be seen as
% the degree of complexity of a chaotic map, if a process is not chaotic,
% it will reject the null hypothesis for all orders (L, m, q), else, it
% will accept the null for some orders where the chaotic map get more
% complex.
%

            if Lambda1 >= Lambda
                
                Lambda  = Lambda1;
                Orders  =   [L, m, q];
%
% Compute the asymptotic variance of the Lyapunov Exponent for noisy
% systems as computed by Shintani M. and Linton O. (2004).
%

                Eta     =   zeros(M, 1);
                
                % Replace infinite values Inf with realmax, and -Inf with
                % realmin to avoid errors when computing the eigenvalues.
                Zeta    =   Tn(:, :, 1)' * Tn(:, :, 1);
                Zeta    =   min(realmax, Zeta);
                Zeta    =   max(realmin, Zeta);
                
                Eta(1)  =   0.5 * log(max(realmin, max(eig(Zeta)))) - Lambda;

% Ensure that the obtained eigenvalue is not zero, if so, replace it with
% a positive small number REALMIN to avoid 'log of zero' and 'divide by zero'.

                for t = 2:M
                    Zeta1   = Tn(:, :, t)' * Tn(:, :, t);
                    Zeta1   = min(realmax, Zeta1);
                    Zeta1   = max(realmin, Zeta1);
                    
                    Zeta2   = Tn(:, :, t-1)' * Tn(:, :, t-1);
                    Zeta2   = min(realmax, Zeta2);
                    Zeta2   = max(realmin, Zeta2);
                    
                    Eta(t)  =   0.5 * log(max(realmin, max(eig(Zeta1)))./ ...
                        max(realmin, max(eig(Zeta2)))) - Lambda;
                end

                gamm    =   zeros(M, 1);                
                for i = 1:M
                    gamm(i) =   1/M * Eta(M-i+1:M)' * Eta(1:i);
                end

                gamm   =   [gamm(1:end-1); flipud(gamm)];

%
% Compute the Lag truncation parameter. This is the optimal Lag as defined
% by Andrews (1991) p-830 for the Quadratic Spectral Kernel. The coefficient
% is for the QS kernel, Andrews (1991) p-830. The coefficient a(2) as defined
% by Andrews converges to 1: a(2) -> 1, its inverse too converges to 1:
% 1/a(2) -> 1, Andrews (1991) p-837. Because the calculation of a(2) is very
% difficult for neural network model, use the evident coeffiicent a(2) = 1.
% Limit(Sm, M, Inf) = Inf and Limit(Sm/M, M, Inf) = 0.
%

                Sm      =   1.3221 * (1 * M)^(1/5);
                
%
% Compute the kernel function needed to estimate the asymptotic variance of
% LAMBDA. The used kernel function is the Quadratic Spectral Kernel as
% defined by Andrews D. (1991) p-821.
%

                j           =   -M+1:M-1;
                KernelFN    =   ones(size(j));

                z           =   6 * pi * (j./Sm) / 5;

                % The limit(KernelFN(z), z, 0) = 1, so remove all z == 0.
                z(j==0)     =   [];

                KernelFN(j~=0)    =   (3 ./ z.^2) .* (sin(z) ./ z - cos(z));

%
% Compute the asymptotic variance of LAMBDA and prevent it from being
% negative or NaN. Next, compute the statistic of LAMBDA.
%

                varLambda       =   max(realmin, KernelFN * gamm);
                LambdaStatistic =   Lambda / sqrt(varLambda / M);
                
            end

        end
    end
end

%
% The statistic just found is for the one-sided test where the null
% hypothesis: LAMBDA >= 0 (presence of chaos) against the alternative
% LAMBDA < 0 (no chaos). This statistic is asymptotically normal.
% This test is for TAIL = 1 (right-tailed test).
%

pValue   =   normcdf(LambdaStatistic, 0, 1);

%
% Compute the critical value and the confidence interval CI for the true
% LAMBDA only when asked because NORMINV is computationally intensive.
%

if nargout >= 5
    crit     = norminv(1 - alpha, 0, 1) * sqrt(varLambda / M);
    CI       = [(Lambda - crit), Inf];
end

%
% To maintain consistency with existing Statistics Toolbox hypothesis
% tests, returning 'H = 0' implies that we 'Do not reject the null 
% hypothesis of chaotic dynamics at the significance level of alpha'
% and 'H = 1' implies that we 'Reject the null hypothesis of chaotic
% dynamics at significance level of alpha.'
%

H  = (alpha >= pValue);

%-------------------------------------------------------------------------%
%                       Helper function JACOBMATX                         %
%-------------------------------------------------------------------------%

function [y, J] = chaosfn(Theta, X, L, m, q, ActiveFN)
%CHAOSFN is the objective neural net function used to run
%   test for chaos and to compute the Lyapunov exponent.
%   Y = CHAOSFN(THETA, X, L, m, q, ActiveFN) where X is a
%   time series and THETA is a vector of parameters.

A   =   reshape(Theta(1:q*m), m, q)'; % size(A) = [q, m].
B   =   Theta(q*m+1:q*m+q)';          % size(B) = [q, 1].
C   =   Theta(q*m+q+1:q*m+q+q);       % size(C) = [1, q].
D   =   Theta(q*m+2*q+1);             % size(D) = 1.
%E   =   Theta(q*m+2*q+2:q*m+2*q+1+m); % size(E) = [1, m].

XLAG   =  lagmatrix(X, L:L:L*m);   % size(XLAG) = [T, m].
XLAG   =  XLAG(m*L+1:end, :);      % Remove all NaN observation.
T      =  length(X) - m*L;         % New sample size.

u   =   A * XLAG' + repmat(B,1,T); % size(u) = [q, T].

%
% Initialize the analytical Jacobian needed to speed up nonlinear least
% square optimization carried out by LSQNONLIN, this Jacobian is relative
% to parameter THETA: Jacobian = dF(X)/d(THETA). Not to confuse with the
% Jacobian needed to compute the Lyapunov exponent.
%

J   =   ones(length(Theta), T);

switch upper(ActiveFN)
    case 'LOGISTIC'
        % Use the logistic function as the sigmoid activation function.
        y   =   D + C * (1 ./ (1 + exp(- u))); % size(y) = [1, T].
        %y   =   E * XLAG' + D + C * (1 ./ (1 + exp(- u))); % size(y) = [1, T].        
        
        J(1:q*m,:)   =   repmat(C', m, T) .* (repmat(XLAG', q, 1) .* ...
                        repmat(exp(- u) ./ (1 + exp(- u)).^2, m, 1)); % A.

        J(q*m+1:q*m+q,:) =  repmat(C', 1, T) .* (exp(- u) ./ (1 + exp(- u)).^2); % B.
        J(q*m+q+1:q*m+q+q,:) =  1 ./ (1 + exp(- u)); % C.
        %J(q*m+2*q+2:q*m+2*q+1+m,:)  =   XLAG'; % E.
        
    case 'TANH'
        % Use a hyperbolic tangent as the activation function. This is a
        % two layered feed-forward neural network.
        y   =   D + C * tanh(u); % size(y) = [1, T].
        %y   =   E * XLAG' + D + C * tanh(u); % size(y) = [1, T].
        
        J(1:q*m,:)   =   repmat(C', m, T) .* (repmat(XLAG', q, 1) .* ...
                        repmat(sech(u).^2, m, 1)); % A.

        J(q*m+1:q*m+q,:) =  repmat(C', 1, T) .* (sech(u).^2); % B.
        J(q*m+q+1:q*m+q+q,:) =  tanh(u); % C.
        %J(q*m+2*q+2:q*m+2*q+1+m,:)  =   XLAG'; % E.
            
    case 'SIGMOID'
        % Use another type of the activation function.
        y   =   D + C * (u .* (1 + abs(u / 2)) ./ (2 + abs(u) + u.^2 / 2));
        %y   =   E * XLAG' + D + C * (u .* (1 + abs(u / 2)) ./ (2 + abs(u) + u.^2 / 2));
        
        J(1:q*m,:)   =   repmat(C', m, T) .* (repmat(XLAG', q, 1) .* ...
                repmat(8*(1 + abs(u)) ./ (3 + (1 + abs(u)).^2).^2, m, 1)); % A.
       
        J(q*m+1:q*m+q,:) =  repmat(C', 1, T) .* (8*(1 + abs(u)) ./ ...
                            (3 + (1 + abs(u)).^2).^2); % B.
        J(q*m+q+1:q*m+q+q,:) =  u .* (1 + abs(u/2)) ./ (2 + abs(u) + u.^2 / 2); % C.
        %J(q*m+2*q+2:q*m+2*q+1+m,:)  =   XLAG'; % E.
    
    otherwise
        error(' Unrecognized activation function!')
end

%
% The obtained Jacobian is for F(x), transform it to the Jacobian of
% (X - F(X)) so it can be used by LSQNONLIN to estimate THETA.
%

J   =   - J';

%
% CHAOSFN returns (X - F(X)), because the function LSQNONLIN
% minimizes the sum of squares of the objective function.
%

y   =   X(m*L+1:end)' - y;

%-------------------------------------------------------------------------%

function YLag = lagmatrix(Y,lags)
%LAGMATRIX Create matrix of lagged time series
%
% Description:
%
%   Create a matrix of lagged (time-shifted) series. Positive lags
%   correspond to delays; negative lags correspond to leads.
%
% Input Arguments:
%
%   Y - Time series data. Y may be a vector or a matrix. If Y is a vector,
%     it represents a single series. If Y is a numObs-by-numSeries matrix,
%     it represents numObs observations of numSeries series, with
%     observations across any row assumed to occur at the same time. The
%     last observation of any series is assumed to be the most recent.
%
%   lags - Vector of integer delays or leads, of length numLags, applied to
%     each series in Y. The first lag is applied to all series in Y, then
%     the second lag is applied to all series in Y, and so forth. To
%     include an unshifted copy of a series in the output, use a zero lag.
%
% Output Argument:
%
%   YLag - numObs-by-(numSeries*numLags) matrix of lagged versions of the
%     series in Y. Columns of YLag are, in order, all series in Y lagged by
%     the first lag in lags, all series in Y lagged by the second lag in
%     lags, and so forth. Unspecified observations (presample and
%     postsample data) are padded with NaN values.
%

% Check for a vector:

if numel(Y) == length(Y)
   Y = Y(:); % Ensure a column vector
end

% Ensure lags is a vector of integers:

lags = lags(:); % Ensure a column vector

% Cycle through the lags vector and shift the input time series. Positive 
% lags are delays, and can be processed by FILTER. Negative lags are leads,
% and series are flipped (reflected in time), run through FILTER, and then
% flipped again. Series with zero lags are simply copied.

numLags = length(lags); % Number of lags to apply to each time series

[numObs,numSeries] = size(Y);

YLag = nan(numObs,numSeries*numLags); % Preallocate

for i = 1:numLags

    L       = lags(i);
    columns = (numSeries*(i-1)+1):i*numSeries; % Columns to fill, this lag

    if L > 0 % Time delays

       YLag((L + 1):end,columns) = Y(1:(end - L), :);

    elseif L < 0 % Time leads

       YLag(1:(end + L),columns) = Y((1 - L):end, :);

    else % No shifts

       YLag(:,columns) = Y;

    end

end

%-------------------------------------------------------------------------%

function    J   =   jacobmatx(Theta, X, L, m, q, ActiveFN)
%JACOBMATX computes the Jacobian matrix needed to compute the Lyapunov
%   exponent LAMBDA. The jacobian matrix is relative to X: dF(X)/dX. Not
%   to confuse with the jacobian needed to determine the coefficient THETA.
%   The last Jacobian is relative to THETA and not X.

XLAG   =  lagmatrix(X, L:L:L*m); % size(XLAG) = [T, m].
XLAG   =  XLAG(m*L+1:end, :);    % Remove all NaN observation.
T      =  length(X) - m*L;       % New sample size.

A   =   reshape(Theta(1:q*m), m, q)'; % size(A) = [q, m].
B   =   Theta(q*m+1:q*m+q)';          % size(B) = [q, 1].
C   =   Theta(q*m+q+1:q*m+q+q);       % size(C) = [1, q].

u   =   A * XLAG' + repmat(B,1,T);    % size(u) = [q, T].

J   =   zeros(m, m, T);

J(2:end, 1:end-1, :)  =   repmat(eye(m-1), [1 1 T]);

switch upper(ActiveFN)
    case 'LOGISTIC'
        J(1, :, :)  =   A' * (repmat(C', 1, T) .* (exp(- u) ./ ...
            (1 + exp(- u)).^2));
        
    case 'TANH'
        J(1, :, :)  =   A' * (repmat(C', 1, T) .* (sech(u).^2));
        
    case 'SIGMOID'
        J(1, :, :)  =   A' * (8 * repmat(C', 1, T) .* ...
                (1 + abs(u)) ./ (3 + (1 + abs(u)).^2).^2);
            
    otherwise
        error(' Unrecognized activation function!')
end