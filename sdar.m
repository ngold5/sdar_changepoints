function [x1h_tally, Sigmh_tally] = sdar(x, r, k)

% (vector, scalar, scalar) -> (vector, vector)
% 
% Sequentially Discounting Autoregressive Model
% Adapted from "A Unifying Framework For Detecting Outlier and 
% Change Points from Time Series" Takeuchi and Yamanishi 2006
% 
% Input
% x 	time series 
% r 	discounting parameter 0 < r < 1
% k 	past observations parameter
%
% Output
% x1h_tally 	vector of predicted means
% sigmh_tally 	vector of predicted covariances
%
% 

NM = length(x) - k;
N = NM + k; % Total length of time series, t

% Preallocate memory
C0 = zeros(k,1); % C_1,...,C_k
C1 = zeros(k,1); % To be used in calculation
x1h_tally = zeros(1,N); % Vector of predicted means
Sigmh_tally = zeros(1,N); % Vector of predicated covariances
x0 = zeros(N, 1);

% Initialization
mu0 = 1 / (N -k) * sum(x(k+1:N));
mue = mu0;
for j = 1:k
	C0(j,1) = 1 / (N - k) * sum((x(k+1:N) - mu0) .* (x(k+1-j:N-j) - mu0)); % C_1,...,C_k
end
C00 = 1 / (N - k) * sum((x(k+1:N) -mu0) .* (x(k+1:N) - mu0)); % C_0

% Construct Toeplitz matrix
v = [C00; C0(1:end - 1)];
M = toeplitz(v);
% Solve linear system
A = M \ C0;

% Compute Sigma
Sigm = C00 - sum(A .* C0);
Sigme = Sigm;
x0 = x(1:k);

% Compute parameters
for ii = k+1:k+NM
	
	x1 = x(ii);

	mu1 = (1 - r) * mu0 + r * x1;

	for j=1:k
		C1(j, 1) = (1 - r) * C0(j, 1) + r * (x1 - mu1) * (x0(end - j + 1) - mu1);
	end
	C00 = (1 - r) * C0(j, 1) + r * (x1 - mu1) * (x1 - mu1);

	% Solve linear system	
	v = [C00; C1(1:end-1)];
	M = toeplitz(v);
	A1 = M \ C1;

	% Now calculate the predicted mean and predicted covariance
	x1h = sum(A1 .* (x0(end:-1:end - k + 1) - mu1)) + mu1;
	Sigmh = (1 - r) * Sigm + r * (x1 - x1h)*(x1 - x1h);

	% Store results as vector
	x1h_tally(ii) = x1h;
	Sigmh_tally(ii) = Sigmh;

	% Update values for next iteration
	mu0 = mu1;
	C0 = C1;
	x0 = [x0; x1];
	Sigm = Sigmh;
end

x1h_tally = x1h_tally';
Sigmh_tally = Sigmh_tally';
