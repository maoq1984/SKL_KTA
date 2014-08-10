function [acc,ypred]=multiclass_Klrls(L,y,l_ind,u_ind,eps)
% L is graph laplacian matrix
% y is a column vector of labeled data
% l_ind is the index set of labeled data, correponding to the order of L
% u_ind is the index set of unlabeled data
% eps is used to avoid singular, normally set the small scale of labeled
% data 1/n_l


oldy = y;
% final semi-supervised kernel learning for regularized least square

c = unique(y);
num_c = length(c);

encode_y = zeros(size(y));
for i=1:num_c
    IX = find(y==c(i));
    encode_y(IX) = i;
end
y = encode_y;

n = length(y);

Y = zeros(n,num_c);
for i=1:n
    Y(i,y(i)) = 1;
end

[U,V] = eig(L);
Ul = U(l_ind,:);
Uu = U(u_ind,:);

A = Ul' * Y(l_ind,:);
A = A * A';

B = V;
if(find(diag(V)<1e-10))
    B = B + eps;
end

a = diag(A);
b = diag(B);

x_ = sum( a .* sqrt(a./(2 * b)) );
y_ = sum( a ./(2 * b) );
z_ = sum( sqrt(a ./(2 * b)));
u_ = sum( a );
tmp = abs((z_ * u_ - n * x_) / (y_ * u_ - z_ * x_));
sigma = tmp .* sqrt(a./(2 .* b));

Kll = Ul * diag(sigma) * Ul';
I = eye(n);
W = Uu * diag(sigma) * Ul' - I(u_ind,l_ind);

ypred = W * (Kll)^(-1) * Y(l_ind,:);

[my,mIX ]= max(ypred,[],2);    
acc = mean(mIX == y(u_ind));

% % %% plot
% % K = U * diag(sigma) * U';
% % max_k = max(max(K));
% % scaleK = K./max_k;
% % scaleK = scaleK .* 255;
% % figure;
% % image(scaleK);
% % 
% % %% compute KTA
% % Kuu = Uu * diag(sigma) *Uu';
% % up = trace(Kll * (oldy(l_ind) *oldy(l_ind)'));
% % down = sqrt(trace(Kll *Kll')) * length(l_ind);
% % kta_l = up /down;
% % 
% % up = trace(Kuu * (oldy(u_ind) *oldy(u_ind)'));
% % down = sqrt(trace(Kuu * Kuu')) * length(u_ind);
% % kta_u = up / down;
% % 
% % fprintf('kta on labeled data: %f, kta on unlabeled data:%f\n',kta_l,kta_u);
    

