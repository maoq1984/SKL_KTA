This code is implemented by MAO QI for the following paper

Qi Mao, and Ivor W. Tsang. Parameter-Free Spectral Kernel Learning. Proceedings of the 26th Conference on Uncertainty in Artificial Intelligence (UAI 2010), Catalina Island, California, July 2010.

If any problem, please email to maoqi by maoq1984@gmail.com


function [acc,ypred]=multiclass_Klrls(L,y,l_ind,u_ind,eps)

% L is graph laplacian matrix
% y is a column vector of labeled data
% l_ind is the index set of labeled data, correponding to the order of L
% u_ind is the index set of unlabeled data
% eps is used to avoid singular, normally set the small scale of labeled
% data 1/n_l