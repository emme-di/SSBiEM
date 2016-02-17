function [res] = SSBiEM(X,nbic,path,varargin)
% [res] = SSBiEM(X,nbic,path,varargin) retrieves the nbic number of 
% biclusters from the matrix X taken as input. Each iteration of the EM
% procedure is saved in a different file in the path directory.
% 
% The result of SSBiEM is stored in res which is a structure containing the
% following fields:
%    itl    - vector containing the number of iterations of each ALM cycle 
%    itu    - vector containing the number of iterations of each NLBGS cycle 
%    h      - (n x nbic) matrix containing the row indicators for the
%             retrieved biclusters
%    g      - (nbic x m) matrix containing the column indicators for the
%             retrieved biclusters
%    V,Z    - rows and columns components which multiplication provide the
%             reconstructed matrix
%    tau1,tau2   - vectors containing the estimated taus
%    rho1,rho2   - vectors containing the estimated taus
%    alpha1,alpha2 - vectors containing the estimated alphas
%    sig    - scalar representing the estimated sigma
%    ll     - vector containing the log-likelihood of each EM iteration
%    time   - time needed for the computation
%    item   - number of EM iteration
%
%
%
% [res]=SSBiEM(X,nbic,path,'NAME',VALUE,...) allows you to specify 
% 	  optional parameter name/value pairs as follows:
%
%   thr     -   convergence threshold of the ALM/NLBGS part
%   thrl    -   convergence threshold of the EM algorithm
%   nit     -   maximum number of iteration for the ALM/NLBGS
%   nitl    -   maximum number of iteration for the EM algorithm
%   varsdif -   initial gap between the variances (Spike and Slab model)
%   minval  -   minimum value for V and Z
%   delete  -   delete the path files before the analysis
%   alphas  -   initial value for the alphas
%

if nargin == 2
    path = './iterations/';
end
if ~exist(path,'dir')
    mkdir(path)
end

opt = init(varargin);

if opt.delete
    delete([path,'*.mat']);
end

tic
[n,m] = size(X);
x = X(:);
% convergence parameters
emConverged = 0;
upConverged = 0;
itu = 0;
itl = 0;
item = 0;

%v,z,tau1,tau2,rho1,rho2 inizialization
% V = repmat(mean(X,2),[1,k]);
% Z = repmat(mean(X),[k,1]);
[V,S,Z] = svd(X);

V = V(:,1:nbic);
Z = Z(:,1:nbic)';
S = (S((1:nbic),(1:nbic)));

V = V*sqrt(S);
Z = sqrt(S)*Z;

v = V(:);
z = Z(:);

tau1 = std(V).^2;
tau2 = tau1* opt.varsdif;

rho1 = std(Z,[],2).^2;
rho2 = rho1 * opt.varsdif;

alpha1(1:nbic) = opt.alphas;
alpha2(1:nbic) = opt.alphas;

LLfun = ['-n*m*0.5*log(2*pi*sig^2) -sum((x-C(:)).^2)/(2*sig^2) - 0.5* sum(sum(bsxfun(@times,h,log(2*pi*tau1))))',...
    '- 0.5* sum(sum(bsxfun(@times,1-h,log(2*pi*tau2)))) - 0.5* sum(sum(bsxfun(@times,g,log(2*pi*rho1)))) - 0.5* sum(sum(bsxfun(@times,1-g,log(2*pi*rho2)))) - 0.5*transpose(v)*H*v - 0.5 * transpose(z)*G*z + sum(sum(h).*alpha1) + sum(sum(1-h).*(1-alpha1)) + sum(sum(transpose(g)).*alpha2) + sum(sum(1-transpose(g)).*(1-alpha2))'];
LL = [Inf];

res.itl = [];
res.itu = [];

LLold = -Inf;

Asp = speye(n);
Bsp = speye(m);
%EM cycle
while ~emConverged
    item = item+1;
    if ~exist([path,int2str(item),'.mat'],'file');
        %% E-Step
        
        %updating hi
        for k = 1 : nbic
            %updating hi
            h(:,k) = alpha1(k)*normpdf(V(:,k),0,sqrt(tau1(k)))./...
                (alpha1(k)*normpdf(V(:,k),0,sqrt(tau1(k)))+(1-alpha1(k))*normpdf(V(:,k),0,sqrt(tau2(k))));
            %updating gi
            g(k,:) = alpha2(k)*normpdf(Z(k,:),0,sqrt(rho1(k)))./...
                (alpha2(k)*normpdf(Z(k,:),0,sqrt(rho1(k)))+(1-alpha2(k))*normpdf(Z(k,:),0,sqrt(rho2(k))));
        end
        
        %building the H weight matrix
        H = bsxfun(@times,h,1./(tau1)) + bsxfun(@times,1-h,1./(tau2));
        H = spdiags(H(:),0,numel(H),numel(H));
        %building the G weight matrix
        G = bsxfun(@times,g,1./(rho1)) + bsxfun(@times,1-g,1./(rho2));
        G = spdiags(G(:),0,numel(G),numel(G));
        
        %% M-Step
        %derivatives of the parameters to optimize
        
        %updating sigma:
        C = V*Z;
        sig = (sum((x - C(:)).^2)/(n*m))^2;
        
        %estimating v and z using the lagrangian
        %Augmented Lagrangian of an equivalent function of Q
        
        %rho and mu inizialization
        r = 10 ^ -1;
        mul = 1.05;
        
        %inizialization of lagrangian multipliers
        Y = ones(n,m)*10^-3;
        yl = Y (:);
        yold = yl;
        %inizialization of U
        uold = C(:);
        ul = uold;
        
        %inizialization of Z
        zlold = z;
        Al = kronSpeye(Z',n);
        Al2 = kronSpeye(Z*Z',n);
        
        %inizialization of V
        vlold = v;
        lagrConverged = 0;
        while ~lagrConverged
            itl = itl +1;
            while ~upConverged
                itu = itu +1;
                %old version
                %updating v's
                %             vl = (H + r*(Al'*Al))^-1 * (Al'*yl + r*Al'*ul);
                %             vl = (H + r*(Al'*Al))\ (Al'*yl + r*Al'*ul);
                %             [Q,R] = qr(H + r*(Al'*Al));
                %             b = (Al'*yl + r*Al'*ul);
                %             vl2 = R \ (Q'*b);
                %             [Q,R] = qr(H + r*(Al'*Al));
                %             b = (Al'*yl + r*Al'*ul);
                %             vl2 = R \ (Q'*b);
                aaa = speye(size(H)) * 10^-7;
                vl = (H + r*(Al2) + aaa) \(Al'*yl + r*Al'*ul);
                Vl = reshape(vl,[n,nbic]);
                %updating B
                Bl = kron(Bsp,Vl);
                Bl2 = kron(Bsp,Vl'*Vl);
                %updating z's
                %             zl = (G + r*(Bl'*Bl))^-1 * (Bl'*yl + r*Bl'*ul);
                aaa = speye(size(G + r*(Bl2))) * 10^-7;
                zl = (G + r*(Bl2) + aaa)\ (Bl'*yl + r*Bl'*ul);
                %             zl = (G + r*(Bl2) )\ (Bl'*yl + r*Bl'*ul);
                %             fprintf('max: %e, min: %e\n',max(full(diag(G + r*(Bl2)))),min(full(diag(G + r*(Bl2)))));
                Zl = reshape(zl,[nbic,m]);
                %updating A, alternative to Al = kron(Z',Asp);
                Al = kronSpeye(Z',n);
                Al2 = kronSpeye(Z*Z',n);
                %updating u's
                C = Vl*Zl;
                ul = (r*C(:) - yl + 2*x)/(2+r);
                
                if mean([mean(abs((vl - vlold)./vlold)),mean(abs((zl - zlold)./zlold)),mean(abs((uold - ul)./uold))]) > opt.thr && itu <= opt.nit
                    vlold = vl;
                    zlold = zl;
                    uold = ul;
                else
                    upConverged = 1;
                    v = vl;
                    V = reshape(v,[n,nbic]);
                    z = zl;
                    Z = reshape(z,[nbic,m]);
                    %                 fprintf('ciclo piu interno: %d\n',itu);
                end
            end
            yl = yold + r*(ul - C(:));
            r = min(r*mul,10^20);
            
            if mean(abs((yl - yold)./yold))  > opt.thr && itl <= opt.nit
                yold = yl;
                upConverged = 0;
                res.itu(end+1) = itu;
                itu = 0;
            else
                %             fprintf('ciclo lagrange: %d\n',itl);
                lagrConverged = 1;
                upConverged = 0;
                res.itl(end+1) = itl;
%                 fprintf('Lagrange - Media iterazioni Interno: %d - %.2f\n', itl,mean(res.itu));
                itl = 0;
            end
        end

        for k = 1 : nbic
            %updating tau1:
            tau1(k) = V(:,k)'*diag(h(:,k))*V(:,k)/sum(h(:,k));
            
            %updating tau2:
            tau2(k) = V(:,k)'*diag(1-h(:,k))*V(:,k)/sum(1-h(:,k));
            
            %updating rho1:
            rho1(k) = Z(k,:)*diag(g(k,:))*Z(k,:)'/sum(g(k,:));
            
            %updating tau2:
            rho2(k) = Z(k,:)*diag(1-g(k,:))*Z(k,:)'/sum(1-g(k,:));
            
            %updating alpha1:
            alpha1(k) = sum(h(:,k)/(sum(h(:,k))+sum(1-h(:,k))));
            
            %updating alpha2:
            alpha2(k) = sum(g(k,:))/(sum(g(k,:))+sum(1-g(k,:)));
        end
        
        tau1(isnan(tau1)|tau1<opt.minval) = opt.minval;
        tau2(isnan(tau2)|tau2<opt.minval) = opt.minval;
        rho1(isnan(rho1)|rho1<opt.minval) = opt.minval;
        rho2(isnan(rho2)|rho2<opt.minval) = opt.minval;

        
        % LogLikelihood Function
        if isinf(LL(1))
            LL(1) =  -eval(LLfun);
        else
            LL(end+1) = -eval(LLfun);
        end
        %new-converge
        
        if length(LL) > 1 && (abs(LL(end) - LL(end-1)) < opt.thrl || item == opt.nitl)
            emConverged = 1;
        end
        
        if length(LL) > 1 && LL(end) > LL(end-1)
            cont = cont +1;
            if cont == 10;
                emConverged = 1;
            end
        else
            cont = 0;
        end
        
        %save the result of each iteration of the EM
        res.h = round(h);
        res.g = round(g);
        res.V = V;
        res.Z = Z;
        res.tau1 = tau1;
        res.tau2 = tau2;
        res.rho1 = rho1;
        res.rho2 = rho2;
        res.alpha1 =alpha1;
        res.alpha2 = alpha2;
        res.sig = sig;
        res.ll = LL;
        res.time = toc;
        res.item = item;
        save([path,int2str(item),'.mat'],'res');
    else
        load([path,int2str(item),'.mat'])
        
        V = res.V;
        Z = res.Z;
        
        z = Z(:);
        v = V(:);
        
        tau1 = res.tau1;
        tau2 = res.tau2;
        
        rho1 = res.rho1;
        rho2 = res.rho2;
        
        alpha1 = res.alpha1;
        alpha2 = res.alpha2;
        
        hold = res.h;
        gold = res.g;
        vold = res.V(:);
        zold = res.Z(:);
    end
end
end
%-------------OPTIONS MANAGEMENT--------------
function options = init(param)

% define defaults
options.thr = 10^-2;
options.thrl = 10^-2;
options.nit = 1000;
options.nitl = 20;
options.varsdif = 0.1;
options.minval = 10^-12;
options.delete = 1;
options.alphas = 0.5;
% read acceptable names
validnames = fieldnames(options);

% check argument pairs
nargs = length(param);
if round(nargs/2)~=nargs/2
    error('MAT2TEX needs name/value pairs options');
end

% parse arguments
for pair = reshape(param,2,[]) %now pair is {paramName;paramValue}
    name = lower(pair{1});
    if any(strcmpi(name, validnames))
        % overwrite option
        options.(name) = pair{2};
    else
        error('%s is not a recognized parameter name', name);
    end
end

end