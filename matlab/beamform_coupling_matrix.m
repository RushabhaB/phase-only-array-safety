%% Loading the mutual coupling matrix

dirName='./';
fileName='SimpleVivaldi_24x4_BruteForce18Pts.mat';
% fileName='SimpleVivaldi_24x4.mat';
load([dirName,fileName],'Smat');

%excitation and active match
dx=.33/12/3.28;     %meters
nrows=4; ncols=24;  %elements
[rr, cc]=find(ones(nrows,ncols));%note: smatrix is column major, not row major
elLoc=cc*dx-1i*rr*dx; elLoc=elLoc-mean(elLoc);
nEl=length(cc);

freq=Smat.f_GHz;
bands=[1 2 4 8 16]; bandIdx=1+round((bands-1)/diff(freq(1:2)));
Sf=Smat.S;
idx=(rr-1)*ncols + cc;%note: smatrix is column major, not row major, need to remap
Sf=Sf(idx,idx,:);

%% Creating the array manifold


rng('default');   % optional: reset to MATLAB’s default settings
rng(1234);          % now lock in the seed
%bb=1;
bb=5;
C_mat = Sf(:,:,bandIdx(bb));
fc=bands(bb)*1e9;
c=3e8;eF=1.2;
nPts=100^2;
ang=bsxfun(@plus,1:sqrt(nPts),1i*(1:sqrt(nPts))'); 
ang=ang(:)-mean(ang(:)); 
ang=ang.'/max(real(ang));
ePat=(1-abs(ang).^2).^(eF/2); ePat(abs(ang)>1)=nan;
ePat(~isnan(ePat)) = 1;
A =exp(1i*2*pi*(real(elLoc) *real(ang)+imag(elLoc) *imag(ang))*fc/c);

%% Sanity check 

%C_mat = eye(nEl);


%% Creating the desired response vector 
valid_angles = find(~isnan(ePat));
desired_resp = eye(length(ang));
desired_resp = desired_resp(:,valid_angles);

column_indices = 1:size(desired_resp,2);
row_indices = randi(size(desired_resp,1),1,length(column_indices));
lin_indices = sub2ind(size(desired_resp),row_indices,column_indices);

desired_resp(lin_indices) = 1;

%% Creating the weight vector

w_uc = A(:,valid_angles);
w_uc = w_uc ./ vecnorm(w_uc,2);
response_uc = ePat.' .* abs(A' * C_mat * w_uc).^2;
response_c = ePat.' .* abs(A' * C_mat * pinv(C_mat)*w_uc).^2;

w_c =  pinv(A'*C_mat) * desired_resp;
w_c = w_c ./ vecnorm(w_c,2);

response_ls_c = ePat.' .* abs(A' * C_mat * w_c).^2;

%% Phase only beamformer 

n_iter = 1000; 
B = A';
z_po = exp(1j*angle(pinv(B)*desired_resp));
beta = 0.25;

j = 4501;
alpha_2 = beta / max(abs(desired_resp(:,j)).^2);
u = ones(size(desired_resp(:,j),1),1);
theta = u;
ind = find(desired_resp(:,j));
for i=1:n_iter
    mod_y = desired_resp(:,j).*u;
    s = z_po(:,j)' * (B' * mod_y )/ norm(B*z_po(:,j),2).^2  ; 
    lambda_max = max(eig(abs(s)^2 * (B' * B)))';
    alpha = beta / lambda_max;
    grad = conj(s)* B' * (mod_y - s * B * z_po(:,j));
    eeta = z_po(:,j) + alpha * (grad - real(grad.*conj(z_po(:,j))).*z_po(:,j)) ;
    z_po(:,j) = exp(1j*angle(eeta));
    
   theta(ind) = u(ind) - alpha_2 * conj(desired_resp(ind,j)).*(mod_y(ind)-s*B(ind,:)*z_po(:,j));
   u = exp(1j*angle(theta));
end

w_po(:,j) = exp(1j*angle(pinv(C_mat) * z_po(:,j))); 
response_po_c = ePat.' .* abs(A' * C_mat * w_po).^2;

%% Plotting the response for all three beamformers

point_ind = j;
point_angle  = ang(desired_resp(:,j)==1);

fig=figure(1); set(gcf,'name','pattern'); clf; 
set(fig,'Units','inches','Position',[0 0 25 10]); 

subplot(1,4,1);hold on;
uc_data = 10*log10(abs(reshape(response_uc(:,point_ind)/max(abs(response_uc(:,point_ind))),sqrt(nPts),sqrt(nPts))));
imagesc(real(ang(1:sqrt(nPts):end)),imag(ang(1:sqrt(nPts))),uc_data);
plot(exp(1i*2*pi*(0:360)/360),'k'); 
scatter(real(point_angle),imag(point_angle),120,'red','x');
title('Uninformed Conjugate Beamformer','FontSize',20); axis equal tight; grid on;


subplot(1,4,2);hold on;
c_data = 10*log10(abs(reshape(response_c(:,point_ind)/max(abs(response_c(:,point_ind))),sqrt(nPts),sqrt(nPts))));
imagesc(real(ang(1:sqrt(nPts):end)),imag(ang(1:sqrt(nPts))),c_data);
plot(exp(1i*2*pi*(0:360)/360),'k');
scatter(real(point_angle),imag(point_angle),120,'red','x')
title('Informed Conjugate beamformer','FontSize',20); axis equal tight; grid on;

subplot(1,4,3);hold on;
c_ls_data = 10*log10(abs(reshape(response_ls_c(:,point_ind)/max(abs(response_ls_c(:,point_ind))),sqrt(nPts),sqrt(nPts))));
imagesc(real(ang(1:sqrt(nPts):end)),imag(ang(1:sqrt(nPts))),c_ls_data);
plot(exp(1i*2*pi*(0:360)/360),'k');
scatter(real(point_angle),imag(point_angle),120,'red','x')
title('Informed LS beamformer','FontSize',20); axis equal tight; grid on;


subplot(1,4,4);hold on;
c_po_data = 10*log10(abs(reshape(response_po_c(:,point_ind)/max(abs(response_po_c(:,point_ind))),sqrt(nPts),sqrt(nPts))));
imagesc(real(ang(1:sqrt(nPts):end)),imag(ang(1:sqrt(nPts))),c_po_data);
plot(exp(1i*2*pi*(0:360)/360),'k');
scatter(real(point_angle),imag(point_angle),120,'red','x')
title('Phase Only Informed Conjugate beamformer','FontSize',20); axis equal tight; grid on;

% Add a common colorbar
cb = colorbar('Position', [0.95 0.25 0.02 0.6]); % adjust position as needed
cb.Label.String = 'dB';
cb.Label.FontSize = 14;

minColorLimit = min([min(uc_data(:)),min(c_ls_data(:)),min(c_data(:)),min(c_po_data(:))]);
maxColorLimit = max([max(uc_data(:)),max(c_ls_data(:)),max(c_data(:)),max(c_po_data(:))]);
clim([minColorLimit,maxColorLimit]);    

set(fig,'PaperUnits','inches');            % same units
set(fig,'PaperPosition',get(fig,'Position'))  % copy screen size to paper
set(fig,'PaperPositionMode','manual');     % use it (important!)


%saveas(gcf,['./Figures/beampattern_mutual_coupling_' num2str(mc) '_freq_' num2str(bands(bb)) '_GHz_two_source.png'])
%saveas(gcf,['./Figures/beampattern_mutual_coupling_' num2str(mc) '_freq_' num2str(bands(bb)) '_GHz_two_source.fig'])
%close(fig)

