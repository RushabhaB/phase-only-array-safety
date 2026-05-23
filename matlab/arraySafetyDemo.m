function toy090822
%look at smatrix params

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

figure(1); clf; hold on;
set(gcf,'name','S matrix');
for(ii=1:length(bands))
    subplot(2,3,ii);
    imagesc(10*log10(abs(Sf(:,:,bandIdx(ii)))));
%     imagesc(10*log10(abs(V(:,1:45)*D(1:45,1:45))));
    caxis([-20 0]);
    title(sprintf('%d GHz',bands(ii)));
end
xlabel('element');ylabel('element'); grid on;

% figure(2); clf; hold on;
% set(gcf,'name','invS matrix');
% for(ii=1:length(bands))
%     subplot(2,3,ii);
%     imagesc(10*log10(abs(inv(Sf(:,:,bandIdx(ii))))));
%     caxis([-15 15]);
%     title(sprintf('%d GHz',bands(ii)));
% end
% xlabel('element');ylabel('element'); grid on;


%eigen analysis
figure(3); clf; hold on;
set(gcf,'name','eigenvectors');
for(ii=1:length(bands))
    subplot(2,3,ii);
    [V, D]=eig(squeeze(Sf(:,:,bandIdx(ii))));
    D=diag(D);[~,indx]=sort(abs(D),'descend'); V=V(:,indx); D=diag(D(indx));
    imagesc(10*log10(abs(V*D)));
%     imagesc(10*log10(abs(V(:,1:45)*D(1:45,1:45))));
    caxis([-15 0]);
    title(sprintf('%d GHz',bands(ii)));
end
xlabel('eigenvalue');ylabel('element'); grid on;

figure(4); clf; hold on;
set(gcf,'name','eigenspectrum');
for(ii=1:length(bands))
    plot(10*log10(sort(abs(eig(squeeze(Sf(:,:,bandIdx(ii))))),'descend')));
end
str=sprintf('''%d GHz'',',bands);
eval(['legend(' str(1:end-1) ');']);legend('location','southwest');
xlabel('eigenvalue');ylabel('dB'); title('Freq Eigenspectrum'); grid on;

experiment=5;
switch(experiment)
    case 1, sa=zeros(nEl,1);%boresight
            ta=ones(nEl,1);
    case 2, sa=(sind(45)+1i*cosd(45)*sind(-20))*ones(nEl,1);%high angle
            ta=ones(nEl,1);
    case 3, sa=zeros(nEl,1);%phase discontinuity
            ta=[ones(nrows,ncols/2),-ones(nrows,ncols/2)]; ta=ta(:);
    case 4, sa=(sind(45)+1i*cosd(45)*sind(-20))*ones(nEl,1);%high angle + discontinuity
            ta=[ones(nrows,ncols/2),-ones(nrows,ncols/2)]; ta=ta(:);
    case 5, sa=[sind(45)*ones(nEl/2,1);-sind(45)*ones(nEl/2,1)];%cross the beams
            ta=ones(nEl,1);
    case 6, sa=[ ];%mimo
end
ta=ta/sqrt(sum(abs(ta)))*sqrt(nEl);

bb=3;
fc=bands(bb)*1e9;
c=3e8;eF=1.2;
nPts=100^2;
ang=bsxfun(@plus,1:sqrt(nPts),1i*(1:sqrt(nPts))'); 
ang=ang(:)-mean(ang(:)); 
ang=ang.'/max(real(ang));
ePat=(1-abs(ang).^2).^(eF/2); ePat(abs(ang)>1)=nan;
illum =exp(1i*2*pi*(real(elLoc) *real(ang)+imag(elLoc) *imag(ang))*fc/c);
excite=ta.*exp(1i*2*pi*(real(elLoc).*real(sa) +imag(elLoc).*imag(sa) )*fc/c);
% repair=phaseOnlyNull(eye(length(elLoc))*lambda+Sf(:,:,bandIdx(bb)),excite);
% repair=(eye(length(elLoc))*lambda+Sf(:,:,bandIdx(bb)))\excite;
% repair=repair/sqrt(repair'*excite)*sqrt(nEl);

repair=excite;
figure(10); clf; hold on;set(gcf,'name','phasors');
plot(repair);
xlabel('real'); ylabel('imag'); grid on; 
axis equal tight;
fprintf('run(elem):  mag\n');
nCmap=64;cc=parula(nCmap); cc(end,:)=[1 0 0];%set the highest value to red, 

for(ii=1:20)
   reflect=Sf(:,:,bandIdx(bb))*repair;
   [val, jj]=max(abs(reflect));
   if(val>(nCmap-1)/nCmap),%nCmap is arbitrary: tied to colormap-> make the red not red
       grad=(reflect(jj)-reflect(jj)/val*(nCmap-2)/nCmap)*Sf(jj,:,bandIdx(bb))';
       repair=repair-grad;
       repair=repair./abs(repair);
   else, break; end
   plot(repair);
%    fprintf('%02d(%02d): %3.2f+1i*%3.2f = %3.2fexp(1i*%3.2f)\n',ii,jj,...
%        real(reflect(jj)),imag(reflect(jj)),abs(reflect(jj)),angle(reflect(jj)));
   fprintf('% 3d(% 4d): %3.2f\n',ii,jj,abs(reflect(jj)));
end

pat=(ePat.*illum)'*[excite repair];

figure(7); set(gcf,'name','pattern'); clf; 
subplot(1,2,1);hold on;
imagesc(real(ang(1:sqrt(nPts):end)),imag(ang(1:sqrt(nPts))),10*log10(abs(reshape(pat(:,1),sqrt(nPts),sqrt(nPts)))));
plot(exp(1i*2*pi*(0:360)/360),'k'); 
title('Before'); axis equal tight; grid on;
subplot(1,2,2);hold on;
imagesc(real(ang(1:sqrt(nPts):end)),imag(ang(1:sqrt(nPts))),10*log10(abs(reshape(pat(:,2),sqrt(nPts),sqrt(nPts)))));
%plot(exp(1i*2*pi*(0:360)/360),'k');
title('After'); axis equal tight; grid on;

figure(5); set(gcf,'name','steer');clf; 
subplot(3,1,1);hold on;
imagesc(abs(reshape(excite,nrows,ncols)));
axis equal tight;title('Before');colorbar;
% caxis([0 1]); colormap(cc);
subplot(3,1,2);hold on;
imagesc(abs(reshape(repair,nrows,ncols)));
axis equal tight;title('After');colorbar;
% caxis([0 1]); colormap(cc);
subplot(3,1,3);hold on;
plot(abs([excite, repair]));
xlabel('element'); ylabel('mag'); grid on;

figure(6); set(gcf,'name','reflected power');clf; 
subplot(3,1,1);hold on;
imagesc(abs(reshape(squeeze(Sf(:,:,bandIdx(bb)))*excite,nrows,ncols)));
axis equal tight;title('Before');colorbar;
caxis([0 1]); colormap(cc);
subplot(3,1,2);hold on;
imagesc(abs(reshape(squeeze(Sf(:,:,bandIdx(bb)))*repair,nrows,ncols)));
axis equal tight;title('After');colorbar;
caxis([0 1]); colormap(cc);
subplot(3,1,3);hold on;
plot(abs(squeeze(Sf(:,:,bandIdx(bb)))*[excite, repair]));
xlabel('element'); ylabel('mag'); grid on;




return

function w=phaseOnlyNull(R,v,plotsOn)
if(nargin<3), plotsOn=false; end
nEl=length(v);

J=toeplitz([1;zeros(nEl-1,1)],[1 -1 zeros(1,nEl-2)]);J=J(1:end-1,:);
n=5;
t=[0 10.^(linspace(-15,20,30)/10)];
w2=repmat(t,nEl,1);
SINR2=t;
grad2=w2;
tIlast=0; tI=1; ii=0;w=v;
while(ii<100 && (tI~=1 || tIlast~=tI))
    ii=ii+1;
    if(~tIlast)%first run
        a=w'*v;
        b=R*w;
        gam=w'*b;
        SINR=abs(a)^2/gam;
        grad=2*(imag((a'*v).*conj(w))-SINR*imag(b.*conj(w)))/gam;
        h=grad;
        gradOld=grad;
    end
    
    %vector operations
    for(jj=1:length(t))
        w2(:,jj)=exp(1i*h*t(jj)).*w;
        a=w2(:,jj)'*v;
        b=R*w2(:,jj);
        gam=w2(:,jj)'*b;
        SINR2(jj)=abs(a).^2./gam;%SINR
        grad2(:,jj)=2*(imag((a'*v).*conj(w2(:,jj)))-...
            SINR2(jj)*imag(b.*conj(w2(:,jj))))/gam;
    end
    
    %set w equal to max SINR
    tIlast=tI;
    [SINR, tI]=max(abs(SINR2));
    index(ii)=tI;
    SINR1(ii)=SINR;
    grad=grad2(:,tI);w=w2(:,tI);
    gam=((grad-gradOld).'*grad)/abs(gradOld).^2;
    gradOld=grad;
    
    %reset gradiant
    if(~mod(ii,n))
        h=grad;
    else
        h=grad+gam*h;
    end
    
    if(plotsOn)
        nFFT=256;
        figure(101);
        plot(linspace(-1,1-2/nFFT,nFFT),20*log10(abs(fftshift(fft([v wA w],nFFT),1))));
        ylim(10*log10(nEl)*[-2 1]);hold on;
        plot([saJ;saJ], get(gca,'ylim')'*ones(1,length(saJ)),'m--');hold off;
        title(sprintf('Iteration- %d',ii));
        xlabel('sin(\alpha)'); ylabel('Gain (dB)');
        legend('UnAdapted','Gain/Phase','\phi(1)');legend('off');
    end
end
%{
ii=0;clear SINR2;log(0);
SINRmax=SINR;wmax=w;w1=w;
while(~ii || sum(abs(h))>1)
% while(SINR>=SINRmax)
    ii=ii+1;
    a=w'*v;
    b=R*w;
    gam=w'*b;
    SINR=abs(a)^2/gam;
    SINR2(ii)=SINR;
    grad=2*(imag((a'*v).*conj(w))-SINR*imag(b.*conj(w)))/gam;

    A=real((v*v'-SINR*R).*conj(w*w'));
%     A=real((v*v'+SINR*R).*conj(w*w'));
    c=2*imag(b.*conj(w));
%     d=2*imag((conj(a)*v).*conj(w))-SINR*c;  %error in paper? doesn't converge
    d=2*imag((conj(a)*v).*conj(w))+SINR*c;
    Hess=2*(A-diag(sum(A)))/gam-(c*d.'+d*c.')/gam^2;

    h=-J.'*((J*Hess*J.')\J)*grad;

%     h=-Hess*grad;
    w=exp(1i*h).*w;

    if(SINR>SINRmax)
        wmax=w;
        SINRmax=SINR;
    end;
    
        if(plotsOn)
    nFFT=256;
    figure(102);
    plot(linspace(-1,1-2/nFFT,nFFT),20*log10(abs(fftshift(fft([v wA w1 wmax w],nFFT),1))));
    ylim(10*log10(nEl)*[-2 1]);hold on;
    plot([saJ;saJ], get(gca,'ylim')'*ones(1,length(saJ)),'m--');hold off;
    title(sprintf('Iteration- %d',ii));
    xlabel('sin(\alpha)'); ylabel('Gain (dB)');
    legend('UnAdapted','Complex','\phi(1)','\phi_{best}(2)','\phi{last}(2)');legend('off');
        end
end
%}
return