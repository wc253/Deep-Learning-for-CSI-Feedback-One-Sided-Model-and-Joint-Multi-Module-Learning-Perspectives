close all;clear all;
%config
ed = 32;
B = 3;
bw_size = 16;
data_size = 2000;
mod = '256QAM';             %QPSK-2bits,16QAM-4bits,64QAM-6bits,256QAM-8bits
switch mod
    case 'QPSK'
        mod_bits = 2;
    case '16QAM'
        mod_bits = 4;
    case '64QAM'
        mod_bits = 6;
    case '256QAM'
        mod_bits = 8;
    otherwise
        mod_bits=0;
end    

%load data
H_data_r = load('data_npz/H.mat').H_test;
H_data_c = H_data_r(:,:,:,1)+1j*H_data_r(:,:,:,2);
qen_data = load(['data_qen/qen_ed',num2str(ed),'_B',num2str(B),'.mat']).qen;

for snrdB=-10:10
    qen_bi_size = ed*B;
    dec = zeros(data_size,ed);
    eq_count = 0;
    for idx=1:data_size
        H_sample = reshape(H_data_c(idx,:,:),256,32);
        H_bw_sample = H_sample(1:bw_size,:);
        %gen bit stream
        qen_sample = double(qen_data(idx,:)');
        qen_bi = de2bi(qen_sample);
        uci_bits = reshape(qen_bi, qen_bi_size, 1);
        E = bw_size*mod_bits;
        % uci channel coding and modulation
        encUCI = nrUCIEncode(uci_bits,E,mod);
        encUCI(encUCI==-1) = 1;
        encUCI(encUCI==-2) = encUCI(find(encUCI==-2)-1);
        modOut = nrSymbolModulate(encUCI,mod);
        %SIMO fading channel
        fadingSig = modOut.*H_bw_sample;
        %AWGN channel
        rxSig = awgn(fadingSig,snrdB);
        %MRC at the receiver
        H_norm2 = sqrt(sum(abs(H_bw_sample).^2,2));
        W = H_bw_sample./H_norm2;
        mrcSig = sum(conj(W).*rxSig, 2);
        mrcSig_norm = mrcSig./H_norm2;
        % uci decode
        rxSoftBits = nrSymbolDemodulate(mrcSig_norm,mod);
        dec_bits = nrUCIDecode(rxSoftBits,qen_bi_size,mod);
        if isequal(dec_bits,uci_bits)
            eq_count = eq_count+1;
        end
        dec_bi = reshape(dec_bits,ed,B);
        dec_sample = bi2de(dec_bi)'; 
        dec(idx,:)=dec_sample;
        idx
    end
    save(['data_dec/dec_bw', num2str(bw_size),'_ed',num2str(ed),'_B', num2str(B),'_mod',num2str(mod_bits),'_snr', num2str(snrdB),'.mat'],'dec', 'eq_count');
end