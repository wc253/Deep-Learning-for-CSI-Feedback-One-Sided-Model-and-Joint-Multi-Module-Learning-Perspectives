clear all;
close all;
%% parameter initial
UE_num = 2;
BS_a = 32;
UE_a = 4;
RB_num = 52;
data_num = 20000;
P_tx = 10^(1.6);
No = 10^(-13.4);

load('E:/DJSCC_dataset_109e/data_uma_npz/MU_data/RB_test.mat');
% load('test_UMa_tx32_c624_UE6_p1.mat')
RB_H_1 = RB_1;
RB_H_2 = RB_2;

% load('E:/DJSCC_dataset_109e/data_uma_npz/MU_data/MU_MIMO_UMa_UE2_V_test.mat');
% B1 = V1;
% B2 = V2;
% B1_check = V2;
% B2_check = V1;
for ed=6:9
    for snr=-10:5:10
        load(['E:/DJSCC_dataset_109e/data_uma_npz/MU_data/DJSCC_V_hat_snr',num2str(snr),'_ed',num2str(2^ed),'.mat']);
        B1 = V_1;
        for i =1:20000
            for j=1:13
                V = squeeze(V_1(i,:,j));
                norm = sqrt(sum(V.^2));
                B1(i,:,j)=V./norm;
            end
        end

        B2 = V_2;
        for i =1:20000
            for j=1:13
                V = squeeze(V_2(i,:,j));
                norm = sqrt(sum(V.^2));
                B2(i,:,j)=V./norm;
            end
        end
        B1_check = B2;
        B2_check = B1;

        for i =1:data_num
            for j = 1:RB_num/4
            B(1,:) = squeeze(B1(i,:,j))';  % 1*32
            B1_check_signal(1,:) = squeeze(B1_check(i,:,j))';
            [U_check, S_check, V_check] = svd(B1_check_signal);
            V_check_0 = V_check(:,2:end);  % 32*31
            BV_check_0 = B * V_check_0;  % 1*31
            [U, S, V] = svd(BV_check_0);
            V_1 = V(:,1);  % 31*1
            MS = V_check_0 * V_1;  % 32*1
            norm_para = sqrt(sum(abs(MS).^2, 1));
            MS_norm = MS ./ norm_para;
            Ms_1(i,j,:,:) = MS_norm;
            end
        end

        for i =1:data_num
            for j = 1:RB_num/4
            B(1,:) = squeeze(B2(i,:,j))';  % 1*32
            B2_check_signal(1,:) = squeeze(B2_check(i,:,j))';
            [U_check, S_check, V_check] = svd(B2_check_signal);
            V_check_0 = V_check(:,2:end);  % 32*31
            BV_check_0 = B * V_check_0;  % 1*31
            [U, S, V] = svd(BV_check_0);
            V_2 = V(:,1);  % 31*1
            MS = V_check_0 * V_2;  % 32*1
            norm_para = sqrt(sum(abs(MS).^2, 1));
            MS_norm = MS ./ norm_para;
            Ms_2(i,j,:,:) = MS_norm;
            end
        end

        %% caculate R sum
        % R1
        R1 = 0;
        for i =1:data_num
            for j = 1:RB_num
                V1 = squeeze(Ms_1(i,fix((j-1)/4)+1,:,:));
                V2 = squeeze(Ms_2(i,fix((j-1)/4)+1,:,:));
                H_1 = squeeze(RB_H_1(i,j,:,:));
                HV_1 = H_1*V1;
                HV_norm2 = sqrt(sum(abs(HV_1).^2));
                W_1 = HV_1./ HV_norm2;
                nom = 0.5*P_tx*(W_1'*H_1*V1)*(W_1'*H_1*V1)';
                denorm = No + 0.5*P_tx*(W_1'*H_1*V2)*(W_1'*H_1*V2)';
                R1 = R1 + real(log2(1+nom/denorm));
            end
        end
        R1_avg = R1/(data_num*RB_num);
        % R2
        R2 = 0;
        for i =1:data_num
            for j = 1:RB_num
                V1 = squeeze(Ms_1(i,fix((j-1)/4)+1,:,:));
                V2 = squeeze(Ms_2(i,fix((j-1)/4)+1,:,:));
                H_2 = squeeze(RB_H_2(i,j,:,:));
                HV_2 = H_2*V2;
                HV_norm2 = sqrt(sum(abs(HV_2).^2));
                W_2 = HV_2./ HV_norm2;
                nom = 0.5*P_tx*(W_2'*H_2*V2)*(W_2'*H_2*V2)';
                denorm = No + 0.5*P_tx*(W_2'*H_2*V1)*(W_2'*H_2*V1)';
                R2 = R2 + real(log2(1+nom/denorm));
            end
        end
        R2_avg = R2/(data_num*RB_num);
        R_sum = R1_avg + R2_avg;
        disp(2^ed);
        disp(snr);
        disp(R_sum);
        save(['E:/DJSCC_dataset_109e/data_uma_npz/MU_data/DJSCC_BD/BD_max_R_Ms_SNR',num2str(snr),'_ed',num2str(2^ed)],'Ms_1','Ms_2');
    end
end