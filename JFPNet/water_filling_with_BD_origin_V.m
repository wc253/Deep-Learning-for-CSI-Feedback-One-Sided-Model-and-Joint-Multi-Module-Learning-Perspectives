clear all;
close all;
%% parameter initial
UE_num = 2;
BS_a = 32;
UE_a = 4;
RB_num = 52;
subband_num = 13;
data_num = 20000;
P_tx = 10^(1.6);
No = 10^(-13.4);

% load('E:/DJSCC_dataset_109e/data_uma_npz/MU_data/BD_max_R_Ms.mat');
load('E:/DJSCC_dataset_109e/data_uma_npz/MU_data/MU_MIMO_UMa_UE2_V_test.mat');
Ms_1 = permute(V1,[1,3,2]);
Ms_2 = permute(V2,[1,3,2]);
load('E:/DJSCC_dataset_109e/data_uma_npz/MU_data/RB_test.mat');
load('E:/DJSCC_dataset_109e/data_uma_npz/MU_data/MU_MIMO_UMa_UE2_eig_test.mat');
RB_H_1 = RB_1;
RB_H_2 = RB_2;
% load('test_UMa_tx32_c624_UE6_p1.mat');
% RB_H_1 = squeeze(H_dl(:,1,:,:,:));
% RB_H_2 = squeeze(H_dl(:,2,:,:,:));
%% calculate power distribute matrix
MS_1 = ones(32,1);
MS_2 = ones(32,1);
P = ones(data_num, subband_num, 2);
for i = 1:data_num
    for j = 1:subband_num
        MS_1(:,1) = squeeze(Ms_1(i,j,:));  % 32*1
        MS_2(:,1) = squeeze(Ms_2(i,j,:));  % 32*1
        % nom_1
        nom_1 = eig_1(i,j);
        % nom_2
        nom_2 = eig_2(i,j);
        % u
        u = (P_tx + No/nom_1 + No/nom_2 )/UE_num;
        % P_matrix
        P_1 = max(0, u-No/nom_1);
        P_2 = max(0, u-No/nom_2);
        if P_1 == 0
            P_2 = P_2 + u-No/nom_1;
        end
        if P_2 == 0
            P_1 = P_1 + u-No/nom_2;
        end

        P(i,j,1) = P_1;
        P(i,j,2) = P_2;
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
        nom = P(i,fix((j-1)/4)+1,1)*(W_1'*H_1*V1)*(W_1'*H_1*V1)';
        denorm = No + P(i,fix((j-1)/4)+1,2)*(W_1'*H_1*V2)*(W_1'*H_1*V2)';
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
        nom = P(i,fix((j-1)/4)+1,2)*(W_2'*H_2*V2)*(W_2'*H_2*V2)';
        denorm = No + P(i,fix((j-1)/4)+1,1)*(W_2'*H_2*V1)*(W_2'*H_2*V1)';
        R2 = R2 + real(log2(1+nom/denorm));
    end
end
R2_avg = R2/(data_num*RB_num);
R_sum = R1_avg + R2_avg;
disp(R1_avg);
disp(R2_avg);
disp(R_sum);
