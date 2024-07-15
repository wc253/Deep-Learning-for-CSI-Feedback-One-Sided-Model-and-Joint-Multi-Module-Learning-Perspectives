clear all;
close all;
%% parameter initial
UE_num = 2;
BS_a = 32;
UE_a = 4;
RB_num = 52;
data_num = 1000;
P_tx = 10^(1.6);
No = 10^(-13.4);

load('E:/DJSCC_dataset_109e/data_uma_npz/MU_data/RB_test.mat');
RB_H_1 = RB_1;
RB_H_2 = RB_2;

H_check_1 = RB_H_2;
H_check_2 = RB_H_1;
for i =1:data_num
    for j = 1:RB_num
    H = squeeze(RB_H_1(i,j,:,:));
    [U_check, S_check, V_check] = svd(squeeze(H_check_1(i,j,:,:)));
    V_check_0 = V_check(:,5:end);
    HV_check_0 = H * V_check_0;
    [U, S, V] = svd(HV_check_0);
    V_1 = V(:,1:4);
    MS = V_check_0 * V_1;
    norm_para = sum(abs(MS).^2, 1);
    MS_norm = MS ./ norm_para;
    Ms_1(i,j,:,:) = MS_norm;
    end
end

for i =1:data_num
    for j = 1:RB_num
    H = squeeze(RB_H_2(i,j,:,:));
    [U_check, S_check, V_check] = svd(squeeze(H_check_2(i,j,:,:)));
    V_check_0 = V_check(:,5:end);
    HV_check_0 = H * V_check_0;
    [U, S, V] = svd(HV_check_0);
    V_1 = V(:,1:4);
    MS = V_check_0 * V_1;
    norm_para = sum(abs(MS).^2, 1);
    MS_norm = MS ./ norm_para;
    Ms_2(i,j,:,:) = MS_norm;
    end
end

%% caculate R sum
% R1
R1 = 0;
for i =1:data_num
    for j = 1:RB_num
        V1 = squeeze(Ms_1(i,j,:,1));
        V2 = squeeze(Ms_2(i,j,:,1));
        H_1 = squeeze(RB_H_1(i,j,:,:));
        HV_1 = H_1*V1;
        HV_norm2 = sqrt(abs(HV_1).^2);
        W_1 = HV_1./ HV_norm2;
        nom = (W_1'*H_1*V1)*(W_1'*H_1*V1)';
        denorm = No + 0.5*P_tx*(W_1'*H_1*V2)*(W_1'*H_1*V2)';
        R1 = R1 + real(log2(1+0.5*P_tx*nom/denorm));
    end
end
R1_avg = R1/(data_num*RB_num);
% R2
R2 = 0;
for i =1:data_num
    for j = 1:RB_num
        V1 = squeeze(Ms_1(i,j,:,1));
        V2 = squeeze(Ms_2(i,j,:,1));
        H_2 = squeeze(RB_H_2(i,j,:,:));
        HV_2 = H_2*V2;
        HV_norm2 = sqrt(abs(HV_2).^2);
        W_2 = HV_2./ HV_norm2;
        nom = (W_2'*H_2*V2)*(W_2'*H_2*V2)';
        denorm = No + 0.5*P_tx*(W_2'*H_2*V1)*(W_2'*H_2*V1)';
        R2 = R2 + real(log2(1+0.5*P_tx*nom/denorm));
    end
end
R2_avg = R2/(data_num*RB_num);
R_sum = R1_avg + R2_avg;
disp(R_sum);

% save('E:/DJSCC_dataset_109e/data_uma_npz/MU_data/BD_max_R_Ms','Ms_1','Ms_2');