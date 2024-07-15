function [Ms_1,Ms_2] = BD_Partial_CSI(V1,V2)
%% parameter initial
RB_num = 52;
data_num = size(V1);
data_num = data_num(1);

B1 = V1;
B2 = V2;
B1_check = V2;
B2_check = V1;

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
end

