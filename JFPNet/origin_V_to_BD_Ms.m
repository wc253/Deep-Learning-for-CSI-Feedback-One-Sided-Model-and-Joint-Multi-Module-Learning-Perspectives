%% 该文件用于对于624*32*4的CSI信息做SVD分解，得到待压缩的特征向量
close all;
clear all;
clc;
data_type = 'test';
RB = 52;
TX = 32;
RX = 4;
scenario = 'UMa';
data_path = 'H:/ZTE_ADJSCC/MU_data/vector';
switch data_type
    case 'train'
        data_no = 10e4;
    case 'val'
        data_no = 3e4;
    case 'test'
        data_no = 2e4;
end
part_no = 1000;
for part_i=1:data_no/part_no %data_no/part_no
    filename = [data_path,'/','MU_MIMO_UMa_UE6_H_dl_vector_',data_type,'_p',num2str(part_i),'.mat'];
    load(filename);
    V1 = squeeze(H_dl_vector(:,1,:,:));
    V2 = squeeze(H_dl_vector(:,2,:,:));
    [MS1,MS2] = BD_Partial_CSI(V1, V2);
    save(['H:/ZTE_ADJSCC/MU_data/BD_Partial_Ms/MU_MIMO_UMa_BD_Partial_CSI_Ms',data_type,'_p',num2str(part_i),'.mat'],'MS1','MS2');
    fprintf(sprintf('%d finished\n',part_i));
end

    
