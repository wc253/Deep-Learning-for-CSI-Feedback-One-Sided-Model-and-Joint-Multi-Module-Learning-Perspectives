数据集处理
1.	对CSI特征向量数据集：对每一个样本进行归一化后存储为.npz文件，存储数据包括训练数据集、验证数据集、测试数据集及三者的归一化因子。数据集存在data_npz文件夹中
	对应文件：data_pro_CSI.py
2.	对上行信道进行处理：对每个样本进行归一化，等效于功放处理，存储数据包括训练数据集、验证数据集、测试数据集及三者的归一化因子。数据集存在data_npz文件夹中
	对应文件：data_pro_2D_H_ul.py
3.	对RB级别的下行CSI进行处理：对每个样本做归一化，等效于功放。此数据集用于计算下行信道容量。，存储数据包括训练数据集、验证数据集、测试数据集及三者的归一化因子。数据集存在data_npz文件夹中
	对应文件：data_pro_Hdl_RB.py

JNPNet网络测试
修改数据集位置，指向自己的数据集（.npz文件）。修改文件中ed(压缩后的CSI维度)、snr_low（上行信最低噪比，单位为dB）、snr_high（上行信最高噪比，单位为dB）、P_sum（下行发射功率）、noise_power（下行全带宽噪声功率），进行训练。

当前网络为用户数为2的情况，当增加用户数时需要修改数据载入和网络框架部分，参考两用户即可。

MU_seperate_DJSCC_net2_PA.py:DJSCC网络+联合预编码模块2+ PA module

传统算法
BD_max_R_perfect_CSI.mat：完美反馈，全信道状态信息
BD_max_R_part_CSI.mat: DJSCC+基于部分信道状态信息的BD，存储BD预编码后预编码向量
NN_with_BD_part_CSI：BD+PF
origin_V_to_BD_Ms：完美反馈，使用部分CSI生成预编码矩阵
water_filling_with_BD_origin_V：完美反馈+BD+WF
water_filling_with_BD_part_CSI：DJSCC+BD+WF，使用BD_max_R_part_CSI.mat生成的预编码向量做注水并计算下行和速率


SFPNet网络测试
先训练DJSCC网络，使用ADJSCC中的DJSCC网络即可。
再训练预编码模块， 使用文件stage2_JMP.py
最后做测试，使用stage3_test_JFPNet.py