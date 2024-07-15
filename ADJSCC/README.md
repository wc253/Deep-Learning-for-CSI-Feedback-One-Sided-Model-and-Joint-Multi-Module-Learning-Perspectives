# Deep Joint Source-Channel Coding for CSI Feedback: An End-to-End Approach

**The code of "Deep Joint Source-Channel Coding for CSI Feedback: An End-to-End Approach" [[IEEE]](https://ieeexplore.ieee.org/abstract/document/9954153) proposed a attention-aided deep joint source-channel coding network fo CSI feedaback task.** If our work is helpful for your research, we would hope you give us a star and citation. Thanks!

This repository contains the implementation code for paper:

**1. Deep Joint Source-Channel Coding for CSI Feedback: An End-to-End Approach [[IEEE]](https://ieeexplore.ieee.org/abstract/document/9954153)**  
**2. Deep Learning for CSI Feedback: One-Sided Model and Joint Multi-Module Learning Perspectives [[arXiv]](https://arxiv.org/abs/2405.05522)**
## Abstract

The increased throughput brought by MIMO technology relies on the knowledge of channel state information (CSI) acquired in the base station (BS). To make the CSI feedback overhead affordable for the evolution of MIMO technology (e.g., massive MIMO and ultra-massive MIMO), deep learning (DL) is introduced to deal with the CSI compression task. In traditional communication systems, the compressed CSI bits is treated equally and expected to be transmitted accurately over the noisy channel. While the errors occur due to the limited bandwidth or low signal-to-noise ratios (SNRs), the reconstruction performance of the CSI degrades drastically. As a branch of semantic communications, deep joint source-channel coding (DJSCC) scheme performs better than the separate source-channel coding (SSCC) scheme—the cornerstone of traditional communication systems—in the limited bandwidth and low SNRs. In this paper, we propose a DJSCC based framework for the CSI feedback task. In particular, the proposed method can simultaneously learn from the CSI source and the wireless channel. Instead of truncating CSI via Fourier transform in the delay domain in existing methods, we apply non-linear transform networks to compress the CSI. Furthermore, we adopt an SNR adaption mechanism to deal with wireless channel variations. The extensive experiments demonstrate the validity, adaptability, and generality of the proposed framework.

## Dependencies
* tensorflow (>=2.5)
* numpy
* argparse
* scipy

## Install

### SSCC Network
Experimental steps:
1. Splice and normalize the small dataset of uplink CSI matrix information and store it as .npz form, the stored data includes the training dataset, validation dataset, test dataset and the normalization factors of the three.
2. Process the small data set of downlink CSI matrix information, perform domain transformation, change the CSI matrix from the null-frequency domain to the time-delay angle domain, and perform cropping, stitch the result and store it in the form of .npz after normalization, and store the data including the training data set, the validation data set, the test data set, and the normalization factor of the three.
3. Modify the dataset location to point to your own dataset (.npz file). Modify ed in CsiNet_plus.py, which is the compressed CSI dimension, and train to get CsiNet_plus_ed { }_b.h5, CsiNet_plus_ed { }_en_b.h5, CsiNet_plus_ed { }_de_b.h5
 corresponds to the file CsiNet_plus.py
4. modify the dataset location to point to your own dataset (.npz file). Modify ed, B in CsiNet_plus_Q_stage1.py, where B is the number of quantization bits. Perform training to get CsiNet_plus_Q_stage1_offset_ed{}_B{}_b.h5
 Corresponding file: CsiNet_plus_Q_stage1.py
5. Modify the dataset location to point to your own dataset (.npz file). Modify ed, B in CsiNet_plus_Q_stage2.py and train to get CsiNet_plus_Q_stage2_offset_ed{}_B{}_b.h5.
 Corresponding file: CsiNet_plus_Q_stage2.py
6. Modify ed, B in eval_CsiNet_plus_Q_encoder.py, train, get qen_ed{}_B{}.mat, store in data_qen folder.
 Corresponding file: eval_CsiNet_plus_Q_encoder.py
7. Modify the dataset location to point to your own channel dataset (.npz file). Run convert_HtoMat.py to convert the channel file from .npz format to a .mat file.
 Corresponding file: convert_HtoMat.py
8. Modify ed, B, bw_size, mod in uci_codec.m, run it, get dec_bw{ }_ed { }_ B { }_ mod { }_ snr{-10:1:10}.mat, store it in data_dec folder.
 Corresponding file: uci_codec.m
9. Modify ed, B, bw_size, mod in eval_CsiNet_plus_Q_decoder.py, run it, get bw{}_ed{}_B{}_mod{}.json and store it in data_eval folder.
 Corresponding file: eval_CsiNet_plus_Q_decoder.py

### DJSCC Network
Experimental steps:
1. Splice and normalize the small dataset of uplink CSI matrix information and store it as .npz form, the stored data includes the training dataset, validation dataset, test dataset and the normalization factors of the three.
2. Process the small dataset of downstream CSI matrix information and store the result in .npz form after splicing and normalization, the stored data include the training dataset, validation dataset, test dataset and the normalization factor of the three.
3. Modify the dataset location to point to your own dataset (.npz file). Modify ed (compressed CSI dimension), snr (received signal-to-noise ratio in dB) in DJSCC.py for training. Obtain the joint end-to-end network based on CsiNet+ for downlink CSI feedback with fixed SNR, and the corresponding test results (NMSE), which are stored in data_eval.

### ADJSCC Network
Experimental Steps:
1. The small data set of uplink CSI matrix information is spliced and normalized and stored as .npz form, the stored data includes the training data set, validation data set, test data set and the normalization factors of the three.
2. Process the small dataset of downstream CSI matrix information and store the result in .npz form after splicing and normalization, the stored data include the training dataset, validation dataset, test dataset and the normalization factor of the three.
3. Modify the dataset location to point to your own dataset (.npz file). Modify ed (compressed CSI dimension), snr (received signal-to-noise ratio in dB) in ADJSCC.py for training. Obtain the joint end-to-end network based on CsiNet+ with downlink CSI feedback at fixed SNR, and the corresponding test results (NMSE), which are stored in data_eval

### dataset
We provide a small training set test, test set and validation set for reference in the folder named "data_npz". The larger dataset can be obtained by connecting the authors of the paper.

## Citation

If you are interested in our repository and our paper, please cite the following paper:

```
@ARTICLE{9954153,
  author={Xu, Jialong and Ai, Bo and Wang, Ning and Chen, Wei},
  journal={IEEE Journal on Selected Areas in Communications}, 
  title={Deep Joint Source-Channel Coding for CSI Feedback: An End-to-End Approach}, 
  year={2023},
  volume={41},
  number={1},
  pages={260-273},
  keywords={Encoding;Task analysis;Downlink;Channel coding;Transforms;Signal to noise ratio;Uplink;CSI feedback;deep joint source-channel coding;autoencoder;deep learning},
  doi={10.1109/JSAC.2022.3221963}}


@article{guo2024deep,
  title={Deep Learning for CSI Feedback: One-Sided Model and Joint Multi-Module Learning Perspectives},
  author={Guo, Yiran and Chen, Wei and Sun, Feifei and Cheng, Jiaming and Matthaiou, Michail and Ai, Bo},
  journal={arXiv preprint arXiv:2405.05522},
  year={2024}
}
```
