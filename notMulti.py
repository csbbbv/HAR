#!/usr/bin/python
# -*- coding:utf-8 -*-
from __future__ import print_function
import time
import timeit
import numpy as np
from scipy.fftpack import fft,ifft
import scipy.stats as stats
from scipy import signal
from astropy.stats import median_absolute_deviation
from scipy.stats import iqr
from scipy.stats import entropy
from smbus2 import SMBus
import multiprocessing as mp
from multiprocessing import Queue
from datetime import datetime

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
# def getdatafromsensors():
Gyro = [0, 0, 0]
Accel = [0, 0, 0]
Mag = [0, 0, 0]
pitch = 0.0
roll = 0.0
yaw = 0.0
pu8data = [0, 0, 0, 0, 0, 0, 0, 0]
U8tempX = [0, 0, 0, 0, 0, 0, 0, 0, 0]
U8tempY = [0, 0, 0, 0, 0, 0, 0, 0, 0]
U8tempZ = [0, 0, 0, 0, 0, 0, 0, 0, 0]
GyroOffset = [0, 0, 0]
Ki = 1.0
Kp = 4.50
q0 = 1.0
q1 = q2 = q3 = 0.0
angles = [0.0, 0.0, 0.0]
true = 0x01
false = 0x00
# define ICM-20948 Device I2C address
I2C_ADD_ICM20948 = 0x68
I2C_ADD_ICM20948_AK09916 = 0x0C
I2C_ADD_ICM20948_AK09916_READ = 0x80
I2C_ADD_ICM20948_AK09916_WRITE = 0x00
# define ICM-20948 Register
# user bank 0 register
REG_ADD_WIA = 0x00
REG_VAL_WIA = 0xEA
REG_ADD_USER_CTRL = 0x03
REG_VAL_BIT_DMP_EN = 0x80
REG_VAL_BIT_FIFO_EN = 0x40
REG_VAL_BIT_I2C_MST_EN = 0x20
REG_VAL_BIT_I2C_IF_DIS = 0x10
REG_VAL_BIT_DMP_RST = 0x08
REG_VAL_BIT_DIAMOND_DMP_RST = 0x04
REG_ADD_PWR_MIGMT_1 = 0x06
REG_VAL_ALL_RGE_RESET = 0x80
REG_VAL_RUN_MODE = 0x01  # Non low-power mode
REG_ADD_LP_CONFIG = 0x05
REG_ADD_PWR_MGMT_1 = 0x06
REG_ADD_PWR_MGMT_2 = 0x07
REG_ADD_ACCEL_XOUT_H = 0x2D
REG_ADD_ACCEL_XOUT_L = 0x2E
REG_ADD_ACCEL_YOUT_H = 0x2F
REG_ADD_ACCEL_YOUT_L = 0x30
REG_ADD_ACCEL_ZOUT_H = 0x31
REG_ADD_ACCEL_ZOUT_L = 0x32
REG_ADD_GYRO_XOUT_H = 0x33
REG_ADD_GYRO_XOUT_L = 0x34
REG_ADD_GYRO_YOUT_H = 0x35
REG_ADD_GYRO_YOUT_L = 0x36
REG_ADD_GYRO_ZOUT_H = 0x37
REG_ADD_GYRO_ZOUT_L = 0x38
REG_ADD_EXT_SENS_DATA_00 = 0x3B
REG_ADD_REG_BANK_SEL = 0x7F
REG_VAL_REG_BANK_0 = 0x00
REG_VAL_REG_BANK_1 = 0x10
REG_VAL_REG_BANK_2 = 0x20
REG_VAL_REG_BANK_3 = 0x30

# user bank 1 register
# user bank 2 register
REG_ADD_GYRO_SMPLRT_DIV = 0x00
REG_ADD_GYRO_CONFIG_1 = 0x01
REG_VAL_BIT_GYRO_DLPCFG_2 = 0x10  # bit[5:3]
REG_VAL_BIT_GYRO_DLPCFG_4 = 0x20  # bit[5:3]
REG_VAL_BIT_GYRO_DLPCFG_6 = 0x30  # bit[5:3]
REG_VAL_BIT_GYRO_FS_250DPS = 0x00  # bit[2:1]
REG_VAL_BIT_GYRO_FS_500DPS = 0x02  # bit[2:1]
REG_VAL_BIT_GYRO_FS_1000DPS = 0x04  # bit[2:1]
REG_VAL_BIT_GYRO_FS_2000DPS = 0x06  # bit[2:1]
REG_VAL_BIT_GYRO_DLPF = 0x01  # bit[0]
REG_ADD_ACCEL_SMPLRT_DIV_2 = 0x11
REG_ADD_ACCEL_CONFIG = 0x14
REG_VAL_BIT_ACCEL_DLPCFG_2 = 0x10  # bit[5:3]
REG_VAL_BIT_ACCEL_DLPCFG_4 = 0x20  # bit[5:3]
REG_VAL_BIT_ACCEL_DLPCFG_6 = 0x30  # bit[5:3]
REG_VAL_BIT_ACCEL_FS_2g = 0x00  # bit[2:1]
REG_VAL_BIT_ACCEL_FS_4g = 0x02  # bit[2:1]
REG_VAL_BIT_ACCEL_FS_8g = 0x04  # bit[2:1]
REG_VAL_BIT_ACCEL_FS_16g = 0x06  # bit[2:1]
REG_VAL_BIT_ACCEL_DLPF = 0x01  # bit[0]

# user bank 3 register
REG_ADD_I2C_SLV0_ADDR = 0x03
REG_ADD_I2C_SLV0_REG = 0x04
REG_ADD_I2C_SLV0_CTRL = 0x05
REG_VAL_BIT_SLV0_EN = 0x80
REG_VAL_BIT_MASK_LEN = 0x07
REG_ADD_I2C_SLV0_DO = 0x06
REG_ADD_I2C_SLV1_ADDR = 0x07
REG_ADD_I2C_SLV1_REG = 0x08
REG_ADD_I2C_SLV1_CTRL = 0x09
REG_ADD_I2C_SLV1_DO = 0x0A

# define ICM-20948 Register  end

# define ICM-20948 MAG Register
REG_ADD_MAG_WIA1 = 0x00
REG_VAL_MAG_WIA1 = 0x48
REG_ADD_MAG_WIA2 = 0x01
REG_VAL_MAG_WIA2 = 0x09
REG_ADD_MAG_ST2 = 0x10
REG_ADD_MAG_DATA = 0x11
REG_ADD_MAG_CNTL2 = 0x31
REG_VAL_MAG_MODE_PD = 0x00
REG_VAL_MAG_MODE_SM = 0x01
REG_VAL_MAG_MODE_10HZ = 0x02
REG_VAL_MAG_MODE_20HZ = 0x04
REG_VAL_MAG_MODE_50HZ = 0x05
REG_VAL_MAG_MODE_100HZ = 0x08
REG_VAL_MAG_MODE_ST = 0x10
# define ICM-20948 MAG Register  end

MAG_DATA_LEN = 6

#####################################################################################################
#####################       计算特征    ##############################################################
####################################################################################################

def Euclidean_norm(x,y,z):
    return (x**2+y**2+z**2)**(1/3)
def CalMag(sigs):
    Mag=[]
    for i in range(len(sigs[0])):
        Mag.append(Euclidean_norm(sigs[0][i],sigs[1][i],sigs[2][i]))
    return Mag
# Mag=CalMag(sig)
# ax5.plot(x, Mag)
# ax5.set_title('Magsignal')
# ax5.axis([0, 1, -1, 1])


########Wn=2*截止频率/采样频率
#########中值滤波+三阶巴特沃斯滤波器20hz去噪###20*2/100hz
def MidFilter(sigs):
    mid = signal.medfilt(sigs)
    # sos =signal.butter(3, 0.4, 'lp', output='sos')
    # filtered = signal.sosfilt(sos, mid,axis=0)
    # b, a = signal.butter(3, 0.4, 'lowpass')
    # filtered = signal.filtfilt(b, a, mid,axis=1)#data为要过滤的信号
    sos20 = signal.butter(3, 0.4, 'lp', fs=100, output='sos')
    filtered= signal.sosfilt(sos20, sigs)
    return filtered
# ax2.plot(x, filtered)
# ax2.set_title('After 20 Hz low-pass filter')
# ax2.axis([0, 1, -1, 1])
# ax2.set_xlabel('Time [seconds]')


######三阶巴特沃斯滤波器 0.3hz分离重力加速度和身体加速度

#####tbodyacc##########
def splitGandB(sigs):
    sos2 =signal.butter(1, 0.3, 'lp', fs=100, output='sos')
    filtered1 = signal.sosfilt(sos2, sigs,axis=0)
    # ax3.plot(x, filtered1)
    # ax3.set_title('After 0.3 Hz low-pass filter')
    # ax3.axis([0, 1, -0.2, 0.2])
    # ax3.set_xlabel('Time [seconds]')
    #

    sos3=signal.butter(1, 0.3, 'hp', fs=100, output='sos')
    filtered2 = signal.sosfilt(sos3, sigs)
    # ax4.plot(x, filtered2)
    # ax4.set_title('After 0.3 Hz high-pass filter')
    # ax4.axis([0, 1, -0.2, 0.2])
    # ax4.set_xlabel('Time [seconds]')
    # plt.tight_layout()
    # plt.show()
    return filtered1,filtered2


##########Jerk##############3
def jerk(sigs):
    Jerksig=[]
    for i in range(1,len(sigs)):
        Jerksig.append((sigs[i]-sigs[i-1])/0.01)
    # print(Jerksig)
    return Jerksig

#######################
#####傅里叶##########f

def fly(sigs,x,fs):#x----linspace
    fs=2000
    yy =fft(sigs,fs)
    yf=abs(yy)                # 取绝对值
    yf1=abs(yy/len(x) )          #归一化处理
    yf2 = yf1[1:int(len(x)/2)]  #取一半区间
    dc=yf1[0]
    xf = np.arange(0,1.,1/fs)
    xf.tolist()# 频率
    xf1 = xf
    xf2 = xf[1:int(len(x)/2)]  #取一半区间
    return dc,xf2,yf2




#mean（）：平均值
def Meansig(sigs):
    m=np.mean(sigs)
    return m
#std（）：标准偏差
def std(sigs):
    sig_std = np.std(sigs, ddof=1)
    return sig_std
#中位c查
def MAD(sigs):
    mads=median_absolute_deviation(sigs)
    return mads
# max（）：数组中的最大值
def sigMAX(sigs):
    return max(sigs)
# min（）：数组中的最小值
def sigMIN(sigs):
    return min(sigs)
# sma（）：信号幅度区域###3-dims
def sigSMA(sigs):
    all=0.
    if len(sigs)==1:
        for i in sigs:
            all+=abs(i)
        all/=len(sigs)
    if len(sigs)==3:
        for i in range(len(sigs[0])):
            all+=abs(sigs[0][i])+abs(sigs[1][i])+abs(sigs[2][i])
        all/=len(sigs[0])
    return all
# energy（）：能量度量平方和除以数量。####1 dim
#####归一化#######################################################################
#####
# %% 将数据归一化到[a,b]区间的方法
# a=0.1;
# b=0.5;
# Ymax=max(y);%计算最大值
# Ymin=min(y);%计算最小值
# k=(b-a)/(Ymax-Ymin);
# norY=a+k*(y-Ymin);
####################################################################################3
def norm(sigs):
    newsig=sigs
    Max=max(sigs)
    Min=min(sigs)
    a=-1
    b=1
    k=(b-a)/(Max-Min)
    for i in range(len(newsig)):
        newsig[i]=a+k*(newsig[i]-Min)
    return newsig
def energy(sigs):

    en=0.
    for i in sigs:
        en+=i**2
    en/=len(sigs)
    return en

# iqr（）：四分位数范围
def iqrs(sigs):
    return iqr(sigs)


# entropy（）：信号熵
def sigEntropy(sigs):
    for i in range(len(sigs)):
        if sigs[i] < 0:
            sigs[i] = -sigs[i]
    return entropy(sigs,base=2)
# arCoeff（）：Burg阶等于4的自回归系数
def arCoeff(u):
    # AR, P, k = arburg(sigs, 4)
    # AR=abs(AR)
    N = len(u)
    # print("数据长度为:" + str(N))
    k = 4  # 阶数

    # 数据初始化
    fO = u[:]  # 0阶前向误差
    bO = u[:]  # 0阶反向误差
    f = u[:]  # 用于更新的误差变量
    b = u[:]
    a = np.array(np.zeros((k + 1, k + 1)))  # 模型参数初始化
    for i in range(k + 1):
        a[i][0] = 1
    # 计算P0 1/N*sum(u*2)
    P0 = 0
    for i in range(N):
        P0 += u[i] ** 2
    P0 /= N
    # print("P0:" + str(P0))
    P = [P0]

    # Burg 算法更新模型参数
    for p in range(1, k + 1):
        Ka = 0  # 反射系数的分子
        Kb = 0  # 反射系数的分母
        for n in range(p, N):
            Ka += f[n] * b[n - 1]
            Kb = Kb + f[n] ** 2 + b[n - 1] ** 2
        K = 2 * Ka / Kb
        # print("第%d阶反射系数:%f" % (p, K))
        # 更新前向误差和反向误差
        fO = f[:]
        bO = b[:]
        for n in range(p, N):
            b[n] = -K * fO[n] + bO[n - 1]
            f[n] = fO[n] - K * bO[n - 1]
        # 更新此时的模型参数
        # print("第%d阶模型参数：" % p)
        for i in range(1, p + 1):
            if (i == p):
                a[p][i] = -K
            else:
                a[p][i] = a[p - 1][i] - K * a[p - 1][p - i]
            # print("a%d=%f" % (i, a[p][i]))
        P.append((1 - K ** 2) * P[p - 1])

    # print('cor!!!!!!!!!!!!!!!!',AR[0],AR[1],AR[2],AR[3])
    return [P[0],P[1],P[2],P[3]]
# related（）：两个信号之间的相关系数
def corr(sigs):
    a,b,c=sigs[0],sigs[1],sigs[2]
    x1=np.corrcoef(a,b)[1][0]
    x2=np.corrcoef(a,c)[1][0]
    x3=np.corrcoef(b,c)[1][0]
    return [x1,x2,x3]

# maxInds（）：幅度最大的频率分量的索引###对FFT后
def maxinds(sigs,fs):
    if (type(sigs).__name__ != 'list'):
        sig=sigs.tolist()
    else:
        sig=sigs
    idx,m= sig.index(max(sig)),max(sig)
    return (0.5-(idx/fs))*m
# meanFreq（）：获得平均频率的频率分量的加权平均值###FFT后

def meanFreq(sigs):
    sum=0.
    for i in sigs:
        sum+=i
    mean=sum/len(sigs)
    tot=0.
    for i in sigs:
        tot+=i*i/mean
    return tot/len(sigs)


# kurtosis（）：频域信号的峰度
# skewness（）：频域信号的偏斜度
#FFT
def SkewAndKur(sigs):
    sk=stats.skew(sigs, axis=0, bias=True)
    ku = stats.kurtosis(sigs)
    return [abs(sk),abs(ku)]


# bandsEnergy（）：每个窗口的FFT的64个bin内的频率间隔的能量。
def bandEnergy(sigs,fs):  ####FFT
    # print(len(sig))
    inter = fs//8  #
    res = []
    for i in range(8):
        x = 0
        # for j in range(8):
            # for i in range(inter):
            #     x += sig[j * inter + i] ** 2
            # res.append(x / inter)
        for j in sigs[i*inter:inter*(i+1)]:
            x+=j
        res.append(x/inter)
    res.append((res[0]+res[1])/2)
    res.append((res[2]+res[3])/2)
    res.append((res[4]+res[5])/2)
    res.append((res[6] + res[7])/2)
    res.append((res[8]+res[9])/2)
    res.append((res[10] + res[11]) / 2)
    return res###14个数

# angle（）：矢量之间的角度。
def angle(x,y):
    if x=='X':
        x=[1,0,0]
    elif x=='Y':
        x=[0,1,0]
    elif x=='Z':
        x=[0,0,1]
    absx,absy,xy=0,0,0
    for i in range(3):
        absx+=x[i]**2
        absy+=y[i]**2
        xy+=x[i]*y[i]
    absx=absx**0.5
    absy=absy**0.5
    angle=xy/(absy*absx)
    return angle
############MEAN ###############################################
##Additional vectors obtained by averaging the signals in a ####
# signal window sample. These are used on the angle() variable:##
#################################################################
def calMEAN(sigs):#signal with 3 dims
    xyz=[]
    for i in range(3):
        xyz.append(np.mean(sigs[i]))
    return xyz




# def GyroAndAcc(dir):
#     Gyrosig,x1=generrate_signal(gyrodir)
#     Accsig,x2=generrate_signal(accdir)
#     return Gyrosig,Accsig,x2,x1
def ALLFeature(fs,tAcc,tgyro,xacc):
    FEATURE=[]
    xgro=xacc
    # tgyro,tAcc,xacc,xgro=GyroAndAcc(accdir,gyrdir)#acc,gyro
    # for i in range(3):
    #     tgyro[i]=MidFilter(tgyro[i])
    #     tAcc[i]=MidFilter(tAcc[i])
    # tBodyAcc - XYZ
    # tGravityAcc - XYZ
    tBodyAcc, tGravity,tBodyAccJerk,tBodyGyroJerk,fbodyacc,fbodyaccjerk,fbodygyro,fbodygyrojerk=[],[],[],[],[],[],[],[]
    for i in range(3):
        tgyro[i] = MidFilter(tgyro[i])
        # tAcc[i] = MidFilter(tAcc[i])
        tBodyAcc1,tGravity1=splitGandB(tAcc[i])
        tBodyAcc1, tGravity1=MidFilter(tBodyAcc1),MidFilter(tGravity1)
        tbodyAccjerk1=jerk(tBodyAcc1)
        tbodyGyroJerk1=jerk(tgyro[i])
        tBodyAcc.append(tBodyAcc1)
        tGravity.append(tGravity1)
        tBodyAccJerk.append(tbodyAccjerk1)
        tBodyGyroJerk.append(tbodyGyroJerk1)
        ###FFT
        # fBodyAcc - XYZ
        # fBodyAccJerk - XYZ
        # fBodyGyro - XYZ
        fbodyaccDC,fbodyacc_x,fbodyacc_y=fly(tBodyAcc1,xacc,fs)
        fbodyaccjerkDC,fbodyaccjerk_x,fbodyaccjerk_y=fly(tbodyAccjerk1,xacc,fs)
        fbodygyroDC,fbodygyro_x,fbodygyro_y=fly(tgyro[i],xgro,fs)
        fbodygyrojerkDC, fbodygyrojerk_x, fbodygyrojerk_y = fly(tbodyAccjerk1,xgro,fs)
        fbodygyrojerk.append(fbodygyrojerk_y)
        fbodyacc.append(fbodyacc_y)
        fbodyaccjerk.append(fbodyaccjerk_y)
        fbodygyro.append(fbodygyro_y)



    # tBodyAccMag
    # tGravityAccMag
    # tBodyAccJerkMag
    # tBodyGyroMag
    # tBodyGyroJerkMag
    tBodyAccMag=CalMag(tBodyAcc)
    tGravityAccMag=CalMag(tGravity)
    tBodyAccJerkMag=CalMag(tBodyAccJerk)
    tBodyGyroMag=CalMag(tgyro)
    tBodyGyroJerkMag=CalMag(tBodyGyroJerk)

    # fBodyAccMag
    # fBodyAccJerkMag
    # fBodyGyroMag
    # fBodyGyroJerkMag
    fBodyAccMag=CalMag(fbodyacc)
    fBodyAccJerkMag=CalMag(fbodyaccjerk)
    fBodyGyroMag=CalMag(fbodygyro)
    # fbodygyrojerk=jerk(fbodygyro)
    # print('*****************',fbodygyrojerk,'**********')
    # print(fbodygyrojerk[1],'^^^^^^^^^^^^^^^^^^^^^^^^^')
    # print(fbodygyrojerk[2],'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2')

    fBodyGyroJerkMag=CalMag(fbodygyrojerk)
########################################################################
####  [tgyro, tBodyAcc, tGravity,tBodyAccJerk,tBodyGyroJerk,fbodyacc,fbodyaccjerk,fbodygyro,fbodygyrojerk,tBodyAccMag
    # tGravityAccMag, tBodyAccJerkMag,tBodyGyroMag,tBodyGyroJerkMag,fBodyAccMag, fBodyAccJerkMag, fBodyGyroMag
#### fBodyGyroJerkMag  ]
##############################################################################
    TimefeatureList=[tBodyAcc,tGravity,tBodyAccJerk,tgyro,tBodyGyroJerk,tBodyAccMag, tGravityAccMag,tBodyAccJerkMag,tBodyGyroMag,tBodyGyroJerkMag]
    FreqfeatureList=[fbodyacc,fbodyaccjerk,fbodygyro,fBodyAccMag,fBodyAccJerkMag,fBodyGyroMag,fBodyGyroJerkMag]

    #################
    # mean（）：平均值
    # std（）：标准偏差
    # mad（）：中值绝对偏差
    # max（）：数组中的最大值
    # min（）：数组中的最小值
    # t    # sma（）：信号幅度区域
    # energy（）：能量度量平方和除以数量。
    # iqr（）：四分位数范围
    # entropy（）：信号熵
    # arCoeff（）：Burg阶等于4的自回归系数
    # related（）：两个信号之间的相关系数
    #####################

    for i in TimefeatureList:
        mean3,std3,mad3,min3,max3,sma1,energy3,iqr3,entropy3,arCoeff12,correction3,=[],[],[],[],[],[],[],[],[],[],[]
        if len(i)==3:
            for j in range(len(i)):
                mean3.append(Meansig(i[j]))
                std3.append(std(i[j]))
                mad3.append(MAD(i[j]))
                min3.append(sigMIN(i[j]))
                max3.append(sigMAX(i[j]))
                energy3.append(energy(i[j]))
                iqr3.append(iqrs(i[j]))
                entropy3.append(sigEntropy(i[j]))
                arCoeff12+=arCoeff(i[j])
            correction3+=corr(i)
            sma1.append(sigSMA(i))
            # print('mean3',len(mean3),'mean3',len(std3),'mad3',len(mad3),'mad3',len(max3),'min3',len(min3),'sma1',len(sma1),'energy3',len(energy3),'iqr',len(iqr3),'entropy',len(entropy3),'arCoeff12',len(arCoeff12),'correction3',len(correction3))
            FEATURE += mean3 + std3 + mad3 + max3 + min3 + sma1 + energy3 + iqr3 + entropy3 + arCoeff12 + correction3
            # print(len(FEATURE))
        else:
            mean3.append(Meansig(i))
            std3.append(std(i))
            mad3.append(MAD(i))
            min3.append(sigMIN(i))
            max3.append(sigMAX(i))
            sma1.append(sigSMA(i))
            energy3.append(energy(i))
            iqr3.append(iqrs(i))
            entropy3.append(sigEntropy(i))
            arCoeff12 += arCoeff(i)
            # print('mean31',len(mean3),'mean31',len(std3),'mad3',len(mad3),'mad3',len(max3),'min3',len(min3),'sma1',len(sma1),'energy3',len(energy3),'iqr',len(iqr3),'entropy',len(entropy3),'arCoeff12',len(arCoeff12))
            FEATURE += mean3 + std3 + mad3 + max3 + min3 + sma1 + energy3 + iqr3 + entropy3 + arCoeff12
            # print(len(FEATURE))

    #####################
    # maxInds（）：幅度最大的频率分量的索引
    # meanFreq（）：获得平均频率的频率分量的加权平均值
    # skewness（）：频域信号的偏斜度
    # f   # kurtosis（）：频域信号的峰度
    # bandsEnergy（）：每个窗口的FFT的64个bin内的频率间隔的能量。
#348
    #####################79*3
    for i in FreqfeatureList:
        mean33,std33,mad33,max33,min33,sma11,energy33,iqr33,entropy33,maxinds33,meanFreq33,skewnessAndKurtosis66,bandsEnergy42=[],[],[],[],[],[],[],[],[],[],[],[],[]
        if len(i)==3:
            for j in range(len(i)):
                mean33.append(Meansig(i[j]))
                std33.append(std(i[j]))
                mad33.append(MAD(i[j]))
                min33.append(sigMIN(i[j]))
                max33.append(sigMAX(i[j]))
                energy33.append(energy(i[j]))
                iqr33.append(iqrs(i[j]))
                entropy33.append(sigEntropy(i[j]))
                maxinds33+=[maxinds(i[j],fs)]
                meanFreq33.append(meanFreq(i[j]))
                skewnessAndKurtosis66+=SkewAndKur(i[j])
                bandsEnergy42+=bandEnergy(i[j],len(i[j]))
            sma11.append(sigSMA(i))
            FEATURE += mean33 + std33 + mad33 + min33 + max33 + sma11 + energy33 + iqr33 + entropy33 + maxinds33 + meanFreq33 + skewnessAndKurtosis66 + bandsEnergy42
            # print(len(FEATURE))
        else:
            mean33.append(Meansig(i))
            std33.append(std(i))
            mad33.append(MAD(i))
            min33.append(sigMIN(i))
            max33.append(sigMAX(i))
            sma11.append(sigSMA(i))
            energy33.append(energy(i))
            iqr33.append(iqrs(i))
            entropy33.append(sigEntropy(i))
            maxinds33 += [maxinds(i, fs)]
            meanFreq33.append(meanFreq(i))
            skewnessAndKurtosis66 += SkewAndKur(i)
            FEATURE+=mean33+std33+mad33+min33+max33+sma11+energy33+iqr33+entropy33+maxinds33+meanFreq33+skewnessAndKurtosis66

    # angle（）：矢量之间的角度。
    # 555
    # angle(tBodyAccMean, gravity)
    tBodyAccMean=calMEAN(tBodyAcc)
    gravity=calMEAN(tGravity)
    angle1=angle(tBodyAccMean,gravity)
    FEATURE.append(angle1)
    # 556
    # angle(tBodyAccJerkMean), gravityMean)
    tBodyAccJerkMean=calMEAN(tBodyAccJerk)
    angle2=angle(tBodyAccJerkMean,gravity)
    FEATURE.append(angle2)
    # 557
    # angle(tBodyGyroMean, gravityMean)
    tBodyGyroMean=calMEAN(tgyro)
    angle3=angle(tBodyGyroMean,gravity)
    FEATURE.append(angle3)
    # 558
    # angle(tBodyGyroJerkMean, gravityMean)
    tBodyGyroJerkMean=calMEAN(tBodyGyroJerk)
    angle4=angle(tBodyGyroJerkMean,gravity)
    FEATURE.append(angle4)
    # 559
    # angle(X, gravityMean)
    angle5=angle('X',gravity)
    FEATURE.append(angle5)
    # 560
    # angle(Y, gravityMean)
    angle6 = angle('Y', gravity)
    FEATURE.append(angle6)
    # 561
    # angle(Z, gravityMean)
    angle7 = angle('Z', gravity)
    FEATURE.append(angle7)
    FEATURE=norm(FEATURE)
    return FEATURE

####################################################################################################
###################     采集数据    #################################################################
#######################################################################################################
class ICM20948(object):
    def __init__(self, address=I2C_ADD_ICM20948):
        self._address = address
        self._bus = SMBus(1)
        bRet = self.icm20948Check()  # Initialization of the device multiple times after power on will result in a return error
        # while true != bRet:
        #   print("ICM-20948 Error\n" )
        #   time.sleep(0.5)
        # print("ICM-20948 OK\n" )
        time.sleep(0.05)  # We can skip this detection by delaying it by 500 milliseconds
        # user bank 0 register
        self._write_byte(REG_ADD_REG_BANK_SEL, REG_VAL_REG_BANK_0)
        self._write_byte(REG_ADD_PWR_MIGMT_1, REG_VAL_ALL_RGE_RESET)
        time.sleep(0.01)
        self._write_byte(REG_ADD_PWR_MIGMT_1, REG_VAL_RUN_MODE)
        # user bank 2 register
        self._write_byte(REG_ADD_REG_BANK_SEL, REG_VAL_REG_BANK_2)
        self._write_byte(REG_ADD_GYRO_SMPLRT_DIV, 0x07)
        self._write_byte(REG_ADD_GYRO_CONFIG_1,
                         REG_VAL_BIT_GYRO_DLPCFG_6 | REG_VAL_BIT_GYRO_FS_1000DPS | REG_VAL_BIT_GYRO_DLPF)
        self._write_byte(REG_ADD_ACCEL_SMPLRT_DIV_2, 0x07)
        self._write_byte(REG_ADD_ACCEL_CONFIG,
                         REG_VAL_BIT_ACCEL_DLPCFG_6 | REG_VAL_BIT_ACCEL_FS_2g | REG_VAL_BIT_ACCEL_DLPF)
        # user bank 0 register
        self._write_byte(REG_ADD_REG_BANK_SEL, REG_VAL_REG_BANK_0)
        time.sleep(0.01)
        self.icm20948GyroOffset()
        self.icm20948MagCheck()
        self.icm20948WriteSecondary(I2C_ADD_ICM20948_AK09916 | I2C_ADD_ICM20948_AK09916_WRITE, REG_ADD_MAG_CNTL2,
                                    REG_VAL_MAG_MODE_20HZ)

    def icm20948_Gyro_Accel_Read(self):
        self._write_byte(REG_ADD_REG_BANK_SEL, REG_VAL_REG_BANK_0)
        data = self._read_block(REG_ADD_ACCEL_XOUT_H, 12)
        self._write_byte(REG_ADD_REG_BANK_SEL, REG_VAL_REG_BANK_2)
        Accel[0] = (data[0] << 8) | data[1]
        Accel[1] = (data[2] << 8) | data[3]
        Accel[2] = (data[4] << 8) | data[5]
        Gyro[0] = ((data[6] << 8) | data[7]) - GyroOffset[0]
        Gyro[1] = ((data[8] << 8) | data[9]) - GyroOffset[1]
        Gyro[2] = ((data[10] << 8) | data[11]) - GyroOffset[2]
        if Accel[0] >= 32767:  # Solve the problem that Python shift will not overflow
            Accel[0] = Accel[0] - 65535
        elif Accel[0] <= -32767:
            Accel[0] = Accel[0] + 65535
        if Accel[1] >= 32767:
            Accel[1] = Accel[1] - 65535
        elif Accel[1] <= -32767:
            Accel[1] = Accel[1] + 65535
        if Accel[2] >= 32767:
            Accel[2] = Accel[2] - 65535
        elif Accel[2] <= -32767:
            Accel[2] = Accel[2] + 65535
        if Gyro[0] >= 32767:
            Gyro[0] = Gyro[0] - 65535
        elif Gyro[0] <= -32767:
            Gyro[0] = Gyro[0] + 65535
        if Gyro[1] >= 32767:
            Gyro[1] = Gyro[1] - 65535
        elif Gyro[1] <= -32767:
            Gyro[1] = Gyro[1] + 65535
        if Gyro[2] >= 32767:
            Gyro[2] = Gyro[2] - 65535
        elif Gyro[2] <= -32767:
            Gyro[2] = Gyro[2] + 65535

    def icm20948ReadSecondary(self, u8I2CAddr, u8RegAddr, u8Len):
        u8Temp = 0
        self._write_byte(REG_ADD_REG_BANK_SEL, REG_VAL_REG_BANK_3)  # swtich bank3
        self._write_byte(REG_ADD_I2C_SLV0_ADDR, u8I2CAddr)
        self._write_byte(REG_ADD_I2C_SLV0_REG, u8RegAddr)
        self._write_byte(REG_ADD_I2C_SLV0_CTRL, REG_VAL_BIT_SLV0_EN | u8Len)

        self._write_byte(REG_ADD_REG_BANK_SEL, REG_VAL_REG_BANK_0)  # swtich bank0

        u8Temp = self._read_byte(REG_ADD_USER_CTRL)
        u8Temp |= REG_VAL_BIT_I2C_MST_EN
        self._write_byte(REG_ADD_USER_CTRL, u8Temp)
        time.sleep(0.01)
        u8Temp &= ~REG_VAL_BIT_I2C_MST_EN
        self._write_byte(REG_ADD_USER_CTRL, u8Temp)

        for i in range(0, u8Len):
            pu8data[i] = self._read_byte(REG_ADD_EXT_SENS_DATA_00 + i)

        self._write_byte(REG_ADD_REG_BANK_SEL, REG_VAL_REG_BANK_3)  # swtich bank3

        u8Temp = self._read_byte(REG_ADD_I2C_SLV0_CTRL)
        u8Temp &= ~((REG_VAL_BIT_I2C_MST_EN) & (REG_VAL_BIT_MASK_LEN))
        self._write_byte(REG_ADD_I2C_SLV0_CTRL, u8Temp)

        self._write_byte(REG_ADD_REG_BANK_SEL, REG_VAL_REG_BANK_0)  # swtich bank0

    def icm20948WriteSecondary(self, u8I2CAddr, u8RegAddr, u8data):
        u8Temp = 0
        self._write_byte(REG_ADD_REG_BANK_SEL, REG_VAL_REG_BANK_3)  # swtich bank3
        self._write_byte(REG_ADD_I2C_SLV1_ADDR, u8I2CAddr)
        self._write_byte(REG_ADD_I2C_SLV1_REG, u8RegAddr)
        self._write_byte(REG_ADD_I2C_SLV1_DO, u8data)
        self._write_byte(REG_ADD_I2C_SLV1_CTRL, REG_VAL_BIT_SLV0_EN | 1)

        self._write_byte(REG_ADD_REG_BANK_SEL, REG_VAL_REG_BANK_0)  # swtich bank0

        u8Temp = self._read_byte(REG_ADD_USER_CTRL)
        u8Temp |= REG_VAL_BIT_I2C_MST_EN
        self._write_byte(REG_ADD_USER_CTRL, u8Temp)
        time.sleep(0.001)
        u8Temp &= ~REG_VAL_BIT_I2C_MST_EN
        self._write_byte(REG_ADD_USER_CTRL, u8Temp)

        self._write_byte(REG_ADD_REG_BANK_SEL, REG_VAL_REG_BANK_3)  # swtich bank3

        u8Temp = self._read_byte(REG_ADD_I2C_SLV0_CTRL)
        u8Temp &= ~((REG_VAL_BIT_I2C_MST_EN) & (REG_VAL_BIT_MASK_LEN))
        self._write_byte(REG_ADD_I2C_SLV0_CTRL, u8Temp)

        self._write_byte(REG_ADD_REG_BANK_SEL, REG_VAL_REG_BANK_0)  # swtich bank0

    def icm20948GyroOffset(self):
        s32TempGx = 0
        s32TempGy = 0
        s32TempGz = 0
        for i in range(0, 32):
            self.icm20948_Gyro_Accel_Read()
            s32TempGx += Gyro[0]
            s32TempGy += Gyro[1]
            s32TempGz += Gyro[2]
            time.sleep(0.001)
        GyroOffset[0] = s32TempGx >> 5
        GyroOffset[1] = s32TempGy >> 5
        GyroOffset[2] = s32TempGz >> 5

    def _read_byte(self, cmd):
        return self._bus.read_byte_data(self._address, cmd)

    def _read_block(self, reg, length=1):
        return self._bus.read_i2c_block_data(self._address, reg, length)

    def _read_u16(self, cmd):
        LSB = self._bus.read_byte_data(self._address, cmd)
        MSB = self._bus.read_byte_data(self._address, cmd + 1)
        return (MSB << 8) + LSB

    def _write_byte(self, cmd, val):
        self._bus.write_byte_data(self._address, cmd, val)
        time.sleep(0.0001)

    def icm20948Check(self):
        bRet = false
        if REG_VAL_WIA == self._read_byte(REG_ADD_WIA):
            bRet = true
        return bRet

    def icm20948MagCheck(self):
        self.icm20948ReadSecondary(I2C_ADD_ICM20948_AK09916 | I2C_ADD_ICM20948_AK09916_READ, REG_ADD_MAG_WIA1, 2)
        if (pu8data[0] == REG_VAL_MAG_WIA1) and (pu8data[1] == REG_VAL_MAG_WIA2):
            bRet = true
            return bRet

    # def icm20948CalAvgValue(self):
    #     MotionVal[0] = Gyro[0] / 32.8
    #     MotionVal[1] = Gyro[1] / 32.8
    #     MotionVal[2] = Gyro[2] / 32.8
    #     MotionVal[3] = Accel[0]
    #     MotionVal[4] = Accel[1]
    #     MotionVal[5] = Accel[2]
    #     MotionVal[6] = Mag[0]
    #     MotionVal[7] = Mag[1]
    #     MotionVal[8] = Mag[2]



        # with open("tryyyyyyyy.csv", "a", newline='') as csvfile:  ##use 0.0016148999999927582s
        #     writer = csv.writer(csvfile, delimiter=' ')
        #         # writer.writerow([None,None,None,None,None,None])
        #     writer = csv.writer(csvfile)
        #         # for row in rows:
        #     writer.writerow(row)
        # csvfile.close()
            #    count=0
############################################################################################################
############################################################################################################
############                                           ########################################################
###########                  网络                      ######################################################
############################################################################################################

class HARmodel(nn.Module):
    def __init__(self):
        super(HARmodel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 100, 2),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            # nn.Dropout(),
            nn.MaxPool1d(8),
            nn.Conv1d(100, 100, 1),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            # nn.Dropout()
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(100, 50, 2),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            # nn.Dropout(),
            nn.MaxPool1d(8))
        self.fc1 = nn.Linear(400, 100)
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(100,6)

    def forward(self, x):
        # input.shape:(16,1,425)
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)  # torch.Size([16, 400])
        # self.len_Linear=len(out)
        out = self.fc1(out)
        out=self.relu(out)
        out=self.fc2(out)
        return out

def restore_param():
    net1=HARmodel()

    return net1


def GetDataFromSensor(ICM20948,ACCX,ACCY,ACCZ,GYROX,GYROY,GYROZ):
    icm20948 = ICM20948()
    for i in range(100):
        # print("gettingdata..............")
        # for i in range(100):
            # print("\nSense HAT Test Program ...\n")
            # MotionVal=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

        icm20948.icm20948_Gyro_Accel_Read()
                    # icm20948.icm20948MagRead()
                    # icm20948.icm20948CalAvgValue()
        time.sleep(0.003)
                    # icm20948.imuAHRSupdate(MotionVal[0] * 0.0175, MotionVal[1] * 0.0175,MotionVal[2] * 0.0175,
                    # MotionVal[3],MotionVal[4],MotionVal[5],
                    # MotionVal[6], MotionVal[7], MotionVal[8])
                    # pitch = math.asin(-2 * q1 * q3 + 2 * q0* q2)* 57.3
                    # roll  = math.atan2(2 * q2 * q3 + 2 * q0 * q1, -2 * q1 * q1 - 2 * q2* q2 + 1)* 57.3
                    # yaw   = math.atan2(-2 * q1 * q2 - 2 * q0 * q3, 2 * q2 * q2 + 2 * q3 * q3 - 1) * 57.3
            # NowTime = datetime.now()
            # print("\r\n /-------------------------------------------------------------/ \r\n")
                    # print('\r\n Roll = %.2f , Pitch = %.2f , Yaw = %.2f\r\n'%(roll,pitch,yaw))
            # print('\r\nAcceleration:  X = %d , Y = %d , Z = %d\r\n' % (Accel[0], Accel[1], Accel[2]))
            # print('\r\nGyroscope:     X = %d , Y = %d , Z = %d\r\n' % (Gyro[0], Gyro[1], Gyro[2]))
                    # print('\r\nMagnetic:      X = %d , Y = %d , Z = %d'%((Mag[0]),Mag[1],Mag[2]))
                # row = [NowTime, Accel[0], Accel[1], Accel[2], Gyro[0], Gyro[1], Gyro[2]]
            # if ACCX.full()==True:
            #     ACCX.get()
            #     ACCY.get()
            #     ACCX.get()
            #     GYROX.get()
            #     GYROY.get()
            #     GYROZ.get()
            # try:Takedata(ACCX,ACCY,ACCZ,GYROX,GYROY,GYROZ,loc)

        # try:
        ACCX.append(Accel[0])
        ACCY.append(Accel[1])
        ACCZ.append(Accel[2])
        GYROX.append(Gyro[0])
        GYROY.append(Gyro[1])
        GYROZ.append(Gyro[2])
        # except:
    return ACCX, ACCY, ACCZ, GYROX, GYROY, GYROZ
            #     return



def Takedata(ACCX,ACCY,ACCZ,GYROX,GYROY,GYROZ):
    # while True:
    # print("claculing.......................................................................")
    ACClis,GYROlis=[[]]*3,[[]]*3

    # for i in range(100):
    ACClis[0]=ACCX
    ACClis[1] = ACCY
    ACClis[2] = ACCZ
    GYROlis[0]=GYROX
    GYROlis[1] = GYROY
    GYROlis[2] = GYROZ


        # loc.acquire()
        # print(0.00000000000000000000000000000000000001)
        # if ACCX.full():
        # for i in range(100):
        #         ACClis[0].append(ACCX.get())
        #         ACClis[1].append(ACCY.get())
        #         ACClis[2].append(ACCZ.get())
        #         GYROlis[0].append(GYROX.get())
        #         GYROlis[1].append(GYROY.get())
        #         GYROlis[2].append(GYROZ.get())
    # print(ACClis,GYROlis)
        # loc.release()
        # print(0.00000000000000000000000000000000000000002)
        #################
        ##?????????????
        # fs=50##HZ
    fs = 100
    xacc=np.linspace(0,1,100)
    features=ALLFeature(fs, ACClis, GYROlis, xacc)
    model1 = restore_param()
    model1.eval()
    model1.load_state_dict(torch.load('classi_epoch_826.pt'))
    a = []
    for i in features:
            i = float(i)
            a.append(i)
    x = np.array(a)
    x = np.array([[x]])
    x = torch.from_numpy(x).float()
    y = model1(x)
    probability = torch.nn.functional.softmax(y, dim=1)  # 计算softmax，即该图片属于各类的概率
    """
         standing 0;walk 1;laying 2;run 3；down 4;up 5
     """
    max_value, index = torch.max(probability, 1)
    index = index.numpy()  # 找到最大概率对应的索引号，该图片即为该索引号对应的类别
    NowTime = datetime.now()
    if index == 0:
            print(NowTime)
            print("standing")
            # acc+=1
    elif index == 1:
            print(NowTime)
            print("walk")
    elif index == 2:
            print(NowTime)
            print("laying")
            # acc+=1
    elif index == 3:
            print(NowTime)
            print("running")
            # acc+=1
    elif index == 4:
            print(NowTime)
            print("down")
    elif index == 5:

            print(NowTime)
            print("up")
def main(ICM20948):

    while True:
        ACCX = []
        ACCY = []
        ACCZ = []
        GYROX = []
        GYROY = []
        GYROZ = []
        ACCX1, ACCY1, ACCZ1, GYROX1, GYROY1,GYROZ1 = GetDataFromSensor(ICM20948, ACCX, ACCY, ACCZ, GYROX, GYROY, GYROZ)
        Takedata(ACCX1, ACCY1, ACCZ1, GYROX1, GYROY1, GYROZ1)

    # loc=mp.Lock()
    # print(0.0000000000000000000000000000000000000000000000001)
    # process1=mp.Process(target=GetDataFromSensor,args=(ICM20948,ACCX,ACCY,ACCZ,GYROX,GYROY,GYROZ))
    # GetDataFromSensor(ACCX,ACCY,ACCZ,GYROX,GYROY,GYROZ)
    # process2=mp.Process(target=Takedata,args=(ACCX,ACCY,ACCZ,GYROX,GYROY,GYROZ,loc))
    # process1.start()
    # process2.start()
    # process1.join()
    # time.sleep(1)
    # process2.join()


if __name__ == '__main__':
    # ACCX =[10]*100
    # ACCY =[10]*100
    # ACCZ =[10]*100
    # GYROX=[10]*100
    # GYROY=[10]*100
    # GYROZ=[10]*100

    # for i in range(100):
        # ACCX.append(0)
    #     ACCY.appen
    #     ACCZ.put(0.35)
    #     GYROX.put(0.65)
    #     GYROY.put(0.45)
    #     GYROZ.put(0.85)
    # for i in range(100):
    #     ACCX.get()
    #     print(ACCX.empty())

    main(ICM20948)

    # while True:
    #     main(ICM20948,ACCX,ACCY,ACCZ,GYROX,GYROY,GYROZ)















