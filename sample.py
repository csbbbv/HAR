import pandas as pd
import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import random
import scipy.stats as stats
from scipy import signal
from astropy.stats import median_absolute_deviation
from scipy.stats import iqr
import random
from scipy.stats import entropy
from pylab import plot, log10, linspace, axis
from spectrum import *
from datetime import datetime
import timeit
def generrate_signal(dir):
    df=pd.read_csv(dir,header=None,sep=',')#dir='Gyro_t.csv'
    L=len(df[1][1:])
    sig=[]
    for i in range(2,5):
        tem=[]
        for j in range(1,len(df[i])):
            tem.append(float(df[i][j]))
            #+random.gauss(mu,sigma)
        sig.append(tem)
    x = np.linspace(0, 1, len(df[1][1:]))
    return sig,x
    #
    # fig, (ax1, ax2,ax3,ax4,ax5) = plt.subplots(5, 1, sharex=True)
    # ax1.plot(x, sig[0])
    # ax1.set_title('sinusoids')
    # ax1.axis([0, 1, -1, 1])


####范数
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
    sos2 =signal.butter(3, 0.006, 'lp', fs=100, output='sos')
    filtered1 = signal.sosfilt(sos2, sigs,axis=0)
    # ax3.plot(x, filtered1)
    # ax3.set_title('After 0.3 Hz low-pass filter')
    # ax3.axis([0, 1, -0.2, 0.2])
    # ax3.set_xlabel('Time [seconds]')
    #

    sos3=signal.butter(3, 0.006, 'hp', fs=100, output='sos')
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


# plt.subplot(221)
# plt.plot(x[0:500],sig[0][0:500])
# plt.title('Original wave')
#
# plt.subplot(222)
# plt.plot(xf,yf,'r')
# plt.title('FFT of Mixed wave(two sides frequency range)',fontsize=7,color='#7A378B')
#
# plt.subplot(223)
# plt.plot(xf1,yf1,'g')
# plt.title('FFT of Mixed wave(normalization)',fontsize=9,color='r')
#
# plt.subplot(224)
# plt.plot(xf2,yf2,'b')
# plt.title('FFT of Mixed wave)',fontsize=10,color='#F08080')
#
#
# plt.show()
# print(dft_a)
# plt.plot(dft_a)
# plt.grid(True)
# plt.xlim(0, 15)
# plt.show()
#
# print(df)
# print(df.shape)
# print(df.info())




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
def arCoeff(sigs):
    AR, P, k = arburg(sigs, 4)
    AR=abs(AR)
    # print('cor!!!!!!!!!!!!!!!!',AR[0],AR[1],AR[2],AR[3])
    return [AR[0],AR[1],AR[2],AR[3]]
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




def GyroAndAcc(accdir,gyrodir):
    Gyrosig,x1=generrate_signal(gyrodir)
    Accsig,x2=generrate_signal(accdir)
    return Gyrosig,Accsig,x2,x1
def ALLFeature(fs,accdir,gyrdir):
    FEATURE=[]
    tgyro,tAcc,xacc,xgro=GyroAndAcc(accdir,gyrdir)#acc,gyro
    # for i in range(3):
    #     tgyro[i]=MidFilter(tgyro[i])
    #     tAcc[i]=MidFilter(tAcc[i])
    # tBodyAcc - XYZ
    # tGravityAcc - XYZ
    tBodyAcc, tGravity,tBodyAccJerk,tBodyGyroJerk,fbodyacc,fbodyaccjerk,fbodygyro,fbodygyrojerk=[],[],[],[],[],[],[],[]
    for i in range(3):
        tgyro[i] = MidFilter(tgyro[i])
        tAcc[i] = MidFilter(tAcc[i])
        tBodyAcc1,tGravity1=splitGandB(tAcc[i])
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


if __name__=='__main__':

    # f,a,b=ALLFeature1(50)
    # print(f)
    t1=timeit.Timer('ALLFeature1(100)','from __main__ import ALLFeature1')
    t=t1.timeit(1)
    print(t)