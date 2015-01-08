// recognize1.cpp : Defines the entry point for the console application.
//


//#include "stdafx.h"
#include <stdio.h>
#include <time.h>

#include <iostream>
#include <cstring>
#include <fstream>
#include <vector>

#include "opencv/cv.h"
#include "opencv/highgui.h"
#include <opencv/ml.h>
#include <opencv/cxcore.h>

using namespace std;
using namespace cv;

#define CHARACTER 193  //字符特征数






//* -----------------------字符特征提取(垂特征直方向数据统计、13点法、梯度分布特征)-----------------------------------------//
// --Input：
//               IplImage *imgTest             // 字符图片
//
//-- Output:
//               int *num_t      // 字符特征
//--  Description:

//               1. 13点特征提取法:就是在字符图像中提取13个点作为特征向量。首先将图像平均分成8个方块,
//                  统计每个方块内的灰度为1的个数作为8个特征点,接下来在水平和垂直方向上分别画出三分之一和三分之二的位置,
//                  统计这些位置上的像素点作为4个特征点,最后统计图像中所有的灰度值为1的像素点个数作为1个特征,一共提取了
//                  13个特征。这种方法适应性较好,误差相对来说要小。
//
//               2. 垂直特征提取方法的算法：自左向右对图像进行逐列的扫描，统计每列白色像素的个数，然后自上而下逐行扫描，
//                  统计每行的黑色像素的个数，将统计的结果作为字符的特征向量，如果字符的宽度为 w,长度为 h,则特征向量
//                  的维数为 w+h.
//
//               3. 梯度分布特征：计算图像水平方向和竖直方向的梯度图像，然后通过给梯度图像分划不同的区域，
//                  进行梯度图像每个区域亮度值的统计，算法步骤为：
//                  <1>将字符由RGB转化为灰度，然后将图像归一化到40*20。
//                  <2>定义soble水平检测算子：x_mask=[−1,0,1;−2,0,2;–1,0,1]和竖直方向梯度检测算子y_mask=x_maskT。
//                  <3>对图像分别用mask_x和mask_y进行图像滤波得到SobelX和SobelY，下图分别代表原图像、SobelX和SobelY。
//                  <4>对滤波后的图像，计算图像总的像素和，然后划分4*2的网络，计算每个网格内的像素值的总和。
//                  <5>将每个网络内总灰度值占整个图像的百分比统计在一起写入一个向量，将两个方向各自得到的向量并在一起，组成特征向量。
//
//-------------------------------------------------------------------------*/




float num_character[CHARACTER ]={0};  //定义存放字符特征值的数组num_character

//字符特征提取函数CodeCharacter
float* CodeCharacter(IplImage *imgTest)

{
	float sumMatValue(const Mat& image); // 计算图像中像素灰度值总和

	float num_t[CHARACTER ]={0};  //定义存放字符特征值的数组num_t

	int i=0,j=0,k=0;//循环变量
	int Width = imgTest->width;//图像宽度
	int Height = imgTest->height;//图像高度
	int W = Width/4;//每小块的宽度
	int H = Height/8;//每小块的宽度


	//13点特征法:将图像平分为4*8的小块，统计每块中所有灰度值为255的点数

	for(k=0; k<32; k++)
	{
   		for(j=int(k/4)*H; j<int(k/4+1)*H; j++)
		{
			for(i=(k%4)*W;i<(k%4+1)*W;i++)
			{
			   num_t[k] += CV_IMAGE_ELEM(imgTest,uchar,j,i)/255 ;
			}
		}
 		num_t[32]+= num_t[k];  // 第33个特征：前32个特征的和作为第33个特征值,图像所有灰度值为255的点数

		num_character[k] = num_t[k];
	}
	num_character[32] = num_t[32];


	//垂直特征法：自左向右对图像进行逐列的扫描，统计每列白色像素的个数

	for(i=0;i<Width;i++)
	{
		for(j=0;j<Height;j++)
		{
			num_t[33+i] += CV_IMAGE_ELEM(imgTest,uchar,j,i)/255 ;
			num_character[33+i] = num_t[33+i];
		}
	}

	//垂直特征法：自上而下逐行扫描，统计每行的黑色像素的个数

    for(j=0;j<Height;j++)
	{
		for(i=0;i<Width;i++)
		{
			num_t[33+Width+j] += CV_IMAGE_ELEM(imgTest,uchar,j,i)/255;
			num_character[33+Width+j] = num_t[33+Width+j];
		}
	}


	//梯度分布特征

	// 计算x方向和y方向上的滤波
	float mask[3][3] = { { 1, 2, 1 }, { 0, 0, 0 }, { -1, -2, -1 } };
	Mat y_mask = Mat(3, 3, CV_32F, mask) / 8; //定义soble水平检测算子
	Mat x_mask = y_mask.t(); // 转置,定义竖直方向梯度检测算子

	//用x_mask和y_mask进行对字符图像进行图像滤波得到SobelX和SobelY
	Mat sobelX, sobelY;
	Mat image = imgTest;//IplImage * -> Mat,共享数据
	filter2D(image, sobelX, CV_32F, x_mask);
	filter2D(image, sobelY, CV_32F, y_mask);
	sobelX = abs(sobelX);
	sobelY = abs(sobelY);


	//计算图像总的像素和
	float totleValueX = sumMatValue(sobelX);
	float totleValueY = sumMatValue(sobelY);

	// 将图像划分为10*5共50个格子，计算每个格子里灰度值总和的百分比
	int m=0;
	int n=50;
	for (int i = 0; i < image.rows; i = i + 4)
	{
		for (int j = 0; j < image.cols; j = j + 4)
		{
			Mat subImageX = sobelX(Rect(j, i, 4, 4));
			num_t[33+Width+Height+m] = sumMatValue(subImageX) / totleValueX;
			num_character[33+Width+Height+m] = num_t[33+Width+Height+m];
			Mat subImageY= sobelY(Rect(j, i, 4, 4));
			num_t[33+Width+Height+n] = sumMatValue(subImageY) / totleValueY;
			num_character[33+Width+Height+n] = num_t[33+Width+Height+n];
			m++;
			n++;
		}
	}

	return num_character;
}

// 计算图像中像素灰度值总和
float sumMatValue(const Mat& image)
	{
		float sumValue = 0;
		int r = image.rows;
		int c = image.cols;
		if (image.isContinuous())
		{
			c = r*c;
			r = 1;
		}
		for (int i = 0; i < r; i++)
		{
			const uchar* linePtr = image.ptr<uchar>(i);
			for (int j = 0; j < c; j++)
			{
				sumValue += linePtr[j];
			}
		}
		return sumValue;
	}




int ImageStretchByHistogram(IplImage *src,IplImage *dst)
/*************************************************
Function:      通过直方图变换进行图像增强，将图像灰度的域值拉伸到0-255
src:               单通道灰度图像
dst:              同样大小的单通道灰度图像
*************************************************/
{
    assert(src->width==dst->width);
    double p[256],p1[256],num[256];

    std::memset(p,0,sizeof(p));
    std::memset(p1,0,sizeof(p1));
    std::memset(num,0,sizeof(num));
    int height=src->height;
    int width=src->width;
    long wMulh = height * width;

    //statistics
    for(int x=0;x<src->width;x++)
    {
        for(int y=0;y<src-> height;y++){
            uchar v=((uchar*)(src->imageData + src->widthStep*y))[x];
                num[v]++;
        }
    }
    //calculate probability
    for(int i=0;i<256;i++)
    {
        p[i]=num[i]/wMulh;
    }

    //p1[i]=sum(p[j]);  j<=i;
    for(int i=0;i<256;i++)
    {
        for(int k=0;k<=i;k++)
            p1[i]+=p[k];
    }

    // histogram transformation
    for(int x=0;x<src->width;x++)
    {
        for(int y=0;y<src-> height;y++){
            uchar v=((uchar*)(src->imageData + src->widthStep*y))[x];
                ((uchar*)(dst->imageData + dst->widthStep*y))[x]= p1[v]*255+0.5;
        }
    }
    return 0;
}


// 获取直方图
// 1. pImageData   图像数据
// 2. nWidth       图像宽度
// 3. nHeight      图像高度
// 4. nWidthStep   图像行大小
// 5. pHistogram   直方图
bool GetHistogram(unsigned char *pImageData,int nWidth,int nHeight,int nWidthStep,int *pHistogram)
{
	int i = 0;
	int j = 0;
    unsigned char *pLine = NULL;

    // 清空直方图
    std::memset(pHistogram,0,sizeof(int) * 256);
    for (pLine = pImageData, j = 0; j < nHeight; j++, pLine += nWidthStep)
    {
        for (i = 0; i < nWidth; i++)
        {
            pHistogram[pLine[i]]++;
        }
    }
    return TRUE;
}



// 大津法取阈值
// 1. pImageData   图像数据
// 2. nWidth       图像宽度
// 3. nHeight      图像高度
// 4. nWidthStep   图像行大小
// 函数返回阈值
int Otsu(unsigned char *pImageData, int nWidth, int nHeight, int nWidthStep)
{
    int    i          = 0;
    int    j          = 0;
    int    nTotal     = 0;
    int    nSum       = 0;
    int    A          = 0;
    int    B          = 0;
    double u          = 0;
    double v          = 0;
    double dVariance  = 0;
    double dMaximum   = 0;
    int    nThreshold = 0;
    int    nHistogram[256];
    // 获取直方图
    GetHistogram(pImageData, nWidth, nHeight, nWidthStep, nHistogram);
    for (i = 0; i < 256; i++)
    {
        nTotal += nHistogram[i];
        nSum   += (nHistogram[i] * i);
    }
    for (j = 0; j < 256; j++)
    {
        A = 0;
        B = 0;
        for (i = 0; i < j; i++)
        {
            A += nHistogram[i];
            B += (nHistogram[i] * i);
        }
        if (A > 0)
        {
            u = B / A;
        }
        else
        {
            u = 0;
        }
        if (nTotal - A > 0)
        {
            v = (nSum - B) / (nTotal - A);
        }
        else
        {
            v = 0;
        }
        dVariance = A * (nTotal - A) * (u - v) * (u - v);
        if (dVariance > dMaximum)
        {
            dMaximum = dVariance;
            nThreshold = j;
        }
    }
    return nThreshold;
}


IplImage *Pretreatment(IplImage *img)
{

	//灰度化
	IplImage *dst_gray = cvCreateImage(cvGetSize(img),img->depth,1);//灰度图
	//cvCvtColor(src_image,dst_gray,CV_BGR2GRAY);
	cvCopy(img, dst_gray, 0 );

	/*
	cvNamedWindow("Gray image",CV_WINDOW_AUTOSIZE);
	cvShowImage("Gray image",dst_gray);
	cvWaitKey(0);
	*/

	//灰度拉伸
	IplImage *dst_stretch=cvCreateImage(cvGetSize(dst_gray),dst_gray->depth,dst_gray->nChannels);
	ImageStretchByHistogram(dst_gray,dst_stretch);

	/*
	cvNamedWindow("Stretch image",CV_WINDOW_AUTOSIZE);
	cvShowImage("Stretch image",dst_stretch);
	cvWaitKey(0);
	*/

	// 创建二值图
	IplImage *g_pBinaryImage = cvCreateImage(cvGetSize(dst_stretch),IPL_DEPTH_8U,1);
	// 获取阈值
	unsigned int threshold= Otsu((unsigned char *)dst_stretch->imageData, dst_stretch->width, dst_stretch->height, dst_stretch->widthStep);
	//二值处理
	cvThreshold(dst_stretch, g_pBinaryImage,threshold,255,CV_THRESH_BINARY);

	/*
	cvNamedWindow("Binary image",CV_WINDOW_AUTOSIZE);
	cvShowImage("Binary image",g_pBinaryImage);
	cvWaitKey(0);
	*/

	//锐化
	IplImage *pBlur8UC1 = cvCreateImage(cvGetSize(g_pBinaryImage),IPL_DEPTH_8U,1);
	IplImage *pLablacian = cvCreateImage(cvGetSize(g_pBinaryImage),IPL_DEPTH_32F,1);
	IplImage *pEnhanced = cvCreateImage(cvGetSize(g_pBinaryImage),IPL_DEPTH_8U,1);
	cvSmooth(g_pBinaryImage,pBlur8UC1,CV_GAUSSIAN,3,3,0,0);
	cvLaplace(pBlur8UC1,pLablacian,3);
	cvConvert(pLablacian,pEnhanced);
	cvSub(pBlur8UC1,pEnhanced,pEnhanced,0);

	/*
	cvNamedWindow("Enhanced",1);
	cvShowImage("Enhanced",pEnhanced);
	cvWaitKey(0);
	*/

	//归一化处理
	int NewHeight = 40;
	int NewWidth = 20;
	IplImage* image = cvCreateImage(cvSize(NewWidth,NewHeight),IPL_DEPTH_8U,1);
	cvResize(pEnhanced,image);

	/*
	cvNamedWindow("归一化",1);
	cvShowImage("归一化",image);
	cvWaitKey(0);
	*/

	return image;
}



int main(int argc, char* argv[])
{

	static clock_t BeforeRunTime;//开始处理的时间
	clock_t UsedTime;//处理用去的时间

	BeforeRunTime = clock();

	int train_samples = 50;//每类样本的个数
    int classes = 34;//样本种类
	int nSamples = train_samples*classes;//样本总数

	char file_path[50];//存放文件的路径
    char file[255];//文件名

	Mat classmat(1,nSamples,CV_32FC1);//目标分类结果的矩阵
	IplImage* src_image;
	Mat DataCharacter(nSamples, CHARACTER, CV_32FC1,Scalar(0));//创建大小为 样本总数*特征数 的矩阵DataCharacter，用来存放字符特征

    for(int i =0; i<classes; i++)
    {
		sprintf(file_path ,"%s/samples/%d/",getcwd(NULL, 0),i);
			//"D:\\字母和数字训练样本\\%d\\",i);
		for(int j = 1; j<= train_samples; j++)
		{
			//Load file
			sprintf(file,"%s%d.png",file_path, j);
			src_image = cvLoadImage(file,0);

			classmat.at<float>(0,i*train_samples+j-1) = i;//记录目标分类结果
			// cout<<classmat.at<float>(0,i*train_samples+j-1)<<endl;

			if(!src_image)
			{
				printf("Error: Cant load image %s\n", file);
			}


			//预处理

			IplImage *pre_img = Pretreatment(src_image);

			//提取字符特征

			IplImage* img=cvCreateImage(cvGetSize(pre_img),IPL_DEPTH_8U,1);
			cvCopyImage(pre_img, img);

			float* character = CodeCharacter(img);//提取字符特征
			Mat tempMat = Mat(1, CHARACTER, CV_32FC1, character);//将特征数组转化成特征矩阵，以便于后续处理
			Mat dsttemp = DataCharacter.row(i*train_samples+j-1);
			tempMat.copyTo(dsttemp);//将每个样本的特征值作为一行存入特征矩阵中
			cout<<DataCharacter.row(i*train_samples+j-1)<<endl;
		}
	}


	CvANN_MLP bp;//创建一个3层的神经网络，其中第一层结点数为x1,第二层结点数为x2，第三层结点数为x3
	int x1 = CHARACTER;
	int x2 = 85;
	int x3 = classes;
	int layer_num[3] = { x1, x2, x3 };

	CvMat *layer_size = cvCreateMatHeader( 1, 3, CV_32S );
	cvInitMatHeader( layer_size, 1, 3, CV_32S, layer_num );
	bp.create( layer_size, CvANN_MLP::SIGMOID_SYM, 1, 1 );

	//设定神经网络训练参数
	CvANN_MLP_TrainParams params;
	params.term_crit = cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 90000, 0.00001 );



	params.train_method = CvANN_MLP_TrainParams::BACKPROP;
	params.bp_dw_scale = 0.01;
	params.bp_moment_scale = 0.05;

   	Mat outputs(1700,classes,CV_32FC1);//nSamples,classes,CV_32FC1);//目标输出矩阵

	//第i类所在的位置上的值最大为0.98，其他位置上的值较小，接近于0.02
	for( int m = 0; m <  outputs.rows; m++ )
	{
		for( int k = 0; k < outputs.cols; k++ )
        {
			if( k == classmat.at<float>(0,m) )
				outputs.at<float>(m,k) = 0.98;
            else
                outputs.at<float>(m,k) = 0.02;
        }
		cout<<outputs.row(m)<<endl;
    }

	//神经网络的训练

	//Mat sampleWeights( 1, DataCharacter.rows, CV_32FC1, Scalar::all(1) );

	Mat sampleWeights( 1, DataCharacter.rows, CV_32FC1);
	randu(sampleWeights, Scalar(0), Scalar(1));

	bp.train( DataCharacter, outputs, sampleWeights, Mat(), params );
	//bp.train( DataCharacter, outputs, Mat(), Mat(), params );
	printf(" 训练结束\n");


	//保存训练得到的权值矩阵
    //bp.save("D:\\WorkSpace\\c++\\recognize1\\NN_DATA.xml");


	/**************************************************

	识别测试

	**************************************************/


	//字符识别
	int test_samples = train_samples;
	int test_classes = classes;
	int ncounts = 1700;//nSamples;


	//定义输出矩阵
	Mat nearest(ncounts, test_classes, CV_32FC1, Scalar(0));
	//Mat nearest(nSamples, classes, CV_32FC1, Scalar(0));

	//神经网络识别
	//bp.predict(ImgCharacter, nearest);
	bp.predict(DataCharacter, nearest);


	Mat result = Mat(1, ncounts, CV_32FC1);
	for (int i=0;i<ncounts;i++)
	{
		Mat temp = Mat(1, CHARACTER, CV_32FC1);
		Mat dsttempt = nearest.row(i);
		dsttempt.copyTo(temp);
		cout<<temp.row(0)<<endl;
		Point maxLoc;
		minMaxLoc(temp, NULL, NULL, NULL, &maxLoc);
		result.at<float>(0,i) = maxLoc.x;
	}
	cout<<classmat.row(0)<<endl;
	cout<<result.row(0)<<endl;
	Mat resultd = Mat(1, ncounts, CV_32FC1);
	resultd = classmat - result;

	int countr = countNonZero(resultd);
	float rate = 1-(float)countr/ncounts;
	cout<<rate<<endl; //统计识别率

	UsedTime = (clock() - BeforeRunTime)*1000/CLOCKS_PER_SEC;
	printf("\nUsedTime: %d ms\n",UsedTime);

	return 0;
}
