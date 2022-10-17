#include <iostream>  
#include <opencv2/opencv.hpp>  
#include <fstream> 
#include <sstream>
#include <string>  
#include <cstring>
#include <streambuf> 
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

//
//This program is inspired by an article named "Best-Buddies Similarity for Robust Template Matching CVPR2015"
//After reading this passage, I started to realize this method described in it by using OPENCV and c++.
//And here is the source code I wrote.
//"BBS" is short for "Best-Buddies Similarity", which is a useful, robust, and parameter-free similarity measure between two sets of points.
//BBS is based on counting the number of Best-Buddies Pairs (BBPs)-pairs of points in source and target sets, 
//where each point is the nearest neighbor of the other. 
//BBS has several key features that make it robust against complex geometric deformations and high levels of outliers, 
//such as those arising from background clutter and occlusions. 
//And the output of this source code on the challenging real-world dataset is amazingly precise, far beyond my previous expectation.
//

using namespace cv ;
using namespace std ;

int gamma_, pz, verbose ;

//Gaussian lowpass filter
float Gaussian[]{ 0.0277, 0.1110, 0.0277, 0.1110, 0.4452, 0.1110, 0.0277, 0.1110, 0.0277 };

//convert the image's info into a matrix and store them as a 2-dim vector list.
Mat Im2col(Mat src, int Mrows, int Mcols){ // input or template / pz / pz

    int col = 0;
    int rows = Mrows * Mcols; // 9
    int i, j, k, r ;
    int cols = ceil(src.rows / Mrows) * ceil(src.cols / Mcols); // 160 * 90 = 14400 or 7 * 15 = 105
    Mat ans(rows, cols, CV_32FC3) ;

    for ( j = 0; j * pz < src.cols; j++){
        for ( i = 0; i * pz < src.rows; i++){
            for ( k = 0; k < Mcols; k++){
                for ( r = 0; r < Mrows; r++){
                    ans.at<Vec3f>(r + k*Mrows, col)[0] = src.at<Vec3f>(i * Mcols + k, j * Mrows + r)[0] ;
                    ans.at<Vec3f>(r + k*Mrows, col)[1] = src.at<Vec3f>(i * Mcols + k, j * Mrows + r)[1] ;
                    ans.at<Vec3f>(r + k*Mrows, col)[2] = src.at<Vec3f>(i * Mcols + k, j * Mrows + r)[2] ;
                }
            }
            col++;
        }
    }
    return ans;
}

//the main code
int main(int argc, char *argv[]){

    int mode ;
    mode = 1 ;
    string TName, IName, TxtName, resultName, output_name, logT, logI ;
    Mat RESR, RESG, RESB ;
    Mat RESR2, RESG2, RESB2 ;
    verbose = 0 ;

    // Check Options
    for( int idx = 1; idx < argc; idx++ ){
        if( !strcmp( argv[idx], "-gamma" )) gamma_ = atoi( argv[++idx] ) ;
        else if( !strcmp( argv[idx], "-pz" )) pz = atoi( argv[++idx] ) ;
        else if( !strcmp( argv[idx], "-tmp" )) TName = string( argv[++idx] ) ;
        else if( !strcmp( argv[idx], "-i" )) IName = string( argv[++idx] ) ;
        else if( !strcmp( argv[idx], "-txt" )) TxtName = string( argv[++idx] ) ;
        else if( !strcmp( argv[idx], "-res" )) resultName = string( argv[++idx] ) ;
        else if( !strcmp( argv[idx], "-log" )) output_name = string( argv[++idx] ) ;
        else if( !strcmp( argv[idx], "-logT" )) logT = string( argv[++idx] ) ;
        else if( !strcmp( argv[idx], "-logI" )) logI = string( argv[++idx] ) ;
        else if( !strcmp( argv[idx], "-v" )) verbose = atoi( argv[++idx] ) ;
        else if( !strcmp( argv[idx], "-mode" )) mode = atoi( argv[++idx] ) ;
    }

    Mat Ts = imread(TName);
    cout << TName << endl ;

    Mat Is = imread(IName);
    cout << IName << endl ;

    vector<int> Tcut(4, 0) ;
    ifstream input(TxtName);
    if(mode == 1) input >> Tcut[0] >> Tcut[1] >> Tcut[2] >> Tcut[3] ;
    if(mode == 0){
        string temp, temp2;
        getline(input, temp);
        istringstream ss(temp);
        int i = 0;
        do{
            ss >> Tcut[i++] ;
        }while (getline(ss, temp2, ',')) ;
    }

    cout << TxtName << endl ;
    cout << Tcut[0] << " " << Tcut[1] << " " << Tcut[2] << " " << Tcut[3] << endl ;

    //clipping pictures
    if ((Tcut[2] % pz) < (pz / 2)) Tcut[2] -= (Tcut[2] % pz) ;
    else Tcut[2] += (pz - Tcut[2] % pz) ;
    if ((Tcut[3] % pz) < (pz / 2)) Tcut[3] -= (Tcut[3] % pz) ;
    else Tcut[3] += (pz - Tcut[3] % pz) ;

    cout << Tcut[0] << " " << Tcut[1] << " " << Tcut[2] << " " << Tcut[3] << endl ;

    Mat T ;
    if(mode == 0) T = Ts(Rect(Tcut[0], Tcut[1], Tcut[2], Tcut[3]));
    if(mode == 1) T = Ts ;
    Mat I = Is(Rect(0, 0, (Is.cols - Is.cols % pz), (Is.rows - Is.rows % pz)));

    //imwrite(logI, I) ;
    //imwrite(logT, T) ;

    T.convertTo(T, CV_32FC3, 1.0 / 255.0);
    I.convertTo(I, CV_32FC3, 1.0 / 255.0);

    Mat TMat = Im2col(T, pz, pz);
    Mat IMat = Im2col(I, pz, pz);

    int N = TMat.cols; // 105
    int rowT = T.rows; // 45
    int colT = T.cols; // 21
    int rowI = I.rows; // 270
    int colI = I.cols; // 480

    //pre compute spatial distance component
    vector<vector<float>> Dxy, Drgb, Drgb_prev, D, D_r, BBS;
    Dxy.resize(N);
    Drgb_prev.resize(N);
    Drgb.resize(N);
    D.resize(N);
    D_r.resize(N);

    for (int i = 0; i < N; i++){
        Dxy[i].resize(N);
        Drgb[i].resize(N);
        Drgb_prev[i].resize(N);
        D[i].resize(N);
        D_r[i].resize(N);
    }

    BBS.resize(rowI);
    for (int i = 0; i < static_cast<int>(BBS.size()); i++) BBS[i].resize(colI);

    //Drgb's buffer
    vector<vector<vector<float>>> Drgb_buffer;
    Drgb_buffer.resize(N);
    int bufSize = rowI - rowT ; // 270 - 45 = 225

    for (int i = 0; i < static_cast<int>(Drgb_buffer.size()); i++){
        Drgb_buffer[i].resize(N) ;
        for (int j = 0; j < static_cast<int>(Drgb_buffer[i].size()); j++) Drgb_buffer[i][j].resize(bufSize) ;
    }

    vector<float> xx, yy;
    for (int i = 0; (pz * i) < colT; i++){
        float n = pz * i * 3.0039;
        for (int j = 0; (pz * j) < rowT; j++){
            float m = pz * j * 0.0039;
            xx.push_back(n);
            yy.push_back(m);
        }
    }

    for (int j = 0; j < static_cast<int>(xx.size()); j++){
        for (int i = 0; i < static_cast<int>(xx.size()); i++){
            Dxy[i][j] = pow((xx[i] - xx[j]), 2) + pow((yy[i] - yy[j]), 2);
        }
    }

    vector<vector<int>> IndMat;
    IndMat.resize(I.rows / pz); // 90
    for (int i = 0; i < static_cast<int>(IndMat.size()); i++) IndMat[i].resize(I.cols / pz) ; // 160

    int n = 0;
    for (int j = 0; j < (I.cols / pz); j++){
        for (int i = 0; i < (I.rows / pz); i++){
            IndMat[i][j] = n++ ; // Nmax is 14399
        }
    }

    // loop over image pixels
    cout << "log : " << colI << " " << rowI << endl ;
    cout << "log : " << colT << " " << rowT << endl ;

    std::chrono::system_clock::time_point start1, end1 ;
    start1 = std::chrono::system_clock::now() ;

    //#pragma omp parallel for
    for (int coli = 0; coli < (colI / pz - colT / pz + 1); coli++){ // 154
        for (int rowi = 0; rowi < (rowI / pz - rowT / pz + 1); rowi++){ // 76
            Mat PMat(9, N, CV_32FC3);
            vector<int> v;
            vector<float> w;
            for (int j = coli; j < (coli + colT / pz); j++)
            {
                for (int i = rowi; i < (rowi + rowT / pz); i++)
                {
                    v.push_back(IndMat[i][j]);
                }
            }
            int ptv = 0;
            for (int ix = 0; ix < N; ix++)
            {
                for (int jx = 0; jx < 9; jx++)
                {
                    PMat.at<Vec3f>(jx, ix)[0] = IMat.at<Vec3f>(jx, v[ptv])[0];
                    PMat.at<Vec3f>(jx, ix)[1] = IMat.at<Vec3f>(jx, v[ptv])[1];
                    PMat.at<Vec3f>(jx, ix)[2] = IMat.at<Vec3f>(jx, v[ptv])[2];
                }
                ptv++;
            }

            //compute distance matrix
            for (int idxP = 0; idxP < N; idxP++)
            {
                Mat Temp(9, N, CV_32FC3);
                for (int i = 0; i < Temp.cols; i++)
                {
                    for (int j = 0; j < Temp.rows; j++)
                    {
                        Temp.at<Vec3f>(j, i)[0] = pow(((-TMat.at<Vec3f>(j, i)[0] + PMat.at<Vec3f>(j, idxP)[0])*Gaussian[j]), 2);
                        Temp.at<Vec3f>(j, i)[1] = pow(((-TMat.at<Vec3f>(j, i)[1] + PMat.at<Vec3f>(j, idxP)[1])*Gaussian[j]), 2);
                        Temp.at<Vec3f>(j, i)[2] = pow(((-TMat.at<Vec3f>(j, i)[2] + PMat.at<Vec3f>(j, idxP)[2])*Gaussian[j]), 2);
                    }
                }
                for (int jx = 0; jx < N; jx++)
                {
                    float res = 0;
                    for (int ix = 0; ix < 9; ix++)
                    {
                        if (D[ix][jx] < 1e-4) D[ix][jx] = 0;
                        res += Temp.at<Vec3f>(ix, idxP)[0];
                        res += Temp.at<Vec3f>(ix, idxP)[1];
                        res += Temp.at<Vec3f>(ix, idxP)[2];
                    }
                    Drgb[jx][idxP] = res;
                }
            }

            //make the reversed matrix of distance matrix
            for (int ix = 0; ix < N; ix++)
            {
                for (int jx = 0; jx < N; jx++)
                {
                    //calculate distance
                    D[ix][jx] = Dxy[ix][jx] * gamma_ + Drgb[ix][jx];
                    if (D[ix][jx] < 1e-4) D[ix][jx] = 0;
                    D_r[jx][ix] = D[ix][jx];
                }
            }

            //compute the BBS value of this point
            vector<float> minVal1, minVal2;
            vector<int> idx1, idx2, ii1, ii2;

            for (int ix = 0; ix < N; ix++)
            {
                auto min1 = min_element(begin(D[ix]), end(D[ix]));
                minVal1.push_back(*min1);
                idx1.push_back(distance(begin(D[ix]), min1));

                ii1.push_back((ix * N) + idx1[ix]);
            }
            for (int ix = 0; ix < N; ix++)
            {
                auto min2 = min_element(begin(D_r[ix]), end(D_r[ix]));
                minVal2.push_back(*min2);
                idx2.push_back(distance(begin(D_r[ix]), min2));
                ii2.push_back((ix * N) + idx2[ix]);
            }

            vector<vector<int>> IDX_MAT1, IDX_MAT2;
            IDX_MAT1.resize(N);
            IDX_MAT2.resize(N);
            for (int i = 0; i < N; i++)
            {
                IDX_MAT1[i].resize(N);
                IDX_MAT2[i].resize(N);
            }
            int sum, sum2, pt1, pt2 ;
            sum = sum2 = pt1 = pt2 = 0 ;
            for (int ix = 0; ix < N; ix++)
            {
                for (int jx = 0; jx < N; jx++)
                {
                    IDX_MAT1[ix][jx] = 0;
                    IDX_MAT2[ix][jx] = 999;
                    if ((pt1 < N) && ((ix * N + jx) == ii1[pt1]))
                    {
                        IDX_MAT1[ix][jx] = 1;
                        pt1++;
                    }
                    if ((pt2 < N) && ((jx * N + ix) == ii2[pt2]))
                    {
                        IDX_MAT2[ix][jx] = 1;
                        pt2++;
                    }
                    if (IDX_MAT2[ix][jx] == IDX_MAT1[ix][jx])
                        sum += 1;
                }
            }	
            BBS[rowi][coli] = sum;
            //cout << coli << " " << rowi << endl ;
        }
    }

    end1 = std::chrono::system_clock::now() ;
    const double time = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count() *0.001 ;
    cout << "time " << time << " [msec]" << endl ;

    float max = BBS[0][0] ;
    int markRow, markCol ;
    markCol = markRow = 0 ;

    // search max score
    for (int i = 0; i < rowI; i++){
        for (int j = 0; j < colI; j++){
            if (BBS[i][j] >= max){
                max = BBS[i][j];
                markRow = i ;
                markCol = j ;
            }
        }
    }

    if(verbose){
        //Initialize the output iamge and .txt files
        cout << output_name << endl ;

        ofstream output(output_name);
        output << markRow * pz << " " << markCol * pz << endl ;
        output.close();
    }

    Mat OUTPUT1, OUTPUT2, OUTPUT3 ;
    Mat Is2, Ts2 ;
    if(mode == 1) Is2 = imread(IName, 1) ;
    if(mode == 0){
        Ts2 = imread(TName, 1) ;
        Is2 = imread(IName, 1) ;
    }
    RESR = cv::Mat_<uchar>(rowI, colI) ;
    RESG = cv::Mat_<uchar>(rowI, colI) ;
    RESB = cv::Mat_<uchar>(rowI, colI) ;
    RESR2 = cv::Mat_<uchar>(rowI, colI) ;
    RESG2 = cv::Mat_<uchar>(rowI, colI) ;
    RESB2 = cv::Mat_<uchar>(rowI, colI) ;

    for( int j = 0 ; j < rowI ; j++ ) {
        for( int i = 0 ; i < colI ; i++ ) {
            if(mode == 0){
                RESR.at<uchar>(j,i) = Is2.at<cv::Vec3b>(j,i)[2] ;
                RESG.at<uchar>(j,i) = Is2.at<cv::Vec3b>(j,i)[1] ;
                RESB.at<uchar>(j,i) = Is2.at<cv::Vec3b>(j,i)[0] ;
                RESR2.at<uchar>(j,i) = Ts2.at<cv::Vec3b>(j,i)[2] ;
                RESG2.at<uchar>(j,i) = Ts2.at<cv::Vec3b>(j,i)[1] ;
                RESB2.at<uchar>(j,i) = Ts2.at<cv::Vec3b>(j,i)[0] ;
            }else{
                RESR.at<uchar>(j,i) = Is2.at<cv::Vec3b>(j,i)[2] ;
                RESG.at<uchar>(j,i) = Is2.at<cv::Vec3b>(j,i)[1] ;
                RESB.at<uchar>(j,i) = Is2.at<cv::Vec3b>(j,i)[0] ;
                RESR2.at<uchar>(j,i) = Is2.at<cv::Vec3b>(j,i)[2] ;
                RESG2.at<uchar>(j,i) = Is2.at<cv::Vec3b>(j,i)[1] ;
                RESB2.at<uchar>(j,i) = Is2.at<cv::Vec3b>(j,i)[0] ;
            }
        }
    }

    //mark rectangle
    int si, sj, ei, ej ;
    si  = markCol * pz ;
    sj  = markRow * pz ;
    ei  = si + colT ;
    ej  = sj + rowT ;

    //Rect-Yellow
    cv::rectangle(RESR,cv::Point(si - 1, sj - 1),cv::Point(ei + 1, ej + 1),cv::Scalar(255), 1) ;
    cv::rectangle(RESG,cv::Point(si - 1, sj - 1),cv::Point(ei + 1, ej + 1),cv::Scalar(255), 1) ;
    cv::rectangle(RESB,cv::Point(si - 1, sj - 1),cv::Point(ei + 1, ej + 1),cv::Scalar(0), 1) ;
    cv::rectangle(RESR,cv::Point(si - 2, sj - 2),cv::Point(ei + 2, ej + 2),cv::Scalar(255), 1) ;
    cv::rectangle(RESG,cv::Point(si - 2, sj - 2),cv::Point(ei + 2, ej + 2),cv::Scalar(255), 1) ;
    cv::rectangle(RESB,cv::Point(si - 2, sj - 2),cv::Point(ei + 2, ej + 2),cv::Scalar(0), 1) ;

    //Rect-Blue
    cv::rectangle(RESR,cv::Point(si, sj),cv::Point(ei, ej),cv::Scalar(50), 1) ;
    cv::rectangle(RESG,cv::Point(si, sj),cv::Point(ei, ej),cv::Scalar(255), 1) ;
    cv::rectangle(RESB,cv::Point(si, sj),cv::Point(ei, ej),cv::Scalar(0), 1) ;
    
    //Rect-Yellow
    cv::rectangle(RESR,cv::Point(si + 1, sj + 1),cv::Point(ei - 1, ej - 1),cv::Scalar(255), 1) ;
    cv::rectangle(RESG,cv::Point(si + 1, sj + 1),cv::Point(ei - 1, ej - 1),cv::Scalar(255), 1) ;
    cv::rectangle(RESB,cv::Point(si + 1, sj + 1),cv::Point(ei - 1, ej - 1),cv::Scalar(0), 1) ;
    cv::rectangle(RESR,cv::Point(si + 2, sj + 2),cv::Point(ei - 2, ej - 2),cv::Scalar(255), 1) ;
    cv::rectangle(RESG,cv::Point(si + 2, sj + 2),cv::Point(ei - 2, ej - 2),cv::Scalar(255), 1) ;
    cv::rectangle(RESB,cv::Point(si + 2, sj + 2),cv::Point(ei - 2, ej - 2),cv::Scalar(0), 1) ;

    vector<Mat> color_img1;
    color_img1.push_back(RESB);
    color_img1.push_back(RESG);
    color_img1.push_back(RESR);
    merge(color_img1, OUTPUT1);

    si  = Tcut[0] ;
    sj  = Tcut[1] ;
    ei  = Tcut[0] + Tcut[2] ;
    ej  = Tcut[1] + Tcut[3] ;

    //Rect-Yellow
    cv::rectangle(RESR2,cv::Point(si - 1, sj - 1),cv::Point(ei + 1, ej + 1),cv::Scalar(255), 1) ;
    cv::rectangle(RESG2,cv::Point(si - 1, sj - 1),cv::Point(ei + 1, ej + 1),cv::Scalar(255), 1) ;
    cv::rectangle(RESB2,cv::Point(si - 1, sj - 1),cv::Point(ei + 1, ej + 1),cv::Scalar(0), 1) ;
    cv::rectangle(RESR2,cv::Point(si - 2, sj - 2),cv::Point(ei + 2, ej + 2),cv::Scalar(255), 1) ;
    cv::rectangle(RESG2,cv::Point(si - 2, sj - 2),cv::Point(ei + 2, ej + 2),cv::Scalar(255), 1) ;
    cv::rectangle(RESB2,cv::Point(si - 2, sj - 2),cv::Point(ei + 2, ej + 2),cv::Scalar(0), 1) ;

    //Rect-Red
    cv::rectangle(RESR2,cv::Point(si, sj),cv::Point(ei, ej),cv::Scalar(0), 1) ;
    cv::rectangle(RESG2,cv::Point(si, sj),cv::Point(ei, ej),cv::Scalar(0), 1) ;
    cv::rectangle(RESB2,cv::Point(si, sj),cv::Point(ei, ej),cv::Scalar(255), 1) ;
    
    //Rect-Yellow
    cv::rectangle(RESR2,cv::Point(si + 1, sj + 1),cv::Point(ei - 1, ej - 1),cv::Scalar(255), 1) ;
    cv::rectangle(RESG2,cv::Point(si + 1, sj + 1),cv::Point(ei - 1, ej - 1),cv::Scalar(255), 1) ;
    cv::rectangle(RESB2,cv::Point(si + 1, sj + 1),cv::Point(ei - 1, ej - 1),cv::Scalar(0), 1) ;
    cv::rectangle(RESR2,cv::Point(si + 2, sj + 2),cv::Point(ei - 2, ej - 2),cv::Scalar(255), 1) ;
    cv::rectangle(RESG2,cv::Point(si + 2, sj + 2),cv::Point(ei - 2, ej - 2),cv::Scalar(255), 1) ;
    cv::rectangle(RESB2,cv::Point(si + 2, sj + 2),cv::Point(ei - 2, ej - 2),cv::Scalar(0), 1) ;

    vector<Mat> color_img2 ;
    color_img2.push_back(RESB2);
    color_img2.push_back(RESG2);
    color_img2.push_back(RESR2);
    merge(color_img2, OUTPUT2);

    hconcat(OUTPUT1, OUTPUT2, OUTPUT3);
    cout << resultName << endl << endl ;
    imwrite(resultName, OUTPUT3);

    return 0 ;
}