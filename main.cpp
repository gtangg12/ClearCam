/**
   main.cpp takes a pair of hazy stereo images, computes the dispairty
   map using 4-state dynammic programming, and runs our proposed post-processing
   algorithm that enhances disparity map quality, which degrades in haze.

   @author George Tang
 */

#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
using namespace std;

/**
   Image processing constants:

   @const SZ, WS match function patch dim (WS = 2*SZ+1)
   @const ROWS, COLS image dim
   @const FMAX max dispairty constant (COLS/4 is sufficent)
   @const WDTH num of cols involved in processing (COLS-2*SZ)
   @const INF not INT_MAX b/c need to support addition
 */
#define SZ 2
#define WS 5
#define ROWS 370
#define COLS 410
#define FMAX 64
#define WDTH 406
#define INF 100000.0

/**
   toArray writes opencv color image to 3-channel image array
   toImage writes 3-channel image array to opencv color image
   norm normalizes 1-channel image array
 */
void toArray(float (&dst)[ROWS][COLS][3], cv::Mat &src) {
	for (int i=0; i<ROWS; i++)
		for (int j=0; j<COLS; j++) {
			cv::Vec3b& color = src.at<cv::Vec3b>(i, j);
			for (int k=0; k<3; k++)
				dst[i][j][k] = color[k];
		}
}

void toImage(float (&src)[ROWS][COLS][3], cv::Mat dst) {
	for (int i=0; i<ROWS; i++)
		for (int j=0; j<COLS; j++) {
			cv::Vec3b& color = dst.at<cv::Vec3b>(i, j);
			for (int k=0; k<3; k++)
				color[k] = src[i][j][k];
		}
}

void norm(float (&img)[ROWS][COLS]) {
	float mx = 0;
	for (int i=0; i<ROWS; i++)
		for (int j=0; j<COLS; j++)
			mx = max(mx, img[i][j]);
	for (int i=0; i<ROWS; i++)
		for (int j=0; j<COLS; j++)
			img[i][j]=max(0.0, min(255.0, img[i][j]*255.0));
}

/**
   mean computes mean of crop
   stdv computes standard deviation of the crop given mean

   @param r, c top left corner of crop
   @param ch channel of crop
   @param m mean of crop
 */
float mean(float (&img)[ROWS][COLS][3], int r, int c, int ch) {
	float sum = 0;
	for (int i=r; i<r+WS; i++)
		for (int j=c; j<c+WS; j++)
			sum+=img[i][j][ch];
	return sum/(WS*WS);
}

float stdv(float (&img)[ROWS][COLS][3], float m, int r, int c, int ch) {
	float sum = 0;
	for (int i=r; i<r+WS; i++)
		for (int j=c; j<c+WS; j++)
			sum+=pow(img[i][j][ch]-m, 2);
	return sqrt(sum/(WS*WS));
}

/**
   Match function. Computes dissimilarity score between two crops in the form of
   0.5*(1-NCC), where NCC is the normalized cross-corelation

   @param img_a, img_b two images involved
   @param ra, ca top left corner of crop from img_a
   @param rb, cb top left corner of crop from img_b
   @param ch channel of crops
   @return value in [0, 1], 1 indicates maximum dissimilarity
 */
float dncc(float (&img_a)[ROWS][COLS][3], float (&img_b)[ROWS][COLS][3], int ra, int ca, int rb, int cb, int ch) {
	float p1m, p2m, product;
	p1m = mean(img_a, ra, ca, ch);
	p2m = mean(img_b, rb, cb, ch);
	for (int i=0; i<WS; i++)
		for (int j=0; j<WS; j++)
			product+=(img_a[ra+i][ca+j][ch]-p1m)*(img_b[rb+i][cb+j][ch]-p2m);
	product/=(WS*WS);
	float stds = stdv(img_a, p1m, ra, ca, ch)*stdv(img_b, p2m, rb, cb, ch);
	if (stds == 0)
		return 0;
	product/=stds;
	return 0.5*(1.0-product);
}

/**
   cost stores the match scores of patches with top left corner at the pixels (r-SZ, c)
   and (r-SZ, c+d) corresponding to cost array indicies (r, c, 0) and (r, c, d).
   Each r defines a 2D matrix of size (WDTH+1, FMAX+1) called a disparity space
   image (DSI). Note r in [0, SZ), [ROWS-SZ, ROWS) are not used as patch would be
   out of bounds.
 */
float cost[ROWS][WDTH+1][FMAX+1];

/**
   Given row that defines a DSI, computes and stores match score of every pixel
   with top left corner (r-SZ, c) with their possible matching pixels (r-SZ, c+d)
   by averaging dncc across all channels.

   @param img_a, img_b two images involved
   @param r given row in [SZ, ROWS-SZ) that defines DSI
 */
void computeCost(float (&img_a)[ROWS][COLS][3], float (&img_b)[ROWS][COLS][3], int r) {
	for (int i=0; i<=WDTH; i++)
		for (int j=0; j<=FMAX; j++) {         // col1 = s+i, col2 = min(s+i+j, COLS-s)
			if (i==0 || j==0)
				continue;
			if (2*SZ+i+j>COLS)
				continue;
			cost[r][i][j] = dncc(img_a, img_b, r-SZ, i, r-SZ, i+j, 0)+
			                dncc(img_a, img_b, r-SZ, i, r-SZ, i+j, 1)+
			                dncc(img_a, img_b, r-SZ, i, r-SZ, i+j, 2);
			cost[r][i][j]/=3.0;
		}
}

/**
   @const RW, CW defines (2*RW+1, 2*CW+1) gaussian blur convolution
   gr, gc contain the values of the seperable convolution
   costblur stores the result after reducing streaking by passing a gaussian blur
   along the diagonal of stacked DSIs in cost
 */
#define RW 3
#define CW 1
float gr[2*RW+1] = {0.106595, 0.140367, 0.165569, 0.174938, 0.165569, 0.140367, 0.106595};
float gc[2*CW+1] = {0.319466, 0.361069, 0.319466};
float costblur[ROWS][WDTH+1][FMAX+1];

/**
   Given diagonal of stacked DSIs stored in cost defined by row, applies seperable
   guassian blur to diagonal to reduce streaking

   @param r given row that defines diagonal
 */
void blur(int r) {
	float temp[WDTH+1][FMAX+1];
	for (int k=0; k<=FMAX; k++) {
		for (int j=0; j<=WDTH; j++) {
			float sum = 0;
			for (int i=-RW; i<=RW; i++) {
				int cr = r+i;
				if (cr<0 || cr>=ROWS)
					continue;
				sum+=gr[i+RW]*cost[cr][j][k];
			}
			temp[j][k] = sum;
		}
		for (int j=0; j<=WDTH; j++) {
			float sum = 0;
			for (int i=-CW; i<=CW; i++) {
				int cc = j+i;
				if (cc<0 || cc>WDTH)
					continue;
				sum+=gc[i+CW]*temp[cc][k];
			}
			costblur[r][j][k] = sum;
		}
	}
}

/**
   @const A, B, Y parameters for 4-state dynammic programming
   dp is the tabulation array. The 4 channels are LO, LM, RM, RO.
 */
#define A 0.7
#define B 1.0
#define Y 0.25
float dp[4][ROWS][WDTH+1][FMAX+1];

/**
   Given row that defines DSI, performs 4-state dyanmmic programming and reconstructs
   the min-cost path to get the disparity values for given row

   @param dis disparity map to write to
   @param r given row that defines DSI
 */
void reconstruct(float (&dis)[ROWS][COLS], int r) {
	int par[4][WDTH+1][FMAX+1];
	memset(par, 0, sizeof(int)*4*(WDTH+1)*(FMAX+1));
	for (int i=0; i<=WDTH; i++)
		for (int j=0; j<=FMAX; j++) {         // col1 = s+i, col2 = min(s+i+j, COLS-s)
			if (i==0 || j==0) {
				for (int m=0; m<4; m++) {
					dp[m][r][i][j] = (m==3 && j==0) ? A*i : INF;
					par[m][i][j] = -1;
				}
				continue;
			}
			if (2*SZ+i+j>COLS)
				continue;
			for (int m=0; m<4; m++)
				dp[m][r][i][j] = INF;
			float cmin, min1, min2, min3, min4;
			int mx = (j+1>FMAX) ? 2 : 4;
			for (int m=0; m<mx; m++) {
				if (m==0) {
					min1 = dp[0][r][i][j-1]+A;
					min2 = dp[1][r][i][j-1]+B;
					min3 = dp[2][r][i][j-1]+B;
					cmin = min(min1, min(min2, min3));
					if (cmin == min1) par[m][i][j] = 0;
					if (cmin == min2) par[m][i][j] = 1;
					if (cmin == min3) par[m][i][j] = 2;
				}
				if (m==1) {
					min1 = dp[0][r][i][j-1]+B;
					min2 = dp[1][r][i][j-1]+Y;
					min3 = dp[2][r][i][j-1];
					min4 = dp[3][r][i][j-1]+B;
					cmin = min(min1, min(min2, min(min3, min4)));
					if (cmin == min1) par[m][i][j] = 0;
					if (cmin == min2) par[m][i][j] = 1;
					if (cmin == min3) par[m][i][j] = 2;
					if (cmin == min4) par[m][i][j] = 3;
					cmin+=costblur[r][i][j];
				}
				if (m==2) {
					min1 = dp[3][r][i-1][j+1]+B;
					min2 = dp[2][r][i-1][j+1]+Y;
					min3 = dp[1][r][i-1][j+1];
					min4 = dp[0][r][i-1][j+1]+B;
					cmin = min(min1, min(min2, min(min3, min4)));
					if (cmin == min1) par[m][i][j] = 13;
					if (cmin == min2) par[m][i][j] = 12;
					if (cmin == min3) par[m][i][j] = 11;
					if (cmin == min4) par[m][i][j] = 10;
					cmin+=costblur[r][i][j];
				}
				if (m==3) {
					min1 = dp[3][r][i-1][j+1]+A;
					min2 = dp[2][r][i-1][j+1]+B;
					min3 = dp[1][r][i-1][j+1]+B;
					cmin = min(min1, min(min2, min3));
					if (cmin == min1) par[m][i][j] = 13;
					if (cmin == min2) par[m][i][j] = 12;
					if (cmin == min3) par[m][i][j] = 11;
				}
				dp[m][r][i][j] = cmin;
			}
		}
	int m=3, cr = WDTH-1, cc = 1;
	while(true) {
		if (par[m][cr][cc] == -1)
			break;
		int p = par[m][cr][cc];
		if (p%10==1 || p%10==2)
			dis[r][SZ+cr] = (float)cc/FMAX;
		if (p<10) {
			cc--;
			m = p;
		}
		else {
			p%=10;
			cr--;
			cc++;
			m = p;
		}
	}
}

/**
   TODO: implement process on CUDA

   Computes DSIs, blurs DSIs, and then executes 4-state dynammic programming. Each
   step is embrassingly parallel.

   @param img_a, img_b two images involved
   @param dis disparity map to write to
 */
void dismap(float (&img_a)[ROWS][COLS][3], float (&img_b)[ROWS][COLS][3], float (&dis)[ROWS][COLS]) {
	memset(cost, 0, sizeof(float)*ROWS*(WDTH+1)*(FMAX+1));
	memset(costblur, 0, sizeof(float)*ROWS*(WDTH+1)*(FMAX+1));
	for (int i=SZ; i<ROWS-SZ; i++)
		computeCost(img_a, img_b, i);
	for (int i=SZ; i<ROWS-SZ; i++)
		blur(i);
	for (int i=SZ; i<ROWS-SZ; i++)
		reconstruct(dis, i);
}

/**
   @const MAXC max number of components able to handle
   @const K max size of row/cols. Used for iterative dfs when labeling components
   dr, dc define unit vectors in the cardinal directions
 */
#define MAXC 10000
int K = 1000;
int dr[4] = {0, -1, 0, 1};
int dc[4] = {-1, 0, 1, 0};

/**
   cnt stores pixel intensity histograms for each component
   label stores the component label of each pixel
   vis is used in the iterative dfs
 */
int cnt[MAXC][256];
int label[ROWS][COLS];
bool vis[ROWS][COLS];

/**
   Proposed post-processing algorithm. First performs edge detection and labels
   components based on edges. Each component should have either 1) similiar disparities
   or 2) varying disparities. Histograms of each component are refined and final
   disparity map is determined. Disparity maps are of better quality that just
   dynammic programming.

   @param dis disparity map
   @param img hazy image to compute edge map from
 */
void blob(float (&dis)[ROWS][COLS], cv::Mat img) {
	// edge detection
	cv::Mat em;
	cv::blur(img, img, cv::Size(3,3));
	cv::Canny(img, em, 75, 150, 5);
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(1, 1));
	dilate(em, em, kernel);
	//cv::imshow("EdgeMap", em);
	//cv::waitKey(0);
	float edge[ROWS][COLS];
	for (int i=0; i<ROWS; i++)
		for (int j=0; j<COLS; j++)
			edge[i][j] = (int)em.at<uchar>(i,j);
	int cmp = 0;
	memset(cnt, 0, sizeof(int)*MAXC*256);
	memset(vis, 0, sizeof(bool)*ROWS*COLS);
	stack<int> stk;
	for (int i=SZ; i<ROWS-SZ; i++)
		for (int j=SZ; j<COLS-SZ; j++) {
			if (vis[i][j] || edge[i][j]>=128)
				continue;
			stk.push(K*i+j);
			while(stk.size()>0) {
				int idx = stk.top();
				stk.pop();
				int r = idx/K;
				int c = idx%K;
				vis[r][c] = true;
				cnt[cmp][(int)dis[r][c]]++;
				label[r][c] = cmp;
				for (int k=0; k<4; k++) {
					int y = r+dr[k];
					int x = c+dc[k];
					if (y>=0 && y<ROWS && x>=0 && x<COLS && !vis[y][x] && edge[y][x]<128)
						stk.push(K*y+x);
				}
			}
			cmp++;
		}
	for (int i=0; i<ROWS; i++)
		for (int j=1; j<COLS; j++)
			if (edge[i][j]>128)
				label[i][j] = label[i][j-1];
	for (int i=0; i<ROWS; i++)
		for (int j=COLS-2; j>=0; j--)
			if (edge[i][j]>128)
				label[i][j] = label[i][j+1];
	int mean[MAXC], lb[MAXC], ub[MAXC];
	int SW = 20;
	for (int i=0; i<cmp; i++) {
		int psum[256];
		psum[0] = 0;
		for (int j=20; j<256; j++)
			psum[j]=cnt[i][j]+psum[j-1];
		lb[i] = SW;
		ub[i] = 255;
		if (psum[255]-psum[1] < 1600)         //min component area
			continue;
		int mx = 0;
		for (int j=30; j<256-SW; j++)
			if (psum[j+SW]-psum[j] > mx) {
				mx = psum[j+SW]-psum[j];
				mean[i] = j;
			}
		int idx = mean[i];
		int avg = 0;
		for (int j=idx-SW/2; j<idx+SW/2; j++)
			avg = max(avg, cnt[i][j]);
		float F = 3.0;
		while(idx>10) {
			if (cnt[i][idx]>10 && cnt[i][idx] < avg/F) {
				lb[i] = idx;
				break;
			}
			idx--;
		}
		idx = mean[i];
		while(idx<255) {
			if (cnt[i][idx]>10 && cnt[i][idx] < avg/F) {
				ub[i] = idx;
				break;
			}
			idx++;
		}
	}
	for (int i=SZ; i<ROWS-SZ; i++)
		for (int j=SZ; j<COLS-SZ; j++) {
			int cc = label[i][j];
			if (dis[i][j]<lb[cc])
				dis[i][j] = lb[cc];
			else if (dis[i][j]>ub[cc])
				dis[i][j] = ub[cc];
		}
}

/**
   Given stereo images, compute disparity map
 */
void run(string src_folder, string dst_folder, string name, int cnt) {
	cv::Mat li, ri;
	li = cv::imread(src_folder+"/"+name+"_"+to_string(cnt)+"_0.png", cv::IMREAD_COLOR);
	ri = cv::imread(src_folder+"/"+name+"_"+to_string(cnt)+"_1.png", cv::IMREAD_COLOR);
	//cout << "Rows: " << li.rows << ' ' << "Cols: " << li.cols << endl;
	float left_img[ROWS][COLS][3], right_img[ROWS][COLS][3], dis[ROWS][COLS];
	memset(dis, 0, sizeof(float)*ROWS*COLS);
	toArray(left_img, li);
	toArray(right_img, ri);
	dismap(right_img, left_img, dis);
	norm(dis);
	blob(dis, ri);
	float final[ROWS][COLS][3];
	for (int i=0; i<ROWS; i++)
		for (int j=0; j<COLS; j++)
			for (int k=0; k<3; k++)
				final[i][j][k] = dis[i][j];
	cv::Mat ans(ROWS, COLS, CV_8UC3, cv::Scalar(0, 0, 0));
	toImage(final, ans);
	//cv::imshow("Frame"+to_string(cnt), ans);
	//cv::waitKey(0);
	cv::imwrite(dst_folder+"/"+name+"?"+to_string(cnt)+".png", ans);
}

/**
   hazy_images stereo data
 */
string names[19] = {"Aloe", "Rocks2", "Midd1", "Bowling1", "Cloth3", "Cloth4", "Baby1",
	                "Lampshade1", "Wood2", "Rocks1", "Midd2", "Flowerpots", "Bowling2", "Baby2", "Baby3",
	                "Plastic", "Cloth1", "Wood1", "Lampshade2"};
/**
   Compute disparity maps for all stereo images in stereo_data/hazy_images
 */
void computeDismaps() {
	for (int s=0; s<19; s++)
		for (int c=0; c<10; c++) {
			cout << s << endl;
			run("stereo_data/hazy_images", "stereo_data/disparity_maps", names[s], c);
		}
}

/**
   Run demo; stores dimap in demo/dismaps. Note demo is not run in parallel.
 */
void runDemo() {
	cout << "Computing Disparity Map" << endl;
	run("demo", "demo/dismaps", "rocks", 0);
	cv::imshow("Demo_Dismap", cv::imread("demo/dismaps/rocks?0.png", 1));
	cv::waitKey(0);
}

int main() {
	//computeDismaps();
	runDemo();
}
