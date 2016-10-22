#include <stdio.h>
/*
 * create S_x functions from input file
 * create Pval(x) for each S_x using ensemble
 * find global minimum for each Pval(x)
 *   -> can be used to create distribution S(xValMinima)
 *
 * compile with:
 *  gcc -O3 -fPIC -shared -o optimizeSx.so optimizeSx.c
 *
 * Modification History:
 *  Written by Z Knight, 2016.09.26
 *  Corrected 2d array indexing problem in optSx variable SxVecs:
 *    apparently passing python numpy arrays into SxVecs using 
 *    python's ctypes ndpointer inadvertantly flattened the array.
 *    Fixed by rewriting to expect 1d array; ZK, 2016.09.28
 *    Modified xstart value to avoid oddness near x=-1.0; ZK, 2016.09.30
 *    Added xStart and xEnd to optSx function to constrain search range; ZK, 2016.10.06
 *
 */

double linearInterp(double x1, double y1, double x2, double y2, double xMid) {
  /*
   * x1,y1: coordinates of left point
   * x2,y2: coordinates of right (x2>x1) point
   * xMid: point to interpolate at (x1<xMid<x2)
   *  (ok if outside range, but that would actually be extrapolation)
   */
  double slope;
  double intercept;
  slope = (y2-y1)/(x2-x1);
  intercept = y2-slope*x2;
  //printf("x1: %f, x2: %f, x2-x1: %f, y2: %f, y1: %f, y2-y1: %f\n",x1,x2,x2-x1,y2,y1,y2-y1);
  //printf("slope: %f, intercept: %f\n",slope,intercept);
  return slope*xMid+intercept;
}


double arrayLinearInterp(const double *xVec, const double *yVec, const double xInterp, size_t xSize) {
  /*
   * Note: not for use with negative y values, due to error return code -1
   * Inputs:
   *  xVec: an array of points to interpolate between, in decreasing order
   *  yVec: the corresponding y values
   *  xInterp: the point to interpolate at
   *    must be within range of highest and lowest values in xVec
   *  xSize: the number of elements in x array
   * Returns:
   *  the interpolated y value, or -1 if xInterp was outside range
   */
  // due to decreasing order of xVec, signs are not as expected:
  //if (xInterp < xVec[0] || xInterp > xVec[xSize-1]) {
  if (xInterp > xVec[0] || xInterp < xVec[xSize-1]) {
    printf("%f, %f, %f",xVec[0],xInterp,xVec[xSize-1]);
    puts("xInterp outside of interpolation range.  Exiting.");
    return -1;
  }
  size_t i = 0; //loop index
  while (i < xSize) {
    // due to decreasing order of xVec, signs are not as expected:
    if (xInterp >= xVec[i]) 
    //  return linearInterp(xVec[i],yVec[i],xVec[i+1],yVec[i+1],xInterp);
    //if (xInterp <= xVec[i+1]) 
      return linearInterp(xVec[i+1],yVec[i+1],xVec[i],yVec[i],xInterp);
    i++;
  }
  return -1; // this line is just to keep the compiler warnings quiet

}

double PvalOfX(double xVal, size_t nSim, const double *xVec, const double *SxVecs, size_t xSize, size_t nSims) {
  /*
   * Purpose: calculate p-values for each S_x, using ensemble
   *  Pval: probability of result equal to or more extreme
   * Inputs:
   *  xVal: the x value to calculate p-values for
   *  nSim: which of the simulations to evaluate for
   *    (this replaces the python version's list of functions)
   *  xVec, SxVecs, xSize: for forwarding to arrayLinearInterp
   *  nSims: number of sims in SxVecs
   * Returns:
   *  the p-value
   */
  int nUnder = 0;
  int nOver  = 0;
  // sending pointer to flattened array: nSim*xSize points to the [0] element in sim nSim 
  double threshold = arrayLinearInterp(xVec,&SxVecs[nSim*xSize],xVal,xSize);
  double mySx;  // for calculating Sx values
  for (size_t mySim = 0; mySim < nSims; mySim++) {
    mySx = arrayLinearInterp(xVec,&SxVecs[mySim*xSize],xVal,xSize);
    if (mySx > threshold) {
      nOver++;
      //printf("Over! mySx: %f, threshold: %f\n",mySx,threshold);
    }
    else                  {
      nUnder++;
      //printf("Under! mySx: %f, threshold: %f\n",mySx,threshold);
    }
  }
  //printf("nUnder: %d, nOver: %d\n",nUnder,nOver);
  return nUnder/(double)(nUnder+nOver);

}

void optSx(const double *xVec, size_t xSize, const double *SxVecs, size_t nSims, double xStart, double xEnd, int nSearch, double *PvalMinima, double *XvalMinima)  {
  /*
   * Name:
   *  optSx
   * Purpose:
   *  find global minimum for each Pval(x)
   * Procedure:
   *  simply use xVals defined on equal intervals and search for min. from left
   *  Note: if there are equal p-values along the range, the one with the
   *    lowest xVal will be reported
   * Inputs:
   *  xVec: an array of x points for S_x values
   *  xSize: the number of points in xVec array
   *  SxVecs: array of arrays of S_x values
   *    This can be thought of as a 2d array, but really it's flattened into 1d.
   *    Its length is nSims*xSize; its index is nSim*xSize+nX
   *  nSims: the number of simulations (rows) in SxVecs
   *  xStart: the start of the x range to search
   *    must be >= -1 and < xEnd.  If -1 is entered, -0.999 will be used 
   *      instead to avoid chaotic behavior of Pval(x) near x=-1
   *  xEnd: the end of the x range to search
   *    must be <= 1 and > xStart
   *  nSearch: the number of points to search for along x
   *  PvalMinima,XvalMinima: arrays of length nSims to contain P(x) and x values
   * Returns:
   *  void
   *  However, variables PvalMinma and XvalMinima will contain return values
   *
   */

  //for (size_t i = 0; i < xSize; i++) {
  //  printf("%.10e ",SxVecs[5*xSize+i]);
  //}
  //printf("\nThat was c's SxVecs[5] Press enter.\n");
  //getchar();

  // create empty pVal array and linspace x array
  //double myPvals[nSearch];
  double myPval;
  double myXvals[nSearch];
  //double xStart = -0.999; //-1.0; //start for range (-1 <= x <= 1)
  if (xStart == -1) {
    xStart = -0.999; //avoid chaotic Pval(x) near -1
  }
  double deltaX = xEnd - xStart;
  for (int n = 0; n < nSearch; n++) {
    myXvals[n] = xStart + deltaX*(n/(double)(nSearch-1));
  }

  //size_t nSim; // loop index
  for (size_t nSim = 0; nSim < nSims; nSim++) {
    printf("starting minimum Pval(x) search for sim %zd of %zd\n",nSim+1,nSims);
    PvalMinima[nSim] = PvalOfX(xStart,nSim,xVec,SxVecs,xSize,nSims);
    XvalMinima[nSim] = xStart;
    for (int n = 1; n < nSearch; n++) { // can omit 0 due to above PvalOfX

      myPval = PvalOfX(myXvals[n],nSim,xVec,SxVecs,xSize,nSims);
      //printf("nSim: %zd, n: %d, myPval: %f, PvalMinima[nSim]: %f\n",nSim,n,myPval,PvalMinima[nSim]);
      if (myPval < PvalMinima[nSim] && myXvals[n] > -0.999) {
        // -0.999 to avoid instabililility at endpoint
        PvalMinima[nSim] = myPval;
        XvalMinima[nSim] = myXvals[n];
      } 
    }    
    //printf("Finished sim %zd of %zd.  Press enter to continue\n",nSim+1,nSims);
    //getchar();
  }
  // return PvalMinima, XvalMinima; //nope, return in parameters instead

}


int main( int argc, char *argv[]) {
  /*
   * Originally this was going to be a program that took a file name as
   *  an argument, opened it, and wrote another file back to disk.
   * Instead, this program is now intended to be used as a shared object
   *  that is linked to from the optimizeSx.py python program.
   *
   * So, this main function does almost nothing, since it doesn't get used.
   *
   */

  if( argc == 2 ) {
    printf("The argument supplied is %s\n",argv[1]);
  }
  else if( argc > 2 ) {
    printf("Too many arguments supplied.\n");
  }
  else {
    printf("One argument expected.\n");
  }
}


