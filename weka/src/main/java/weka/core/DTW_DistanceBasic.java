package weka.core;
/**

Basic DTW implementation for Weka. /Each instance is assumed to be a time series. Basically we
pull all the data out and proceed as usual!
  NOTE: Need to implement the early abandon, no point doing all the sums. Also nee to implement a pre calculated version
  
  Needs black box debug.
  
 **/

import java.util.ArrayList;
import java.util.Enumeration;

import weka.core.neighboursearch.PerformanceStats;

public class DTW_DistanceBasic extends EuclideanDistance{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	double r=1;	//Warping window size percentage, between 0 and 1
	double[][] matrixD;
	int endX=0;
	int endY=0;
	public DTW_DistanceBasic(){
		super();
		m_DontNormalize=true;
	}
	  public DTW_DistanceBasic(Instances data) {	
		  super(data);
			m_DontNormalize=true;
	  }
	
	//Needs overriding to avoid cutoff check
	  public double distance(Instance first, Instance second){
		  return distance(first, second, Double.POSITIVE_INFINITY, null, false);
	  }
	  public double distance(Instance first, Instance second, PerformanceStats stats) { //debug method pls remove after use
		    return distance(first, second, Double.POSITIVE_INFINITY, stats, false);
		  }

	  
/* ASSUMES THE CLASS INDEX IS THE LAST DATA FOR THE NORMALISATION. But dont do the normalisation here anyway.
 * 
	Euclidean distance normalises to [0..1] based on the flag m_DontNormalize, but it does this in the calculation.
More efficient to do this as a filter, otherwise you are repeatedly recalculating.
Basic normalisation is implemented, but advised  not to use it, so flag m_DontNormalize set by default to true. 

*/
	  public double distance(Instance first, Instance second, double cutOffValue, PerformanceStats stats){
		  return distance(first,second,cutOffValue,stats,false);
		  }
	  public double distance(Instance first, Instance second, double cutOffValue, PerformanceStats stats, boolean print) {
		  	//Get the double arrays
		  return distance(first,second,cutOffValue);
	  }
	  
public double distance(Instance first, Instance second, double cutOffValue) {
		  double[] f;
		  double[] s;
		  
//		  System.out.println(" In DTW Basic distance, cutoff ="+cutOffValue);
//Need to remove class value if present. 
		  int fClass=first.classIndex();
		  if(fClass>0) {
			  f=new double[first.numAttributes()-1];
			  int count=0;
			  for(int i=0;i<f.length+1;i++){
				  if(i!=fClass){
					  f[count]=first.value(i);
					  count++;
				  }
			  }
		  }
		  else
			  f=first.toDoubleArray();
		  int sClass=second.classIndex();
		  if(sClass>0) {
			  s=new double[second.numAttributes()-1];
			  int count=0;
			  for(int i=0;i<s.length;i++){
				  if(i!=sClass){
					  s[count]=second.value(i);
					  count++;
				  }
			  }
		  }
		  else
			  s=second.toDoubleArray();
/* Dont use this
 		  if(!m_DontNormalize){
			  for(int i=0;i<f.length;i++){
				  int index=first.index(i);
				//ASSUME THE CLASS INDEX IS THE LAST DATA
				  f[i]=(f[i]-m_Ranges[index][R_MIN])/(m_Ranges[index][R_MAX] - m_Ranges[index][R_MIN]);
			  }
			  for(int i=0;i<s.length;i++){
				  int index=second.index(i);
				//ASSUME THE CLASS INDEX IS THE LAST DATA
				  s[i]=(s[i]-m_Ranges[index][R_MIN])/(m_Ranges[index][R_MAX] - m_Ranges[index][R_MIN]);
			  }
		  }
*/
		  return distance(f,s,cutOffValue);
	  }

	  /* DTW Distance: Need to implement the early abandon
	  * 
	  */ 
	  public double distance(double[] a,double[] b, double cutoff){
		double dist=0, minDist;
// Set the longest series to a
		double[] temp;
		if(a.length<b.length){
			temp=a;
			a=b;
			b=temp;
		}
		int n=a.length;
		int m=b.length;
		
		
/*  Parameter 0<=r<=1. 0 == no warp, 1 == full warp 
 generalised for variable window size
 * */
		int windowSize = getWindowSize(0,0,n);
			
		
		//Set all to max
		matrixD = new double[n][m];
		for(int i=0;i<n;i++)
			for(int j=0;j<m;j++)
				matrixD[i][j]=Double.MAX_VALUE;
		matrixD[0][0]=(a[0]-b[0])*(a[0]-b[0]);
//Base cases for warping 0 to all with max interval	r	
//Warp a[0] onto all b[1]...b[r+1]
		for(int j=1;j<windowSize && j<n;j++)
			matrixD[0][j]=matrixD[0][j-1]+(a[0]-b[j])*(a[0]-b[j]);

//	Warp b[0] onto all a[1]...a[r+1]
		for(int i=1;i<windowSize && i<n;i++)
			matrixD[i][0]=matrixD[i-1][0]+(a[i]-b[0])*(a[i]-b[0]);
		
//Warp the rest,
		int start,end;
		for (int i=1;i<n;i++){
			windowSize = getWindowSize(i,1,n);
			if(i-windowSize<1)
				start=1;
			else
				start=i-windowSize+1;			

			if(start+windowSize>=m)
				end=m;
			else
				end=start+windowSize;
			
		    for (int j = start;j<end;j++)
		    {
//Find the min of matrixD[i][j-1],matrixD[i-1][j] and matrixD[i-1][j-1]
		    	minDist=matrixD[i][j-1];
		    	if(matrixD[i-1][j]<minDist)
		    		minDist=matrixD[i-1][j];
		    	if(matrixD[i-1][j-1]<minDist)
		    		minDist=matrixD[i-1][j-1];
		    	matrixD[i][j]=minDist+(a[i]-b[j])*(a[i]-b[j]);

//Is this a correct early		    	
		    }
		}
/*		for(int i=0;i<n;i++){
			System.out.print("Row ="+i+" = ");
			for(int j=0;j<n;j++){
				System.out.print(" "+matrixD[i][j]);
			}
			System.out.print("\n");
		}
*/			
//Find the minimum distance at the end points, within the warping window. Maybe should not have to match the ends? 
		return matrixD[n-1][m-1];
/*		int x=n-1,y=0;
		dist=matrixD[n-1][0];
		for(int j=1;j<m;j++){
			if(matrixD[n-1][j]<dist){
				dist=matrixD[n-1][j];
				y=j;
			}
		}
		for(int i=0;i<n;i++){
			if(matrixD[i][m-1]<dist){
				x=i;
				y=n-1;
				dist=matrixD[i][m-1];
			}
		}
		endX=x;
		endY=y;
		return dist;
*/	}
// This could be generalised to allow for different window algorithms
	  int getWindowSize(int i, int j, int n){
		int w=(int)(Math.floor(this.r*n));
			//No Warp, windowSize=1
		if(w<1) w=1;
			//Full Warp : windowSize=n, otherwise scale between		
		else if(w<n)
			w++;
		return w;	
	  }
	void printPath(){
		//Find Path backwards in pairs?			
		int n=matrixD.length;
		int m=matrixD[0].length;
		int x=n-1,y=m-1;
		
		
		int count=0;
		System.out.println(count+"END  Point  = "+x+","+y+" value ="+matrixD[x][y]);
		while(x>0 && y>0)
		{
			//Look along
			double min=matrixD[x-1][y-1];
			if(min<=matrixD[x-1][y] && min<=matrixD[x][y-1]){
				x--;
				y--;
			}
			else if(matrixD[x-1][y] < matrixD[x][y-1])
				x--;
			else
				y--;
			count++;
			System.out.println(count+" Point  = "+x+","+y+" value ="+matrixD[x][y]);
		}
		while(x>0){
			x--;
			System.out.println(count+" Point  = "+x+","+y+" value ="+matrixD[x][y]);
		}
		while(y>0){
			y--;
			System.out.println(count+" Point  = "+x+","+y+" value ="+matrixD[x][y]);
		}
		
	}
	  public String toString() {
		    return "DTW BASIC. r="+r;
	  }
	public String globalInfo() {
		return " DTW Basic Distance";
	}
	public String getRevision() {
	
		return null;
	}
	public void setR(double x){ r=x;}
	
/* JML Implementation version
 * 
 */
     public double measure(double[] ts1, double[] ts2) {
        /** Build a point-to-point distance matrix */
        double[][] dP2P = new double[ts1.length][ts2.length];
        for (int i = 0; i < ts1.length; i++) {
            for (int j = 0; j < ts2.length; j++) {
                dP2P[i][j] = (ts1[i]-ts2[j])*(ts1[i]-ts2[j]);
            }
        }
        /** Check for some special cases due to ultra short time series */
        if (ts1.length == 0 || ts2.length == 0) {
            return Double.NaN;
        }
        if (ts1.length == 1 && ts2.length == 1) {
            return dP2P[0][0];
        }

        /**
         * Build the optimal distance matrix using a dynamic programming
         * approach
         */
        double[][] D = new double[ts1.length][ts2.length];
        D[0][0] = dP2P[0][0]; // Starting point

        for (int i = 1; i < ts1.length; i++) { // Fill the first column of our
            // distance matrix with optimal
            // values
            D[i][0] = dP2P[i][0] + D[i - 1][0];
        }

        if (ts2.length == 1) { // TS2 is a point
            double sum = 0;
            for (int i = 0; i < ts1.length; i++) {
                sum += D[i][0];
            }
            return sum;
        }

        for (int j = 1; j < ts2.length; j++) { // Fill the first row of our
            // distance matrix with optimal
            // values
            D[0][j] = dP2P[0][j] + D[0][j - 1];
        }

        if (ts1.length == 1) { // TS1 is a point
            double sum = 0;
            for (int j = 0; j < ts2.length; j++) {
                sum += D[0][j];
            }
            return sum;
        }
        System.out.println(" DTW Distances =");
        for (int i = 1; i < ts1.length; i++) { // Fill the rest
            System.out.print(" ROW "+i+" =");
            for (int  j = 1; j < ts2.length; j++) {
                double[] steps = { D[i - 1][j - 1], D[i - 1][j], D[i][j - 1] };
                double min = Math.min(steps[0], Math.min(steps[1], steps[2]));
                D[i][j] = dP2P[i][j] + min;
                System.out.print(D[i][j]+" ");
            }
            System.out.print("\n");
        }

        /**
         * Calculate the distance between the two time series through optimal
         * alignment. 
         */
        
//This step is surely just wrong!!        
        int i = ts1.length - 1;
        int j = ts2.length - 1;
        int k = 1;
        double dist = D[i][j];

        while (i + j > 2) {
            if (i == 0) {
                j--;
            } else if (j == 0) {
                i--;
            } else {
                double[] steps = { D[i - 1][j - 1], D[i - 1][j], D[i][j - 1] };
                double min = Math.min(steps[0], Math.min(steps[1], steps[2]));

                if (min == steps[0]) {
                    i--;
                    j--;
                } else if (min == steps[1]) {
                    i--;
                } else if (min == steps[2]) {
                    j--;
                }
            }
            k++;
            dist += D[i][j];
        }
        return dist;
    }

	
/* OLD IMPLEMENTATION	
	double r=0.05;	//Warping window size percentage, between 0 and 1
	protected DynamicTimeWarping(){}
	public DynamicTimeWarping(double r){this.r=r;}

	public double distance(Complex[] a,Complex[] b){
		double dist=0, minDist;
// Set the longest series to a
		Complex[] temp;
		if(a.length<b.length)
		{
			temp=a;
			a=b;
			b=temp;
		}
		int n=a.length;
		int m=b.length;
		//Use this to make it the same as Euclidean when this.r=0
		int r = (int)(Math.floor(this.r*n)+1);
//Set all to max
		double[][] matrixD = new double[n][m];
		for(int i=0;i<n;i++)
			for(int j=0;j<m;j++)
				matrixD[i][j]=Double.MAX_VALUE;
		matrixD[0][0]=Complex.distance(a[0],b[0]);
//Base cases for warping 0 to all 		
//Warp a[0] onto all b[1]...b[r+1]
		for(int j=1;j<=r && j<n;j++)
			matrixD[0][j]=matrixD[0][j-1]+Complex.distance(a[0],b[j]);
//	Warp b[0] onto all a[1]...a[r+1]
		for(int i=1;i<=r && i<n;i++)
			matrixD[i][0]=matrixD[i-1][0]+Complex.distance(a[i],b[0]);
		
//Warp the rest,
		int start,end;
		for (int i=1;i<n;i++)
		{
			if(i-r<1)
				start=1;
			else
				start=i-r;
			if(i+r>m)
				end=m;
			else
				end=i+r;
		    for (int j = start;j<end;j++)
		    {
		    	//Find the min of matrixD[i][j-1],matrixD[i-1][j] and matrixD[i-1][j-1]
		    	minDist=matrixD[i][j-1];
		    	if(matrixD[i-1][j]<minDist)
		    		minDist=matrixD[i-1][j];
		    	if(matrixD[i-1][j-1]<minDist)
		    		minDist=matrixD[i-1][j-1];
		    	matrixD[i][j]=minDist+Complex.distance(a[i],b[j]);
		    	
		    }
			
		}
//Find the minimum distance
		dist=matrixD[n-1][0];
		for(int j=1;j<m;j++)
			if(matrixD[n-1][j]<dist)
				dist=matrixD[n-1][j];
		for(int i=0;i<n;i++)
			if(matrixD[i][m-1]<dist)
				dist=matrixD[i][m-1];
		return dist;
	}

	public double distance(DataPoint dp1,DataPoint dp2){
		double[] a = dp1.getData();
		double[] b= dp2.getData();
		return distance(a,b);
	}
	public double distance(ComplexDataPoint dp1,ComplexDataPoint dp2){
		double[] a1 = dp1.getReal();
		double[] a2 = dp1.getImaginary();
		double[] b1= dp2.getReal();
		double[] b2= dp2.getImaginary();
		Complex[] a = new Complex[a1.length];
		for(int i=0;i<a.length;i++)
			a[i]=new Complex((float)a1[i],(float)a2[i]);
		Complex[] b = new Complex[b1.length];
		for(int i=0;i<b.length;i++)
			b[i]=new Complex((float)b1[i],(float)b2[i]);
		return distance(a,b);
	}
	public double distance(double[] a,double[] b){
		double dist=0, minDist;
// Set the longest series to a
		double[] temp;
		if(a.length<b.length)
		{
			temp=a;
			a=b;
			b=temp;
		}
		int n=a.length;
		int m=b.length;
		int r = (int)(Math.floor(this.r*n)+1);
//		int r = (int)Math.ceil(this.r*n);
//		System.out.println(" r = "+r);

		//Set all to max
		double[][] matrixD = new double[n][m];
		for(int i=0;i<n;i++)
			for(int j=0;j<m;j++)
				matrixD[i][j]=Double.MAX_VALUE;
		matrixD[0][0]=(a[0]-b[0])*(a[0]-b[0]);
//Base cases for warping 0 to all 		
//Warp a[0] onto all b[1]...b[r+1]
		for(int j=1;j<=r && j<n;j++)
			matrixD[0][j]=matrixD[0][j-1]+(a[0]-b[j])*(a[0]-b[j]);

//	Warp b[0] onto all a[1]...a[r+1]
		for(int i=1;i<=r && i<n;i++)
			matrixD[i][0]=matrixD[i-1][0]+(a[i]-b[0])*(a[i]-b[0]);
		
//Warp the rest,
		int start,end;
		for (int i=1;i<n;i++)
		{
			if(i-r<1)
				start=1;
			else
				start=i-r;
			if(i+r>m)
				end=m;
			else
				end=i+r;
		    for (int j = start;j<end;j++)
		    {
		    	//Find the min of matrixD[i][j-1],matrixD[i-1][j] and matrixD[i-1][j-1]
		    	minDist=matrixD[i][j-1];
		    	if(matrixD[i-1][j]<minDist)
		    		minDist=matrixD[i-1][j];
		    	if(matrixD[i-1][j-1]<minDist)
		    		minDist=matrixD[i-1][j-1];
		    	matrixD[i][j]=minDist+(a[i]-b[j])*(a[i]-b[j]);
		    	
		    }
			
		}
//Find the minimum distance
		dist=matrixD[n-1][0];
		for(int j=1;j<m;j++)
			if(matrixD[n-1][j]<dist)
				dist=matrixD[n-1][j];
		for(int i=0;i<n;i++)
			if(matrixD[i][m-1]<dist)
				dist=matrixD[i][m-1];
		return dist;
	}

	public double distance(short[] a,short[] b){
		short dist=0, minDist;
// Set the longest series to a
		short[] temp;
		if(a.length<b.length)
		{
			temp=a;
			a=b;
			b=temp;
		}
		int n=a.length;
		int m=b.length;
		int r = (int)(Math.floor(this.r*n)+1);
//		int r = (int)Math.ceil(this.r*n);
//		System.out.println(" r = "+r);

		//Set all to max
		short[][] matrixD = new short[n][m];
		for(int i=0;i<n;i++)
			for(int j=0;j<m;j++)
				matrixD[i][j]=Short.MAX_VALUE;
		matrixD[0][0]=(short)((a[0]-b[0])*(a[0]-b[0]));
//Base cases for warping 0 to all 		
//Warp a[0] onto all b[1]...b[r+1]
		for(int j=1;j<=r && j<n;j++)
			matrixD[0][j]=(short)(matrixD[0][j-1]+(short)((a[0]-b[j])*(a[0]-b[j])));

//	Warp b[0] onto all a[1]...a[r+1]
		for(int i=1;i<=r && i<n;i++)
			matrixD[i][0]=(short)(matrixD[i-1][0]+(short)((a[i]-b[0])*(a[i]-b[0])));
		
//Warp the rest,
		int start,end;
		for (int i=1;i<n;i++)
		{
			if(i-r<1)
				start=1;
			else
				start=i-r;
			if(i+r>m)
				end=m;
			else
				end=i+r;
		    for (int j = start;j<end;j++)
		    {
		    	//Find the min of matrixD[i][j-1],matrixD[i-1][j] and matrixD[i-1][j-1]
		    	minDist=matrixD[i][j-1];
		    	if(matrixD[i-1][j]<minDist)
		    		minDist=matrixD[i-1][j];
		    	if(matrixD[i-1][j-1]<minDist)
		    		minDist=matrixD[i-1][j-1];
		    	matrixD[i][j]=(short)(minDist+(a[i]-b[j])*(a[i]-b[j]));
		    	
		    }
			
		}
//Find the minimum distance
		dist=matrixD[n-1][0];
		for(int j=1;j<m;j++)
			if(matrixD[n-1][j]<dist)
				dist=matrixD[n-1][j];
		for(int i=0;i<n;i++)
			if(matrixD[i][m-1]<dist)
				dist=matrixD[i][m-1];
		return dist;
	}
	public static void main(String[] args)
	{
		System.out.println(" Very basic test for DTW distance");
		double[] a ={1,2,3,4,5,6,7,8};
		double[] b ={2,3,4,5,6,7,8,9};
		for(int i=0;i<a.length;i++)
			System.out.print(a[i]+",");
		System.out.println("\n************");
		for(int i=0;i<b.length;i++)
			System.out.print(b[i]+",");
		System.out.println("\n Euclidean distance is 8, DTW should be 0??");
		DynamicTimeWarping dtw= new DynamicTimeWarping(0.05);
		System.out.println(" DTW Distance ="+dtw.distance(a,b));

		
		
	}
*/	
}
