package weka.core;
/**

Basic DTW implementation for Weka. /Each instance is assumed to be a time series. Basically we
pull all the data out and proceed as usual!
  NOTE: Need to implement the early abandon, no point doing all the sums. Also nee to implement a pre calculated version
  
  Needs black box debug.
  
 **/

import java.util.Enumeration;

import weka.core.neighboursearch.PerformanceStats;

public class DTW_DistanceSpaceEfficient extends DTW_DistanceBasic{
		double[] row1;
		double[] row2;

	 /* DTW Distance: 
	  * 
	  * This implementation is more memory efficient in that it only stores
	  * two rows. 
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
		int r = (int)(Math.floor(this.r*n)+1);
		row1=new double[m];
		row2=new double[m];
		//Set all to max
//		double[][] matrixD = new double[n][m];
		for(int i=0;i<m;i++){
			row1[i]=Double.MAX_VALUE;
		}
		row1[0]=(a[0]-b[0])*(a[0]-b[0]);
//Base cases for warping 0 to all along row 1		
		//Warp a[0] onto all b[1]...b[r+1]
				for(int j=1;j<=r && j<m;j++)
					row1[j]=row1[j-1]+(a[0]-b[j])*(a[0]-b[j]);

		double rowMinDist;
		int start,end;
		
		if(r>=m)
			end=m-1;
		else
			end=r;
		rowMinDist=row1[end];	
//		System.out.println(" First row distance ="+rowMinDist);
		int rowMinPos=0;

//For each remaining row, warp row i
		for (int i=1;i<n;i++){
			row2=new double[m];
//			warp a[i] onto b[0]
			row2[0]=row1[0]+(a[i]-b[0])*(a[i]-b[0]);
			//Set all the rest of row 2 to max	
			for(int j=1;j<m;j++)
				row2[j]=Double.MAX_VALUE;
//Find the interval for warping			
			if(i-r<1)
				start=1;
			else
				start=i-r;
			if(i+r>m)
				end=m;
			else
				end=i+r;
//Warp a[i] onto b[j=start..end]
		    for (int j = start;j<end;j++)
		    {
//Find the min of row2[j-1],row1[j] and row1[j-1]
		    	minDist=row2[j-1];
		    	if(row1[j]<minDist)
		    		minDist=row1[j];
		    	if(row1[j-1]<minDist)
		    		minDist=row1[j-1];
		    	row2[j]=minDist+(a[i]-b[j])*(a[i]-b[j]);
		    }
		    //Record the min dist of the end of the row
//			System.out.println(i+" end row distance ="+row2[end-1]);
		    if(rowMinDist<row2[end-1]){
		    	rowMinDist=row2[end-1];
				rowMinPos=i;
		    }
		    //Swap row 2 into row 1.
		    row1=row2;
		}
//Find min distance overall		
		dist=rowMinDist;
		for(int j=0;j<m;j++){
//			System.out.println(j+" end col distance ="+row2[j]);
			if(row2[j]<dist){
				dist=row2[j];
			}
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
