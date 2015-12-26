package simulators;
//Model ARMA series naively as Box Jenkins representation

// Note really need infinite AR representation

import distributions.*;
import fileIO.*;
import java.util.*;

public class HMM_Model extends Model{

//Number of generating distributions
	int N;	

// family of generating distributions
	Distribution[] f;
//Discrete transition matrix		
	double[][] T;

//Discrete initial values distribution		
	double[] pi;

//Current state
	int state;

//Stats recorded for debugging
	int n=200;	//Default series length
	ArrayList stateRecord;
	ArrayList dataRecord;

	public HMM_Model()
	{
		System.out.println("SHOULD NOT BE HERE!!");
	
	}
	public HMM_Model(int models)
	{
		N=models;
		f=new NormalDistribution[N];
		T=new double[N][N];
		pi = new double[N];
	}
	public HMM_Model(int models, double[] means, double[] stDevs, double[][]trans, double[] inits)
//Default constructor assumes normal distribution
	{
		N=models;
		f=new NormalDistribution[N];
		T=new double[N][N];
		pi = new double[N];
		
		for(int i=0;i<N;i++)
		{
			f[i]= new NormalDistribution(means[i],stDevs[i]);
			for(int j=0;j<N;j++)
				T[i][j]=trans[i][j];
			pi[i]=inits[i];	
		}
		//Set initial state
		initialise();
				
	}
	public void initialise(){
		double r=Distribution.RNG.nextDouble();
		state = 0;
		double s=0;					
		boolean finished=false;
		do{
			s+=pi[state];
			if(s>r)
				finished=true;
			else
				state++;	
		}while(!finished && state<N);
	
		stateRecord = new ArrayList(n);
		dataRecord = new  ArrayList(n);		
	}	

	public double generate(double x){
		System.out.println("Error, generate not implemented for HMM");
		System.exit(0);
		return -1;
	}	

	public double generate(){
	
//Determine state
		double r=Distribution.RNG.nextDouble();
		double s=0;
		int i=0;
		boolean finished=false;
		do{
			s+=T[state][i];
			if(s>r)
				finished=true;
			else
				i++;	
		}while(!finished && i<N);
		state=i;
//Sample Distribution
		return f[state].simulate();
	}



	public String toString()
	{
		String str="";
		return str;
	}		


	static public void main(String[] args){
	
		System.out.println("Testing HMM Models");
		System.out.println("Generating Data: from 2 normal dists, variance 1");
		OutFile f = new OutFile("SmytheHMMData.csv");
/*		double[][] d = GenerateData.simulateSmytheData(3,200);
		for(int j=0;j<d[0].length;j++)
		{
			for(int i=0;i<d.length;i++)	
				f.writeString(d[i][j]+",");
			f.writeString("\n");	
		}	
*/
	}

    @Override
    public void setParameters(double[] p) {
        throw new UnsupportedOperationException("Not supported yet.");
    }
}