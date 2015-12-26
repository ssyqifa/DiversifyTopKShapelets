package simulators;
import distributions.*;

abstract public class Model{
	protected double t=0;
	Distribution error;
        public Model(){
            error=new NormalDistribution(0,1);
        }
/*Generate a single data
//Assumes a model independent of previous observations. As
//such will not be relevant for ARMA or HMM models, which just return -1.
* Should probably remove. 
*/  
	abstract double generate(double x);

//This will generate the next sequence after currently stored t value
	abstract double generate();

	public void reset(){ t=0;}
	public void setError(Distribution d){ error = d;}
//Generates a series of length n
	public	double[] generateSeries(int n)
	{
           double[] d = new double[n];
           for(int i=0;i<n;i++)
              d[i]=generate();
           return d;
        }
/**
 * Subclasses must implement this, how they take them out of the array is their business.
 * @param p 
 */        
        abstract public void setParameters(double[] p);


 }