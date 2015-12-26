package weka.classifiers.lazy;

import utilities.AttributeFilterBridge;
import weka.core.*;
import weka.core.neighboursearch.NearestNeighbourSearch;

/** Nearest neighbour classifier that extends the weka one but can take alternative distance functions.
 * 
 * 
 * 
 * @author ajb
 * @version 1.0
 * @since 5/4/09
 * 
 * 
 * TO DO:
 * 	1. Check Euclidean distance works: Check normalised flag works: DONE: 
 * 			Notes: 	Flag for normalise is in EuclideanDistance, NOT in DistributionFunction, which is rubbish. it is default on
 * 					Normalisation is onto fixed interval 0..1 by (x-min)/(max-min)
 Done 7/4/09
 * 
 * NOTES ON DESIGN: The class EuclideanDistance by default normalizes to range [0..1] by dividing by max, 
 * 
 * This class allows for a built in attribute filter 
 * 
 * 
 * */

public class kNN extends IBk {
	protected DistanceFunction dist;
	double[][] distMatrix;
	boolean storeDistance;
	public kNN(){
//Defaults to Euclidean distance		
		super();
		dist=new EuclideanDistance();
		super.setKNN(1);
	}
	public kNN(int k){
		super(k);
		setDistanceFunction(new EuclideanDistance());
	}
	public kNN(DistanceFunction df){
			super();
			setDistanceFunction(df);
	}
	
	public final void setDistanceFunction(DistanceFunction df){
		dist=df;
		NearestNeighbourSearch s = super.getNearestNeighbourSearchAlgorithm();
		try{
			s.setDistanceFunction(df);
		}catch(Exception e){
			System.err.println(" Exception thrown setting distance function ="+e+" in "+this);
                        e.printStackTrace();
                        System.exit(0);
		}
	}
//Need to implement the early abandon for the search?	
	public double distance(Instance first, Instance second) {  
		  return dist.distance(first, second);
	  }
//Only use with a Euclidean distance method
	public void normalise(boolean v){
		if(dist instanceof NormalizableDistance)
			((NormalizableDistance)dist).setDontNormalize(!v);
		else
			System.out.println(" Not normalisable");
	}
	static String path="C:\\Research\\Data\\WekaTest\\";

	
	public double testKNN(Instances test){
		
		
		return 0;
	}
	
//FILTER CODE 	
	boolean filterAttributes=false;
	double propAtts=0.5;
	int nosAtts=0;
	AttributeFilterBridge af;
	public void setFilterAttributes(boolean f){ filterAttributes=f;}
//	public void setEvaluator(ASEvaluation a){ eval=a;}
	public void setProportion(double f){propAtts=f;}
	public void setNumber(int n){nosAtts=n;}
	
	private Instances filter(Instances d){
//Search method: Simple rank, evaluating in isolation
		af=new AttributeFilterBridge(d);
		af.setProportionToKeep(propAtts);
		Instances d2=af.filter();
//		Instances d2=new Instances(d);
//Remove all attributes not in the list. Are they sorted??			
		return d2;
	}

    @Override
	public void buildClassifier(Instances d){
		Instances d2=d;
		if(filterAttributes){
			d2=filter(d);
		}
		dist.setInstances(d2);
		try{
			super.buildClassifier(d2);
		}catch(Exception e){
			System.out.println("Exception thrown in kNN build Classifier = "+e);
                        e.printStackTrace();
                        System.exit(0);
		}
	}
    @Override
  public double [] distributionForInstance(Instance instance) throws Exception {
	  if(af!=null){
		  Instance newInst=af.filterInstance(instance);
		  return super.distributionForInstance(newInst);
	  }
	  else
		  return super.distributionForInstance(instance);
	
  }	
	public double[] getPredictions(Instances test){
		double[] pred=new double[test.numInstances()];
		try{
			for(int i=0;i<test.numInstances();i++){
				pred[i]=classifyInstance(test.instance(i));
				System.out.println("Pred = "+pred[i]);
			}
		}catch(Exception e){
			System.out.println("Exception thrown in getPredictions in kNN = "+e);
                        e.printStackTrace();
                        System.exit(0);
		}
		return pred;
	}
	public double measureAccuracy(Instances test){
		double[] pred=getPredictions(test);
		double accuracy=0;
		for(int i=0;i<test.numInstances();i++)
			if(pred[i]==test.instance(i).classValue())
				accuracy++;
		return accuracy/test.numInstances();
	}
	public static void main(String[] args){
		
		
		Instances data=weka.classifiers.evaluation.ClassifierTools.loadData(path+"iris");
//		kNN c=new kNN(new DTW_DistanceBasic());
//		kNN c2=new kNN(new DTW_DistanceEfficient());

/*		data.randomize(new Random());
		Instances train=data.trainCV(3,1);
		Instances test=data.testCV(3,1);
		c.buildClassifier(train);
		c.setKNN(3);
		c2.setKNN(3);
		c2.buildClassifier(train);
		double a1=c.measureAccuracy(test);
		double a2=c.measureAccuracy(test);
		System.out.println("Accuracy basic = "+a1+" space efficient = "+a2);
//		kNN c=new kNN(new EuclideanDistance());
*/		
	//Test the filter	
		kNN c3=new kNN();
		c3.setFilterAttributes(true);
		c3.buildClassifier(data);
		
		
		//		c.normalise(false);
		Instance first=data.instance(0);
		Instance second=data.instance(1);
//		System.out.println(" Basic Distance between "+first+" and  "+second+" = "+c.distance(first,second));
//		System.out.println(" Space Efficient Distance between "+first+" and  "+second+" = "+c.distance(first,second));

	}
}
