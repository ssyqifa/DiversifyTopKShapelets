package utilities;

import java.util.Random;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.AttributeEvaluator;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.ChiSquaredAttributeEval;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;
import weka.core.Instances;
import fileIO.OutFile;

public class AttributeSelectionTools {

	
	/** 

This methods scores each attribute with IG, IGR and CHI and writes the results to file
	Filters:
	ChiSquaredAttributeEval, GainRatioAttributeEval, InfoGainAttributeEval, 
	Single attribute classifiers:
	OneRAttributeEval, ReliefFAttributeEval, SVMAttributeEval, 
	Dont know what these are!
	SymmetricalUncertAttributeEval, UnsupervisedAttributeEvaluator
	NOTE: I have pulled in Chi Squared from an older weka version, might need testing ... 

 * @param tr: training instances to evaluate
 * @param file: full path for the output


	**/
	
	public static void attributeScoring(Instances tr, String file)
	{
		try{
			OutFile f=new OutFile(file);
			double e1,e2,e3;
			AttributeEvaluator as = new InfoGainAttributeEval();
			AttributeEvaluator as2 = new GainRatioAttributeEval();
			AttributeEvaluator as3 = new ChiSquaredAttributeEval();
			f.writeLine("INDEX,NAME,IG,IGR,CHI");
//			as.buildEvaluator(tr);
//			as2.buildEvaluator(tr);
//			as3.buildEvaluator(tr);
			for(int i=0;i<tr.numAttributes();i++)
			{
				e1= as.evaluateAttribute(i);
				e2= as2.evaluateAttribute(i);
				e3= as3.evaluateAttribute(i);
	
				f.writeLine(i+","+tr.attribute(i).name()+","+e1+","+e2+","+e3);
				System.out.println(i+","+tr.attribute(i).toString()+e1+","+e2+","+e3);
			}
		}catch(Exception e)
		{
			System.out.println("Exception thrown in attributeScoring = "+e.toString());
		}
	}	

	/** This class takes a particular Evaluator, evaluates each attribute, then returns the ranked
	 * list. Note we dont return the filter scores, I'm assuming this is done already in attributeScoring
THIS NEEDS TESTING

	 * * @param tr: training instances to evaluate
 * @param as: attribute scoring technique
	 */
	public static int[] simpleAttributeRanking(Instances tr,ASEvaluation as)
	{
		int size=tr.numInstances()-1;
		int[] rankings;
		AttributeSelection a;
		Instances trFiltered=null;
		ASSearch rank= new Ranker();
		int[] att=null;
		try{
			as.buildEvaluator(tr);
			System.out.println("Attribute Selector built");
			rank.search(as,tr);
			a=new AttributeSelection();
			a.setEvaluator(as);
			a.setSearch(rank);
			System.out.println("Attribute Selector set in AttributeSelection");
			System.out.println("Attribute Selector "+a);
			System.out.println("Ranker "+rank);
			System.out.println("ASEvaluator "+as);
			a.SelectAttributes(tr);
			att= a.selectedAttributes();
//			for(int i=0;i<att.length;i++)
//				System.out.print(att[i]+" ");
//			System.out.println(a.toResultsString());
		}
		catch(Exception e)
		{
			System.out.println(" Error in  simpleAttributeRanking  "+e.toString());
		}
		return att;
	}
	

	/** This class takes the entire data set and does a forward selection of attributes with a fixed proportion
	 * of test train.
	 * NEEDS TESTING
 * * @param allData: training instances to base selection on
* @param c: classifier to rank attributes with 
 * @param testingProportion: test/train prop
 * @return array of doubles. Really not sure what this does!!
	 */
	
	public static double[] attributeForwardSelection(Instances allData, Classifier c, double testingProportion)
	{
		ASEvaluation as = new GainRatioAttributeEval();
//Split data into Test/Train		
		allData.randomize(new Random());
		int size=allData.numInstances();
		int testSize=(int)(testingProportion*size);
		int trainSize=size-testSize;
		Instances trainData=new Instances(allData,0,trainSize);
		Instances testData=new Instances(allData,trainSize,testSize);
		double[] accuracies= new double[allData.numAttributes()];
//Rank attributes on training data		
		int[] atts=simpleAttributeRanking(trainData,as);
		int removalPos;
// Remove an attribute one at a time in reverse order of importance
// Construct classifier on train, evaluate on test and store		

		for(int i=0;i<accuracies.length;i++)
		{

			try{
				c.buildClassifier(trainData);
				accuracies[i]=ClassifierTools.accuracy(testData,c);
			//Record accuracies
				System.out.println("Nos Attributes train = "+trainData.numAttributes()+"  test = "+testData.numAttributes()+" Accuracy = "+accuracies[i]);
			//Remove attribute 
				removalPos=atts[atts.length-2-i];	
				trainData.deleteAttributeAt(removalPos);
				testData.deleteAttributeAt(removalPos);
				//Recalculate all the postions! What a pain, will be very inefficient with competition data
				for(int j=0;j<atts.length;j++)
				{
					if(atts[j]>=removalPos)
						atts[j]--;
				}
			}
			catch(Exception e){System.out.println("Error in build classifier XXSS");}
		}
		return accuracies;
	}


}
