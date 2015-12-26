/*
 * This class contains a static method for generating parameterised simulated
 * time-series datasets. The datasets are designed for shapelet approaches.
 * This will produce a two-class problem.
 */
package simulators;

import java.util.Random;
import weka.core.Instances;

/**
 *
 * @author Jon Hills
 * j.hills@uea.ac.uk
 */
public class SimulateDataset {
       
    
    /**
     * This method creates and returns a set of Instances representing a
     * simulated two-class time-series problem.
     * 
     * @param casesPerClass An array of two integers indicating the number of 
     * instances of class 0 and class 1.
     * @param seriesLength The length of the series. All time series in the
     * dataset are the same length.
     * @return Instances representing the time-series dataset. The Instances
     * returned will be empty if the casesPerClass parameter does not contain
     * exactly two values.
     */
    public static Instances getShapeletData(int []casesPerClass, int seriesLength)
    {
        
        if( casesPerClass.length != 2)
        {
            System.err.println("Incorrect parameters, dataset will not be co"
                    + "rrect.");
            int[] tmp = {0,0};
            casesPerClass = tmp;
            
        }
        
            ShapeletModel[] shapeMod = new ShapeletModel[2];
            populateShapeletArray(shapeMod, seriesLength);
            DataSimulator sim = new DataSimulator(shapeMod);
            return sim.generateDataSet(seriesLength, casesPerClass);          
    }
    
    /**
     * This is a support method for getShapeletData
     * 
     * @param array An array of two ShapeletModel2 models, representing the 
     * simulated shapes inserted into the respective classes.
     * @param seriesLength The length of the series.
     */
    private static void populateShapeletArray(ShapeletModel [] s, int seriesLength)
    {
        
        double[] p1={1,seriesLength};
        double[] p2={1,seriesLength};
        
//Create two ShapeleModels with different base Shapelets        
        s[0]=new ShapeletModel(p1);        
        ShapeletModel.ShapeType st=s[0].getShapeType();
        s[1]=new ShapeletModel(p2);
        while(st==s[1].getShapeType())
            s[1]=new ShapeletModel(p2);
    }
    
    /**
     * This method converts a set of Instances into two sets of instances,
     * randomly divided into training and test sets.
     * 
     * @param orig The full set of Instances.
     * @param trainSize The number of cases in the training set.
     * @return An array of Instances of length two.
     */
    public static Instances[] trainTestSplit(Instances orig, int trainSize)
    {
       
        Random r = new Random();
        
        orig.randomize(r);
        
        Instances tr = new Instances(orig,0,trainSize);
        Instances te = new Instances(orig,trainSize,orig.numInstances()-(trainSize));
        
         Instances[] tt = {tr,te};
              
        return tt;
    }
    
    /**
     * 
     * This creates a set of Instances representing a two-class problem with
     * a 50/50 balance of classes, 1100 instances of length 500. The set is 
     * then split into training and testing 100/1000.
     */
    public static void main(String[] args)
    {
        int[] casesPerClass = {550,550};
        int seriesLength = 500;
        int trainSize = 100;
        
        Instances data = SimulateDataset.getShapeletData(casesPerClass, seriesLength);
        Instances [] trainTest = SimulateDataset.trainTestSplit(data, trainSize);
    }
    
}
