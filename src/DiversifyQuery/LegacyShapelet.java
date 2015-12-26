/*
 * A legacy shapelet class to recreate the results from the following paper:
 * Classification of Time Series by Shapelet Transformation,
 * Hills, J., Lines, J., Baranauskas, E., Mapp, J., and Bagnall, A.
 * Data Mining and Knowledge Discovery (2013)
 * 
 */
package DiversifyQuery;



import java.util.*;
import weka.core.shapelet.*;
import weka.core.*;
/**
 *
 * @author Jon Hills, j.hills@uea.ac.uk
 */
public class LegacyShapelet implements Comparable<LegacyShapelet>  {
    
    public double separationGap;
    public double splitThreshold;
    public double[] content;
    public int seriesId;
    public int startPos;
    public int length;
    public QualityMeasures.ShapeletQualityMeasure qualityType;
    public double qualityValue;
    
    public LegacyShapelet()
    {   }
    public LegacyShapelet(int seriesId, int pos, int length,double gain, double gap,double threshold){
        this.seriesId=seriesId;
        this.startPos=pos;
        this.length=length;
        this.qualityValue=gain;
        this.separationGap=gap;
        this.splitThreshold=threshold;
    }
    public LegacyShapelet(double[] content)
    {
        this.content = content;
    }
    
    public LegacyShapelet(double[] content, double qualValue, int seriesId, int startPos)
    {
        this.content = content;
        this.seriesId = seriesId;
        this.startPos = startPos;
        this.qualityValue = qualValue;
    }
    
    public LegacyShapelet(double[] content, int seriesId, int startPos, QualityMeasures.ShapeletQualityMeasure qualityChoice)
    {
        this.content = content;
        this.seriesId = seriesId;
        this.startPos = startPos;
        this.qualityType = qualityChoice;
    }
    
    public void calculateQuality(ArrayList<OrderLineObj> orderline, TreeMap<Double, Integer> classDistribution){
        this.qualityValue = this.qualityType.calculateQuality(orderline, classDistribution);
    }
    
    public void calcInfoGainAndThreshold(ArrayList<OrderLineObj> orderline, TreeMap<Double, Integer> classDistribution){
            // for each split point, starting between 0 and 1, ending between end-1 and end
            // addition: track the last threshold that was used, don't bother if it's the same as the last one
            double lastDist = orderline.get(0).getDistance(); // must be initialised as not visited(no point breaking before any data!)
            double thisDist = -1;

            double bsfGain = -1;
            double threshold = -1;

            for(int i = 1; i < orderline.size(); i++){
                thisDist = orderline.get(i).getDistance();
                if(i==1 || thisDist != lastDist){ // check that threshold has moved(no point in sampling identical thresholds)- special case - if 0 and 1 are the same dist

                    // count class instances below and above threshold
                    TreeMap<Double, Integer> lessClasses = new TreeMap<Double, Integer>();
                    TreeMap<Double, Integer> greaterClasses = new TreeMap<Double, Integer>();

                    for(double j : classDistribution.keySet()){
                        lessClasses.put(j, 0);
                        greaterClasses.put(j, 0);
                    }

                    int sumOfLessClasses = 0;
                    int sumOfGreaterClasses = 0;

                    //visit those below threshold
                    for(int j = 0; j < i; j++){
                        double thisClassVal = orderline.get(j).getClassVal();
                        int storedTotal = lessClasses.get(thisClassVal);
                        storedTotal++;
                        lessClasses.put(thisClassVal, storedTotal);
                        sumOfLessClasses++;
                    }

                    //visit those above threshold
                    for(int j = i; j < orderline.size(); j++){
                        double thisClassVal = orderline.get(j).getClassVal();
                        int storedTotal = greaterClasses.get(thisClassVal);
                        storedTotal++;
                        greaterClasses.put(thisClassVal, storedTotal);
                        sumOfGreaterClasses++;
                    }

                    int sumOfAllClasses = sumOfLessClasses + sumOfGreaterClasses;

                    double parentEntropy = QualityMeasures.InformationGain.entropy(classDistribution);

                    // calculate the info gain below the threshold
                    double lessFrac =(double) sumOfLessClasses / sumOfAllClasses;
                    double entropyLess = QualityMeasures.InformationGain.entropy(lessClasses);
                    // calculate the info gain above the threshold
                    double greaterFrac =(double) sumOfGreaterClasses / sumOfAllClasses;
                    double entropyGreater = QualityMeasures.InformationGain.entropy(greaterClasses);

                    double gain = parentEntropy - lessFrac * entropyLess - greaterFrac * entropyGreater;

                    if(gain > bsfGain){
                        bsfGain = gain;
                        threshold =(thisDist - lastDist) / 2 + lastDist;
                    }
                }
                lastDist = thisDist;
            }
            if(bsfGain >= 0){
//                this.informationGain = bsfGain;
                this.splitThreshold = threshold;
                this.separationGap = calculateSeparationGap(orderline, threshold);
            }
        }
    
    private double calculateSeparationGap(ArrayList<OrderLineObj> orderline, double distanceThreshold){

            double sumLeft = 0;
            double leftSize = 0;
            double sumRight = 0;
            double rightSize = 0;

            for(int i = 0; i < orderline.size(); i++){
                if(orderline.get(i).getDistance() < distanceThreshold){
                    sumLeft += orderline.get(i).getDistance();
                    leftSize++;
                } else{
                    sumRight += orderline.get(i).getDistance();
                    rightSize++;
                }
            }

            double thisSeparationGap = 1 / rightSize * sumRight - 1 / leftSize * sumLeft; //!!!! they don't divide by 1 in orderLine::minGap(int j)

            if(rightSize == 0 || leftSize == 0)
            {
                return -1; 
            }
            else
            {
                return thisSeparationGap;
            }
            
        }
    

    @Override
    public int compareTo(LegacyShapelet shapelet) {
        final int BEFORE = -1;
        final int EQUAL = 0;
        final int AFTER = 1;

        if(this.qualityValue != shapelet.qualityValue){
            if(this.qualityValue > shapelet.qualityValue){
                return BEFORE;
            }else{
                return AFTER;
            }
        }else{// if this.informationGain == shapelet.informationGain
                if(this.separationGap != shapelet.separationGap){
                    if(this.separationGap > shapelet.separationGap){
                        return BEFORE;
                    }else{
                        return AFTER;
                    }
                } else if(this.content.length != shapelet.content.length){
            if(this.content.length < shapelet.content.length){
                return BEFORE;
            }else{
                return AFTER;
            }
        }else{
            return EQUAL;
        }
    }
    } 
    
    public boolean isSimilar(LegacyShapelet shapelet, Instances data ){
        
        double dist=0;
        if(this.content.length>shapelet.content.length)
            dist=DivTopK.subsequenceDistance(shapelet.content, this.content);
        else
            dist=DivTopK.subsequenceDistance(this.content, shapelet.content);
        
        if(   (data.instance(this.seriesId).classValue()==data.instance(shapelet.seriesId).classValue())  && 
                (  (this.qualityValue>=shapelet.qualityValue && dist<this.splitThreshold) 
                || (this.qualityValue<=shapelet.qualityValue && dist<shapelet.splitThreshold) )   )
            return true;
        else
            return false;
    }
}

