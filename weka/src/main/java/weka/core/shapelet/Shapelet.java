    /**
     * A  class to represent a Shapelet object
     *      * copyright: Anthony Bagnall

     */
package weka.core.shapelet;

import java.io.File;
import java.util.ArrayList;
import java.util.TreeMap;


public class Shapelet implements Comparable<Shapelet>{
    public double[] content;
    public int seriesId;
    public int startPos;
    public QualityMeasures.ShapeletQualityMeasure qualityType;
    public double qualityValue;

    
    public Shapelet(double[] content, int seriesId, int startPos, QualityMeasures.ShapeletQualityMeasure qualityChoice){
        this.content = content;
        this.seriesId = seriesId;
        this.startPos = startPos;
        this.qualityType = qualityChoice;
    }

    //used when processing has been carried out in initial stage, then the shapelets read in via csv later
    public Shapelet(double[] content){
        this.content = content;
    }

    public void calculateQuality(ArrayList<OrderLineObj> orderline, TreeMap<Double, Integer> classDistribution){
        this.qualityValue = this.qualityType.calculateQuality(orderline, classDistribution);
    }

    // comparison 1: to determine order of shapelets in terms of info gain, then separation gap, then shortness
    @Override
    public int compareTo(Shapelet shapelet) {
        final int BEFORE = -1;
        final int EQUAL = 0;
        final int AFTER = 1;

        if(this.qualityValue != shapelet.qualityValue){
            if(this.qualityValue > shapelet.qualityValue){
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


    public double[] getContent(){    return content;    }

    public double getQualityValue(){    return qualityValue;    }

    public int getSeriesId(){    return seriesId;    }

    public int getStartPos(){    return startPos;    }


}



