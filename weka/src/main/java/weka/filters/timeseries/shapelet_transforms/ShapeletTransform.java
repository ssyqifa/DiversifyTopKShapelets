/*
      * copyright: Anthony Bagnall
* NOTE: As shapelet extraction can be time consuming, there is an option to output shapelets
 * to a text file (Default location is in the root dir of the project, file name "defaultShapeletOutput.txt").
 *
 * Default settings are TO NOT PRODUCE OUTPUT FILE - unless file name is changed, each successive filter will
 * overwrite the output (see "setLogOutputFile(String fileName)" to change file dir and name).
 *
 * To reconstruct a filter from this output, please see the method "createFilterFromFile(String fileName)".
 */

package weka.filters.timeseries.shapelet_transforms;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Scanner;
import java.util.TreeMap;
import weka.core.*;
import weka.core.shapelet.*;
import weka.filters.SimpleBatchFilter;
/**
 * A filter to transform a dataset by k shapelets. Once built on a training set, the
 * filter can be used to transform subsequent datasets using the extracted shapelets.
 * <p>
 * See <a href="http://delivery.acm.org/10.1145/2340000/2339579/p289-lines.pdf?ip=139.222.14.198&acc=ACTIVE%20SERVICE&CFID=221649628&CFTOKEN=31860141&__acm__=1354814450_3dacfa9c5af84445ea2bfd7cc48180c8">Lines, J., Davis, L., Hills, J., Bagnall, A.: A shapelet transform for time series classification. In: Proc. 18th ACM SIGKDD (2012)</a>
 * @author Jason Lines
 */
public class ShapeletTransform extends SimpleBatchFilter{

    @Override
    public String globalInfo() {
        throw new UnsupportedOperationException("Not supported yet.");
    }


    protected boolean supressOutput = false; // defaults to print in System.out AS WELL as file, set to true to stop printing to console

    protected int minShapeletLength;
    protected int maxShapeletLength;
    protected int numShapelets;
    protected boolean shapeletsTrained;
    protected ArrayList<Shapelet> shapelets;
    protected String ouputFileLocation = "defaultShapeletOutput.txt"; // default store location
    protected boolean recordShapelets = true; // default action is to write an output file
    
    public static int DEFAULT_NUMSHAPELETS=100;
    public static int DEFAULT_MINSHAPELETLENGTH=3;
    public static int DEFAULT_MAXSHAPELETLENGTH=30;
  
    protected QualityMeasures.ShapeletQualityMeasure qualityMeasure;
    protected QualityMeasures.ShapeletQualityChoice qualityChoice;
    protected boolean useCandidatePruning;
    protected int candidatePruningStartPercentage;
            
    //Variables for experiments
    private static long subseqDistOpCount;
    
    /**
     * Default constructor; Quality measure defaults to information gain.
     */
    public ShapeletTransform(){
        this(DEFAULT_NUMSHAPELETS,DEFAULT_MINSHAPELETLENGTH,DEFAULT_MAXSHAPELETLENGTH,QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);
    }

    /**
     * Single param constructor: 
     * Quality measure defaults to information gain.
     * @param k the number of shapelets to be generated
     */
    public ShapeletTransform(int k){
        this(k,DEFAULT_MINSHAPELETLENGTH,DEFAULT_MAXSHAPELETLENGTH,QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);
    }
    /**
     * Full constructor to create a usable filter. Quality measure defaults to information gain.
     *
     * @param k the number of shapelets to be generated
     * @param minShapeletLength minimum length of shapelets
     * @param maxShapeletLength maximum length of shapelets
     */
    public ShapeletTransform(int k, int minShapeletLength, int maxShapeletLength){
        this(k,minShapeletLength,maxShapeletLength,QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);

    }

    /**
     * Full, exhaustive, constructor for a filter. Quality measure set via enum, invalid 
     * selection defaults to information gain.
     *
     * @param k the number of shapelets to be generated
     * @param minShapeletLength minimum length of shapelets
     * @param maxShapeletLength maximum length of shapelets
     * @param qualityChoice the shapelet quality measure to be used with this filter
     */
    public ShapeletTransform(int k, int minShapeletLength, int maxShapeletLength, weka.core.shapelet.QualityMeasures.ShapeletQualityChoice qualityChoice){
        this.minShapeletLength = minShapeletLength;
        this.maxShapeletLength = maxShapeletLength;
        this.numShapelets = k;
        this.shapelets = new ArrayList<Shapelet>();
        this.shapeletsTrained = false;
        this.useCandidatePruning = false;
        this.qualityChoice=qualityChoice;
        switch(qualityChoice){
            case F_STAT:
                this.qualityMeasure = new QualityMeasures.FStat();
                break;
            case KRUSKALL_WALLIS:
                this.qualityMeasure = new QualityMeasures.KruskalWallis();
                break;
            case MOODS_MEDIAN:
                this.qualityMeasure = new QualityMeasures.MoodsMedian();
                break;
            default:
                this.qualityMeasure = new QualityMeasures.InformationGain();
        }
    }

    /**
     * Supresses filter output to the console; useful when running timing experiments.
     */
    public void supressOutput(){
        this.supressOutput=true;
    }

    /**
     * Use candidate pruning technique when checking candidate quality. This 
     * speeds up the transform processing time. 
     */
    public void useCandidatePruning(){
        this.useCandidatePruning = true;
        this.candidatePruningStartPercentage = 10;
    }
    public void setCandidatePruning(boolean f){
        this.useCandidatePruning = f;
        if(f)
            this.candidatePruningStartPercentage = 10;
        else    //Not necessary
            this.candidatePruningStartPercentage = 100;
            
    }
       
    /**
     * Use candidate pruning technique when checking candidate quality. This 
     * speeds up the transform processing time. 
     * @param percentage the percentage of data to be precocessed before pruning
     * is initiated. In most cases the higher the percentage the less effective 
     * pruning becomes
     */
    public void useCandidatePruning(int percentage){
        this.useCandidatePruning = true;
        this.candidatePruningStartPercentage = percentage;
    }
    
    /**
     * Mutator method to set the number of shapelets to be stored by the filter.
     *
     * @param k the number of shapelets to be generated
     */
    public void setNumberOfShapelets(int k){
        this.numShapelets = k;
    }
    public int getNumberOfShapelets(){ return numShapelets;
    }

    /**
     *  Mutator method to set the minimum and maximum shapelet lengths for the filter.
     *
     * @param minShapeletLength minimum length of shapelets
     * @param maxShapeletLength maximum length of shapelets
     */
    public void setShapeletMinAndMax(int minShapeletLength, int maxShapeletLength){
        this.minShapeletLength = minShapeletLength;
        this.maxShapeletLength = maxShapeletLength;
    }

    /**
     * Mutator method to set the quality measure used by the filter. As with constructors, default 
     * selection is information gain unless another valid selection is specified.
     *
     * @param qualityChoice the quality measure to be used with this filter
     */
    public QualityMeasures.ShapeletQualityChoice getQualityMeasure(){
        return qualityChoice;
    }
    public void setQualityMeasure(QualityMeasures.ShapeletQualityChoice qualityChoice){
        this.qualityChoice=qualityChoice;
        switch(qualityChoice){
            case F_STAT:
                this.qualityMeasure = new QualityMeasures.FStat();
                break;
            case KRUSKALL_WALLIS:
                this.qualityMeasure = new QualityMeasures.KruskalWallis();
                break;
            case MOODS_MEDIAN:
                this.qualityMeasure = new QualityMeasures.MoodsMedian();
                break;
            default:
                this.qualityMeasure = new QualityMeasures.InformationGain();
        }
    }
    
    /**
     * Sets the format of the filtered instances that are output. I.e. will include k attributes each shapelet 
     * distance and a class value
     *
     * @param inputFormat the format of the input data
     * @return a new Instances object in the desired output format
     * @throws Exception if all required parameters of the filter are not initialised correctly
     */
    @Override
    protected Instances determineOutputFormat(Instances inputFormat) throws Exception{

        if(this.numShapelets < 1){
            throw new Exception("ShapeletFilter not initialised correctly - please specify a value of k that is greater than or equal to 1");
        }

        //Set up instances size and format.
        //int length = this.numShapelets;
        int length = this.shapelets.size();
        FastVector atts = new FastVector();
        String name;
        for(int i = 0; i < length; i++){
            name = "Shapelet_" + i;
            atts.addElement(new Attribute(name));
        }

        if(inputFormat.classIndex() >= 0){ //Classification set, set class
            //Get the class values as a fast vector
            Attribute target = inputFormat.attribute(inputFormat.classIndex());

            FastVector vals = new FastVector(target.numValues());
            for(int i = 0; i < target.numValues(); i++){
                vals.addElement(target.value(i));
            }
            atts.addElement(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(), vals));
        }
        Instances result = new Instances("Shapelets" + inputFormat.relationName(), atts, inputFormat.numInstances());
        if(inputFormat.classIndex() >= 0){
            result.setClassIndex(result.numAttributes() - 1);
        }
        return result;
    }


    /**
     * The main logic of the filter; when called for the first time, k shapelets are extracted from the input Instances 'data'.
     * The input 'data' is transformed by the k shapelets, and the filtered data is returned as an output.
     * <p>
     * If called multiple times, shapelet extraction DOES NOT take place again; once k shapelets are established from the initial
     * call to process(), the k shapelets are used to transform subsequent Instances. 
     * <p>
     * Intended use: <p>
     * 1. Extract k shapelets from raw training data to build filter; <p>
     * 2. Use the filter to transform the raw training data into transformed training data; <p>
     * 3. Use the filter to transform the raw testing data into transformed testing data (e.g. filter never extracts shapelets from training data, therefore avoiding bias); <p>
     * 4. Build a classifier using transformed training data, perform classification on transformed test data.
     *
     * @param data the input data to be transformed (and to find the shapelets if this is the first run)
     * @return the transformed representation of data, according to the distances from each instance to each of the k shapelets
     * @throws Exception if the number of shapelets or the length parameters specified are incorrect
     */
    @Override
    public Instances process(Instances data) throws Exception{
        if(this.numShapelets < 1){
            throw new Exception("Number of shapelets initialised incorrectly - please select value of k greater than or equal to 1 (Usage: setNumberOfShapelets");
        }

        int maxPossibleLength = data.instance(0).numAttributes() - 1;
        if(data.classIndex() < 0) {
            throw new Exception("Require that the class be set for the ShapeletTransform");
        }

        if(this.minShapeletLength < 1 || this.maxShapeletLength < 1 || this.maxShapeletLength < this.minShapeletLength || this.maxShapeletLength > maxPossibleLength){
            throw new Exception("Shapelet length parameters initialised incorrectly");
        }

        //Sort data in round robin order
        data = QualityBound.roundRobinData(data);
        
        if(this.shapeletsTrained == false){ // shapelets discovery has not yet been caried out, so do so
            this.shapelets = findBestKShapeletsCache(this.numShapelets, data, this.minShapeletLength, this.maxShapeletLength); // get k shapelets ATTENTION
            this.shapeletsTrained = true;
            if(!supressOutput){
                System.out.println(shapelets.size()+" Shapelets have been generated");
            }
        }
        
        Instances output = determineOutputFormat(data);

        // for each data, get distance to each shapelet and create new instance
        for(int i = 0; i < data.numInstances(); i++){ // for each data
            Instance toAdd = new Instance(this.shapelets.size() + 1);
            int shapeletNum = 0;
            for(Shapelet s: this.shapelets){
                double dist = subseqDistance(s.content, data.instance(i));
                toAdd.setValue(shapeletNum++, dist);
            }
            toAdd.setValue(this.shapelets.size(), data.instance(i).classValue());
            output.add(toAdd);
        }
        return output;
    }


    /**
     * Set file path for the filter log. Filter log includes shapelet quality, seriesId, startPosition, and content for each shapelet.
     * @param fileName the updated file path of the filter log
     */
    public void setLogOutputFile(String fileName){
        this.recordShapelets = true;
        this.ouputFileLocation = fileName;
    }

    /**
     * Turns off log saving; useful for timing experiments where speed is essential.
     */
    public void turnOffLog(){
        this.recordShapelets = false;
    }

    /**
     * Private method for extracting k shapelets.
     *
     * @param numShapelets the target number of shapelets to generate
     * @param data the data that the shapelets will be taken from
     * @param minShapeletLength the minimum length of possible shapelets
     * @param maxShapeletLength the maximum length of possible shapelets
     * @return an ArrayList of ShapeletTransform objects in order of their fitness (by infoGain, seperationGap then shortest length)
     */
    public ArrayList<Shapelet> findBestKShapeletsCache(int numShapelets, Instances data, int minShapeletLength, int maxShapeletLength)throws Exception{
                            
        ArrayList<Shapelet> kShapelets = new ArrayList<Shapelet>();         // store (upto) the best k shapelets overall
        ArrayList<Shapelet> seriesShapelets;                                // temp store of all shapelets for each time series
        
        /* new version to allow caching:
         * - for all time series, calculate the gain of all candidates of all possible lengths
         * - insert into a strucutre in order of fitness - arraylist with comparable implementation of shapelets
         * - once all candidates for a series are established, integrate into store of k best
        */

        TreeMap<Double, Integer> classDistributions = getClassDistributions(data); // used to calc info gain
                
        //for all time series
        if(!supressOutput){
            System.out.println("Processing data: ");
        }

        int numInstances = data.numInstances();
        for(int i = 0; i < numInstances; i++){

            if(!supressOutput && (i==0 || i%(numInstances/4)==0)){
                System.out.println("Currently processing instance "+(i+1)+" of "+ numInstances);
            }

            double[] wholeCandidate = data.instance(i).toDoubleArray();
            seriesShapelets = new ArrayList<Shapelet>();

            for(int length = minShapeletLength; length <= maxShapeletLength; length++){

                //for all possible starting positions of that length
                for(int start = 0; start <= wholeCandidate.length - length-1; start++){ //-1 = avoid classVal - handle later for series with no class val
                    // CANDIDATE ESTABLISHED - got original series, length and starting position
                    // extract relevant part into a double[] for processing
                    double[] candidate = new double[length];
                    for(int m = start; m < start + length; m++){
                        candidate[m - start] = wholeCandidate[m];
                    }

                    // znorm candidate here so it's only done once, rather than in each distance calculation
                    candidate = zNorm(candidate, false);
                        
                    //Initialize bounding algorithm for current candidate
                    QualityBound.ShapeletQualityBound qualityBound = initializeQualityBound(classDistributions);
        
                    //Set bound of the bounding algorithm
                    if(qualityBound != null && kShapelets.size() == numShapelets){
                        qualityBound.setBsfQulity(kShapelets.get(numShapelets-1).qualityValue);
                    }
                            
                    Shapelet candidateShapelet = checkCandidate(candidate, data, i, start, classDistributions, qualityBound);
                    
                    //If shapelet was pruned then null will be returned so need to check for that
                    if(candidateShapelet != null){
                        seriesShapelets.add(candidateShapelet);
                    }
                }
            }
            // now that we have all shapelets, self similarity can be fairly assessed without fear of removing potentially
            // good shapelets
            Collections.sort(seriesShapelets);
            seriesShapelets = removeSelfSimilar(seriesShapelets);
            kShapelets = combine(numShapelets,kShapelets,seriesShapelets);
        }
        
        this.numShapelets = kShapelets.size();

        if(this.recordShapelets){
            FileWriter out = new FileWriter(this.ouputFileLocation);
            for(int i = 0; i < kShapelets.size();i++){
                out.append(kShapelets.get(i).qualityValue+","+kShapelets.get(i).seriesId+","+kShapelets.get(i).startPos+"\n");

                double[] shapeletContent = kShapelets.get(i).content;
                for(int j = 0; j < shapeletContent.length; j++){
                    out.append(shapeletContent[j]+",");
                }
                out.append("\n");
            }
            out.close();
        }
        if(!supressOutput){
            System.out.println();
            System.out.println("Output Shapelets:");
            System.out.println("-------------------");
            System.out.println("informationGain,seriesId,startPos");
            System.out.println("<shapelet>");
            System.out.println("-------------------");
            System.out.println();
            for(int i = 0; i < kShapelets.size();i++){
                System.out.println(kShapelets.get(i).qualityValue+","+kShapelets.get(i).seriesId+","+kShapelets.get(i).startPos);
                double[] shapeletContent = kShapelets.get(i).content;
                for(int j = 0; j < shapeletContent.length; j++){
                    System.out.print(shapeletContent[j]+",");
                }
                System.out.println();
            }
        }


        return kShapelets;
    }

    protected QualityBound.ShapeletQualityBound initializeQualityBound(TreeMap<Double, Integer> classDist){
        if(useCandidatePruning){
            if(qualityMeasure instanceof QualityMeasures.InformationGain){
                return new QualityBound.InformationGainBound(classDist, candidatePruningStartPercentage);
            }else if(qualityMeasure instanceof QualityMeasures.MoodsMedian){
                return new QualityBound.MoodsMedianBound(classDist, candidatePruningStartPercentage);    
            }else if(qualityMeasure instanceof QualityMeasures.FStat){
                return new QualityBound.FStatBound(classDist, candidatePruningStartPercentage);
            }else if(qualityMeasure instanceof QualityMeasures.KruskalWallis){
                return new QualityBound.KruskalWallisBound(classDist, candidatePruningStartPercentage);
            }
        }
        return null;
    }
    
    /**
     * protected method to remove self-similar shapelets from an ArrayList (i.e. if they come from the same series
     * and have overlapping indicies)
     *
     * @param shapelets the input Shapelets to remove self similar ShapeletTransform objects from
     * @return a copy of the input ArrayList with self-similar shapelets removed
     */
    protected static ArrayList<Shapelet> removeSelfSimilar(ArrayList<Shapelet> shapelets){
        // return a new pruned array list - more efficient than removing
        // self-similar entries on the fly and constantly reindexing
        ArrayList<Shapelet> outputShapelets = new ArrayList<Shapelet>();
        boolean[] selfSimilar = new boolean[shapelets.size()];

        // to keep track of self similarity - assume nothing is similar to begin with
        for(int i = 0; i < shapelets.size(); i++){
            selfSimilar[i] = false;
        }

        for(int i = 0; i < shapelets.size();i++){
            if(selfSimilar[i]==false){
                outputShapelets.add(shapelets.get(i));
                for(int j = i+1; j < shapelets.size(); j++){
                    if(selfSimilar[j]==false && selfSimilarity(shapelets.get(i),shapelets.get(j))){ // no point recalc'ing if already self similar to something
                        selfSimilar[j] = true;
                    }
                }
            }
        }
        return outputShapelets;
    }


    /**
     * Private method to combine two ArrayList collections of ShapeletTransform objects.
     *
     * @param k the maximum number of shapelets to be returned after combining the two lists
     * @param kBestSoFar the (up to) k best shapelets that have been observed so far, passed in to combine with shapelets from a new series
     * @param timeSeriesShapelets the shapelets taken from a new series that are to be merged in descending order of fitness with the kBestSoFar
     * @return an ordered ArrayList of the best k (or less) ShapeletTransform objects from the union of the input ArrayLists
     */

    //NOTE: could be more efficient here
    protected ArrayList<Shapelet> combine(int k, ArrayList<Shapelet> kBestSoFar, ArrayList<Shapelet> timeSeriesShapelets){

        ArrayList<Shapelet> newBestSoFar = new ArrayList<Shapelet>();
        for(int i = 0; i < timeSeriesShapelets.size();i++){
            kBestSoFar.add(timeSeriesShapelets.get(i));
        }
        Collections.sort(kBestSoFar);
        if(kBestSoFar.size()<k) { // no need to return up to k, as there are not k shapelets yet
            return kBestSoFar;
        } 

        for(int i = 0; i < k; i++){
            newBestSoFar.add(kBestSoFar.get(i));
        }

        return newBestSoFar;
    }

    /**
     *  Private method to calculate the class distributions of a dataset. Main purpose is for computing shapelet qualities.
     *
     * @param data the input data set that the class distributions are to be derived from
     * @return a TreeMap<Double, Integer> in the form of <Class Value, Frequency>
     */
    public static TreeMap<Double, Integer> getClassDistributions(Instances data){
        TreeMap<Double, Integer> classDistribution = new TreeMap<Double, Integer>();
        double classValue;
        for(int i = 0; i < data.numInstances(); i++){
            classValue = data.instance(i).classValue();
            boolean classExists = false;
            for(Double d : classDistribution.keySet()){
                if(d == classValue){
                    int temp = classDistribution.get(d);
                    temp++;
                    classDistribution.put(classValue, temp);
                    classExists = true;
                }
            }
            
            if(classExists == false){
                classDistribution.put(classValue, 1);
            }
        }
        return classDistribution;
    }

    /**
     * protected method to check a candidate shapelet. Functions by passing in the raw data, and returning an assessed ShapeletTransform object.
     *
     * @param candidate the data from the candidate ShapeletTransform
     * @param data the entire data set to compare the candidate to
     * @param seriesId series id from the dataset that the candidate came from
     * @param startPos start position in the series where the candidate came from
     * @param classDistribution a TreeMap<Double, Integer> in the form of <Class Value, Frequency> to describe the dataset composition
     * @return a fully-computed ShapeletTransform, including the quality of this candidate
     */
    protected Shapelet checkCandidate(double[] candidate, Instances data, int seriesId, int startPos, TreeMap classDistribution, QualityBound.ShapeletQualityBound qualityBound){
        
        // create orderline by looping through data set and calculating the subsequence
        // distance from candidate to all data, inserting in order.
        ArrayList<OrderLineObj> orderline = new ArrayList<OrderLineObj>();

        boolean pruned = false;
        
        for(int i = 0; i < data.numInstances(); i++){
            //Check if it is possible to prune the candidate
            if(qualityBound != null){
                if(qualityBound.pruneCandidate()){
                    pruned = true;
                    break;
                }
            }

            double distance = 0.0;
            if(i != seriesId){
                distance = subseqDistance(candidate, data.instance(i));  
            }
            
            double classVal = data.instance(i).classValue();
            // without early abandon, it is faster to just add and sort at the end
            orderline.add(new OrderLineObj(distance, classVal));
            
            //Update qualityBound - presumably each bounding method for different quality measures will have a different update procedure.
            if(qualityBound != null){
                qualityBound.updateOrderLine(orderline.get(orderline.size()-1));
            }
        }

        // note: early abandon entropy pruning would appear here, but has been ommitted
        // in favour of a clear multi-class information gain calculation. Could be added in
        // this method in the future for speed up, but distance early abandon is more important
        
        //If shapelet is pruned then it should no longer be considered in further processing
        if(pruned){
            return null;
        }else{
            // create a shapelet object to store all necessary info, i.e.
            Shapelet shapelet = new Shapelet(candidate, seriesId, startPos, this.qualityMeasure); 
            shapelet.calculateQuality(orderline, classDistribution);
            return shapelet;
        }
    }
    
    /**
     * Calculate the distance between a candidate series and an Instance object
     *
     * @param candidate a double[] representation of a shapelet candidate
     * @param timeSeriesIns an Instance object of a whole time series
     * @return the distance between a candidate and a time series
     */
    protected double subseqDistance(double[] candidate, Instance timeSeriesIns){
        return subsequenceDistance(candidate, timeSeriesIns);
    }    
    
    public static double subsequenceDistance(double[] candidate, Instance timeSeriesIns){
        double[] timeSeries = timeSeriesIns.toDoubleArray();
        return subsequenceDistance(candidate, timeSeries);
    }


    /**
     * Calculate the distance between a shapelet candidate and a full time series (both double[]).
     *
     * @param candidate a double[] representation of a shapelet candidate
     * @param timeSeries a double[] representation of a whole time series (inc. class value)
     * @return the distance between a candidate and a time series
     */
    public static double subsequenceDistance(double[] candidate, double[] timeSeries){

        double bestSum = Double.MAX_VALUE;
        double sum;
        double[] subseq;

        // for all possible subsequences of two
        for(int i = 0; i <= timeSeries.length - candidate.length - 1; i++){
            sum = 0;
            // get subsequence of two that is the same lenght as one
            subseq = new double[candidate.length];

            for(int j = i; j < i + candidate.length; j++){
                subseq[j - i] = timeSeries[j];
                
                //Keep count of fundamental ops for experiment
                subseqDistOpCount++;
            }
            subseq = zNormalise(subseq, false); // Z-NORM HERE
            
            //Keep count of fundamental ops for experiment
            subseqDistOpCount += 3 * subseq.length;
            
            for(int j = 0; j < candidate.length; j++){
                sum +=(candidate[j] - subseq[j]) *(candidate[j] - subseq[j]);
                
                //Keep count of fundamental ops for experiment
                subseqDistOpCount++;
            }
            if(sum < bestSum){
                bestSum = sum;
            }
        }
        return (bestSum == 0.0) ? 0.0 : (1.0 / candidate.length * bestSum); 
    }
    protected double[] zNorm(double[] input, boolean classValOn){        
        return ShapeletTransform.zNormalise(input, classValOn);
    }
       
    /**
     * Z-Normalise a time series
     *
     * @param input the input time series to be z-normalised
     * @param classValOn specify whether the time series includes a class value (e.g. an full instance might, a candidate shapelet wouldn't)
     * @return a z-normalised version of input
     */
    public static double[] zNormalise(double[] input, boolean classValOn){
        double mean;
        double stdv;

        double classValPenalty = 0;
        if(classValOn){
            classValPenalty = 1;
        }
        double[] output = new double[input.length];
        double seriesTotal = 0;

        for(int i = 0; i < input.length - classValPenalty; i++){
            seriesTotal += input[i];
        }

        mean = seriesTotal /(input.length - classValPenalty);
        stdv = 0;
        for(int i = 0; i < input.length - classValPenalty; i++){
            stdv +=(input[i] - mean) *(input[i] - mean);
        }

        stdv = stdv / (input.length - classValPenalty);
        stdv = Math.sqrt(stdv);

        for(int i = 0; i < input.length - classValPenalty; i++){
            output[i] =(input[i] - mean) / stdv;
        }

        if(classValOn == true){
            output[output.length - 1] = input[input.length - 1];
        }

        return output;
    }

    /**
     * Load a set of Instances from an ARFF
     *
     * @param fileName the file name of the ARFF
     * @return a set of Instances from the ARFF
     */
    public static Instances loadData(String fileName){
        Instances data = null;
        try{
            FileReader r;
            r = new FileReader(fileName);
            data = new Instances(r);

            data.setClassIndex(data.numAttributes() - 1);
        } catch(Exception e){
            System.out.println(" Error =" + e + " in method loadData");
            e.printStackTrace();
        }
        return data;
    }

    /**
     * A private method to assess the self similarity of two ShapeletTransform objects (i.e. whether they have overlapping indicies and
     * are taken from the same time series).
     *
     * @param shapelet the first ShapeletTransform object (in practice, this will be the dominant shapelet with quality >= candidate)
     * @param candidate the second ShapeletTransform
     * @return
     */
    private static boolean selfSimilarity(Shapelet shapelet, Shapelet candidate){
        if(candidate.seriesId == shapelet.seriesId){
            if(candidate.startPos >= shapelet.startPos && candidate.startPos < shapelet.startPos + shapelet.content.length){ //candidate starts within exisiting shapelet
                return true;
            }
            if(shapelet.startPos >= candidate.startPos && shapelet.startPos < candidate.startPos + candidate.content.length){
                return true;
            }
        }
        return false;
    }
    /**
     * A method to read in a ShapeletTransform log file to reproduce a ShapeletTransform
     * <p>
     * NOTE: assumes shapelets from log are Z-NORMALISED
     *
     * @param fileName the name and path of the log file
     * @return a duplicate ShapeletTransform to the object that created the original log file
     * @throws Exception
     */
    public static ShapeletTransform createFilterFromFile(String fileName) throws Exception{
        return createFilterFromFile(fileName, Integer.MAX_VALUE);
    }
    

    /**
     * A method to read in a ShapeletTransform log file to reproduce a ShapeletTransform,
     * <p>
     * NOTE: assumes shapelets from log are Z-NORMALISED
     *
     * @param fileName the name and path of the log file
     * @param maxShapelets
     * @return a duplicate ShapeletTransform to the object that created the original log file
     * @throws Exception
     */
    public static ShapeletTransform createFilterFromFile(String fileName, int maxShapelets) throws Exception{

        File input = new File(fileName);
        Scanner scan = new Scanner(input);
        scan.useDelimiter("\n");

        ShapeletTransform sf = new ShapeletTransform();
        ArrayList<Shapelet> shapelets = new ArrayList<Shapelet>();

        String shapeletContentString;
        ArrayList<Double> content;
        double[] contentArray;
        Scanner lineScan;

        int shapeletCount = 0;

        while(shapeletCount < maxShapelets && scan.hasNext()){
            scan.next();
            shapeletContentString = scan.next();

            lineScan = new Scanner(shapeletContentString);
            System.out.println(shapeletContentString);
            lineScan.useDelimiter(",");

            content = new ArrayList<Double>();
            while(lineScan.hasNext()){
                String next = lineScan.next().trim();
                if(!next.isEmpty()){
                    content.add(Double.parseDouble(next));
                    }
            }

            contentArray = new double[content.size()];
            for(int i = 0; i < content.size(); i++){
                contentArray[i] = content.get(i);
            }


            contentArray = zNormalise(contentArray, false);


            Shapelet s = new Shapelet(contentArray);
            shapelets.add(s);
            shapeletCount++;
        }
        sf.shapelets = shapelets;
        sf.shapeletsTrained = true;
        sf.numShapelets=shapelets.size();
        sf.setShapeletMinAndMax(1, 1);

        return sf;
    }

    public boolean foundShapelets(){ return shapeletsTrained;}
    
    /**
     * A method to obtain time taken to find a single best shapelet in the data set
     * @param data the data set to be processed
     * @param minShapeletLength minimum shapelet length
     * @param maxShapeletLength maximum shapelet length
     * @return time in seconds to find the best shapelet
     * @throws Exception 
     */
    public double timingForSingleShapelet(Instances data, int minShapeletLength, int maxShapeletLength) throws Exception {
        data = QualityBound.roundRobinData(data);
        long startTime = System.nanoTime();
        findBestKShapeletsCache(1, data, minShapeletLength, maxShapeletLength);
        long finishTime = System.nanoTime();
        return (double)(finishTime - startTime) / 1000000000.0;
    }
       
    public long opCountForSingleShapelet(Instances data, int minShapeletLength, int maxShapeletLength) throws Exception {
        data = QualityBound.roundRobinData(data);
        subseqDistOpCount = 0;
        findBestKShapeletsCache(1, data, minShapeletLength, maxShapeletLength);
        return subseqDistOpCount;
    }
        
    /**
     * An example use of a ShapeletTransform
     * @param args command line args. arg[0] should spcify a set of training instances to transform
     */
   public static void main(String[] args){
        try{
            // mandatory requirements:  numShapelets (k), min shapelet length, max shapelet length, input data
            // additional information:  log output dir

            // example filter, k = 10, minLength = 20, maxLength = 40, data = , output = exampleOutput.txt
            int k = 10;
            int minLength = 10;
            int maxLength = 20;
//            Instances data= ShapeletTransform.loadData("ItalyPowerDemand_TRAIN.arff"); // for example
            Instances data= ShapeletTransform.loadData(args[0]);

            ShapeletTransform sf = new ShapeletTransform(k, minLength, maxLength);
            sf.setQualityMeasure(QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);
            sf.setLogOutputFile("exampleOutput.txt"); // log file stores shapelet output

            // Note: sf.process returns a transformed set of Instances. The first time that
            //      thisFilter.process(data) is called, shapelet extraction occurs. Subsequent calls to process
            //      uses the previously extracted shapelets to transform the data. For example:
            //
            //      Instances transformedTrain = sf.process(trainingData); -> extracts shapelets and can be used to transform training data
            //      Instances transformedTest = sf.process(testData); -> uses shapelets extracted from trainingData to transform testData
            Instances transformed = sf.process(data);
        }catch(Exception e){
            e.printStackTrace();
        }
    }

}


