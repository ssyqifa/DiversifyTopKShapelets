package weka.filters.timeseries.shapelet_transforms;

import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.TreeMap;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.shapelet.OrderLineObj;
import weka.core.shapelet.QualityBound;
import weka.core.shapelet.QualityMeasures.ShapeletQualityChoice;
import weka.core.shapelet.Shapelet;



/**
 * An optimised filter to transform a dataset by k shapelets.
     * copyright: Anthony Bagnall

 * @author Edgaras Baranauskas
 */
public class ShapeletTransformDistCaching extends ShapeletTransform{
    
    protected Stats stats;
    protected double[][] data;
    
    /**
     * Default constructor; Quality measure defaults to information gain.
     */
    public ShapeletTransformDistCaching(){
        super();
        stats = null;
        data = null;
    }

    /**
     * Single param constructor: filter is unusable until min/max params are initialised.
     * Quality measure defaults to information gain.
     * @param k the number of shapelets to be generated
     */
    public ShapeletTransformDistCaching(int k){
        super(k);
        stats = null;
        data = null;
    }

    /**
     * Full constructor to create a usable filter. Quality measure defaults to information gain.
     *
     * @param k the number of shapelets to be generated
     * @param minShapeletLength minimum length of shapelets
     * @param maxShapeletLength maximum length of shapelets
     */
    public ShapeletTransformDistCaching(int k, int minShapeletLength, int maxShapeletLength){
        super(k, minShapeletLength, maxShapeletLength);
        stats = null;
        data = null;
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
    public ShapeletTransformDistCaching(int k, int minShapeletLength, int maxShapeletLength, ShapeletQualityChoice qualityChoice){
        super(k, minShapeletLength, maxShapeletLength, qualityChoice);
        stats = null;
        data = null;
    }
    
    @Override
    public Instances process(Instances dataInst) throws Exception{
        if(this.numShapelets < 1){
            throw new Exception("Number of shapelets initialised incorrectly - please select value of k (Usage: setNumberOfShapelets");
        }

        int maxPossibleLength;
        if(dataInst.classIndex() < 0) {
            maxPossibleLength = dataInst.instance(0).numAttributes();
        }else{
            maxPossibleLength = dataInst.instance(0).numAttributes() - 1;
        }

        if(this.minShapeletLength < 1 || this.maxShapeletLength < 1 || this.maxShapeletLength < this.minShapeletLength || this.maxShapeletLength > maxPossibleLength){
            throw new Exception("Shapelet length parameters initialised incorrectly");
        }

        //Sort data in round robin order
        dataInst = QualityBound.roundRobinData(dataInst);
            
        if(this.shapeletsTrained == false){ // shapelets discovery has not yet been caried out, so do so
            this.shapelets = findBestKShapeletsCache(this.numShapelets, dataInst, this.minShapeletLength, this.maxShapeletLength); // get k shapelets ATTENTION
            this.shapeletsTrained = true;
            if(!supressOutput){
                System.out.println(shapelets.size()+" Shapelets have been generated");
            }
        }else{
            stats = null;
            data = null;
        }
        
        Instances output = determineOutputFormat(dataInst); 
        
        for(int i = 0; i < shapelets.size() + 1; i++){
            Shapelet s = null;
            if(i < shapelets.size()){
                s = shapelets.get(i);
                if(data != null && stats != null){
                    stats.computeStats(s.getSeriesId(), data);
                }
            }
            
           for(int j = 0; j < dataInst.numInstances(); j++){
               if(i < shapelets.size()){
                   double dist;
                   if(data != null && stats != null){
                        stats.setCurrentY(j);
                        dist = cachedSubsequenceDistance(s.getStartPos(), s.getContent().length, data[j].length, stats);
                   }else{
                        dist = subseqDistance(s.getContent(), dataInst.instance(j));
                   }

                    if(i == 0){
                         output.add(new Instance(this.shapelets.size() + 1));
                         output.instance(j).setValue(i, dist);
                    }else{
                         output.instance(j).setValue(i, dist); 
                    }
               }else{
                   output.instance(j).setValue(i, dataInst.instance(j).classValue());
               }
           }
        }
        return output;
    }
        
    @Override
    public ArrayList<Shapelet> findBestKShapeletsCache(int numShapelets, Instances dataInst, int minShapeletLength, int maxShapeletLength)throws Exception{
        ArrayList<Shapelet> kShapelets = new ArrayList<Shapelet>();                     // store (upto) the best k shapelets overall
        ArrayList<Shapelet> seriesShapelets;                                            // temp store of all shapelets for each time series
        TreeMap<Double, Integer> classDistributions = getClassDistributions(dataInst);  // used to calc info gain
                
        //Initialise stats object
        stats = new Stats();
        
        //Normalise all time series for furhter processing
        data = new double[dataInst.numInstances()][];
        for(int i = 0; i < dataInst.numInstances(); i++){
            data[i] = ShapeletTransform.zNormalise(dataInst.instance(i).toDoubleArray(), true);
        }
        
        if(!supressOutput) {
            System.out.println("Processing data: ");
        }
       
        //Every time series instance
        int numInstances = dataInst.numInstances();
        for(int i = 0; i < numInstances; i++){

            if(!supressOutput && (i==0 || i%(numInstances/4)==0)){
                System.out.println("Currently processing instance "+(i+1)+" of "+ numInstances);
            }

            seriesShapelets = new ArrayList<Shapelet>();

            //Compute statistics for the candidate series and every instance
            stats.computeStats(i, data);
            
            //Every possible lengh
            for(int length = minShapeletLength; length <= maxShapeletLength; length++){

                //for all possible starting positions of that length
                for(int start = 0; start <= data[i].length - length-1; start++){ //-1 = avoid classVal - handle later for series with no class val
                    // CANDIDATE ESTABLISHED - got original series, length and starting position
                    // extract relevant part into a double[] for processing
                    double[] candidate = new double[length];
                    for(int m = start; m < start + length; m++){
                        candidate[m - start] = data[i][m];
                    }

                    candidate = zNorm(candidate, false);
                    
                    //Initialize bounding algorithm for current candidated
                    QualityBound.ShapeletQualityBound qualityBound = initializeQualityBound(classDistributions);
                           
                    //Set bound of the bounding algorithm
                    if(qualityBound != null && kShapelets.size() == numShapelets){
                        qualityBound.setBsfQulity(kShapelets.get(numShapelets-1).qualityValue);
                    }
                    
                    Shapelet candidateShapelet = checkCandidate(candidate, dataInst, i, start, classDistributions, qualityBound);
                    
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

        
        //Record shapelets to log if required 
        if(this.recordShapelets){
            FileWriter out = new FileWriter(this.ouputFileLocation);
            for(int i = 0; i < kShapelets.size();i++){
                out.append(kShapelets.get(i).getQualityValue()+","+kShapelets.get(i).getSeriesId()+","+kShapelets.get(i).getStartPos()+"\n");

                double[] shapeletContent = kShapelets.get(i).getContent();
                for(int j = 0; j < shapeletContent.length; j++){
                    out.append(shapeletContent[j]+",");
                }
                out.append("\n");
            }
            out.close();
        }
        
        //Print processing information to command window if required
        if(!supressOutput){
            System.out.println();
            System.out.println("Output Shapelets:");
            System.out.println("-------------------");
            System.out.println("informationGain,seriesId,startPos");
            System.out.println("<shapelet>");
            System.out.println("-------------------");
            System.out.println();
            for(int i = 0; i < kShapelets.size();i++){
                System.out.println(kShapelets.get(i).getQualityValue()+","+kShapelets.get(i).getSeriesId()+","+kShapelets.get(i).getStartPos());
                double[] shapeletContent = kShapelets.get(i).getContent();
                for(int j = 0; j < shapeletContent.length; j++){
                    System.out.print(shapeletContent[j]+",");
                }
                System.out.println();
            }
        }

        return kShapelets;
    }
    
    @Override
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
                stats.setCurrentY(i);
                distance = cachedSubsequenceDistance(startPos, candidate.length, data.instance(i).numAttributes(), stats);    
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
     * Calculate the distance between a shapelet candidate and a full time series (both double[]).
     *
     * @param startPos start position of the candidate in the whole candidate series
     * @param subLength candidate length
     * @param seriesLength  series length
     * @param stats Stats object containing statistics computed for candidate and all time series
     * @return the distance beween a candidate and a time series
     */
    public static double cachedSubsequenceDistance(int startPos, int subLength, int seriesLength, Stats stats){
        
        double minSum = Double.MAX_VALUE;
        double xMean = stats.getMeanX(startPos, subLength);
        double xStdDev = stats.getStdDevX(startPos, subLength);
        double yMean;
        double yStdDev;
        double crossProd;
        
        if(xStdDev == 0.0){
            xStdDev = 1.0;
        }
        // Scan through all possible subsequences of two
        for(int v = 0; v < seriesLength - subLength; v++){
            yMean = stats.getMeanY(v, subLength);
            yStdDev = stats.getStdDevY(v, subLength);
            crossProd = stats.getSumOfProds(startPos, v, subLength);
            
            if(yStdDev == 0.0){
                yStdDev = 1.0;
            }
                    
            double cXY = (crossProd - (subLength * xMean * yMean))/(subLength * xStdDev * yStdDev);
            double dist = 2 * (1 - cXY);
            
            if(dist < minSum){
                minSum = dist;
            }
        }
        
        return minSum; 
    }
    
    /**
     * A class for holding relevant statistics for any given candidate series and
     * all time series
     */
    public static class Stats{
        
        private double[][] cummSums;
        private double[][] cummSqSums;
        private double[][][] crossProds;
        private int xIndex;
        private int yIndex;
        
        /**
         * Default constructor
         */
        public Stats(){
            cummSums = null;
            cummSqSums = null;
            crossProds = null;
            xIndex = -1;
            yIndex = -1;
        }
        
        /**
         * A method to retrieve cumulative sums for all time series processed so far
         * @return cumulative sums
         */
        public double[][] getCummSums(){
            return cummSums;
        }
        
        /**
         * A method to retrieve cumulative square sums for all time series processed so far
         * @return cumulative square sums
         */
        public double[][] getCummSqSums(){
            return cummSqSums;
        }
           
        /**
         * A method to retrieve cross products for candidate series. The cross
         * products are computed between candidate and all time series.
         * @return cross products
         */
        public double[][][] getCrossProds(){
            return crossProds;
        }
          
        /**
         * A method to set current time series that is being examined.
         * @param yIndex time series index
         */
        public void setCurrentY(int yIndex){
            this.yIndex = yIndex;
        }
        
        /**
         * A method to retrieve the mean value of a whole candidate sub-series.
         * @param startPos start position of the candidate
         * @param subLength length of the candidate
         * @return mean value of sub-series
         */
        public double getMeanX(int startPos, int subLength){
            return (cummSums[xIndex][startPos + subLength] - cummSums[xIndex][startPos]) / subLength;
        }
        
        /**
         * A method to retrieve the mean value for time series sub-series. Note
         * that the current Y must be set prior invoking this method.
         * @param startPos start position of the sub-series
         * @param subLength length of the sub-series
         * @return mean value of sub-series
         */
        public double getMeanY(int startPos, int subLength){
            return (cummSums[yIndex][startPos + subLength] - cummSums[yIndex][startPos]) / subLength;
        }
                
        /**
         * A method to retrieve the standard deviation of a whole candidate sub-series.
         * @param startPos start position of the candidate
         * @param subLength length of the candidate
         * @return standard deviation of the candidate sub-series
         */
        public double getStdDevX(int startPos, int subLength){
            return Math.sqrt(((cummSqSums[xIndex][startPos + subLength] - cummSqSums[xIndex][startPos]) / subLength) - (getMeanX(startPos, subLength) * getMeanX(startPos, subLength)));
        }
          
        /**
         * A method to retrieve the standard deviation for time series sub-series. 
         * Note that the current Y must be set prior invoking this method.
         * @param startPos start position of the sub-series
         * @param subLength length of the sub-series
         * @return standard deviation of sub-series
         */
        public double getStdDevY(int startPos, int subLength){
            return Math.sqrt(((cummSqSums[yIndex][startPos + subLength] - cummSqSums[yIndex][startPos]) / subLength) - (getMeanY(startPos, subLength) * getMeanY(startPos, subLength)));
        }
        
        /**
         * A method to retrieve the cross product of whole candidate sub-series 
         * and time series sub-series. Note that the current Y must be set prior 
         * invoking this method.
         * @param startX start position of the whole candidate sub-series
         * @param startY start position of the time series sub-series
         * @param length length of the both sub-series
         * @return sum of products for a given overlap between two sub=series
         */
        public double getSumOfProds(int startX, int startY, int length){
            return crossProds[yIndex][startX+length][startY+length] - crossProds[yIndex][startX][startY];
        }
                
        private double[][] computeCummSums(double[] currentSeries){

            double[][] output = new double[2][];
            
            //Compute stats for a given series instance
            for(int i = 0; i < currentSeries.length; i++){
                if(i == 0){ //Arrays have left margin of zero
                    //Allocate memory to store sums for current instance
                    output[0] = new double[currentSeries.length];
                    output[1] = new double[currentSeries.length];
                    output[0][i] = 0;
                    output[1][i] = 0;
                }else{
                    output[0][i] = output[0][i-1] + currentSeries[i-1];                         //Sum of vals
                    output[1][i] = output[1][i-1] + (currentSeries[i-1] * currentSeries[i-1]);  //Sum of squared vals
                }
            }

            return output;
        }
    
        private double[][] computeCrossProd(double[] x, double[] y){

            double[][] output = new double[x.length][y.length];


            for(int u = 0; u < x.length; u++){
                for(int v = 0; v < y.length; v++){
                    int t;  //abs(u-v)
                    if(u == 0 || v == 0){//Arrays have left margin of zero
                        output[u][v] = 0;
                    }else if(v < u){
                        t = u - v;
                        output[u][v] = output[u-1][v-1] + (x[v-1+t] * y[v-1]);
                    }else{//else v >= u
                        t = v - u;
                        output[u][v] = output[u-1][v-1] + (x[u-1] * y[u-1+t]);
                    }
                } 
            }

            return output;
        }
    
        /**
         * A method to compute statistics for a given candidate series index and
         * normalised time series
         * @param candidateInstIndex index of the candidate within the time series
         * database
         * @param data the normalised database of time series
         */
        public void computeStats(int candidateInstIndex, double[][] data){

            xIndex = candidateInstIndex;
            
            //Initialise stats caching arrays
            if(cummSums == null || cummSqSums == null){
                cummSums = new double[data.length][];
                cummSqSums = new double[data.length][];
            }
            
            crossProds = new double[data.length][][];

            //Process all instances
            for(int i = 0; i < data.length; i++){
                
                //Check if cummulative sums are already stored for corresponding instance
                if(cummSums[i] == null || cummSqSums[i] == null){
                    double[][] sums = computeCummSums(data[i]);
                    cummSums[i] = sums[0];
                    cummSqSums[i] = sums[1];
                }

                //Compute cross products between candidate series and current series
                crossProds[i] = computeCrossProd(data[candidateInstIndex], data[i]);
            }
        }
    }
    
    /**
     *
     * @param args
     */
    public static void main(String[] args){        
        //################ Test 1 ################
        System.out.println("1) Testing cached subsequence distance: ");
        
        //Create some time series for testing
        System.out.println("\n1.1) Example series: ");
        int seriesLength = 11;
        int numOfSeries = 5;
        double[][] data = new double[numOfSeries][seriesLength];
        
        int min = -5;
        int max = 5;
        for(int i = 0; i < numOfSeries; i++){
            for(int j = 0; j < seriesLength; j++){
                if(j == seriesLength-1){
                    data[i][j] = 0;
                }else{
                    data[i][j] = min + (int)(Math.random() * ((max - min) + 1));
                }
            }
            ShapeletTransformOnlineNorm.printSeries(data[i]);
        }
        
        //Normalise test time series
        System.out.println("\n1.2) Normalised example series: ");
        for(int i = 0; i < numOfSeries; i++){
            data[i] = ShapeletTransform.zNormalise(data[i], true);
            ShapeletTransformOnlineNorm.printSeries(data[i]);
        }
         
        double total = 0.0;
        for(int i = 0; i < numOfSeries; i++){
            for(int j = 0; j < seriesLength; j++){
                total += data[i][j];
            }
            
            System.out.println("sum for series "+ i+": " + total);
            total = 0.0;
        }
            
        seriesLength--;
        
        //Create stats object
        System.out.println("\n1.3) Unequal distances: ");
        Stats stats = new Stats();
        
        int minShapeletLength = seriesLength/2;
        int maxShapeletLength = seriesLength/2;
        
        //Every time series instance
        for(int i = 0; i < numOfSeries; i++){
            //Compute statistics for the candidate series and every instance
            stats.computeStats(i, data);
            
            //Every possible lengh
            for(int length = minShapeletLength; length <= maxShapeletLength; length++){

                //for all possible starting positions of that length
                for(int start = 0; start <= data[i].length - length-1; start++){ //-1 = avoid classVal - handle later for series with no class val
                    // CANDIDATE ESTABLISHED - got original series, length and starting position
                    // extract relevant part into a double[] for processing
                    double[] candidate = new double[length];
                    for(int m = start; m < start + length; m++){
                        candidate[m - start] = data[i][m];
                    }
                    
                    //Check individual components for completeness.
                    //System.out.println("MEAN: " + computeMean(candidate, false) + " = " + stats.getMeanX(start, length));
                    //System.out.println("STDV: " + computeStdv(candidate, false) + " = " + stats.getStdDevX(start, length));
                    
                    //Compute distance for each candidate
                    for(int j = 0; j < numOfSeries; j++){
                        stats.setCurrentY(j);
                        
                        double distanceCached = cachedSubsequenceDistance(start, candidate.length, data[j].length, stats);
                        double distanceOriginal = ShapeletTransform.subsequenceDistance(ShapeletTransform.zNormalise(candidate, false), data[j]);
                        if(Math.abs(distanceCached - distanceOriginal) > 0.0000000000000000001){
                            System.out.println("Candidate = " + i + ", startPos = " + start +", series = "+j+":\t"+ distanceCached + " = "+ distanceOriginal);
                        }
                    }
                }
            }
        }
    }
    
    //Used for testing
    private static double computeMean(double[] input, boolean classValOn){
        double mean;
        
        double classValPenalty = 0;
        if(classValOn){
            classValPenalty = 1;
        }
        
        double seriesTotal = 0;
        
        for(int i = 0; i < input.length - classValPenalty; i++){
            seriesTotal += input[i];
        }        
        mean = seriesTotal /(input.length - classValPenalty);
         
        return mean;
    }
    
    //Used for testing
    private static double computeStdv(double[] input, boolean classValOn){
        double mean;
        double stdv;
        
        double classValPenalty = 0;
        if(classValOn){
            classValPenalty = 1;
        }
        
        double seriesTotal = 0;
        double seriesTotal2 = 0;
        
        for(int i = 0; i < input.length - classValPenalty; i++){
            seriesTotal += input[i];
            seriesTotal2 += (input[i] * input[i]);
        }        
        mean = seriesTotal /(input.length - classValPenalty);
        stdv = Math.sqrt((seriesTotal2 - (mean * mean * (input.length - classValPenalty))) / (input.length - classValPenalty));
        return stdv;
    }
}
    
   
