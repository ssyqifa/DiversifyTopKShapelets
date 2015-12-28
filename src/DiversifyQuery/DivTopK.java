/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package DiversifyQuery;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Scanner;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IB1;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.shapelet_trees.FStatShapeletTreeWithInfoGain;
import weka.classifiers.trees.shapelet_trees.KruskalWallisTree;
import weka.classifiers.trees.shapelet_trees.MoodsMedianTree;
import weka.classifiers.trees.shapelet_trees.MoodsMedianTreeWithInfoGain;
import weka.classifiers.trees.shapelet_trees.ShapeletTreeClassifier;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author sun
 */
public class DivTopK {

    protected static final double ROUNDING_ERROR_CORRECTION = 0.000000000000001;

    public static ArrayList<Dresult<LegacyShapelet>> DResultSet = new ArrayList<>();

    private static Instances dataTrainTransformed;
    private static Instances dataTestTransformed;

    private static Classifier classifiers[];
    private static String classifierNames[];

    private static int classifierToProcessIndex;
private static String outFileName;
    /**
     * Load a set of Instances from an ARFF
     *
     * @param fileName the file name of the ARFF
     * @return a set of Instances from the ARFF
     */
    public static Instances loadData(String fileName) {
        Instances data = null;
        try {
            FileReader r;
            r = new FileReader(fileName);
            data = new Instances(r);

            data.setClassIndex(data.numAttributes() - 1);
        } catch (Exception e) {
            System.out.println(" Error =" + e + " in method loadData");
            e.printStackTrace();
        }
        return data;
    }

    /*
     * @param data the input data to be transformed (and to find the shapelets if this is the first run)
     * @return the transformed representation of data, according to the distances from each instance to each of the k shapelets
     * @throws Exception if the number of shapelets or the length parameters specified are incorrect
     */
    public Instances transformData(Instances data) throws Exception {
        ArrayList<LegacyShapelet> shapelets = new ArrayList<>();
        for (int i = 5; i <= 1; i--) {
            if (DResultSet.get(i).result.size() == i) {
                shapelets.addAll(DResultSet.get(i).result);
            }
        }
        if (shapelets.size() < 1) {
            throw new Exception("Number of shapelets initialised incorrectly - please select value of k greater than or equal to 1 (Usage: setNumberOfShapelets");
        }

        if (data.classIndex() < 0) {
            throw new Exception("Require that the class be set for the ShapeletTransform");
        }

        Instances output = determineOutputFormat(data, shapelets);

        // for each data, get distance to each shapelet and create new instance
        for (int i = 0; i < data.numInstances(); i++) { // for each data
            Instance toAdd = new Instance(shapelets.size() + 1);
            int shapeletNum = 0;
            for (LegacyShapelet s : shapelets) {
                double dist = subsequenceDistance(s.content, data.instance(i));
                toAdd.setValue(shapeletNum++, dist);
            }
            toAdd.setValue(shapelets.size(), data.instance(i).classValue());
            output.add(toAdd);
        }
        return output;
    }

    /**
     * Sets the format of the filtered instances that are output. I.e. will
     * include k attributes each shapelet distance and a class value
     *
     * @param inputFormat the format of the input data
     * @return a new Instances object in the desired output format
     * @throws Exception if all required parameters of the filter are not
     * initialised correctly
     */
    protected Instances determineOutputFormat(Instances inputFormat, ArrayList<LegacyShapelet> shapelets) throws Exception {

        //Set up instances size and format.
        //int length = this.numShapelets;
        int length = shapelets.size();
        FastVector atts = new FastVector();
        String name;
        for (int i = 0; i < length; i++) {
            name = "Shapelet_" + i;
            atts.addElement(new Attribute(name));
        }

        if (inputFormat.classIndex() >= 0) { //Classification set, set class
            //Get the class values as a fast vector
            Attribute target = inputFormat.attribute(inputFormat.classIndex());

            FastVector vals = new FastVector(target.numValues());
            for (int i = 0; i < target.numValues(); i++) {
                vals.addElement(target.value(i));
            }
            atts.addElement(new Attribute(inputFormat.attribute(inputFormat.classIndex()).name(), vals));
        }
        Instances result = new Instances("Shapelets" + inputFormat.relationName(), atts, inputFormat.numInstances());
        if (inputFormat.classIndex() >= 0) {
            result.setClassIndex(result.numAttributes() - 1);
        }
        return result;
    }

    public static ArrayList<LegacyShapelet> readShapelets(String fileName, Instances data) {
        ArrayList<LegacyShapelet> shapeletsList = new ArrayList<>();
        LegacyShapelet shapelet = new LegacyShapelet();
        int obj, pos, length;
        double gain, gap, distanceThreshold;
        try {
            Scanner sc = new Scanner(new File("shapeletsResult.txt"));
            while (sc.hasNext()) {
                shapelet = new LegacyShapelet(sc.nextInt(), sc.nextInt(), sc.nextInt(), sc.nextDouble(), sc.nextDouble(), sc.nextDouble());
                double[] contentValue = new double[shapelet.length];
                for (int i = 0; i < shapelet.length; i++) {
                    contentValue[i] = data.instance(shapelet.seriesId).value(shapelet.startPos + i);
                }
                shapelet.content = contentValue;
                shapeletsList.add(shapelet);
            }
        } catch (Exception e) {
            System.out.println("读取shapelets文件失败!");
            e.printStackTrace();
        }
        return shapeletsList;
    }

    /**
     *
     * @param candidate
     * @param timeSeriesIns
     * @return
     */
    public static double subsequenceDistance(double[] candidate, Instance timeSeriesIns) {
        double[] timeSeries = timeSeriesIns.toDoubleArray();
        return subsequenceDistance(candidate, timeSeries);
    }

    /**
     * Calculate the distance between a shapelet candidate and a full time
     * series (both double[]).
     *
     * @param candidate a double[] representation of a shapelet candidate
     * @param timeSeries a double[] representation of a whole time series (inc.
     * class value)
     * @return the distance between a candidate and a time series
     */
    public static double subsequenceDistance(double[] candidate, double[] timeSeries) {

        double bestSum = Double.MAX_VALUE;
        double sum;
        double[] subseq;
        candidate = zNormalise(candidate, false);

        // for all possible subsequences of two
        for (int i = 0; i <= timeSeries.length - candidate.length - 1; i++) {
            sum = 0;

            // get subsequence of two that is the same lenght as one
            subseq = new double[candidate.length];
            for (int j = i; j < i + candidate.length; j++) {
                subseq[j - i] = timeSeries[j];
            }
            subseq = zNormalise(subseq, false); // Z-NORM HERE

            for (int j = 0; j < candidate.length; j++) {
                sum += (candidate[j] - subseq[j]) * (candidate[j] - subseq[j]);
            }
            if (sum < bestSum) {
                bestSum = sum;
            }
        }
        return (bestSum == 0.0) ? 0.0 : Math.sqrt(1.0 / candidate.length * bestSum);
    }

    /**
     * Z-Normalise a time series
     *
     * @param input the input time series to be z-normalised
     * @param classValOn specify whether the time series includes a class value
     * (e.g. an full instance might, a candidate shapelet wouldn't)
     * @return a z-normalised version of input
     */
    public static double[] zNormalise(double[] input, boolean classValOn) {
        double mean;
        double stdv;

        double classValPenalty = 0;
        if (classValOn) {
            classValPenalty = 1;
        }
        double[] output = new double[input.length];
        double seriesTotal = 0;

        for (int i = 0; i < input.length - classValPenalty; i++) {
            seriesTotal += input[i];
        }

        mean = seriesTotal / (input.length - classValPenalty);
        stdv = 0;

        for (int i = 0; i < input.length - classValPenalty; i++) {
            stdv += (input[i] - mean) * (input[i] - mean);
        }

        stdv = stdv / (input.length - classValPenalty);
        if (stdv < ROUNDING_ERROR_CORRECTION) {
            stdv = 0.0;
        } else {
            stdv = Math.sqrt(stdv);
        }

        for (int i = 0; i < input.length - classValPenalty; i++) {
            if (stdv == 0.0) {
                output[i] = 0.0;
            } else {
                output[i] = (input[i] - mean) / stdv;
            }
        }

        if (classValOn == true) {
            output[output.length - 1] = input[input.length - 1];
        }

        return output;
    }

    //构造一个图，图顶点为shapelet，各顶点之间是否有边以两个shapelets是否相似为依据
    public ArrayList<GraphNode<LegacyShapelet>> constructShapeletGraph(ArrayList<LegacyShapelet> seriesShapelets, Instances data) {

        //读取文件
        ArrayList<GraphNode<LegacyShapelet>> Graph = new ArrayList<>();
        Collections.sort(seriesShapelets); //降序排序，
        for (int i = 0; i < seriesShapelets.size(); i++) {
            GraphNode node = new GraphNode();
            node.setVertex(seriesShapelets.get(i));
            Graph.add(node);
        }
        for (int i = 0; i < seriesShapelets.size(); i++) {
            for (int j = i + 1; j < seriesShapelets.size(); j++) {
                if (seriesShapelets.get(i).isSimilar(seriesShapelets.get(j), data)) {
                    if (Graph.get(i).getAdj() == null) {
                        ArrayList<LegacyShapelet> adjecentShapelets = new ArrayList<>();
                        adjecentShapelets.add(seriesShapelets.get(j));
                        Graph.get(i).setAdj(adjecentShapelets);
                    } else {
                        Graph.get(i).getAdj().add(seriesShapelets.get(j));
                    }
                    if (Graph.get(j).getAdj() == null) {
                        ArrayList<LegacyShapelet> adjecentShapelets = new ArrayList<LegacyShapelet>();
                        adjecentShapelets.add(seriesShapelets.get(i));
                        Graph.get(j).setAdj(adjecentShapelets);
                    } else {
                        Graph.get(j).getAdj().add(seriesShapelets.get(i));
                    }
                }
            }
        }

        for (int m = 0; m < Graph.size(); m++) {
            System.out.printf("第 %d 个顶点的相邻节点： \n", m + 1);
            System.out.println("id   pos     len     ");
            for (int n = 0; Graph.get(m).getAdj() != null && n < Graph.get(m).getAdj().size(); n++) {

                System.out.print(Graph.get(m).getAdj().get(n).seriesId + "    ");
                System.out.print(Graph.get(m).getAdj().get(n).startPos + "    ");
                System.out.print(Graph.get(m).getAdj().get(n).length + "    \n");
            }
            System.out.println("\n");
        }

        return Graph;
    }

    public ArrayList<Dresult<LegacyShapelet>> divAstar(ArrayList<GraphNode<LegacyShapelet>> G, int k) {
        MaxHeap<Entry> H = new MaxHeap<>();
        H.insert(new Entry());

        for (int i = 0; i < k + 1; i++) {
            Dresult<LegacyShapelet> d = new Dresult<>();
            d.score = -1;
            DResultSet.add(d);
        }
        for (int j = k; j >= 1; j--) {
            AStarSearch(G, H, k);
            ArrayList<Entry> arrayEntrys = H.getArray();
            for (int m = 1; m <= H.getCurrentSize(); m++) {
                Entry entry = new Entry();
                entry = arrayEntrys.get(m);

                double bound = AstarBound(G, entry, k);
                entry.setBound(bound);
                H.update(arrayEntrys, m, entry);
            }
        }
        return DResultSet;
    }

    public void AStarSearch(ArrayList<GraphNode<LegacyShapelet>> G, MaxHeap<Entry> H, int k) {
        while ((!H.isEmpty()) && H.getMax().getBound() > maxDresultSet(DResultSet)) {
            Entry e = new Entry();
            e = H.deleteMax();
            for (int i = e.pos + 1; i < G.size(); i++) {
                if (!andSet(G.get(i).getAdj(), e.solution)) {
                    Entry e_ = new Entry();
                    e_.solution.addAll(e.solution);
                    e_.solution.add(G.get(i).getVertex());
                    e_.pos = i;
                    e_.score = e.score + G.get(i).getVertex().qualityValue;
                    e_.bound = AstarBound(G, e_, k);
                    H.insert(e_);

                    if (DResultSet.get(e_.solution.size()).score < e_.score) {
                        DResultSet.get(e_.solution.size()).result = e_.solution;
                        DResultSet.get(e_.solution.size()).score = e_.score;
                    }
                }

            }
        }
    }

    public double AstarBound(ArrayList<GraphNode<LegacyShapelet>> G, Entry e, int k) {
        int p, i;
        double bound;

        p = e.solution.size();
        i = e.pos + 1;
        bound = e.score;
        while (p < k && i < G.size()) {
            if (!andSet(G.get(i).getAdj(), e.solution)) {
                bound = bound + G.get(i).getVertex().qualityValue;
                p = p + 1;
            }
            i = i + 1;
        }
        return bound;
    }

    public double maxDresultSet(ArrayList<Dresult<LegacyShapelet>> dresultSet) {
        double max = -2;
        for (int i = 0; i < dresultSet.size(); i++) {
            if (max < dresultSet.get(i).score) {
                max = dresultSet.get(i).score;
            }
        }
        return max;
    }

    public boolean andSet(ArrayList<LegacyShapelet> a, ArrayList<LegacyShapelet> b) {
        if (a == null || b == null) {
            return false;
        }
        for (int i = 0; i < a.size(); i++) {
            for (int j = 0; j < b.size(); j++) {
                if (a.get(i) == b.get(j)) {
                    return true;
                }
            }
        }
        return false;
    }

    public static void table4_5() throws Exception {

        // Initialise classifiers required for this experiment
        classifiers = new Classifier[8];
        classifiers[0] = new ShapeletTreeClassifier("infoTree.txt");
        classifiers[1] = new J48();
        classifiers[2] = new IB1();
        classifiers[3] = new NaiveBayes();
        classifiers[4] = new BayesNet();
        classifiers[5] = new RandomForest();
        classifiers[6] = new RotationForest();
        classifiers[7] = new SMO();

        // Set up names for the classifiers - only used for output
        classifierNames = new String[8];
        classifierNames[0] = "ShapeletTree";
        classifierNames[1] = "C4.5";
        classifierNames[2] = "1NN";
        classifierNames[3] = "Naive Bayes";
        classifierNames[4] = "Bayesian Network";
        classifierNames[5] = "Random Forest";
        classifierNames[6] = "Rotation Forest";
        classifierNames[7] = "SVM (linear)";

//        if ((classifierToProcessIndex < 1 || classifierToProcessIndex > classifiers.length) && classifierToProcessIndex != -1) {
//            throw new IOException("Invalid classifier identifier.");
//        } else {
//            if (classifierToProcessIndex != -1) {
//                classifierToProcessIndex--;
//            }
//        }

        // Compute classifier accuracies for each classifier
        double accuracies[] = new double[classifiers.length];

        for (int i = 1; i < classifiers.length; i++) {
            
            //if (i == classifierToProcessIndex || classifierToProcessIndex == -1) {
                accuracies[i] = classifierAccuracy(i,true, true);
            
        }

        // Write experiment output to file 
        writeFileContent(accuracies);
    }

    /**
     * A method to write content to a given file.
     * 
     * @param fileName file name including extension
     * @param content  content of the file
     */
   private static void writeFileContent(double content[][]){
        // Check if file name is provided.
        if(outFileName == null || outFileName.isEmpty()){
            outFileName = "Table_" + tableToProduceIndex + 
                          "_File_" + (fileToProcessIndex+1) + 
                          "_Classifier_" + (classifierToProcessIndex+1) + ".csv";            
        }
            
       // If a file with given name does not exists then create one and print
       // the header to it, which inlcudes all the classifier names used in the
       // experiment. 
       StringBuilder sb = new StringBuilder();
        if(!isFileExists(outFileName)){
            sb.append("Data Set, "); 
            for(int i = 0; i < classifierNames.length; i++){
                if(i == classifierToProcessIndex || classifierToProcessIndex == -1){
                    sb.append(classifierNames[i]);
                }
                
                if(-1 == classifierToProcessIndex && i != classifierNames.length - 1){
                    sb.append(", ");
                }
            }
            writeToFile(outFileName, sb.toString(), false);
        }
        
        // Print the experiment results to the file.
        sb = new StringBuilder();
        for(int i = 0; i < fileNames.length; i++){
            if(fileToProcessIndex == i || fileToProcessIndex == -1){
                for(int k = 0; k < classifiers.length; k++){
                    if(k == 0){
                        sb.append(fileNames[i]);
                        sb.append(", ");
                    }
                    
                    if(k == classifierToProcessIndex || classifierToProcessIndex == -1 ){
                        sb.append(content[k][i]);
                    }
                    
                    if(-1 == classifierToProcessIndex && k != classifiers.length - 1){
                        sb.append(", ");
                    }
                }
            }
        }
        writeToFile(outFileName, sb.toString(), true);
   }
   
   /**
    * A method to write text into a file.
    * @param filename file name including the extension.
    * @param text content to be written into the file.
    * @param append flag indicating whether a file should be appended (true) or
    *        replaced (false).
    */
    private static void writeToFile(String filename, String text, boolean append) {
        
        BufferedWriter bufferedWriter = null;
        
        try {       
            //Construct the BufferedWriter object
            bufferedWriter = new BufferedWriter(new FileWriter(filename, append));
            
            //Start writing to the output stream
            bufferedWriter.write(text);
            bufferedWriter.newLine();
            
        } catch (FileNotFoundException ex) {
            ex.printStackTrace();
        } catch (IOException ex) {
            ex.printStackTrace();
        } finally {
            //Close the BufferedWriter
            try {
                if (bufferedWriter != null) {
                    bufferedWriter.flush();
                    bufferedWriter.close();
                }
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        }
    }
    
    /**
     * A method to check if file with a given name exists.
     * 
     * @param filename file name including the extension.
     * @return true if file with given file name exists, otherwise false.
     */
    private static boolean isFileExists(String filename){
        File f = new File(filename);
        if(f.isFile() && f.canWrite()) {
            return true;
        }else{
            return false;
        }   
    }
    /**
     * A method to validate a given classifier
     *
     * @param classifierIndex index of the classifier to be validated
     * @param useTransformedData flag indicating what type of data to use.
     * Shapelet is used for data transformation.
     * @param computeErrorRate flag indicating whether error rate is required
     * rather than classifier accuracy.
     * @param usePercentage flag indicating whether an accuracy/error rate
     * should be converted to percentage.
     * @return classifier accuracy/error rate
     */
    private static double classifierAccuracy(int classifierIndex,  boolean computeErrorRate, boolean usePercentage) {
        try {
            classifiers[classifierIndex].buildClassifier(dataTrainTransformed);
        } catch (Exception e) {
            System.out.println("build classifier error!");
            e.printStackTrace();
        }
        
        
        double accuracy=0.0;

        for(int j=0;j<dataTestTransformed.numInstances();j++){
            double classifierPrediction=0.0;
            try {
                classifierPrediction=classifiers[classifierIndex].classifyInstance(dataTestTransformed.instance(j));
                
            } catch (Exception e) {
                System.out.println("classification error!");
                e.printStackTrace();
            }
            
            double actualClass=dataTestTransformed.instance(j).classValue();
            if (classifierPrediction==actualClass) {
                accuracy++;
            }
        }
        accuracy /=dataTestTransformed.numInstances();

        if (computeErrorRate) {
            accuracy = 1 - accuracy;
        }

        if (usePercentage) {
            accuracy *= 100;
        }

        return accuracy;
    }

    public static void main(String[] args) {
        try {

            int k = 10;
            DivTopK divTopK = new DivTopK();
            Instances dataTrain = DivTopK.loadData(args[0]); //训练数据
            Instances dataTest = divTopK.loadData(args[2]);   //测试数据
            ArrayList<LegacyShapelet> shapeletsList = divTopK.readShapelets(args[1], dataTrain);//候选shapelets

//            System.out.println("----------------");
//            for (int i = 0; i < shapeletsList.size(); i++) {
//
//                System.out.print(i + 1 + " " + shapeletsList.get(i).seriesId + " " + shapeletsList.get(i).startPos + " " + shapeletsList.get(i).length);
//                System.out.print(" " + shapeletsList.get(i).qualityValue + " ");
//                System.out.println(shapeletsList.get(i).separationGap + "  " + shapeletsList.get(i).splitThreshold);
//
//            }
            ArrayList<GraphNode<LegacyShapelet>> graph = divTopK.constructShapeletGraph(shapeletsList, dataTrain);
            DResultSet = divTopK.divAstar(graph, k);

            Instances dataTrainTransformed = divTopK.transformData(dataTrain);
            Instances dataTestTransformed = divTopK.transformData(dataTest);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
