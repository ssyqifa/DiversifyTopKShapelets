/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package DiversifyTopKShaepelet;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.awt.Point;
import java.lang.Math;
import java.util.Iterator;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Scanner;
import java.util.TreeMap;
import java.util.HashSet;
import java.util.HashMap;
import java.util.Map;
import java.util.AbstractMap;
import java.util.Comparator;
import java.util.Random;
import weka.associations.CARuleMiner;
import weka.core.*;
import weka.core.shapelet.*;
import weka.filters.SimpleBatchFilter;

/**
 *
 * @author sun
 */
public class DiversifyTopKShaepelet extends SimpleBatchFilter {

    @Override
    public String globalInfo() {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    protected boolean supressOutput = false;
    protected int minShapeletLength;
    protected int maxShapeletLength;
    protected int numShapelets;
    protected boolean shapeletsTrained;
    protected ArrayList<LegacyShapelet> shapelets;
    protected String outputFileLocation = "defaultShapeletOutput.txt";
    protected boolean recordShapelets = true;

    public static int DEFAULT_NUMSHAPELETS = 10;
    public static int DEFAULT_MINSHAPELETLENGTH = 3;
    public static int DEFAULT_MAXSHAPELETLENGTH = 30;

    public static HashMap<Integer, USAXElmentType> USAXMap = new HashMap<Integer, USAXElmentType>();
    public static ArrayList<Map.Entry<Integer, Double>> scoreList = new ArrayList<>();
    public static ArrayList<Dresult> DResultSet = new ArrayList<Dresult>();

    protected QualityMeasures.ShapeletQualityMeasure qualityMeasure;
    protected QualityMeasures.ShapeletQualityChoice qualityChoice;
    protected boolean useCandidatePruning;
    protected int candidatePruningStartPercentage;

    protected static final double ROUNDING_ERROR_CORRECTION = 0.000000000000001;
    protected int[] dataSourceIDs;

    //Variables for experiments
    private static long subseqDistOpCount;

    /**
     * Default constructor; Quality measure defaults to information gain.
     */
    public DiversifyTopKShaepelet() {
        this(DEFAULT_NUMSHAPELETS, DEFAULT_MINSHAPELETLENGTH, DEFAULT_MAXSHAPELETLENGTH, QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);

    }

    /**
     * Single param constructor: Quality measure defaults to information gain.
     *
     * @param k the number of shapelets to be generated
     */
    public DiversifyTopKShaepelet(int k) {
        this(k, DEFAULT_MINSHAPELETLENGTH, DEFAULT_MAXSHAPELETLENGTH, QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);

    }

    /**
     * Full constructor to create a usable filter. Quality measure defaults to
     * information gain.
     *
     * @param k the number of shapelets to be generated
     * @param minShapeletLength minimum length of shapelets
     * @param maxShapeletLength maximum length of shapelets
     */
    public DiversifyTopKShaepelet(int k, int minShapeletLength, int maxShapeletLength) {
        this(k, minShapeletLength, maxShapeletLength, QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);

    }

    /**
     * Full, exhaustive, constructor for a filter. Quality measure set via enum,
     * invalid selection defaults to information gain.
     *
     * @param k the number of shapelets to be generated
     * @param minShapeletLength minimum length of shapelets
     * @param maxShapeletLength maximum length of shapelets
     * @param qualityChoice the shapelet quality measure to be used with this
     * filter
     */
    public DiversifyTopKShaepelet(int k, int minShapeletLength, int maxShapeletLength, weka.core.shapelet.QualityMeasures.ShapeletQualityChoice qualityChoice) {

        this.minShapeletLength = minShapeletLength;
        this.maxShapeletLength = maxShapeletLength;
        this.numShapelets = k;
        this.shapelets = new ArrayList<>();
        this.shapeletsTrained = false;
        this.useCandidatePruning = false;
        this.qualityChoice = qualityChoice;
        switch (qualityChoice) {
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
     * Supresses filter output to the console; useful when running timing
     * experiments.
     */
    public void supressOutput() {
        this.supressOutput = true;
    }

    /**
     * Use candidate pruning technique when checking candidate quality. This
     * speeds up the transform processing time.
     */
    public void useCandidatePruning() {
        this.useCandidatePruning = true;
        this.candidatePruningStartPercentage = 10;
    }

    /**
     *
     * @param f
     */
    public void setCandidatePruning(boolean f) {
        this.useCandidatePruning = f;
        if (f) {
            this.candidatePruningStartPercentage = 10;
        } else //Not necessary
        {
            this.candidatePruningStartPercentage = 100;
        }

    }

    /**
     * Use candidate pruning technique when checking candidate quality. This
     * speeds up the transform processing time.
     *
     * @param percentage the percentage of data to be precocessed before pruning
     * is initiated. In most cases the higher the percentage the less effective
     * pruning becomes
     */
    public void useCandidatePruning(int percentage) {
        this.useCandidatePruning = true;
        this.candidatePruningStartPercentage = percentage;
    }

    /**
     * Mutator method to set the number of shapelets to be stored by the filter.
     *
     * @param k the number of shapelets to be generated
     */
    public void setNumberOfShapelets(int k) {
        this.numShapelets = k;
    }

    /**
     *
     * @return
     */
    public int getNumberOfShapelets() {
        return numShapelets;
    }

    /**
     * Mutator method to set the minimum and maximum shapelet lengths for the
     * filter.
     *
     * @param minShapeletLength minimum length of shapelets
     * @param maxShapeletLength maximum length of shapelets
     */
    public void setShapeletMinAndMax(int minShapeletLength, int maxShapeletLength) {
        this.minShapeletLength = minShapeletLength;
        this.maxShapeletLength = maxShapeletLength;
    }

    /**
     * Mutator method to set the quality measure used by the filter. As with
     * constructors, default selection is information gain unless another valid
     * selection is specified.
     *
     * @return
     */
    public QualityMeasures.ShapeletQualityChoice getQualityMeasure() {
        return qualityChoice;
    }

    /**
     *
     * @param qualityChoice
     */
    public void setQualityMeasure(QualityMeasures.ShapeletQualityChoice qualityChoice) {
        this.qualityChoice = qualityChoice;
        switch (qualityChoice) {
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
     * Sets the format of the filtered instances that are output. I.e. will
     * include k attributes each shapelet distance and a class value
     *
     * @param inputFormat the format of the input data
     * @return a new Instances object in the desired output format
     * @throws Exception if all required parameters of the filter are not
     * initialised correctly
     */
    @Override
    protected Instances determineOutputFormat(Instances inputFormat) throws Exception {

        if (this.numShapelets < 1) {
            throw new Exception("ShapeletFilter not initialised correctly - please specify a value of k that is greater than or equal to 1");
        }

        //Set up instances size and format.
        //int length = this.numShapelets;
        int length = this.shapelets.size();
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

    @Override
    public Instances process(Instances data) throws Exception {
        if (this.numShapelets < 1) {
            throw new Exception("Number of shapelets initialised incorrectly - please select value of k greater than or equal to 1 (Usage: setNumberOfShapelets");
        }

        int maxPossibleLength = data.instance(0).numAttributes() - 1;
        if (data.classIndex() < 0) {
            throw new Exception("Require that the class be set for the ShapeletTransform");
        }

        if (this.minShapeletLength < 1 || this.maxShapeletLength < 1 || this.maxShapeletLength < this.minShapeletLength || this.maxShapeletLength > maxPossibleLength) {
            throw new Exception("Shapelet length parameters initialised incorrectly");
        }

        //Sort data in round robin order
        dataSourceIDs = new int[data.numInstances()];

        for (int i = 0; i < data.numInstances(); i++) {
            dataSourceIDs[i] = i;
        }
//        data = roundRobinData(data, dataSourceIDs);

        if (this.shapeletsTrained == false) { // shapelets discovery has not yet been caried out, so do so
            this.shapelets = findDiversityTopKShapelets(this.numShapelets, data, this.minShapeletLength, this.maxShapeletLength); // get k shapelets ATTENTION
            this.shapeletsTrained = true;
            if (!supressOutput) {
                System.out.println(shapelets.size() + " Shapelets have been generated");
            }
        }

        Instances output = determineOutputFormat(data);

        // for each data, get distance to each shapelet and create new instance
        for (int i = 0; i < data.numInstances(); i++) { // for each data
            Instance toAdd = new Instance(this.shapelets.size() + 1);
            int shapeletNum = 0;
            for (LegacyShapelet s : this.shapelets) {
                double dist = subseqDistance(s.content, data.instance(i));
                toAdd.setValue(shapeletNum++, dist);
            }
            toAdd.setValue(this.shapelets.size(), data.instance(i).classValue());
            output.add(toAdd);
        }
        return output;
    }

    /**
     * Set file path for the filter log. Filter log includes shapelet quality,
     * seriesId, startPosition, and content for each shapelet.
     *
     * @param fileName the updated file path of the filter log
     */
    public void setLogOutputFile(String fileName) {
        this.recordShapelets = true;
        this.outputFileLocation = fileName;
    }

    /**
     * Turns off log saving; useful for timing experiments where speed is
     * essential.
     */
    public void turnOffLog() {
        this.recordShapelets = false;
    }

    public ArrayList<LegacyShapelet> findDiversityTopKShapelets(int numShapelets, Instances data, int minShaepeletLength, int maxShapeletLength) throws Exception {

        ArrayList<LegacyShapelet> kShapelets = new ArrayList<LegacyShapelet>();  //store up to k shapeles overall
        ArrayList<LegacyShapelet> tempKShapelets;   //store temporary k shapelets each iteration
        ArrayList<LegacyShapelet> seriesShapelets = new ArrayList<LegacyShapelet>(); //store all temporary k shapelets each itreration for diversifying process

        int saxLENGTH = 15;
        int w = 4;
        int R = 10;
        double percentMask = 0.25;
        int topK = 10;

        TreeMap<Double, Integer> classDistributions = getClassDistributions(data); //calc info gain//calc info gain//calc info gain//calc info gain

        int numClass = classDistributions.size();

        if (!supressOutput) {
            System.out.println("Processing data: ");
        }

        int numInstances = data.numInstances();

        for (int length = minShaepeletLength; length <= maxShapeletLength; length++) {

            createSAXList(length, saxLENGTH, w, data);

            randomProjection(R, percentMask, saxLENGTH);

            scoreAllSAX(R, numClass, data);

            tempKShapelets = findBestTopKSAX(length, topK, data, numClass);

            for (int i = 0; i < tempKShapelets.size(); i++) {
                seriesShapelets.add(tempKShapelets.get(i));
            }

            USAXMap.clear();
            scoreList.clear();

        }

        ArrayList<GraphNode> Graph = new ArrayList<GraphNode>();
        Graph = constructShapeletGraph(seriesShapelets, data);
        kShapelets = DiversifyTopKQuery(Graph, numShapelets);

        return kShapelets;
//        return seriesShapelets;
    }

    public ArrayList<LegacyShapelet> DiversifyTopKQuery(ArrayList<GraphNode> graph, int k) {
        ArrayList<Dresult> resultsList = new ArrayList<Dresult>();
        resultsList = divAstar(graph, k);
        
        for (int i = k; i <= 1; i--) {
            if (resultsList.get(i).resultShapelets.size() == k) {
                return resultsList.get(i).resultShapelets;
            }
        }
        return null;
    }

    public ArrayList<LegacyShapelet> findBestTopKSAX(int subsequenceLength, int top_k, Instances data, int numClass) {
        int numObject = data.numInstances();
        ArrayList<Point> Dist = new ArrayList<>(numObject);
        int word;
        int kk;
        double gain, distanceThreshold, gap;
        int qObject, qPosition;
        USAXElmentType usax;

        TreeMap<Double, Integer> classDistributions = getClassDistributions(data); // used to calc info gain

        double[] candidate = new double[subsequenceLength];
        ArrayList<LegacyShapelet> shapelets = new ArrayList<LegacyShapelet>();
        if (top_k > 0) {
            Collections.sort(scoreList, new Comparator<Map.Entry<Integer, Double>>() {
                @Override
                public int compare(Map.Entry<Integer, Double> a, Map.Entry<Integer, Double> b) {
                    return ((Double) b.getValue()).compareTo((Double) a.getValue());
                }
            });
        }
        for (int k = 0; k < Math.min(top_k, (int) scoreList.size()); k++) {

            word = scoreList.get(k).getKey();
            usax = USAXMap.get(word);
            for (kk = 0; kk < Math.min((int) usax.SAXIdArrayList.size(), 1); kk++) {
                qObject = usax.SAXIdArrayList.get(kk).x;
                qPosition = usax.SAXIdArrayList.get(kk).y;

                for (int i = 0; i < subsequenceLength; i++) {
                    candidate[i] = data.instance(qObject).value(qPosition + i);
                }
                candidate = zNorm(candidate, false);
                LegacyShapelet candidateShapelet = checkCandidate(candidate, data, qObject, qPosition, classDistributions, null);
                shapelets.add(candidateShapelet);
            }
        }
        return shapelets;
    }

    public int sortScore(Map.Entry a, Map.Entry b) {
        return ((Double) a.getValue()).compareTo((Double) b.getValue());

    }

    public ArrayList<Dresult> divAstar(ArrayList<GraphNode> G, int k) {
        MaxHeap<Entry> H = new MaxHeap<Entry>();
        H.insert(new Entry());

        for (int i = 0; i < 1110; i++) {
            Dresult d = new Dresult();
            d.score = -1;
            DResultSet.add(d);
        }
//        for (int j = k; j >= 1; j--) {
            AStarSearch(G, H, k);
//            ArrayList<Entry> arrayEntrys = H.getArray();
//            for (int m = 0; m < H.getCurrentSize(); m++) {
//                Entry entry=arrayEntrys.get(m);
//                
//                if(entry==null) continue;
//                
//                double bound = AstarBound(G,entry , k);
//                entry.setBound(bound);
//                H.update(arrayEntrys, m, entry);
//            }
//        }
        return DResultSet;
    }

    public void AStarSearch(ArrayList<GraphNode> G, MaxHeap<Entry> H, int k) {
        while ((!H.isEmpty()) && H.getMax().getBound() > maxDresultSet(DResultSet)) {
            Entry e = new Entry();
            e = H.deleteMax();
            for (int i = e.pos + 1; i < G.size(); i++) {
                if (!andSet(G.get(i).getAdjShapelets(), e.solution)) {
                    Entry e_ = new Entry();
                    e_.solution = e.solution;
                    e_.solution.add(G.get(i).getVertexShapelet());
                    e_.pos = i;
                    e_.score = e.score + G.get(i).getVertexShapelet().qualityValue;
                    e_.bound = AstarBound(G, e_, k);
                    H.insert(e_);

                    if (DResultSet.get(e_.solution.size()).score < e_.score) {
                        DResultSet.get(e_.solution.size()).resultShapelets = e_.solution;
                        DResultSet.get(e_.solution.size()).score = e_.score;
                    }
                }

            }
        }
    }

    public double AstarBound(ArrayList<GraphNode> G, Entry e, int k) {
        int p, i;
        double bound;
        
        p = e.solution.size();
        i = e.pos + 1;
        bound = e.score;
        while (p < k && i < G.size()) {
            if (!andSet(G.get(i).getAdjShapelets(), e.solution)) {
                bound = bound + G.get(i).getVertexShapelet().qualityValue;
                p = p + 1;
            }
            i = i + 1;
        }
        return bound;
    }

    public double maxDresultSet(ArrayList<Dresult> dresultSet) {
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

    //构造一个图，图顶点为shapelet，各顶点之间是否有边以两个shapelets是否相似为依据
    public ArrayList<GraphNode> constructShapeletGraph(ArrayList<LegacyShapelet> seriesShapelets, Instances data) {

        ArrayList<GraphNode> Graph = new ArrayList<GraphNode>();
        Collections.sort(seriesShapelets); //降序排序，
        for (int i = 0; i < seriesShapelets.size(); i++) {
            GraphNode node = new GraphNode();
            node.setVertexShapelet(seriesShapelets.get(i));
            Graph.add(node);
        }
        for (int i = 0; i < seriesShapelets.size(); i++) {
            for (int j = i + 1; j < seriesShapelets.size(); j++) {
                if (seriesShapelets.get(i).isSimilar(seriesShapelets.get(j), data)) {
                    if (Graph.get(i).getAdjShapelets() == null) {
                        ArrayList<LegacyShapelet> adjecentShapelets = new ArrayList<LegacyShapelet>();
                        adjecentShapelets.add(seriesShapelets.get(j));
                        Graph.get(i).setAdjShapelet(adjecentShapelets);
                    } else {
                        Graph.get(i).getAdjShapelets().add(seriesShapelets.get(j));
                    }
                    if (Graph.get(j).getAdjShapelets() == null) {
                        ArrayList<LegacyShapelet> adjecentShapelets = new ArrayList<LegacyShapelet>();
                        adjecentShapelets.add(seriesShapelets.get(i));
                        Graph.get(j).setAdjShapelet(adjecentShapelets);
                    } else {
                        Graph.get(j).getAdjShapelets().add(seriesShapelets.get(i));
                    }
                }
            }
        }
        return Graph;
    }

    protected void createSAXList(int subsequenceLength, int saxLength, int w, Instances data) {

        w = (int) Math.ceil((double) subsequenceLength / saxLength);
        saxLength = (int) Math.ceil((double) subsequenceLength / w);

        double ex, ex2, mean, std;
        double[] sumSegment = new double[saxLength]; //sumsegment为每段内数据之和
        int[] elementSegment = new int[saxLength];
        int j, jSt, k, slot, objectId;
        double dataPoint;
        int word, previousWord;
        for (k = 0; k < saxLength; k++) {
            elementSegment[k] = w;
        }
        elementSegment[saxLength - 1] = subsequenceLength - (saxLength - 1) * w; // w为宽度，表示

        for (objectId = 0; objectId < data.numInstances(); objectId++) {
            ex = ex2 = 0;
            previousWord = -1;

            for (k = 0; k < saxLength; k++) {
                sumSegment[k] = 0;
            }
            double[] timeSeriesObject = data.instance(objectId).toDoubleArray();

            //case 1: Initial
            for (j = 0; (j < timeSeriesObject.length - 1) && (j < subsequenceLength); j++) {
                dataPoint = timeSeriesObject[j];
                ex += dataPoint;
                ex2 += dataPoint * dataPoint;
                slot = (int) Math.floor(j / w);     //slot为第几段，w为每段的宽度
                sumSegment[slot] += dataPoint;      // 
            }
            //case 2: slightly update
            for (j = j; j <= timeSeriesObject.length - 1; j++) {
                jSt = j - subsequenceLength;
                mean = ex / subsequenceLength;
                std = Math.sqrt(ex2 / subsequenceLength - mean * mean);

                //create SAX from sumSegment
                word = createSAXWord(sumSegment, elementSegment, mean, std, saxLength);

                if (word != previousWord) {
                    previousWord = word;
                    if (!(USAXMap.containsKey(word))) {
                        USAXMap.put(word, null);
                        USAXElmentType usax = new USAXElmentType();
                        usax.objectHashSet.add(objectId);
                        usax.SAXIdArrayList.add(new Point(objectId, jSt));
                        USAXMap.put(word, usax);
                    } else {
                        USAXMap.get(word).objectHashSet.add(objectId);
                        USAXMap.get(word).SAXIdArrayList.add(new Point(objectId, jSt));   ////////待修改
                    }
                }
                /// for next updata
                if (j < timeSeriesObject.length - 1) {
                    ex -= timeSeriesObject[jSt];
                    ex2 -= timeSeriesObject[jSt] * timeSeriesObject[jSt];

                    for (k = 0; k < saxLength - 1; k++) {
                        sumSegment[k] -= timeSeriesObject[jSt + k * w];
                        sumSegment[k] += timeSeriesObject[jSt + (k + 1) * w];
                    }
                    sumSegment[k] -= timeSeriesObject[jSt + k * w];
                    sumSegment[k] += timeSeriesObject[jSt + Math.min((k + 1) * w, subsequenceLength)];

                    dataPoint = timeSeriesObject[j];
                    ex += dataPoint;
                    ex2 += dataPoint * dataPoint;
                }
            }

        }

    }

    protected int createSAXWord(double[] sumSegment, int[] eleSegment, double mean, double std, int saxLength) {
        int word = 0, val = 0;
        double d = 0;

        for (int i = 0; i < saxLength; i++) {
            d = (sumSegment[i] / eleSegment[i] - mean) / std;
            if (d < 0) {
                if (d < -0.67) {
                    val = 0;
                } else {
                    val = 1;
                }
            } else if (d < 0.67) {
                val = 2;
            } else {
                val = 3;
            }
            word = (word << 2) | (val);
        }
        return word;
    }

    protected int createMaskWord(int numMask, int wordLength) {
        int a, b;
        a = 0;
        for (int i = 0; i < numMask; i++) {
            Random random = new Random();
            b = 1 << (random.nextInt(wordLength));
            a = a | b;
        }
        return a;
    }

    protected void randomProjection(int R, double percentMask, int saxLength) {
        HashMap<Integer, HashSet<Integer>> hashMarkMap = new HashMap<>();

        Iterator it = USAXMap.entrySet().iterator();
        int word, maskWord, newWord;
        HashSet<Integer> objectSet;
        int numMask = (int) Math.ceil(percentMask * saxLength);

        for (int r = 0; r < R; r++) {
            maskWord = createMaskWord(numMask, saxLength);

            /// random projection and mark non-duplicate boject
            for (Map.Entry<Integer, USAXElmentType> entrySet : USAXMap.entrySet()) {
                word = entrySet.getKey();
                objectSet = entrySet.getValue().objectHashSet;
                newWord = word | maskWord;
                if (!(hashMarkMap.containsKey(newWord))) {
                    HashSet<Integer> temp = new HashSet<>();
                    temp.addAll(objectSet);
                    hashMarkMap.put(newWord, temp);
                } else {
                    hashMarkMap.get(newWord).addAll(objectSet);
                }

            }

            /// hash again for keep the count
            for (Map.Entry<Integer, USAXElmentType> entrySet : USAXMap.entrySet()) {
                word = entrySet.getKey();
                newWord = word | maskWord;

                objectSet = hashMarkMap.get(newWord);

                Iterator objIt = objectSet.iterator();
                while (objIt.hasNext()) {
                    int mappedValue = (Integer) objIt.next();
                    if (entrySet.getValue().objectCountHashMap.containsKey(mappedValue)) {
                        int temp = entrySet.getValue().objectCountHashMap.get(mappedValue);
                        temp++;
                        entrySet.getValue().objectCountHashMap.put(mappedValue, temp);
                    } else {
                        entrySet.getValue().objectCountHashMap.put(mappedValue, 1);
                    }

                }
            }
            hashMarkMap.clear();
        }
    }

    public void scoreAllSAX(int R, int numClass, Instances data) {
        Iterator it = USAXMap.entrySet().iterator();
        int word;
        double score;
        USAXElmentType usax;

        while (it.hasNext()) {
            Map.Entry entry = (Map.Entry) it.next();
            word = (Integer) entry.getKey();
            usax = (USAXElmentType) entry.getValue();
            score = calcScore(usax, R, numClass, data);
            Map.Entry<Integer, Double> tempPair = new AbstractMap.SimpleEntry<>(word, score);
            scoreList.add(tempPair);

        }

    }

    public double calcScore(USAXElmentType usax, int R, int numClass, Instances data) {       //待修改
        double score = -1;
        int cid, count;
        Iterator objectIt = usax.getObjectCountHashMap().entrySet().iterator();

        ArrayList<Double> cIn = new ArrayList<>();
        ArrayList<Double> cOut = new ArrayList<>();

        for (int i = 0; i < numClass; i++) {
            cIn.add(0.0);
            cOut.add(0.0);
        }

        while (objectIt.hasNext()) {
            Map.Entry entry = (Map.Entry) objectIt.next();
            cid = (int) data.instance((int) entry.getKey()).classValue();
            count = (int) entry.getValue();
            cIn.set(cid, cIn.get(cid) + count);
            cOut.set(cid, cOut.get(cid) + (R - count));
        }
        score = calScoreFromObjectCount(cIn, cOut, numClass);
        return score;
    }

    public double calScoreFromObjectCount(ArrayList<Double> cIn, ArrayList<Double> cOut, int numClass) {
        //2 classes only
        //return Math.abs((cIn.get(0)+cOut.get(1))-(cOut.get(0)+cIn.get(1)));

        //multi-class
        double diff, sum = 0, maxValue = -Double.MAX_VALUE, minValue = Double.MIN_VALUE;
        for (int i = 0; i < numClass; i++) {
            diff = cIn.get(i) - cOut.get(i);
            if (diff > maxValue) {
                maxValue = diff;
            }
            if (diff < minValue) {
                minValue = diff;
            }
            sum += Math.abs(diff);
        }
        return (sum - Math.abs(maxValue) - Math.abs(minValue) + Math.abs(maxValue - minValue));
    }

    /**
     * protected method to check a candidate shapelet. Functions by passing in
     * the raw data, and returning an assessed ShapeletTransform object.
     *
     * @param candidate the data from the candidate ShapeletTransform
     * @param data the entire data set to compare the candidate to
     * @param seriesId series id from the dataset that the candidate came from
     * @param startPos start position in the series where the candidate came
     * from
     * @param classDistribution a TreeMap<Double, Integer> in the form of
     * <Class Value, Frequency> to describe the dataset composition
     * @param qualityBound
     * @return a fully-computed ShapeletTransform, including the quality of this
     * candidate
     */
    protected LegacyShapelet checkCandidate(double[] candidate, Instances data, int seriesId, int startPos, TreeMap classDistribution, QualityBound.ShapeletQualityBound qualityBound) {

        // create orderline by looping through data set and calculating the subsequence
        // distance from candidate to all data, inserting in order.
        ArrayList<OrderLineObj> orderline = new ArrayList<OrderLineObj>();

        boolean pruned = false;

        for (int i = 0; i < data.numInstances(); i++) {
            //Check if it is possible to prune the candidate
            if (qualityBound != null) {
                if (qualityBound.pruneCandidate()) {
                    pruned = true;
                    break;
                }
            }

            double distance = 0.0;
            if (i != seriesId) {
                distance = subseqDistance(candidate, data.instance(i));
            }

            double classVal = data.instance(i).classValue();
            // without early abandon, it is faster to just add and sort at the end
            orderline.add(new OrderLineObj(distance, classVal));

            //Update qualityBound - presumably each bounding method for different quality measures will have a different update procedure.
            if (qualityBound != null) {
                qualityBound.updateOrderLine(orderline.get(orderline.size() - 1));
            }
        }

        // note: early abandon entropy pruning would appear here, but has been ommitted
        // in favour of a clear multi-class information gain calculation. Could be added in
        // this method in the future for speed up, but distance early abandon is more important
        //If shapelet is pruned then it should no longer be considered in further processing
        if (pruned) {
            return null;
        } else {
            // create a shapelet object to store all necessary info, i.e.
            LegacyShapelet shapelet = new LegacyShapelet(candidate, seriesId, startPos, this.qualityMeasure);
            shapelet.calculateQuality(orderline, classDistribution);
            shapelet.calcInfoGainAndThreshold(orderline, classDistribution);
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
    protected double subseqDistance(double[] candidate, Instance timeSeriesIns) {
        return subsequenceDistance(candidate, timeSeriesIns);
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

        // for all possible subsequences of two
        for (int i = 0; i <= timeSeries.length - candidate.length - 1; i++) {
            sum = 0;
            // get subsequence of two that is the same lenght as one
            subseq = new double[candidate.length];

            for (int j = i; j < i + candidate.length; j++) {
                subseq[j - i] = timeSeries[j];

                //Keep count of fundamental ops for experiment
                subseqDistOpCount++;
            }
            subseq = zNormalise(subseq, false); // Z-NORM HERE

            //Keep count of fundamental ops for experiment
            subseqDistOpCount += 3 * subseq.length;

            for (int j = 0; j < candidate.length; j++) {
                sum += (candidate[j] - subseq[j]) * (candidate[j] - subseq[j]);

                //Keep count of fundamental ops for experiment
                subseqDistOpCount++;
            }
            if (sum < bestSum) {
                bestSum = sum;
            }
        }
        return (bestSum == 0.0) ? 0.0 : (1.0 / candidate.length * bestSum);
    }

    /**
     *
     * @param input
     * @param classValOn
     * @return
     */
    protected double[] zNorm(double[] input, boolean classValOn) {
        return DiversifyTopKShaepelet.zNormalise(input, classValOn);
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

    /**
     * Private method to calculate the class distributions of a dataset. Main
     * purpose is for computing shapelet qualities.
     *
     * @param data the input data set that the class distributions are to be
     * derived from
     * @return a TreeMap<Double, Integer> in the form of
     * <Class Value, Frequency>
     */
    public static TreeMap<Double, Integer> getClassDistributions(Instances data) {
        TreeMap<Double, Integer> classDistribution = new TreeMap<Double, Integer>();
        double classValue;
        for (int i = 0; i < data.numInstances(); i++) {
            classValue = data.instance(i).classValue();
            boolean classExists = false;
            for (Double d : classDistribution.keySet()) {
                if (d == classValue) {
                    int temp = classDistribution.get(d);
                    temp++;
                    classDistribution.put(classValue, temp);
                    classExists = true;
                }
            }

            if (classExists == false) {
                classDistribution.put(classValue, 1);
            }
        }
        return classDistribution;
    }

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

    /**
     * An example use of a ShapeletTransform
     *
     * @param args command line args. arg[0] should spcify a set of training
     * instances to transform
     */
    public static void main(String[] args) {
        try {
            // mandatory requirements:  numShapelets (k), min shapelet length, max shapelet length, input data
            // additional information:  log output dir

            // example filter, k = 10, minLength = 20, maxLength = 40, data = , output = exampleOutput.txt
            int k = 10;
            int minLength = 20;
            int maxLength = 4;
//            Instances data= ShapeletTransform.loadData("ItalyPowerDemand_TRAIN.arff"); // for example
            Instances data = DiversifyTopKShaepelet.loadData(args[0]);

            DiversifyTopKShaepelet dtks = new DiversifyTopKShaepelet(k, minLength, maxLength);
            dtks.setQualityMeasure(QualityMeasures.ShapeletQualityChoice.INFORMATION_GAIN);
            dtks.setLogOutputFile("exampleOutput.txt"); // log file stores shapelet output

            // Note: sf.process returns a transformed set of Instances. The first time that
            //      thisFilter.process(data) is called, shapelet extraction occurs. Subsequent calls to process
            //      uses the previously extracted shapelets to transform the data. For example:
            //
            ArrayList<LegacyShapelet> finalShapelets = dtks.findDiversityTopKShapelets(k, data, minLength, maxLength);
            //      Instances transformedTrain = sf.process(trainingData); -> extracts shapelets and can be used to transform training data
            //      Instances transformedTest = sf.process(testData); -> uses shapelets extracted from trainingData to transform testData
            System.out.println("-------------------------shapelets---------------------\n");
            for (int i = 0; i < finalShapelets.size(); i++) {
                System.out.println("第" + i + "个shapelets：\n");
                for (int j = 0; j < finalShapelets.get(i).content.length; j++) {
                    System.out.println(finalShapelets.get(i).content[j]);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
