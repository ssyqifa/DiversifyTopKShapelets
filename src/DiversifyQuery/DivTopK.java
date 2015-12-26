/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package DiversifyQuery;

import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Scanner;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author sun
 */
public class DivTopK {

    protected static final double ROUNDING_ERROR_CORRECTION = 0.000000000000001;

    public static ArrayList<Dresult<LegacyShapelet>> DResultSet = new ArrayList<>();

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

        for (int i = 0; i < k+1; i++) {
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

    public static void main(String[] args) {
        try {

            int k = 15;
            DivTopK divTopK = new DivTopK();
            Instances data = DivTopK.loadData(args[0]);
            ArrayList<LegacyShapelet> shapeletsList = divTopK.readShapelets(args[1], data);
//            System.out.println("----------------");
//            for (int i = 0; i < shapeletsList.size(); i++) {
//
//                System.out.print(i + 1 + " " + shapeletsList.get(i).seriesId + " " + shapeletsList.get(i).startPos + " " + shapeletsList.get(i).length);
//                System.out.print(" " + shapeletsList.get(i).qualityValue + " ");
//                System.out.println(shapeletsList.get(i).separationGap + "  " + shapeletsList.get(i).splitThreshold);
//
//            }

            ArrayList<GraphNode<LegacyShapelet>> graph = divTopK.constructShapeletGraph(shapeletsList, data);
            DResultSet = divTopK.divAstar(graph, k);
            double dist=0;
            

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
