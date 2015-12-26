/*
 * k-NN class that uses Dynamic Time Warping to find the Nearest Neighbours
 */
package DTW;

import weka.classifiers.lazy.IBk;
import weka.core.neighboursearch.NearestNeighbourSearch;

/**
 *
 * @author Chris
 */
public class IBkDTW extends IBk{
    
    private BasicDTW DTWSearch = new BasicDTW();
    
    /**
     * Create new kNN classifier with Dynamic Time Warping
     * 
     * @param k value for k 
     */
    public IBkDTW(int k) {
        super(k);
        setup();  
    }  
    
    /**
     * Create new kNN classifier with Dynamic Time Warping
     */
    public IBkDTW() {
        super();
        setup();
    }
    
    /**
     * Sets up the classifier with the correct distance function
     */
    private void setup(){
        NearestNeighbourSearch NNS = this.getNearestNeighbourSearchAlgorithm();
        try {
            NNS.setDistanceFunction(DTWSearch);
        } catch (Exception e) {
            System.out.println(e);
        }

        this.setNearestNeighbourSearchAlgorithm(NNS);  
    }
    
    /**
     * Sets the DTW search algorithm 
     * 
     * @param BasicDTW DTWSearch 
     */
    public void setDTWSearch(BasicDTW DTWSearch){
        this.DTWSearch = DTWSearch;
        setup();
    }
    
    /**
     * Gets the current distance metric being used
     * 
     * @return the current distance metric
     */
    public BasicDTW getDTWSearch(){
        return this.DTWSearch;
    }
}
