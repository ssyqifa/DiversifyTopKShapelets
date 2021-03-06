/*
 * Sakoe Chiba DTW distance metric
 */
package DTW;

/**
 *
 * @author Chris Rimmer
 */
public class SakoeChibaDTW extends BasicDTW {

    private int bandSize;

    /**
     * Creates new Sakoe Chiba Distance metric
     * 
     * @param bandSize warping window width
     * @throws IllegalArgumentException bandSize must be > 0
     */
    public SakoeChibaDTW(int bandSize) throws IllegalArgumentException {
        super();
        setup(bandSize);
    }

    /**
     * Creates new Sakoe Chiba Distance metric
     * 
     * @param bandSize warping window width
     * @param earlyAbandon set early abandon
     * @throws IllegalArgumentException bandSize must be > 0
     */
    public SakoeChibaDTW(int bandSize, boolean earlyAbandon) throws IllegalArgumentException {
        super(earlyAbandon);
        setup(bandSize);
    }

    /**
     * sets up the distance metric
     * 
     * @param bandSize
     * @throws IllegalArgumentException 
     */
    private void setup(int bandSize) throws IllegalArgumentException {
        if (bandSize < 1) {
            throw new IllegalArgumentException("Band Size must be 1 or greater");
        }
        
        this.bandSize = bandSize;
    }

    /**
     * calculates the distance between two instances (been converted to arrays)
     * 
     * @param first instance 1 as array
     * @param second instance 2 as array
     * @param cutOffValue used for early abandon
     * @return distance between instances
     */
    @Override
    public double distance(double[] first, double[] second, double cutOffValue) {
        //create empty array
        this.distances = new double[first.length][second.length];

        //first value
        this.distances[0][0] = (first[0] - second[0]) * (first[0] - second[0]);

        //early abandon if first values is larger than cut off
        if (this.distances[0][0] > cutOffValue && this.isEarlyAbandon) {
            return Double.MAX_VALUE;
        }

        //top row
        for (int i = 1; i < second.length; i++) {
            if (i < this.bandSize) {
                this.distances[0][i] = this.distances[0][i - 1] + ((first[0] - second[i]) * (first[0] - second[i]));
            } else {
                this.distances[0][i] = Double.MAX_VALUE;
            }
        }

        //first column
        for (int i = 1; i < first.length; i++) {
            if (i < this.bandSize) {
                this.distances[i][0] = this.distances[i - 1][0] + ((first[i] - second[0]) * (first[i] - second[0]));
            } else {
                this.distances[i][0] = Double.MAX_VALUE;
            }
        }

        //warp rest
        double minDistance;
        for (int i = 1; i < first.length; i++) {
            boolean overflow = true;
            for (int j = 1; j < second.length; j++) {
                //Checks if i and j are within the band window
                if (i < j + this.bandSize && j < i + this.bandSize) {
                    minDistance = Math.min(this.distances[i][j - 1], Math.min(this.distances[i - 1][j], this.distances[i - 1][j - 1]));
                    //Assign distance
                    if (minDistance > cutOffValue && this.isEarlyAbandon) {
                        this.distances[i][j] = Double.MAX_VALUE;
                    } else {
                        this.distances[i][j] = minDistance + ((first[i] - second[j]) * (first[i] - second[j]));
                        overflow = false;
                    }
                } else {
                    this.distances[i][j] = Double.MAX_VALUE;
                }
            }

            //early abandon
            if (overflow && this.isEarlyAbandon) {
                return Double.MAX_VALUE;
            }
        }
        
        return this.distances[first.length - 1][second.length - 1];
    }

    /**
     * Sets the size of the warping window
     * 
     * @param bandSize band width
     * @throws IllegalArgumentException 
     */
    public void setBandSize(int bandSize) throws IllegalArgumentException {
        setup(bandSize);
    }
    
    /**
     * Gets the current warping window width
     * 
     * @return warping window width
     */
    public int getBandSize() {
        return this.bandSize;
    }

    @Override
    public String toString() {
        return "SakoeChibaDTW{ " + "bandSize=" + this.bandSize + ", earlyAbandon=" + this.isEarlyAbandon + " }";
    }
}