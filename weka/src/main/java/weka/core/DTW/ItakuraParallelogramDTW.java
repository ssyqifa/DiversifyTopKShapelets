/*
 * Itakura Parallelogram DTW distance metric
 */
package DTW;

/**
 *
 * @author Chris Rimmer
 */
public class ItakuraParallelogramDTW extends BasicDTW {
    
    private int maxWidth;
        
    private double AB_A;
    private double AB_B;
    private double AB_C;
    
    private double AC_A;
    private double AC_B;
    private double AC_C;
    
    private double BD_A;
    private double BD_B;
    private double BD_C;
    
    private double CD_A;
    private double CD_B;
    private double CD_C;
    
    private boolean calcLines;
    
    private double halfWidth;
    
    /**
     * Create Itakura Parallelogram distance metric
     * 
     * @param maxWidth maximum width of the warping window
     * @throws IllegalArgumentException 
     */
    public ItakuraParallelogramDTW(int maxWidth) throws IllegalArgumentException{
        super();
        setup(maxWidth);
    }
    
    /**
     * Create Itakura Parallelogram distance metric
     * 
     * @param maxWidth maximum width of the warping window
     * @param earlyAbandon enable early abandon
     * @throws IllegalArgumentException 
     */
    public ItakuraParallelogramDTW(int maxWidth, boolean earlyAbandon) throws IllegalArgumentException{
        super(earlyAbandon);
        setup(maxWidth);
    }
    
    /**
     * setup the distance metric
     * 
     * @param maxWidth maximum width of the warping window
     * @throws IllegalArgumentException 
     */
    private void setup(int maxWidth) throws IllegalArgumentException{
        if(maxWidth < 1){
            throw new IllegalArgumentException("Max Width must be 1 or greater");
        }
        
        this.maxWidth = maxWidth;
        this.calcLines = true;
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
    public double distance(double[] first,double[] second, double cutOffValue){
        //calculate 
        if(this.calcLines){
            calculateSlopes(first);
            this.calcLines = false;
        }
        
        //double halfwidth = Math.floor(this.maxWidth/2);   
        
        //create empty array
        this.distances = new double[first.length][second.length];
        
        //first value
        this.distances[0][0] = (first[0]-second[0])*(first[0]-second[0]);
        
        //early abandon if first values is larger than cut off
        if(this.distances[0][0] > cutOffValue && this.isEarlyAbandon){
            return Double.MAX_VALUE;
        }
                
        //top row
        for(int i = 1; i<first.length; i++){
            double AC_D = AC_A*(i+1)+AC_B+AC_C;
            
            if(AC_D > 0){
                this.distances[0][i] = this.distances[0][i-1]+((first[0]-second[i])*(first[0]-second[i]));
            }else{
                this.distances[0][i] = Double.MAX_VALUE;
            }
        }
        
        //first column
        for(int i = 1; i<first.length; i++){
            double AB_D = AB_A+AB_B*(i+1)+AB_C;
            
            if(AB_D < 0){
                this.distances[i][0] = this.distances[i-1][0]+((first[i]-second[0])*(first[i]-second[0]));
            }else{
                this.distances[i][0] = Double.MAX_VALUE;
            }
        }
        
        double minDistance;
        for(int i = 1; i<first.length; i++){
            boolean overflow = true;
            
            for(int j = 1; j<second.length; j++){
                
                double AB_D = AB_A*(j+1)+AB_B*(i+1)+AB_C;
                double AC_D = AC_A*(j+1)+AC_B*(i+1)+AC_C;
                double BD_D = BD_A*(j+1)+BD_B*(i+1)+BD_C;
                double CD_D = CD_A*(j+1)+CD_B*(i+1)+CD_C;
                double firstHalf = (first.length/2)+this.halfWidth;
                double secondHalf = (first.length/2)-this.halfWidth;
                boolean evenlyDivisible = this.maxWidth%2 == 0;
                                              
                if((i < firstHalf && j < firstHalf && ((!evenlyDivisible && AB_D <= 0 && AC_D >= 0) || (evenlyDivisible && AB_D < 0 && AC_D > 0))) ||
                  (i > secondHalf && j > secondHalf && ((!evenlyDivisible && BD_D <= 0 && CD_D >= 0) || (evenlyDivisible && BD_D < 0 && CD_D > 0))) ||
                  (i == first.length-1 && j == first.length-1)){
                    minDistance = Math.min(this.distances[i][j-1], Math.min(this.distances[i-1][j], this.distances[i-1][j-1]));
                    //Assign distance
                    if(minDistance > cutOffValue && this.isEarlyAbandon){
                        this.distances[i][j] = Double.MAX_VALUE;
                    }else{
                        this.distances[i][j] = minDistance+((first[i]-second[j])*(first[i]-second[j]));
                        overflow = false;
                    }
                }else{
                    this.distances[i][j] = Double.MAX_VALUE;
                }
            }
            //early abandon
            if(overflow && this.isEarlyAbandon){
                return Double.MAX_VALUE;
            }
        }
        
        return this.distances[first.length-1][second.length-1];
    }

    /**
     * Sets the maximum width of the warping window
     * 
     * @param maxWidth width of warping window
     * @throws IllegalArgumentException 
     */
    public void setMaxWidth(int maxWidth) throws IllegalArgumentException {
        setup(maxWidth);
    }

    /**
     * Gets the current size of the warping window
     * 
     * @return maximum size of warping window
     */
    public int getMaxWidth() {
        return maxWidth;
    }
    
    /**
     * pre processing to calculate lines of parallelogram
     * 
     * @param first array of data
     */
    private void calculateSlopes(double[] first){
        double middle = first.length/2;
        this.halfWidth = Math.floor(this.maxWidth/2);   
        
        double Ax = 1;
        double Ay = 1;
        double Bx = middle-this.halfWidth;
        double By = middle+this.halfWidth;
        double Cx = middle+this.halfWidth;
        double Cy = middle-this.halfWidth;
        double Dx = first.length;
        double Dy = first.length;
        
        this.AB_A = -(By-Ay);
        this.AB_B = Bx-Ax;
        this.AB_C = -(AB_A*Ax+AB_B*Ay);
        
        this.AC_A = -(Cy-Ay);
        this.AC_B = Cx-Ax;
        this.AC_C = -(AC_A*Ax+AC_B*Ay);
        
        this.BD_A = -(Dy-By);
        this.BD_B = Dx-Bx;
        this.BD_C = -(BD_A*Bx+BD_B*By);
        
        this.CD_A = -(Dy-Cy);
        this.CD_B = Dx-Cx;
        this.CD_C = -(CD_A*Cx+CD_B*Cy);
    }

    @Override
    public String toString() {
        return "ItakuraParallelogramDTW{ " + "maxWidth=" + this.maxWidth + ", earlyAbandon=" + this.isEarlyAbandon + " }";
    }
    
}