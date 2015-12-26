/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package simulators;
import java.util.*;
import java.io.*;
/**
 *
 * @author u0318701
 */
public class ShapeletModel extends Model {
    
    public enum ShapeType {TRIANGLE,HEADSHOULDERS,SINE, STEP, SPIKE};
    protected ArrayList<Shape> shapes;
    
    private static int DEFAULTNUMSHAPELETS=1;
    private static int DEFAULTSERIESLENGTH=500;
    private static int DEFAULTSHAPELETLENGTH=29;
    private static int DEFAULTMAXSTART=70;
    
    protected int numShapelets;
    protected int seriesLength; 
    protected int shapeletLength;
    protected int maxStart;
    
    
    
    
    //Default Constructor, max start should be at least 29 less than the length
    // of the series if using the default shapelet length of 29
    public ShapeletModel()
    {
        this(new double[]{DEFAULTNUMSHAPELETS,DEFAULTSERIESLENGTH,DEFAULTSHAPELETLENGTH,DEFAULTMAXSTART});
    }
    public final void setDefaults(){
       seriesLength=DEFAULTSERIESLENGTH; 
       numShapelets=DEFAULTNUMSHAPELETS;
       shapeletLength=DEFAULTSHAPELETLENGTH;
       maxStart=DEFAULTMAXSTART;
    }
    public ShapeletModel(double[] param)
    {
        setDefaults();
//PARAMETER LIST: seriesLength,  numShapelets, shapeletLength, maxStart
//Using the fall through for switching, I should be shot!        
        if(param!=null){
            switch(param.length){
                default: case 4:    maxStart=(int)param[3];
                case 3:             shapeletLength=(int)param[2];
                case 2:             numShapelets=(int)param[1];
                case 1:             seriesLength=(int)param[0];
            }
        }
        shapes=new ArrayList<>();
        // Shapes are randomised for type and location; the other characteristics, such as length
        // must be changed in the inner class.
        for(int i=0;i<numShapelets;i++)
        {
           Shape sh = new Shape();
           boolean valid = sh.randomiseShape(maxStart,shapes);
             //This checks that there is sufficient space to fit non-overlapping shapes;
           // If this is not the case, the max start value is increased by 
           //default shapelet length. Be aware that max start may exceed your original
           // value if too many shapes are used.
           if(!valid)
           {
 //              System.out.println("Insufficient space, increasing max start");
               maxStart = maxStart+shapeletLength;
               sh.randomiseShape(maxStart,shapes);
           }
           shapes.add(sh); 
        }       
    }
    //This constructor is used for data of a given length
    public ShapeletModel(int s)
    {
        this(new double[]{(double)s});
    }
    
    //This constructor is used for data of a given length in a two class problem
    //where the shape distinguishing the first class is known
    public ShapeletModel(int seriesLength,Shape shape)
    {
        setDefaults();
        shapes=new ArrayList<Shape>();
        
        maxStart = seriesLength-shapeletLength;
        for(int i=0;i<numShapelets;i++)
        {
           Shape sh = new Shape();
           boolean valid = sh.randomiseShape(maxStart,shapes);
           //This checks that the shape is not of the same type as the
           //first shape.
           while(sh.type==shape.type)
               valid = sh.randomiseShape(maxStart, shapes);

           if(!valid){
//               System.out.println("Insufficient space, increasing max start");
               maxStart = maxStart+shapeletLength;
               sh.randomiseShape(maxStart,shapes);
           }
           
           shapes.add(sh); 
        }       

    }
    // This constructor accepts an ArrayList of shapes for the shapelet model,
    // rather than determining the shapes randomly.
    public ShapeletModel(ArrayList<Shape> s)
    {
        shapes=new ArrayList<Shape>(s);
    }
    
    /*Generate a single data
//Assumes a model independent of previous observations. As
//such will not be relevant for ARMA or HMM models, which just return -1.
* Should probably remove. 
*/
    @Override
	public double generate(double x)
        {
            double value=error.simulate();
            //System.out.println("Value"+value);
            //Slightly inefficient for non overlapping shapes, but worth it for clarity and generality
            for(Shape s:shapes)
            {
                //System.out.println("***********Fired");
                value+=s.generate((int)x);
            }
                
            return value;
        }

//This will generate the next sequence after currently stored t value
    @Override
	public double generate()
        {
            //System.out.println("t"+t);
            double value=generate(t);
            t++;
            return value;
        }
    
    
    /**
 * Subclasses must implement this, how they take them out of the array is their business.
 * @param p 
 */ 
    @Override
    public void setParameters(double[] p)
    {
    }
    
    // The implementation of the reset method should be adjusted appropriately.
    // Currently uses the randomiseLocation() method to implement a random
    // location after reset. 
    public void reset()
    {
        t=0;
        
        for(int i=0;i<shapes.size();i++)
        {
            shapes.get(i).randomiseLocation(maxStart, shapes);
        }
    }
    
    public ShapeType getShapeType()
    {
        return shapes.get(0).type;
    }
    public Shape getShape()
    {
        ShapeType shp = shapes.get(0).type;
        Shape shape = new Shape(shp,1,1,1,1);
        return shape;
    }
    
    // The toString() method has not been changed.
    @Override
    public String toString(){
        return  "nos shapes = "+shapes.size()+" of type: "+shapes.get(0);

    }
    
    // Inner class determining the shape inserted into the shapelet model
    public static class Shape{
        // Type: head and shoulders, spike, step, triangle, or sine wave.
        private ShapeType type;
        //Length of shape
        private int length;
        //Position of shape on axis determined by base (lowest point) and amp(litude).
        private double base;
        private double amp;
        //The position in the series at which the shape begins.
        private int location;
        
        private static int DEFAULTLENGTH=29;
        private static int DEFAULTBASE=-2;
        private static int DEFAULTAMP=4;
        private static int DEFAULTLOCATION=0;
        
        //Default constructor, call randomise shape to get a random instance
        // The default length is 29, the shape extends from -2 to +2, is of 
        // type head and shoulders, and is located at index 0.
        private Shape()
        {
            this(ShapeType.HEADSHOULDERS,DEFAULTLENGTH,DEFAULTBASE,DEFAULTAMP,DEFAULTLOCATION); 
        }  
        //Set length only, default for the others
         private Shape(int length){
            this(ShapeType.HEADSHOULDERS,length,DEFAULTBASE,DEFAULTAMP,DEFAULTLOCATION);      
             
         }       
        // This constructor produces a completely specified shape
        private Shape(ShapeType t,int l, double b, double a, int loc){
            type=t;
            length=l;
            base=b;
            amp=a;
            location=loc;
        }
        
        //Checks the location against the value t, and outputs part of the shape
        // if appropriate.
        private double generate(int t){
            
            if(t<location || t>location+length-1)
            {
                return 0;
            }
            
            int offset=t-location;            
            double value=0;
            
            switch(type){
             case TRIANGLE:
                 
                 if(offset<=length/2)
                 {
                    if(offset==0)
                    {
                       value=base;
                       
                    }                        
                    else
                    {
                       value=((offset/(double)(length/2))*(amp))+base;
                       
                    }
                      
                 }
                 else
                 {
                     if(offset+1==length)
                     {
                         value=base;
                         
                     }
                     else
                     {
                         value=((length-offset-1)/(double)(length/2)*(amp))+base;
                         
                     }
                     
                     
                 }
   
                
                   break;
                case HEADSHOULDERS:
                    
                if(offset<length/3)
                {                                
                    
                        value = ((amp/2)*Math.sin(((2*Math.PI)/((length/3-1)*2))*offset))+base;
                                                                
                    
                }
                else
                {
                    if(offset+1>=(2*length)/3)
                    {
                        
                        if(length%3>0&&offset>=(length/3)*3)
                        {
                            value = base;
                            
                        }
                        else
                        {
                            value = ((amp/2)*Math.sin(((2*Math.PI)/((length/3-1)*2))*(offset+1-(2*length)/3)))+base;
                            
                        }
                       
                    }
                    else
                    {
                        
                        value = ((amp)*Math.sin(((2*Math.PI)/((length/3-1)*2))*(offset-length/3)))+base;
                        
                    }
                }
                
                   break;
                case SINE:
                
                 value=amp*Math.sin(((2*Math.PI)/(length-1))*offset)/2;
                 
                    break;
                case STEP:
                   
                if(offset<length/2)
                {
                    value=base;
                    
                }
                else
                {
                    value=base+amp;
                    
                }
                    break;
                case SPIKE:
                    
                if(offset<=length/4)
                 {
                    if(offset==0)
                    {
                       value=0;
                       
                    }                        
                    else
                    {
                       value=offset/(double)(length/4)*(-amp/2);
                       
                    }
                      
                 }
                if(offset>length/4 && offset<=length/2)
                {
                 
                    if(offset == length/2)
                    {
                        value=0;
                        
                    }
                    else
                    {
                        value=(-amp/2)+((length/4-offset-1)/(double)(length/4)*(-amp/2));
                        
                    }
                    
                }                               
                if(offset>length/2&&offset<=length/4*3)
                {
                    value=(offset-length/2)/(double)(length/4)*(amp/2);
                    
                }
                              
                if(offset>length/4*3)
                 {
                     
                     
                     if(offset+1==length)
                     {
                         value=0;
                         
                     }
                     else
                     {
                         value=(length-offset-1)/(double)(length/4)*(amp/2);
                         
                     }
                     
                     
                 }
                
                break;
            }
            return value;
            
        }
        
        private void setLocation(int newLoc)
        {
            this.location=newLoc;
        }
        
        private int getLocation()
        {
            return location;
        }
        
        private void setType(ShapeType newType)
        {
            this.type=newType;
        }
        
        // Randomises the starting location of a shape. Returns false when
        // there is insufficient space to fit all shapes within the value
        // of maxStart. This is resolved in the constructor.
        private boolean randomiseLocation(int maxStart, ArrayList<Shape> shapes)
        {
            if(shapes.size()*(length*2-1)>maxStart)
            {
                return false;
            }
            
            Random ran =  new Random();
            boolean validStart = false;
            int start =-1;            
            
            while(!validStart)
            {
                
                start = (int)(maxStart*ran.nextDouble());       
                int end = start+length;
                validStart = true;
                
                for(int i=0;i<shapes.size();i++)
                {
                    int locStart = shapes.get(i).getLocation();
                    int locEnd = locStart+length;
                    
                    if((start>=locStart&&start<=locEnd)||(end>=locStart&&end<=locEnd))
                    {
                        validStart=false;
                        break;
                    }
                }
                            
            }
                        
           setLocation(start);
                        
           return true; 
            
            
        }
        
        @Override
        public String toString()
        {
            String shp = ""+this.type;
            return shp;
        }
        
        
        //gives a shape a random type and start position
        private boolean randomiseShape(int maxStart, ArrayList<Shape> shapes)
        {
            
 
            Random ran =  new Random();                        
            boolean valid = randomiseLocation(maxStart, shapes);
                
            if(!valid)
            {
                return false;
            }
                                                                     
            ShapeType [] types = ShapeType.values();            
            int ranType = (int)(types.length*ran.nextDouble());
            setType(types[ranType]);
                       
           return true; 
                
        }
    
}
    
    
    //Test harness
   
    public static void main (String[] args) throws IOException
    {
       ShapeletModel shape = new ShapeletModel();
       
       
       for(int i=0;i<200;i++)
       {
           
           System.out.println(shape.generate());
       }
       
       shape.reset();
       System.out.println(-10);
       
       for(int i=0;i<200;i++)
       {
           
           System.out.println(shape.generate());
       }
                  
    }
    
}
