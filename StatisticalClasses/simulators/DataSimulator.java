/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package simulators;

import java.util.ArrayList;
import java.util.Arrays;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author ajb
 */
public class DataSimulator {
    
    		int nosClasses=2;
                ArrayList<Model> models;
                
        protected DataSimulator(double[][] paras){
            nosClasses=paras.length;
            models=new ArrayList<Model>(nosClasses);
        } 
/**
 * So you can either load the models like this, or you can subclass DataSimulator and redefine the constructor
 * to create the base Models. This is a deep copy, I THINK!?
 * @param m 
 */
        public DataSimulator(ArrayList<Model> m){
            nosClasses=m.size();          
            models.addAll(m);
        }
    @SuppressWarnings("ManualArrayToCollectionCopy")
        public DataSimulator(Model[] m){
            nosClasses=m.length;
            models=new ArrayList<Model>(nosClasses);
            for(int i=0;i<m.length;i++)
                models.add(m[i]);
        }
        public void setModel(ArrayList<Model> m){
            nosClasses=m.size();
            models.addAll(m);
        }
    @SuppressWarnings("ManualArrayToCollectionCopy")
        public void setModel(Model[] m){
            nosClasses=m.length;
            for(int i=0;i<m.length;i++)
                models.add(m[i]);
        }
/**
 * @PRE: All parameters of the model have been set through other means
 * @POST: no change to the model, no instances are stored
 * 
 * @param seriesLength Length of each series, assumed same for all instances
 * @param casesPerClass. nosCases.length specifies the number of cases (which should already be stored), casesPerClass[i] gives the number of cases in class i 
 * @return Set of n=sum(casesPerClass[i]) instances, each seriesLength+1 attributes, the last of which is the class label,
 */
    public Instances generateDataSet(int seriesLength, int[] casesPerClass) {
        Instances data;
        FastVector atts=new FastVector();
        nosClasses=casesPerClass.length;
        int totalCases=casesPerClass[0];
        for(int i=1;i<casesPerClass.length;i++)
                totalCases+=casesPerClass[i];
        for(int i=1;i<=seriesLength;i++){
                atts.addElement(new Attribute("t"+i));
        }
        FastVector fv=new FastVector();
        for(int i=0;i<nosClasses;i++)
                fv.addElement(""+i);
        atts.addElement(new Attribute("Target",fv));
        data = new Instances("AR",atts,totalCases);

        double[] d;
        for(int i=0;i<nosClasses;i++){
                for(int j=0;j<casesPerClass[i];j++){
//Generate the series					
                        initialise();
                        d=generate(seriesLength,i);
//Add to an instance
                        Instance in= new Instance(data.numAttributes());
                        for(int k=0;k<d.length;k++)
                                in.setValue(k,d[k]);
//Add to all instances					
                        data.add(in);
                        in=data.lastInstance();
                        in.setValue(d.length,""+i);
                }

        }
        data.setClassIndex(seriesLength);
        return data;
    }
    public double[] generate(int length, int modelNos){
        double[] d=new double[length];
        Model a=models.get(modelNos);
        d=a.generateSeries(length);
        return d;
    }
/** 
 * This method
 */    
    public void initialise(){
        for(Model a:models)
            a.reset();
    };
/**
 * @return String with all parameter names and values
 */    
   public String getParameters(){
       String str=nosClasses+"\n";
       for(Model m:models)
           str+=m.toString()+"\n";
       return str;
   } 
}
