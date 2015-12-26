/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package simulators;


/**
 *
 * @author ajb
 */
public class SimulatePowerSpectrum extends DataSimulator {
    //Models 

    int minWaves=1;
    int maxWaves=5;

    public SimulatePowerSpectrum(double[][] paras){
        super(paras);
        for(int i=0;i<nosClasses;i++)
            models.add(new SinusoidalModel(paras[i]));
    }

    @Override
    public double[] generate(int length, int modelNos) {
        throw new UnsupportedOperationException("Not supported yet.");
    }

  
    
    
}
