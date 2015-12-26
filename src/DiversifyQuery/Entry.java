/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package DiversifyQuery;


import java.util.ArrayList;

/**
 *
 * @author sun
 */
public class Entry implements Comparable<Entry>{
    ArrayList<LegacyShapelet> solution=new ArrayList<>();
    int pos;
    double score;
    double bound;
    
    public Entry(){
        this.solution.clear();
        this.pos=-1;
        this.score=0;
        this.bound=0;
    }
    
    public double getBound(){
        return bound;
    }
    public void setBound(double b){
        this.bound=b;
    }
    
    public boolean greaterThan(Entry entry){
        
        return this.bound>entry.bound;
    }
    public boolean lessThan(Entry entry){
        
        return this.bound<entry.bound;
    }
    public boolean greaterEqualTo(Entry entry){
        return this.bound>=entry.bound;
    }
    
    @Override
    public int compareTo(Entry entry){
        if(this.bound<entry.bound)
            return -1;
        else
            if(this.bound>entry.bound)
                return 1;
            else
                return 0;
    }
    
}
