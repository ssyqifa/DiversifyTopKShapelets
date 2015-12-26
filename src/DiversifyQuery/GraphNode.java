/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package DiversifyQuery;

import DiversifyTopKShaepelet.*;
import java.util.ArrayList;

/**
 *
 * @author sun
 */
public class GraphNode<T extends Comparable<T>>{
    private T vertex;
    private ArrayList<T> adj;
    
    public T getVertex(){
        return vertex;
    }
    public ArrayList<T> getAdj(){
        return adj;
    }
    
    public void setVertex(T value){
        this.vertex=value;
    }
    public void setAdj(ArrayList<T> adjsArrayList){
        this.adj=adjsArrayList;
    }
    
}
