/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package DiversifyTopKShaepelet;

import java.util.ArrayList;

/**
 *
 * @author sun
 */
public class GraphNode {
    private LegacyShapelet vertexShapelet;
    private ArrayList<LegacyShapelet> adjShapelets;
    
    public LegacyShapelet getVertexShapelet(){
        return vertexShapelet;
    }
    public ArrayList<LegacyShapelet> getAdjShapelets(){
        return adjShapelets;
    }
    
    public void setVertexShapelet(LegacyShapelet shapelet){
        this.vertexShapelet=shapelet;
    }
    public void setAdjShapelet(ArrayList<LegacyShapelet> shapelets){
        this.adjShapelets=shapelets;
    }
    
}
