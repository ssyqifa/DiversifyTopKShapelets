/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package DiversifyTopKShaepelet;

import java.awt.Point;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

/**
 *
 * @author sun
 */
public class USAXElmentType {
    HashSet<Integer> objectHashSet;
    ArrayList<Point> SAXIdArrayList;
    HashMap<Integer, Integer> objectCountHashMap;

    public HashSet<Integer> getObjectHashSet() {
        return objectHashSet;
    }

    public ArrayList<Point> getSAXIdArrayList() {
        return SAXIdArrayList;
    }

    public HashMap<Integer, Integer> getObjectCountHashMap() {
        return objectCountHashMap;
    }
    public USAXElmentType(){
        this.objectHashSet=new HashSet<>();
        this.SAXIdArrayList=new ArrayList<>();
        this.objectCountHashMap=new HashMap<>();
    }
}
