/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package DiversifyQuery;

import DiversifyTopKShaepelet.*;
import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author sun
 */
public class MaxHeap<T extends Comparable<T>>{
    
    private ArrayList<T> array=new ArrayList<>();
    private int currentSize;
    
    public int getCurrentSize()
    {
        return currentSize;
    }
    public ArrayList<T> getArray(){
        return  array;
    }
    public MaxHeap(){
        currentSize=0;
        array.add(null);
    }    
    
    public boolean isEmpty(){
        return currentSize==0;
    }
    
    public int size(){
        return currentSize;
    }
    
    public void insert(T value){
        array.add(null);
        array.set(++currentSize, value);
        swim(currentSize);
    }
    
    public T getMax(){
        return array.get(1);
    }
    
    public T deleteMax(){
        T max=array.get(1);
        exchange(1, currentSize--);
        array.set(currentSize+1, null);
        sink(1);
        return max;
    }
    
    public void update(ArrayList<T> array, int i, T e){
        array.set(i, e);
        if(i==1){
            sink(i);
            return;
        }
        if(less(i/2, i)){
            swim(i);
            return;
        }
        
            sink(i);
         
    }
    
    private boolean less(int i, int j){
        return array.get(i).compareTo(array.get(j))<0;
    }
    private void exchange(int i, int j){
        T temp=array.get(i);
        array.set(i,array.get(j));
        array.set(j, temp);
    }
    
    private void swim(int k){
        while(k>1 && less(k/2, k)){
            exchange(k/2, k);
            k=k/2;
        }
    }
    private void sink(int k){
        while(2*k<=currentSize){
            int j=2*k;
            if(j<currentSize && less(j,j+1))
                j++;
            if(!less(k, j))
                break;
            exchange(k, j);
            k=j;
        }
    }
    
}
