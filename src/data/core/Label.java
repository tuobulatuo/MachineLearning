package data.core;

import java.util.EnumMap;

/**
 * Created by hanxuan on 9/17/15.
 */
public class Label {
    private float[] vector = null;
    private Enum Category = null;
    private EnumMap map = null;


    public Label (float[] data) {
        this.vector = data;
    }

    public Label(float[] data, Enum Category, EnumMap map) {
        this.vector = data;
        this.Category = Category;
        this.map = map;
    }
}
