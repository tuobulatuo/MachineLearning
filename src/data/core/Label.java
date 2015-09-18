package data.core;

import java.util.EnumMap;
import java.util.HashSet;

/**
 * Created by hanxuan on 9/17/15.
 */
public class Label {

    private float[] vector = null;

    private HashSet<String> categories = null;

    private boolean isCategory = false;

    public Label (float[] data) {
        this.vector = data;
    }

    public Label(float[] data, HashSet<String> categories) {
        this.vector = data;
        this.categories = categories;
    }

    public boolean isCategoryLabel() {
        return isCategory;
    }


    public static void main(String[] args) {

    }
}
