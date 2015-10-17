package algorithms.parameterestimate;

import data.DataSet;
import gnu.trove.list.array.TIntArrayList;

import java.util.stream.IntStream;

/**
 * Created by hanxuan on 10/17/15 for machine_learning.
 */
public class DataPointSet {

    private DataSet data = null;

    private int[] rows = null;

    private int[] cols = null;

    public DataPointSet (DataSet data, int classIndex, int[] cols) {

        this.data = data;
        this.cols = cols;

        TIntArrayList rowList = new TIntArrayList(data.getInstanceLength());
        IntStream.range(0, data.getInstanceLength()).forEach(i -> {
            if (data.getLabel(i) == classIndex) {
                rowList.add(i);
            }
        });
        this.rows = rowList.toArray();
    }

    public int size() {
        return rows.length;
    }

    public int width() {
        return cols.length;
    }

    public double[] getI(int i) {

        double[] x = data.getInstance(rows[i]);
        if (x.length == cols.length) {
            return x;
        }else {
            double[] selectedX = new double[cols.length];
            IntStream.range(0, cols.length).forEach(k -> selectedX[k] = x[cols[k]]);
            return selectedX;
        }
    }
}
