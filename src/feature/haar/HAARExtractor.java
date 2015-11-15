package feature.haar;

import gnu.trove.list.array.TFloatArrayList;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Created by hanxuan on 11/15/15 for machine_learning.
 */
public class HAARExtractor {

    private static Logger log = LogManager.getLogger(HAARExtractor.class);

    private int[][] table = null;

    public HAARExtractor(int[][] image) {
        createTable(image);
        log.info("table: ");
        log.info(" {}", table);
    }

    public float[] extract(int windowWidth, int windowHeight, int widthGap, int heightGap) {
        return slide(windowWidth, windowHeight, widthGap, heightGap);
    }

    private void createTable(int[][] image) { // dp solve

        table = new int[image.length][image[0].length];
        for (int i = 0; i < image.length; i++)
            for (int j = 0; j < image[i].length; j++)
                table[i][j] = tableIJ(i, j - 1) + tableIJ(i - 1, j) - tableIJ(i - 1, j - 1) + (image[i][j] > 0 ? 1 : 0);
    }

    private int tableIJ(int i, int j) {
        return (i < 0 || j < 0) ? 0 : table[i][j];
    }

    private float[] slide(int windowWidth, int windowHeight, int widthGap, int heightGap) {

        int capacity = table[0].length * table.length / widthGap / heightGap * 2;
        TFloatArrayList feature = new TFloatArrayList(capacity);
        for (int up = 0; up + windowHeight <= table.length; up += heightGap) { // discard corner cases
            int down = up + windowHeight - 1;
            for (int left = 0; left + windowWidth <= table[up].length; left += widthGap) {
                int right = left + windowWidth - 1;
                int hMid = (left + right) / 2; // horizontal middle point
                feature.add(blackInArea(left, hMid, up, down) - blackInArea(hMid + 1, right, up, down));
                int vMid = (up + down) / 2; // vertical middle point
                feature.add(blackInArea(left, right, up, vMid) - blackInArea(left, right, vMid + 1, down));
            }
        }
        return feature.toArray();
    }

    private int blackInArea(int left, int right, int up, int down) {
        return tableIJ(down, right) - tableIJ(up - 1, right) - tableIJ(down, left - 1) + tableIJ(up - 1, left - 1);
    }

    public static void main(String[] args) {

        int[][] image = new int[][]{
                {0,2,0,4},
                {0,0,5,0},
                {1,5,6,0},
                {4,0,5,0}
        };

        HAARExtractor extractor = new HAARExtractor(image);
        float[] feature = extractor.extract(4,4,2,2);
        log.info("{}", feature);
    }

}
