package feature.haar;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.DataInputStream;
import java.io.FileInputStream;

/**
 * Created by hanxuan on 11/15/15 for machine_learning.
 */
public class MNISTReader {

    private static Logger log = LogManager.getLogger(MNISTReader.class);

    private DataInputStream labelsStream = null;

    private DataInputStream imagesStream = null;

    private static final int LABEL_MAGIC = 2049;

    private static final int IMAGE_MAGIC = 2051;

    private int numRows = 0;

    private int numCols = 0;

    private int total = 0;

    private int counter = 0;

    public MNISTReader (String path, String imageFile, String labelFile) throws Exception{

        labelsStream = new DataInputStream(new FileInputStream(path + "/" + labelFile));
        imagesStream = new DataInputStream(new FileInputStream(path + "/" + imageFile));

        int labelMagicNumber = labelsStream.readInt();
        if (labelMagicNumber != LABEL_MAGIC) {
            log.error("Label file has wrong magic number: {} (should be {})", labelMagicNumber, LABEL_MAGIC);
            System.exit(0);
        }

        int imageMagicNumber = imagesStream.readInt();
        if (imageMagicNumber != IMAGE_MAGIC) {
            log.error("Image file has wrong magic number: {} (should be {})", imageMagicNumber, IMAGE_MAGIC);
            System.exit(0);
        }

        int numLabels = labelsStream.readInt();
        int numImages = imagesStream.readInt();
        if (numLabels != numImages) {
            log.error("Image file and label file do not contain the same number of entries.");
            log.error("  Label file contains: {}", numLabels);
            log.error("  Image file contains: {}", numImages);
            System.exit(0);
        }

        total = numLabels;
        numRows = imagesStream.readInt();
        numCols = imagesStream.readInt();
    }

    public boolean hasNext() throws Exception{
        return (labelsStream.available() > 0 && counter < total);
    }

    public int readNext(int[][] image) throws Exception{

        byte label = labelsStream.readByte();
        for (int colIdx = 0; colIdx < numCols; colIdx++)
            for (int rowIdx = 0; rowIdx < numRows; rowIdx++)
                image[colIdx][rowIdx] = imagesStream.readUnsignedByte();

        counter++;
        return Byte.toUnsignedInt(label);
    }

    public int getNumRows() {
        return numRows;
    }

    public int getNumCols() {
        return numCols;
    }

    public int getTotal() {
        return total;
    }

    public static void main(String[] args) throws Exception{

        String path = "/Users/hanxuan/Dropbox/neu/fall15/machine learning/data/digits.mnist";
        String labelFile = "t10k-labels-idx1-ubyte";
        String imageFile = "t10k-images-idx3-ubyte";

        MNISTReader reader = new MNISTReader(path, imageFile, labelFile);

        int total = reader.getTotal();
        int colNum = reader.getNumCols();
        int rowNum = reader.getNumRows();

        log.info("{} * {} * {} = {} Mbytes", total, colNum, rowNum, total * colNum * rowNum * 4 / 1024 / 1024);

        int counter = 0;

        while (reader.hasNext()) {
            int[][] image = new int[rowNum][colNum];
            int label = reader.readNext(image);
            log.info("{}", label);
            log.info("{}", image);
            if (counter ++ > 10) break;
        }
    }
}

