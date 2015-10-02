package algorithms.gradient;

import data.DataSet;

/**
 * Created by hanxuan on 9/30/15 for machine_learning.
 */
public interface GradientDecent {
    <T> void gGradient(int start, int end, T theta);
}
