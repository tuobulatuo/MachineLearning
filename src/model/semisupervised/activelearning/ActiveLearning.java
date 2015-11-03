package model.semisupervised.activelearning;

/**
 * Created by hanxuan on 11/2/15 for machine_learning.
 */
public interface ActiveLearning {

    default void loop(){
        while (!converge()) {
            activeTrain();
            addData();
        }
    }

    boolean converge();

    void activeTrain();

    void addData();
}
