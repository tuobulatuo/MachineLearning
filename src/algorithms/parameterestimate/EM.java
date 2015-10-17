package algorithms.parameterestimate;

/**
 * Created by hanxuan on 10/17/15 for machine_learning.
 */
public interface EM {

    void initialize();

    void e();

    void m();

    boolean convergence();

    default void run() {

        initialize();
        while (!convergence()) {
            e();
            m();
        }
    }

}
