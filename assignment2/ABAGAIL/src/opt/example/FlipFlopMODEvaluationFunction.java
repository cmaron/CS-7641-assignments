package opt.example;

import util.linalg.Vector;
import opt.EvaluationFunction;
import shared.Instance;

/**
 * A function that counts the ones in the data
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class FlipFlopMODEvaluationFunction implements EvaluationFunction {
    /**
     * @see opt.EvaluationFunction#value(opt.OptimizationData)
     */
	 /* modified to favour a 1 at the start */
	 public long fevals;
    public double value(Instance d) {
        Vector data = d.getData();
        double val = 0;
        for (int i = 0; i < data.size() - 1; i++) {
            if (data.get(i) != data.get(i + 1)) {
                val++;
            }
        }
		if (data.get(0) >0) {
			val=val+10;	
		}
		this.fevals = this.fevals + 1;
        return val;
    }
}
