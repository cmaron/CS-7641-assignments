package func.nn.activation;

/**
 * The tanh sigmoid function
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class RELU
		extends DifferentiableActivationFunction{

   /**
    * @see nn.function.DifferentiableActivationFunction#derivative(double)
    */
	public double derivative(double value) {
		if (value < 0){
			return 0;
		} else {
			return 1;
		}
	}

	/**
	 * @see nn.function.ActivationFunction#activation(double)
	 */
    public double value(double value) {
        
        if (value < 0) {
            return 0;
        } else {
            return value;
        }
	}
	

}
