/**
 * Created by weininghu on 2016-10-15.
 */

import java.math.*;
import java.util.Arrays;
import java.util.Random;

public class NN_Implementation  {

    // Declaration of variables
    int activationType;  // 1-Binary, 2-Bipolar
    int numInput; // 2 for XOR problem
    int numHidden;  // 4 for XOR problem
    double learningRate; // Learning Rate
    double momentum; // Momentum
    double epoch = 0;

    // Weights
    double [][] weightIH; // The weight from input layer to hidden layer
    double [] weightHO; // The weight from hidden layer to output layer

    // Old weights, for calculating weights delta
    double [][] oldWeightIH;
    double [] oldWeightHO;

    // Swapping weights, for updating old weights and new weights
    double [][] swapWeightIH;
    double [] swapWeightHO;

    // Output
    double []initialHidden; // The initial four values
    double []activatedHidden; // The activated four values


    double activatedOutput = 0;
    double outputDelta = 0;
    double [] hiddenDelta;

    double convergenceError = 0.05;
    double totalError = 0;

    // Input
    double [][] trainInput; // A set of
    double [] targetOutput; // 4 targeted output



//
//    private static double trainInput[][] = new double[NUM_PATTERN][NUM_INPUT];
//    private static double targetOutput[] = new double[NUM_PATTERN];
//
//    // Initialize
//    private static double weightIH[][] = new double[NUM_HIDDEN][NUM_INPUT];
//    private static double weightHO[] = new double[NUM_HIDDEN];
//
//    private static double biasIH[] = new double[NUM_HIDDEN];
//    private static double biasHO[] = new double[NUM_OUTPUT];



   // Constructor
    NN_Implementation (
            int activationType,
            int numInput,
            int numHidden,
            double learningRate,
            double momentum ) {
        this.activationType = activationType; // 1 -Binary, 2 - Bipolar
        this.numInput = numInput;
        this.numHidden = numHidden;
        this.learningRate = learningRate;
        this.momentum = momentum;

        weightIH = new double[numHidden][numInput + 1];
        weightHO = new double[numHidden + 1];

        oldWeightIH = new double[numHidden][numInput + 1];
        oldWeightHO = new double[numHidden + 1];

        swapWeightIH = new double[numHidden][numInput + 1];
        swapWeightHO = new double[numHidden + 1];

        initialHidden = new double[numHidden];
        activatedHidden = new double[numHidden];

        trainInput = new double[4][2];
        targetOutput = new double[4];

        hiddenDelta = new double[numHidden];


    }




    /*

     */

    public void defineData() {

        if (activationType == 1) {

            trainInput[ 0 ][ 0 ] = 0;
            trainInput[ 0 ][ 1 ] = 0;
            targetOutput[ 0 ] = 0;

            trainInput[ 1 ][ 0 ] = 0;
            trainInput[ 1 ][ 1 ] = 1;
            targetOutput[ 1 ] = 1;

            trainInput[ 2 ][ 0 ] = 1;
            trainInput[ 2 ][ 1 ] = 0;
            targetOutput[ 2 ] = 1;

            trainInput[ 3 ][ 0 ] = 1;
            trainInput[ 3 ][ 1 ] = 1;
            targetOutput[ 3 ] = 0;
        } else {

            trainInput[ 0 ][ 0 ] = -1;
            trainInput[ 0 ][ 1 ] = -1;
            targetOutput[ 0 ] = -1;

            trainInput[ 1 ][ 0 ] = 1;
            trainInput[ 1 ][ 1 ] = -1;
            targetOutput[ 1 ] = 1;

            trainInput[ 2 ][ 0 ] = -1;
            trainInput[ 2 ][ 1 ] = 1;
            targetOutput[ 2 ] = 1;

            trainInput[ 3 ][ 0 ] = 1;
            trainInput[ 3 ][ 1 ] = 1;
            targetOutput[ 3 ] = -1;

        }


    }

    // Initialize the weights between -0.5 and 0.5

    /**
     *
     * @return Initialize the weight as value between -0.5 and 0.5, including bias
     */
    public void initializeWeights(){
        for (int i = 0; i < numHidden + 1; i ++){
            weightHO[i] = new Random().nextDouble() - 0.5;

        }

        for (int i =0; i < numHidden; i ++){
            for (int j =0; j < numInput + 1; j ++){
                weightIH[i][j] = new Random().nextDouble() - 0.5;
            }
        }

    }

    public void zeroWeights() {
        // Initialize the weight InputToHidden to random values
        for (int i = 0; i < numHidden; i++) {
            for (int j = 0; j < numInput + 1; j++) {
                oldWeightIH[i][j] = 0;
            }
        }

        // Initialize the weight HiddenToOutput to random values
        for (int j = 0; j < numHidden + 1; j++) {
            oldWeightHO[j] = 0;
        }
    }

    /**
     *
     * @param X The input vectors, an array of doubles
     * @return the sigmoid output value
     */

    // step4, step5
    public double outputFor( double[] X ){
//        double initialHidden[] = new double[numHidden];
//        double activatedHidden[] = new double[numHidden];
        Arrays.fill(initialHidden, 0);
        Arrays.fill( activatedHidden, 0 );

        double initialOutput = 0;
        activatedOutput = 0;

        // w0*x0 + w1*x1
        for (int i = 0; i < numHidden; i++){
            initialHidden[i] = 0;
            // w0*x0 + x1*x1 + bias0
            initialHidden[i] += 1.0*weightIH[i][numInput];

            for (int j = 0; j< numInput; j ++){
                initialHidden[i] += X[j]*weightIH[i][j];
            }
            if (activationType == 1) {
                activatedHidden[ i ] = binarySigmoid( initialHidden[i] );
            } else {
                activatedHidden[i] = bipolarSigmoid( initialHidden[i] );
            }
        }

        // wo*x0 + w1*x1 + w2*x2 + w3*x3
        for (int i = 0; i < numHidden; i++){
            initialOutput += activatedHidden[i]*weightHO[i];
        }
        // wo*x0 + w1*x1 + w2*x2 + w3*x3 + bias1
        initialOutput += weightHO[numHidden];
        if (activationType == 1){
            activatedOutput = binarySigmoid( initialOutput );
        } else {
            activatedOutput = bipolarSigmoid( initialOutput );
        }




        return activatedOutput;
    }

    // step 6
    public void bpOutputError(double target){
        outputDelta = 0;
        if (activationType ==1) {
            // Binary error
            // deltaK = (tk - yk)[yk(1-yk)]
            outputDelta = (target - activatedOutput) * activatedOutput * (1 - activatedOutput);
        } else{
            // Bipolar error
            // deltaK = (tk - yk)[0.5*(1 + yk)(1 - yk)]
            outputDelta = 0.5*(target - activatedOutput)*(1+activatedOutput)*(1-activatedOutput);
        }

    }


    // step 7
    public void bpHiddenError() {
        if (activationType == 1){
            // Binary error
            for (int i = 0; i < numHidden; i++){
                hiddenDelta[i] = weightHO[i]*outputDelta*activatedHidden[i]*(1-activatedHidden[i]);
            }

        } else {
            for (int i = 0; i < numHidden; i++){
                hiddenDelta[i] = weightHO[i]*outputDelta*0.5*(1 + activatedHidden[i])*(1-activatedHidden[i]);
            }
        }


    }

    // step 8
    public void updateWeightIH(double[] X) {
        // Load current weights into swapping weights
        for (int i = 0; i < numHidden; i++){
            System.arraycopy( weightIH[i], 0, swapWeightIH[i], 0, weightIH[i].length );
        }

        for (int i = 0; i < numHidden; i++){
            for (int j = 0; j < numInput; j++){
                //Wi,j * = Wi,j + alpha*deltaWi,j + rou*delta_i*uj
                weightIH[i][j]+= momentum*calDeltaWeightIH( i,j ) + learningRate*hiddenDelta[i]*X[j];
            }
            // Update the corresponding bias for each hidden unit
            weightIH[i][numInput] += momentum*calDeltaWeightIH( i,numInput ) + learningRate*hiddenDelta[i];
        }

        // Load swapping weights into old weights
        for (int i = 0; i < numHidden; i++){
            System.arraycopy( swapWeightIH[i], 0, oldWeightIH[i], 0, weightIH[i].length );


        }
    }


    private void updateWeightHO() {
        // backup current weight
        System.arraycopy(weightHO, 0, swapWeightHO, 0, weightHO.length);

        for (int i = 0; i < numHidden; i++) {
            weightHO[i] += learningRate * outputDelta * activatedHidden[i] + momentum * calDeltaWeightHO(i);
            //System.out.println("numHidden: " + i + "  weight: " +weightHiddenToOutput[i]);
        }
        weightHO[numHidden] += learningRate * outputDelta * 1 + momentum * calDeltaWeightHO(numHidden);
        // update previous weight
        System.arraycopy(swapWeightHO, 0,
                oldWeightHO, 0,
                swapWeightHO.length);
    }

    public double calDeltaWeightIH(int i, int j){
        if (oldWeightIH[i][j] != 0){
            return weightIH[i][j] - oldWeightIH[i][j];
        } else{
            return 0;
        }
    }

    public double calDeltaWeightHO(int i){
        if (oldWeightHO[i] != 0){
            return weightHO[i] - oldWeightHO[i];
        } else {
            return 0;
        }
    }


    /**
     *  Return a binary sigmoid of the input x
     * @param x The input
     * @return f(x) = 1/(1 + e(-x))
     */

    public double binarySigmoid( double x ){
        double result;
        result = 1/( 1 + Math.exp( -x ) );
        return result;
    }

    /**
     * Return a bipolar sigmoid of the input X
     * @param x The input
     * @return f(x) = 2 / (1+e(-x)) - 1
     */

    public double bipolarSigmoid( double x ){
        double result;
        result = (2/( 1 + Math.exp( -x ) )) - 1;
        return result;

    }

    public double derivationOfBinarySigmoid( double x ){
        double result;
        result = binarySigmoid( x )*( 1 - binarySigmoid( x ));
        return result;
    }

    public double derivationOfBipolarSigmoid( double x ){
        double result;
        result = 0.5*( 1 + bipolarSigmoid( x ) )*( 1 - bipolarSigmoid( x ) );
        return result;

    }

    public double train(double [] X, double target){
        // Feedforward
        outputFor( X );

        // Backpropagation
        bpOutputError( target );
        bpHiddenError();

        // Update weights
        updateWeightHO();
        updateWeightIH( X );

        return activatedOutput;


    }


    public static void main(String[] args) {
        NN_Implementation bp = new NN_Implementation( 2, 2, 4, 0.2, 0.9 );
        bp.initializeWeights();
        bp.defineData();
        bp.zeroWeights();

        double totalError = 1;
        int epoch;
        double convergenceError = 0.05;

        epoch = 0;
            while ( totalError > convergenceError ) {
                totalError = 0;
                epoch++;
                for ( int i = 0; i < 4; i++ ) {
                    bp.train( bp.trainInput[ i ], bp.targetOutput[ i ] );
//                    System.out.println(bp.weightHO[0]);
//
//                    System.out.println("activated   " + bp.activatedOutput);
                    totalError += 0.5 * Math.pow( (bp.activatedOutput - bp.targetOutput[ i ]), 2 );

//                    System.out.println("train input" + bp.trainInput[i][0] + bp.trainInput[i][1]);
//                    System.out.println(bp.targetOutput[i]);
//                    System.out.println("epoch: " + epoch + "pattern no.: " + i + ", total error: " + totalError);
                }


                //System.out.println("Epoch: " + epoch + "Total Error is " + totalError);
                System.out.println( epoch + "\t" + totalError );

            }
            System.out.println( epoch + "\t");


        }
    }










