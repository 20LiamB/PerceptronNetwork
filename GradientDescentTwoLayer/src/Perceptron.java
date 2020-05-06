import java.io.*;
import java.util.Arrays;
import java.util.Scanner;

/**
 * @author Liam Bakar
 * @date 4/20/2020
 *
 * This class serves as a model for the multilayer Perceptron neural network. This structure has an A-B-C network. It defines methods that find the optimal
 * weights for whichever test case is being used. The output multiplies inputs by weights, adds them, and finds errors.
 * The constructor calls the readFile method which takes the values from the input file and creates an activation matrix
 * with the local variables. It also takes in the range for the initial weights, the error threshold, the lambda, the
 * adapt, the maximum amount of iterations, and a test case for the file.
 */
public class Perceptron
{
   private double [][][] weights;
   private double [][][] deltas;
   private double [][] activations;
   private int [] layers;
   private int numLayers;
   private double[][][] expectedValue;
   private double errorThreshold;
   private double lambda;
   private double lambdaAdaptUp;
   private double lambdaAdaptDown;
   private int iterations;
   private double minWeights;
   private double maxWeights;
   private int numCases;

   private int inputLayer;
   private int numInputs;
   private int outputLayer;
   private int outputLayerSize;
   private int hiddenActivationSize;



   /**
    * Creates an instance of the Perceptron class and sets up an activation matrix.
    *
    * @param fileName the path of the file
    * @throws Exception if the path of the file does not lead to any file
    */
   public Perceptron(String fileName) throws Exception
   {
      readFile(fileName);
   }

   /**
    * A method that reads the input file to retrieve layers, inputs, and weights
    * and assigns them to the local variables.
    *
    * @param fileName the address of the file
    */
   public void readFile(String fileName) throws Exception
   {
      //Creates a file BufferedReader
      File file = new File(fileName);
      BufferedReader br = new BufferedReader(new FileReader(file));
      //Reads and assigns layers to the layers variables
      String line = br.readLine();
      String[] layerInputs = line.split(" ");
      layers = new int[layerInputs.length];
      numLayers = layers.length;
      activations = new double[numLayers][];
      for (int counter = 0; counter < numLayers; counter++)
      {
         layers[counter] = Integer.parseInt(layerInputs[counter]);
         activations[counter] = new double[layers[counter]];
      }

      inputLayer = numLayers - numLayers;
      numInputs = layers[inputLayer];
      outputLayer = numLayers-1;
      outputLayerSize = layers[outputLayer];
      hiddenActivationSize = layers[outputLayer-1];

      //Skips the lign for the manual input
      br.readLine();

      //Reads and assigns weights to the weights local variable
      weights = new double[outputLayer][][];
      deltas = new double[outputLayer][][];
      line = br.readLine();
      if (line.equals("manual"))
      {
         for (int layerCounter = 0; layerCounter < outputLayer; layerCounter++)
         {
            line = br.readLine();
            weights[layerCounter] = new double [layers[layerCounter]][layers[layerCounter+1]];
            String [] weightInputs = line.split(" ");
            for (int left = 0; left < weights[layerCounter].length; left++)
            {
               int counter = 0;
               for (int right = 0; right < weights[layerCounter][left].length; right++)
               {
                  weights[layerCounter][left][right] = Double.parseDouble(weightInputs[counter]);
                  counter++;
               }
            }
         }
      }
      else if (line.equals("random"))
      {
         for (int layerCounter = inputLayer; layerCounter < outputLayer; layerCounter++)
         {
            // In the inputfile the layer structure dictates the weights and deltas, this means we have the weights and
            // deltas as 3 dimentional arrays, the firstLayer has 2 input elements, each with 4 weights (i.e. the zero
            // dimension has two arrays, and each is an array of 4 double element), and the secondLayer has 4 element,
            // each with 1 weight (i.e. the 1 dimension has 4 arrays, each is an array of 1 double element).
            weights[layerCounter] = new double[layers[layerCounter]][layers[layerCounter+1]];
            deltas[layerCounter] = new double[layers[layerCounter]][layers[layerCounter+1]];
         }
         //skips the lines of manual weights
         br.readLine();
         br.readLine();
      }

      //Sets the error threshold of the network
      errorThreshold = Double.parseDouble(br.readLine());

      //Set lambda multiplier of the delta
      lambda = Double.parseDouble(br.readLine());

      //Sets the lambda adaption factor in the up direction of the network
      lambdaAdaptUp = Double.parseDouble(br.readLine());

      //Sets the lambda adaption factor in the down direction of the network
      lambdaAdaptDown = Double.parseDouble(br.readLine());

      //Sets the iterations for optimizing the weights
      iterations = Integer.parseInt(br.readLine());

      //Sets the max value for a weight
      minWeights = Double.parseDouble(br.readLine());

      //Sets the minimum value for a weight
      maxWeights = Double.parseDouble(br.readLine());

      //Sets the number of test cases
      numCases = Integer.parseInt(br.readLine());

      //sets length of expectedValue array equal to however many test cases there are
      expectedValue = new double[numCases][layers[0]][];

      //populates the expectedValue array with the test case
      for (int counter = 0; counter < numCases; counter++)
      {
         line = br.readLine();
         String[] testCaseInputs = line.split(" ");
         expectedValue[counter][0] = new double[numInputs];
         expectedValue[counter][1] = new double[outputLayerSize];

         for (int x = 0; x < testCaseInputs.length-outputLayerSize; x++)
         {
            expectedValue[counter][0][x] = Double.parseDouble(testCaseInputs[x]);
         }
         for (int y = testCaseInputs.length-outputLayerSize; y < testCaseInputs.length; y++)
         {
            expectedValue[counter][1][y-layers[0]] = Double.parseDouble(testCaseInputs[y]);
         }


      }

   } //public void readFile(String fileName) throws Exception

   /**
    * Helper method that calculates the next activations
    *
    * @param layer
    */
   private void calculateTheNexActivation(int layer)
   {
      for (int j = 0; j < activations[layer+1].length; j++)
      {
         double theta = 0.0;

         for (int k = 0; k<activations[layer].length;k++)
         {
            theta += activations[layer][k] * weights[layer][k][j];
         }

         activations[layer+1][j] = thresholdFunction(theta);
      }
   }

   /**
    * This method calculate the output F based on currently set activations and weights
    *
    * @param inputs
    * @output
    */
   public double[] calculateOutput(double[] inputs)
   {
      double[] outputF;
      // First initializes the first layer of the activation aka the input
      for (int i = 0; i < layers[inputLayer]; i++)
      {
         activations[inputLayer][i] = inputs[i];
      }

      // Goes over all the layers and for each calculate the next activation
      for (int l = 0; l < numLayers - 1; l++)
      {
         calculateTheNexActivation(l);
      }

      outputF = activations[outputLayer];
      return outputF;

   } //public double[] calculateOutput(double[] inputs)


   /**
    * This function finds how close the actual output was to the expected output.
    *
    * @return the error between the real and expected output
    */
   public double errorFunction() //change so that it's average of all instead of sum
   {
      double errorSum = 0.0;


      for (int testCase = 0; testCase < numCases; testCase++)
      {
         double[] output = calculateOutput(expectedValue[testCase][0]);
         double testCaseSum = 0;
         double numOutputs = 0.0;

         for (int out = 0; out < expectedValue[testCase][1].length; out++)
         {
            double difference = expectedValue[testCase][1][out] - output[out];
            testCaseSum += difference * difference / 2.0;
            numOutputs += 1.0;

         }

         testCaseSum /= numOutputs;
         errorSum += testCaseSum;
      }

      return Math.sqrt(errorSum);
   } //public double errorFunction()


   /**
    * Finds the sigmoid threshold function for an input
    *
    * @param x the input for the function
    * @return the potential threshold
    */
   private double thresholdFunction(double x)
   {
      return 1.0 / (1.0 + Math.exp(-x));
   }

   /**
    * A function that represents the derivative of the threshold function.
    *
    * @param x the input for the function
    * @return the derivative of the potential threshold
    */
   private double derivativeThresholdFunction(double x)
   {
      double thresholdX = thresholdFunction(x);
      return thresholdX * (1.0 - thresholdX);
   }

   /**
    * Calculates the final output theta.
    *
    * @param localActivations an array of size 2 holding the two inputs
    * @param weights an array of 2 elements, each holding four weights
    *
    * @return the final output theta of the function
    */
   private double[] calculateOutputTheta(double[] localActivations, double[][] weights)
   {
      double[] thetas = new double[outputLayerSize];

      for (int j = 0; j < localActivations.length; j++)
      {
         for (int i = 0; i < outputLayerSize; i++)
         {
            thetas[i] += localActivations[j] * weights[j][i];
         }
      }

      return thetas;
   }

   /**
    * Calculates the hidden activation thetas. The thetas prior to the final output theta.
    *
    * @param localActivations an array of size 2 holding the two inputs
    * @param weights an array of 2 elements, each holding four weights
    *
    * @return the thetas in the hidden layers, the thetas before the final output
    */
   private double[] calculateHiddenActivationTheta(double[] localActivations, double[][] weights)
   {
      double[] thetas = new double[hiddenActivationSize];

      for (int j = 0; j < hiddenActivationSize; j++)
      {
         for (int i = 0; i < localActivations.length; i++)
         {
            thetas[j] += localActivations[i] * weights[i][j];
         }
      }
      return thetas;
   }

   /**
    * The minimization of the error function for an A-B-C network with C amount of outputs to find the deltas that will
    * change the weights.
    *
    * @param testIndex the index for which part of the training set is being used
    */
   private void findDeltas(int testIndex)
   {
      // calculates the output so that it populates the activations array
      calculateOutput(expectedValue[testIndex][0]);

      double[] outputTheta = calculateOutputTheta(activations[outputLayer-1], weights[outputLayer-1]);
      double[] omega = new double[outputLayerSize];
      double[] psi = new double[outputLayerSize];

      for (int x = 0; x < outputLayerSize; x++)
      {
         omega[x] = expectedValue[testIndex][1][x] - thresholdFunction(outputTheta[x]);
         psi[x] = omega[x] * derivativeThresholdFunction(outputTheta[x]);
      }

      for (int i = 0; i < layers[outputLayer-1]; i++)
      {
         for (int j = 0; j < outputLayerSize; j++)
         {
            deltas[outputLayer-1][i][j] = lambda * activations[outputLayer-1][i] * psi[j];
         }
      }

      // we can generalize this by adding a counter instead of +1
      double[] hiddenActivationThetas = calculateHiddenActivationTheta(activations[inputLayer], weights[inputLayer]);

      double capitalOmega = 0.0;

      for (int k = 0; k < layers[inputLayer]; k++) // loop over the element in the input layer
      {
         // loop over the element of the next layer except the output layer
         for (int i = 0; i < layers[inputLayer+1]; i++)
         {
            capitalOmega = 0.0;
            for (int j = 0; j < outputLayerSize; j++)
            {
               capitalOmega += psi[j] * weights[outputLayer-1][i][j];
            }
            //setting the second set of delta values
            deltas[inputLayer][k][i] =
                    lambda * activations[inputLayer][k] *
                            derivativeThresholdFunction(hiddenActivationThetas[i]) *
                            capitalOmega;
         //psi * weights[outputLayer-1][j][0];
         }
      }

   } //public void firstMinimizationFunction(double truth, double real, int testIndex)

   /**
    * Adds the deltas to the weights to create an updated set of weights. 
    */
   private void setNewWeights()
   {
      for (int n = 0; n < numLayers- 1; n++)
      {
         for (int k = 0; k < layers[n]; k++)
         {
            for (int j = 0; j < weights[n][k].length; j++)
            {
               weights[n][k][j] += deltas[n][k][j];
            }
         }
      }
   } //public void setNewWeights()

   /**
    * Optimizes the weights for the network by finding deltas (how much the weight is changed by) and updating the
    * weights by adding the deltas until either the total error is low enough, the max iterations are reached, or lambda
    * is 0. At the end, the function prints the reason for why it stopped changing the weights.
    */
   public void optimizeWeight()
   {
      int counter = 0;
      boolean errorIsTooHigh = true;
      double currentError = 0.0;
      double pastError = 0.0;

      do {
         for (int index = 0; index < numCases; index++)
         {
            findDeltas(index);
            // Sets the weights based on the latest deltas
            setNewWeights();
         }

         // calculates the errors
         currentError = errorFunction();
         if (currentError < errorThreshold)
            errorIsTooHigh = false;
         else
            errorIsTooHigh = true;

         if (pastError > currentError)
            lambda *= lambdaAdaptUp;
         else
            lambda /= lambdaAdaptDown;
         pastError = currentError;
         counter++;

      } while ((counter < iterations) && errorIsTooHigh && (lambda != 0.0));

      if (!errorIsTooHigh)
         System.out.println("\nFinished optimizing weights because the error is below the error threshold. There were "
                 + counter + " iterations");
      else if (counter >= iterations)
         System.out.println("\nFinished optimizing weights because the max amount of iterations was reached.");
      else if (lambda == 0.0)
         System.out.println("\nFinished optimizing weights because lambda was equal to zero.");

   } //public void optimizeWeight(double error)

   /**
    * Replaces the weights in the weight array with random weights to begin the search for optimized weights.
    */
   public void randomizeWeights()
   {
      System.out.println("Initial Weights: ");
      for (int n = 0; n < numLayers - 1; n++)
      {
         for (int k = 0; k < layers[n]; k++)
         {
            for (int j = 0; j < weights[n][k].length; j++)
            {
               weights[n][k][j] = Math.random() * (maxWeights - minWeights) + minWeights;
               System.out.print(weights[n][k][j] + ", ");
            }
         }
      }
   } //public void randomizeWeights()

   /**
    * Prompts the user for the name of an input file. With that file, main prints out the respective inputs, weights,
    * final output, and error.
    *
    * @param args the command line arguments given when running the program
    * @throws Exception if the file path does not lead to a real file
    */
   public static void main(String[] args) throws Exception
   {
      //Prompting for a file name
      String fileName = "InputFile.txt";
      Scanner in = new Scanner(System.in);
      System.out.println("Enter a name for the input file:");
      fileName = in.nextLine();

      //Creates new Perceptron
      Perceptron p = new Perceptron(fileName);
      p.randomizeWeights();
      p.optimizeWeight();
      System.out.println("Final Weights: ");
      for (int n = 0; n < p.numLayers - 1; n++)
      {
         for (int k = 0; k < p.layers[n]; k++)
         {
            for (int j = 0; j < p.weights[n][k].length; j++)
            {
               System.out.print(p.weights[n][k][j] + ", ");
            }
         }
      }
      System.out.println("\nThe layer structure is: ");
      for (int i = 0; i < p.numLayers; i++)
      {
          System.out.print(p.layers[i] + ", ");
      }

      System.out.println("\nThe error threshold for the network was " + p.errorThreshold +".");
      System.out.println("The maximum amount of iterations allowed was " + p.iterations + ".");
      System.out.println("The lambda for the network was " + p.lambda + ".");
      System.out.println("The weights were in the range of [" + p.minWeights + ", " + p.maxWeights + "].");

      System.out.println("Error: " + p.errorFunction());
      for (int i = 0; i < p.expectedValue.length; i++)
      {
         System.out.print("  case " + i + ": ");
         System.out.print(Arrays.toString(p.expectedValue[i][0])+" ");
         System.out.println(Arrays.toString(p.calculateOutput(p.expectedValue[i][0])));
      }

   } //public static void main(String[] args) throws Exception

} //public class Perceptron