import java.io.*;
import java.util.Arrays;
import java.util.Scanner;

/**
 * @author Liam Bakar
 * @date 5/5/2020
 *
 * This class serves as a model for the multilayer Perceptron neural network using back propagation to minimize the
 * error. This structure has 3 layers and can be modeled as an A-B-C-D network. It defines methods that find the optimal
 * weights for whichever test case is being used. The output multiplies inputs by weights, adds them, and finds errors.
 * To minimize the error and optimize the weights, this network uses the back propagation algorithm from the notes in
 * class. The constructor calls the readFile method which takes the values from the input file and creates an activation
 * matrix with the local variables. It also takes in the range for the initial weights, the error threshold, the lambda,
 * the adapt, the maximum amount of iterations, and a test case for the file.
 *
 *
 *
 * Perceptron()                     - Constructs a perceptron network using the input file
 * readFile()                       - Retrieves and reads data from the input file
 * calculateTheNexActivation()      - Calculates the next activations in the network
 * calculateOutput()                - Calculates the output of the network
 * errorFunction()                  - Calculates the error of the test cases
 * thresholdFunction()              - Calculates the sigmoid threshold function for an input
 * derivativeThresholdFunction()    - The derivative of the threshold function
 * dimensionArrayNLayers()          - Sets the dimensions of arrays used in back propagation
 * resetArray()                     - Clears arrays used in back propagation
 * backProp()                       - Optimizes weights for one instance using back propagation
 * optimizeWeight()                 - Trains the weights until the end conditions are satisfied
 * randomizeWeights()               - Randomizes the initial weights of the network
 * main()                           - Runs the network, optimizing the weights for the designated inputs
 *
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
   private Perceptron(String fileName) throws Exception
   {
      readFile(fileName);
   }

   /**
    * A method that reads the input file to retrieve layers, inputs, and weights
    * and assigns them to the local variables.
    *
    * @param fileName the address of the file
    */
   private void readFile(String fileName) throws Exception
   {
      File file = new File(fileName);                             //Creates a file BufferedReader
      BufferedReader br = new BufferedReader(new FileReader(file));

      String line = br.readLine();                                //Reads and assigns layers to the layers variables
      String[] layerInputs = line.split(" ");
      layers = new int[layerInputs.length];
      numLayers = layers.length;
      activations = new double[numLayers][];
      for (int counter = 0; counter < numLayers; counter++)
      {
         layers[counter] = Integer.parseInt(layerInputs[counter]);
         activations[counter] = new double[layers[counter]];
      }

      inputLayer = 0;
      numInputs = layers[inputLayer];
      outputLayer = numLayers-1;
      outputLayerSize = layers[outputLayer];
      hiddenActivationSize = layers[outputLayer-1];

      br.readLine();                                              //Skips the lign for the manual input

      weights = new double[outputLayer][][];                      //Reads and assigns weights to the weights local variable
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
      } //if (line.equals("manual"))
      else if (line.equals("random"))
      {
         for (int layerCounter = inputLayer; layerCounter < outputLayer; layerCounter++)
         {
            weights[layerCounter] = new double[layers[layerCounter]][layers[layerCounter+1]];
            deltas[layerCounter] = new double[layers[layerCounter]][layers[layerCounter+1]];
         }

         br.readLine();                                              //skips the lines of manual weights
         br.readLine();
      } //else if (line.equals("random"))


      errorThreshold = Double.parseDouble(br.readLine());            //Sets the error threshold of the network

      lambda = Double.parseDouble(br.readLine());                    //Set lambda multiplier of the delta

      lambdaAdaptUp = Double.parseDouble(br.readLine());             //Sets the lambda adaption in the up direction

      lambdaAdaptDown = Double.parseDouble(br.readLine());           //Sets the lambda adaption in the down direction

      iterations = Integer.parseInt(br.readLine());                  //Sets the iterations for optimizing the weights

      minWeights = Double.parseDouble(br.readLine());                //Sets the max value for a weight

      maxWeights = Double.parseDouble(br.readLine());                //Sets the minimum value for a weight

      numCases = Integer.parseInt(br.readLine());                    //Sets the number of test cases

      expectedValue = new double[numCases][layers[0]][];             //sets to however many test cases there are


      for (int counter = 0; counter < numCases; counter++)           //populates the array with the test case
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


      } //for (int counter = 0; counter < numCases; counter++)

   } //public void readFile(String fileName) throws Exception

   /**
    * Helper method that calculates the next activations
    *
    * @param layer the layer previous to the one of the activations that will be populated
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
    * @param inputs the inputs that will be summed in the output
    */
   private double[] calculateOutput(double[] inputs)
   {
      double[] outputF;

      for (int i = 0; i < layers[inputLayer]; i++) // First initializes the first layer of the activation aka the input
      {
         activations[inputLayer][i] = inputs[i];
      }


      for (int l = 0; l < numLayers - 1; l++)      // Goes over all the layers; for each calculate the next activation
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
   private double errorFunction()
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

      return Math.sqrt(errorSum) / 10;
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
    * Helper method to build arrays like thetas, omegas, and psis
    *
    * @return a new array with the updated dimensions
    */
   private double[][] dimensionArrayNLayers()
   {
      double[][] array = new double[numLayers][];

      for (int i = 0; i < numLayers; i++)
      {
         array[i] = new double[layers[i]];
      }

      return array;
   }

   /**
    * Resets a 2D array so that all its values are zero
    *
    * @param array the array that's being reset
    */
   public void resetArray(double[][] array)
   {
      for (int i = 0; i < array.length; i++)
      {
         for (int j = 0; j < array[i].length; j++)
         {
            array[i][j] = 0.0;
         }
      }
   }

   /**
    * Uses the back propagation method from the notes to optimize the weights
    *
    * @return the new, updated weights
    */
   private void backProp(int testIndex, double[][] thetas, double[][] omegas, double[][] psis)
   {
      for (int k = 0; k < layers[inputLayer]; k++)
      {
         activations[inputLayer][k] = expectedValue[testIndex][0][k];
      }

      resetArray(thetas);
      resetArray(omegas);
      resetArray(psis);

      for (int a = 1; a < numLayers; a++)
      {
         for (int b = 0; b < layers[a]; b++)
         {
            for (int g = 0; g < layers[a-1]; g++)
            {
               thetas[a][b] += activations[a - 1][g] * weights[a - 1][g][b];
            }

            activations[a][b] = thresholdFunction(thetas[a][b]);

            if (a == outputLayer)
            {
               omegas[a][b] = expectedValue[testIndex][1][b] - activations[a][b];
               psis[a][b] = omegas[a][b] * derivativeThresholdFunction(thetas[a][b]);
            }
         } //for (int b = 0; b < layers[a]; b++)
      } //for (int a = 1; a < numLayers; a++)

      for (int a = outputLayer-1; a >= 0; a--)
      {
         for (int b = 0; b < layers[a]; b++)
         {
            int nextLayer = a+1;
            for (int g = 0; g < layers[nextLayer]; g++)
            {
               omegas[a][b] += psis[nextLayer][g] * weights[a][b][g];
               weights[a][b][g] += lambda * activations[a][b] * psis[nextLayer][g];
            }
            psis[a][b] = omegas[a][b] * derivativeThresholdFunction(thetas[a][b]);
         }
      } //for (int a = outputLayer-1; a >= 0; a--)

   } //public void backProp(int testIndex)

   /**
    * Optimizes the weights for the network by finding deltas (how much the weight is changed by) and updating the
    * weights by adding the deltas until either the total error is low enough, the max iterations are reached, or lambda
    * is 0. At the end, the function prints the reason for why it stopped changing the weights.
    */
   private void optimizeWeight()
   {
      int counter = 0;
      boolean errorIsTooHigh = true;
      double currentError = 0.0;
      double pastError = 0.0;


      //variables required for the backProp method
      double[][] thetas = dimensionArrayNLayers();
      double[][] omegas = dimensionArrayNLayers();
      double[][] psis = dimensionArrayNLayers();

      do {

         for (int index = 0; index < numCases; index++)
         {
            backProp(index, thetas, omegas, psis);
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
   private void randomizeWeights()
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
      String fileName = "InputFile.txt"; //Prompting for a file name
      Scanner in = new Scanner(System.in);
      System.out.println("Enter a name for the input file:");
      //fileName = in.nextLine();

      Perceptron p = new Perceptron(fileName); //Creates new Perceptron
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