package de.htw.ml;

import java.io.IOException;

import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;
import org.jblas.ranges.IntervalRange;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.stage.Stage;

public class Ue07_Christian_Westphal extends Application {

	public static final String title = "Line Chart";
	public static final String xAxisLabel = "iteration";
	public static final String yAxisLabel = "prediction rate";
	public static final FloatMatrix alpha = FloatMatrix.scalar((float) 0.2);
	public static final int TESTPERCENTAGE = 10;
	public static final IntervalRange Y = new IntervalRange(0,1);
	public static final IntervalRange X = new IntervalRange(1,21);
	
	public static void main(String[] args) throws IOException {
		FloatMatrix originalMatrix = FloatMatrix.loadCSVFile("german_credit_jblas.csv");
		FloatMatrix xTrainSet = prepareSets(originalMatrix)[0];
		FloatMatrix yTrainSet = prepareSets(originalMatrix)[1];
		FloatMatrix xTestSet = prepareSets(originalMatrix)[2];		
		FloatMatrix yTestSet = prepareSets(originalMatrix)[3];
		int [] rate = training(xTrainSet, yTrainSet, xTestSet, yTestSet, yTestSet.rows);
		plot(rate); 
	}

	// ---------------------------------------------------------------------------------
	// -------------- All changes from here on happen at your own risk -----------------
	// --------------------------------------------------------------------------------- 
	
	/**
	 * Prepare the test set and the training set.
	 * 1. Splits the original matrix horizontal into two parts depending on y (700 and 300).
	 * 2. Create test set.
	 * 3. Oversample the data without including the data from the test set.
	 * 4. Create train set.
	 * 5. Split the y from the x for the training set and test set.
	 * 6. Normalize the x values of both sets.
	 * @param originalMatrix
	 * @return [normalizedTrainSet, trainSetY, normalizedTestSet, testSetY]
	 */
	
	private static FloatMatrix [] prepareSets(FloatMatrix originalMatrix) {
		FloatMatrix matrix0 = splitZerosFromOnes(originalMatrix)[0];
		FloatMatrix matrix1 = splitZerosFromOnes(originalMatrix)[1];
		int setLength = setLength(matrix0, matrix1);
		int testSetLength = testSetLength(setLength, TESTPERCENTAGE);
		FloatMatrix testSetWithY = prepareTestSet(matrix0, matrix1, TESTPERCENTAGE, testSetLength);
		FloatMatrix oversampledMatrix0 = oversampling(matrix0, matrix1, testSetLength)[0];
		FloatMatrix oversampledMatrix1 = oversampling(matrix0, matrix1, testSetLength)[1];
		FloatMatrix trainSetWithY = prepareTrainSet(oversampledMatrix0, oversampledMatrix1, setLength, testSetLength);
		FloatMatrix testSetX = splitsXAndY(testSetWithY, X, Y)[0];
		FloatMatrix testSetY = splitsXAndY(testSetWithY, X, Y)[1];
		FloatMatrix trainSetX = splitsXAndY(trainSetWithY, X, Y)[0];
		FloatMatrix trainSetY = splitsXAndY(trainSetWithY, X, Y)[1];
		FloatMatrix normalizedTrainSet = normalize(trainSetX);
		FloatMatrix normalizedTestSet = normalize(testSetX);
		return new FloatMatrix[] {normalizedTrainSet, trainSetY, normalizedTestSet, testSetY};
	}
	
	/**
	 * Splits the Matrix into two matrices depending on y-values.  
	 * 
	 * @param originalMatrix 
	 * @return [matrix0, matrix1]
	 */
	private static FloatMatrix [] splitZerosFromOnes(FloatMatrix originalMatrix) {
		FloatMatrix matrix0 = new FloatMatrix(0,originalMatrix.columns);
		FloatMatrix matrix1 = new FloatMatrix(0,originalMatrix.columns);
			for (int index = 0; index < originalMatrix.rows; index++) {
				if (originalMatrix.getColumn(0).get(index) == 0) {
					matrix0 = FloatMatrix.concatVertically(matrix0, originalMatrix.getRow(index));
				}
				else { matrix1 = FloatMatrix.concatVertically(matrix1, originalMatrix.getRow(index)); }
			}	
		return new FloatMatrix [] {matrix0, matrix1};
	}
	
	/**
	 * Takes the splitted set depending on y and returns the double length of the biggest matrix.
	 *    
	 * @param [matrix0, matrix1] 
	 * @return setLength
	 */
	
	private static int setLength(FloatMatrix matrix0, FloatMatrix matrix1) {
		int aLength = matrix0.getRows();
		int bLength = matrix1.getRows();
			if(aLength > bLength) {
				return 2*aLength;
			}
			else {return 2*bLength;}
	}
	
	/**
	 * Calculates the length of the test set.
	 *    
	 * @param setLength
	 * @param testPercentage
	 * @return testSetLength
	 */
	
	private static int testSetLength(int setLength, int testPercentage) {
		int testSetLength = setLength*testPercentage/100/2;
		return testSetLength;
	}
	
	/**
	 * Takes the two values from the splitted originalMatrix  
	 * 
	 * @param [cred0, cred1] 
	 * @param index  
	 * @return xCred0, xCred1
	 */
	
	private static FloatMatrix prepareTestSet(FloatMatrix matrix0, FloatMatrix matrix1, int testPercentage, int testSetLength) {
		FloatMatrix testSet = new FloatMatrix (0,matrix0.columns);
		for(int index = 0; index < testSetLength; index++) {
			testSet = FloatMatrix.concatVertically(testSet, matrix0.getRow(index));
			testSet = FloatMatrix.concatVertically(testSet, matrix1.getRow(index));
		}
		return testSet;
	}	
	
	/**
	 * Oversampling the training data.  
	 * 
	 * @param [cred0, cred1] 
	 * @param index  
	 * @return xCred0, xCred1
	 */
	
	private static FloatMatrix [] oversampling (FloatMatrix matrix0, FloatMatrix matrix1, int testSetLength) {
		while(matrix0.rows != matrix1.rows) {
			if(matrix0.rows < matrix1.rows) {
				matrix0 = FloatMatrix.concatVertically(matrix0, matrix0.getRow(testSetLength));
				testSetLength++;
			}
			else {
				matrix1 = FloatMatrix.concatVertically(matrix1, matrix1.getRow(testSetLength));
				testSetLength++;
			}
		}
		return new FloatMatrix[] {matrix0, matrix1};
	}
	
	/**
	 * Takes the two values from the splitted originalMatrix  
	 * 
	 * @param [cred0, cred1] 
	 * @param index  
	 * @return xCred0, xCred1
	 */
	
	private static FloatMatrix prepareTrainSet(FloatMatrix oversampledMatrix0, FloatMatrix oversampledMatrix1, int doubleSetLength, int testSetLength) {
		int trainSetLength = (doubleSetLength - (testSetLength)) / 2;
		FloatMatrix trainSet = new FloatMatrix (0,oversampledMatrix0.columns);
		for(int index = testSetLength; index < trainSetLength ; index++) {
			trainSet = FloatMatrix.concatVertically(trainSet, oversampledMatrix0.getRow(index));
			trainSet = FloatMatrix.concatVertically(trainSet, oversampledMatrix1.getRow(index));				
		}
		return trainSet;
	}
	
	/**
	 * Splits the Matrix into x and y values  
	 * 
	 * @param set
	 * @param xRange
	 * @param yRange
	 * @return [xSet, ySet]
	 */
	
	private static FloatMatrix [] splitsXAndY(FloatMatrix set, IntervalRange xRange, IntervalRange yRange) {
		FloatMatrix xSet = set.getColumns(xRange);
		FloatMatrix ySet = set.getColumns(yRange);
		return new FloatMatrix [] {xSet, ySet};	
	}
	
	/**
	 * Normalize the x-values  
	 * 
	 * @param matrix
	 * @return normalized matrix
	 */
	
	private static FloatMatrix normalize (FloatMatrix matrix) {
		FloatMatrix min = matrix.columnMins();
		FloatMatrix max = matrix.columnMaxs();
		FloatMatrix normalized = (matrix.subRowVector(min)).divRowVector(max.subRowVector(min));
		return normalized;
	}
	
	public static int [] training(FloatMatrix xTrainSet, FloatMatrix yTrainSet, FloatMatrix xTestSet, FloatMatrix yTestSet, int testSetLength) {
		int [] rate = new int[yTrainSet.length];
		FloatMatrix theta = FloatMatrix.ones(1,xTrainSet.columns).transpose();
		FloatMatrix m = FloatMatrix.scalar(yTrainSet.length);
			for(int index = 0; index < yTrainSet.length; index++) {
				FloatMatrix h = xTrainSet.mmul(theta);
				FloatMatrix sigmoid = sigmoid(h);
				FloatMatrix diff = sigmoid.sub(yTrainSet);
				FloatMatrix delta0 = xTrainSet.transpose().mmul(diff);
				FloatMatrix minifiedDelta = delta0.mmul(alpha).divRowVector(m);
				FloatMatrix newTheta = theta.subColumnVector(minifiedDelta);
				rate[index] = testPredicition(xTestSet, yTestSet, theta, testSetLength);
				theta = newTheta;
			}
		return rate;
	}

	private static int testPredicition(FloatMatrix xTestSet, FloatMatrix yTestSet, FloatMatrix theta, int testSetLength) {
		FloatMatrix h = xTestSet.mmul(theta);
		FloatMatrix sigmoid = sigmoid(h);
		FloatMatrix binary = binaryPredictions(sigmoid);
		int error = predictionError(binary, yTestSet);
		int rate = predictionRate(error, testSetLength);
		return rate;
	}	

	// abgeguckt von Aufgabe 8 
	private static FloatMatrix sigmoid(FloatMatrix h) {
		for (int i = 0; i < h.data.length; i++) {
			h.data[i] = (float) (1. / ( 1. + Math.exp(-h.data[i]) ));
		}
		return h;
	}
	
	private static FloatMatrix binaryPredictions (FloatMatrix sigmoid) {
		for(int index = 0; index < sigmoid.data.length; index++) {
			if(sigmoid.data[index] > 0.5) {
				sigmoid.data[index] = 1;
			}
			else {sigmoid.data[index] = 0;}
		}
		return sigmoid;
	}
	
	private static int predictionError(FloatMatrix binaryError, FloatMatrix creditability) {
		FloatMatrix error = binaryError.sub(creditability);
		FloatMatrix test = 	MatrixFunctions.abs(error);
		FloatMatrix result = test.columnSums();
		int[] resultToInt = result.getRow(0).toIntArray();
		int absResult = Math.abs(resultToInt[0]);
		return absResult;
	}

	private static int predictionRate(float error, int testSetLength) {
		float rate = (float) (testSetLength - error) / testSetLength * 100;
		return (int) rate;
	}
	private static int[] dataY;
	
	/**
	 * Draw the values and start the UI
	 */
	public static void plot(int[] rate) {
		dataY = rate;
		Application.launch(new String[0]);
		}
	
	/**
	 * Draw the UI
	 */
	@SuppressWarnings("unchecked")
	@Override public void start(Stage stage) {

		stage.setTitle(title);
		
		final NumberAxis xAxis = new NumberAxis();
		xAxis.setLabel(xAxisLabel);
        final NumberAxis yAxis = new NumberAxis();
        yAxis.setLabel(yAxisLabel);
        
		final LineChart<Number, Number> sc = new LineChart<>(xAxis, yAxis);

		XYChart.Series<Number, Number> series1 = new XYChart.Series<>();
		series1.setName("Data");
		for (int i = 0; i < dataY.length; i++) {
			series1.getData().add(new XYChart.Data<Number, Number>(i, dataY[i]));
		}

		sc.setAnimated(false);
		sc.setCreateSymbols(true);

		sc.getData().addAll(series1);

		Scene scene = new Scene(sc, 500, 400);
		stage.setScene(scene);
		stage.show();
    }
}
