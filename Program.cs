using intro;
using Microsoft.ML;
using Microsoft.ML.Data;


string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "SalaryData.csv");
MLContext mlContext = new MLContext(seed: 100);
IDataView trainDataView = mlContext.Data.LoadFromTextFile<SalaryData>(_trainDataPath, hasHeader: true, separatorChar: ',');
// Veri setini karıştır ve yüzde 30'unu test veri seti olarak ayır
var trainTestSplit = mlContext.Data.TrainTestSplit(trainDataView, testFraction: 0.3);
var trainSet = trainTestSplit.TrainSet;
var testSet = trainTestSplit.TestSet;
Console.WriteLine("Train Set:");
foreach (var item in mlContext.Data.CreateEnumerable<SalaryData>(trainSet, reuseRowObject: false))
{
    Console.WriteLine($"YearsExperience: {item.YearsExperience}, Salary: {item.Salary}");
}

Console.WriteLine("\nTest Set:");
foreach (var item in mlContext.Data.CreateEnumerable<SalaryData>(testSet, reuseRowObject: false))
{
    Console.WriteLine($"YearsExperience: {item.YearsExperience}, Salary: {item.Salary}");
}




var model = Train(mlContext, trainSet);
ITransformer Train(MLContext mlContext, IDataView dataView)
{
    var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "Salary")
        .Append(mlContext.Transforms.Concatenate("Features", "YearsExperience", "Salary"))
        .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Label", maximumNumberOfIterations: 100));
    var model = pipeline.Fit(dataView);
    return model;
}
Evaluate(mlContext, model, testSet);

void Evaluate(MLContext mlContext, ITransformer model, IDataView testSet)
{
    var predictions = model.Transform(testSet);
    var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");
    Console.WriteLine();
    Console.WriteLine($"*************************************************");
    Console.WriteLine($"*       Model quality metrics evaluation         ");
    Console.WriteLine($"*------------------------------------------------");
    Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");
}

TestSinglePrediction(mlContext, model, testSet);

void TestSinglePrediction(MLContext mlContext, ITransformer model, IDataView testSet)
{
    var predictionFunction = mlContext.Model.CreatePredictionEngine<SalaryData, SalaryPrediction>(model);
    var predictions = model.Transform(testSet);
    var SalarySample = mlContext.Data.CreateEnumerable<SalaryData>(testSet, reuseRowObject: false).First();
    var prediction = predictionFunction.Predict(SalarySample);
    Console.WriteLine($"**********************************************************************");
    Console.WriteLine($"Predicted fare: {prediction.Salary:0.####}, actual fare: {(double)SalarySample.Salary}");
    Console.WriteLine($"**********************************************************************");
}
