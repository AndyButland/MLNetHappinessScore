namespace HappinessScoreMLNet.Program
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using ConsoleTables;
    using HappinessScoreMLNet.Program.Helpers;
    using HappinessScoreMLNet.Program.Models;
    using Microsoft.Data.DataView;
    using Microsoft.ML;
    using Microsoft.ML.Data;
    using Microsoft.ML.Trainers.FastTree;
    using Microsoft.ML.Transforms;

    public class Program
    {
        private const string PredictionLabel = "HappinessScore";
        private static readonly string[] FeatureColumns =
            {
                "Population", "Area", "PopulationDensity", "Coastline", "NetMigration", "InfantMortality", "GDP",
                "Literacy", "Phones", "Arable", "Climate", "Birthrate", "Deathrate"
            };

        private static readonly string InputDataPath = Path.Combine(Environment.CurrentDirectory, "Data\\input", "input.csv");
        private static readonly string ModelPath = Path.Combine(Environment.CurrentDirectory, "Data", "model.zip");

        public static void Main(string[] args)
        {
            var context = new MLContext();

            var data = GetTrainAndTestData(context, InputDataPath);

            var model = TrainModel(context, data.TrainSet);

            ReportOnFeatureImportance(context, model, data.TrainSet);

            EvaluateModel(context, model, data.TestSet);

            TestSinglePrediction(context);

            Console.ReadLine();
        }

        private static TrainCatalogBase.TrainTestData GetTrainAndTestData(MLContext context, string dataPath)
        {
            // Load data from the CSV file.
            var dataView = context.Data.LoadFromTextFile<CountryData>(dataPath, hasHeader: true, separatorChar: ',');

            // Split the data randomly, in 90:10 ratio, for training and evaluation data respectively.
            return context.MulticlassClassification.TrainTestSplit(dataView, testFraction: 0.1);
        }

        private static ITransformer TrainModel(MLContext context, IDataView data)
        {
            // Construct training pipeline:
            // - create a column for the output by copying the one we want to predict to the expected name "Label"
            // - create a column for all features using the expected name "Features"
            // - apply a regression
            var pipeline = context.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: PredictionLabel)
                .Append(context.Transforms.ReplaceMissingValues(
                    new MissingValueReplacingEstimator.ColumnOptions(
                        "Population", 
                        replacementMode: MissingValueReplacingEstimator.ColumnOptions.ReplacementMode.Mean)))
                .Append(context.Transforms.Concatenate("Features", FeatureColumns))
                .Append(context.Regression.Trainers.FastForest());

            var model = pipeline.Fit(data);
            SaveModelAsFile(context, model);
            return model;
        }

        private static void SaveModelAsFile(MLContext context, ITransformer model)
        {
            using (var fileStream = new FileStream(ModelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                context.Model.Save(model, fileStream);
            }

            Console.WriteLine("The model is saved to {0}", ModelPath);
            Console.WriteLine();
        }

        private static void ReportOnFeatureImportance(MLContext context, ITransformer model, IDataView data)
        {
            // Need to cast from the ITransformer interface to gain access to the LastTransformer property.
            var typedModel = (TransformerChain<RegressionPredictionTransformer<FastForestRegressionModelParameters>>)model;

            // Calculate metrics.
            var permutationMetrics = context.Regression.PermutationFeatureImportance(typedModel.LastTransformer, model.Transform(data), PredictionLabel);

            // Combine metrics with feature names and format for display.
            var columnsToExclude = new[] { PredictionLabel, "Code", "Name", "IdPreservationColumn" };
            var featureNames = data.Schema.AsEnumerable()
                .Select(column => column.Name)
                .Where(name => !columnsToExclude.Contains(name))
                .ToArray();
            var results = featureNames
                .Select((t, i) => new FeatureImportance
                    {
                        Name = t,
                        RSquaredMean = Math.Abs(permutationMetrics[i].RSquared.Mean),
                        CorrelationCoefficient = CalculateSingleFactorCorrelationCoefficient(context, data, t)
                })
                .OrderByDescending(x => x.RSquaredMean);

            OutputFeatureImportanceResults(results);
        }

        private static void OutputFeatureImportanceResults(IEnumerable<FeatureImportance> results)
        {
            Console.WriteLine("Feature importance:");

            var table = new ConsoleTable("Feature", "R Squared Mean", "Correlation Coefficient");
            foreach (var result in results)
            {
                table.AddRow(result.Name, result.RSquaredMean.ToString("G4"), result.CorrelationCoefficient.ToString("N2"));
            }

            table.Write();
            Console.WriteLine();
        }

        private static double CalculateSingleFactorCorrelationCoefficient(MLContext context, IDataView data, string featureColumn)
        {
            return CalculateCorrelationCoefficientBetweenValues(GetSingleColumn(context, data, featureColumn), GetSingleColumn(context, data, PredictionLabel));
        }

        private static IEnumerable<float> GetSingleColumn(IHostEnvironment context, IDataView data, string columnName)
        {
            return data.GetColumn<float>(context, columnName);
        }

        private static double CalculateCorrelationCoefficientBetweenValues(IEnumerable<float> featureColumn, IEnumerable<float> resultColumn)
        {
            return StatsHelper.ComputeCorrellationCoefficent(featureColumn.ToArray(), resultColumn.ToArray());
        }

        private static void EvaluateModel(MLContext context, ITransformer model, IDataView data)
        {
            var predictions = model.Transform(data);
            var metrics = context.Regression.Evaluate(predictions, "Label", "Score");

            OutputEvaluationResults(metrics);
        }

        private static void OutputEvaluationResults(RegressionMetrics metrics)
        {
            Console.WriteLine("Evaluation results:");

            var table = new ConsoleTable("R2 Score", "RMS loss");
            table.AddRow(metrics.RSquared.ToString("0.##"), metrics.Rms.ToString("#.##"));
            table.Write();

            Console.WriteLine();
        }

        private static void TestSinglePrediction(MLContext context)
        {
            // For this test we'll load the model back from disk.
            using (var stream = new FileStream(ModelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                var loadedModel = context.Model.Load(stream);

                var predictionFunction = loadedModel.CreatePredictionEngine<CountryData, CountryHappinessScorePrediction>(context);

                // Record of Italy - expected 5.964
                var sample = new CountryData
                    {
                        Population = 58133509,
                        Area = 301230,
                        PopulationDensity = 193,
                        Coastline = 2.52f,
                        NetMigration = 2.07f,
                        InfantMortality = 5.94f,
                        GDP = 26700,
                        Literacy = 98.6f,
                        Phones = 430.9f,
                        Arable = 27.79f,
                        Climate = 0,
                        Birthrate = 8.72f,
                        Deathrate = 10.4f
                    };
                var prediction = predictionFunction.Predict(sample);

                OutputSinglePredictionResult(prediction);
            }
        }

        private static void OutputSinglePredictionResult(CountryHappinessScorePrediction prediction)
        {
            Console.WriteLine("Single prediction result:");

            var table = new ConsoleTable("Predicted", "Expected");
            table.AddRow(prediction.HappinessScore.ToString("0.####"), 5.964);
            table.Write();

            Console.WriteLine();
        }

        private class FeatureImportance
        {
            public string Name { get; set; }

            public double RSquaredMean { get; set; }

            public double CorrelationCoefficient { get; set; }
        }
    }
}
