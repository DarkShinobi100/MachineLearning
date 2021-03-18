using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MachineLearning
{
    class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext();
            IDataView dataView = mlContext.Data.LoadFromTextFile<FlowerData>(@"C:\Users\User\Documents\C++\MachineLearning\MachineLearning\MachineLearning\iris_dataset.csv", hasHeader: true, separatorChar: ',');
            Console.WriteLine("load file!");
            var featureVectorName = "Features";
            var labelColumnName = "Label";
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey(
                inputColumnName: "SepalLength",
                outputColumnName: labelColumnName)
                .Append(mlContext.Transforms.Concatenate(featureVectorName,
                "SepalWidth", "PetalLength", "PetalWidth"))
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.MulticlassClassification.Trainers
                .SdcaMaximumEntropy(labelColumnName, featureVectorName))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
            
            Console.WriteLine("train data!");
            var model = pipeline.Fit(dataView);
            using (var fileStream = new FileStream("model.zip",FileMode.Create,
                FileAccess.Write, FileShare.Write)) { mlContext.Model.Save(model,dataView.Schema,
                    fileStream);}        
        }
    }
}
//input data from file
public class FlowerData
{
    [LoadColumn(0)]
    public float SepalLength { get; set; }

    [LoadColumn(1)]
    public float SepalWidth { get; set; }

    [LoadColumn(2)]
    public float PetalLength { get; set; }

    [LoadColumn(3)]
    public float PetalWidth { get; set; }

    [LoadColumn(4)]
    public string Species { get; set; }
}

//output data
public class FlowerPrediction
{
    [ColumnName("PredictedLabel")]
    public string Flower { get; set; }

    [ColumnName("Score")]
    public float[] Scores { get; set; }
}