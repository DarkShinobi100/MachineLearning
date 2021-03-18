using System;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MachineLearning
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
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