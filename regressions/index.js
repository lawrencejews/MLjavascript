require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");
const LinearRegression = require("./regressionTensor");
const plot = require("node-remote-plot");

let { features, labels, testFeatures, testLabels } = loadCSV("./cars.csv", {
    shuffle: true,
    splitTest: 50,
    dataColumns: ["horsepower", "displacement", "weight"],
    labelColumns: ["mpg"],
});

const regression = new LinearRegression(features, labels, {
    learingRate: 0.1,
    iterations: 100,
    batchSize: 10,
});

regression.train();
const r2 = regression.test(testFeatures, testLabels); //determines better results

plot({
    x: regression.mseHistory.reverse(),
    xLabel: "Iteration #",
    yLabel: "Mean Squared Error",
});

console.log("R2 is", r2);
regression.predict([[120, 380, 2]]).print();
