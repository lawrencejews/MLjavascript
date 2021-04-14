require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");
const logisticRegression = require("./logisticRegression");
const plot = require('node-remote-plot');

const { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv', {
    dataColumns: ['horsepower', 'displacement', 'weight'],
    
    labelColumns: ['passedemissions'],
    shuffle: true,
    splitTest: 50,
    converters: {
        passedemissions: (value) => {
            return value === 'TRUE' ? 1 : 0;               // Converter work with 2 value === One-Hot Encoder
        }
    }
});
const regression = new logisticRegression(features, labels, {
    learningRate: 0.5,
    iterations: 100,
    batchSize: 50,
    decisionBoundary: 0.6
})
 
regression.train();
console.log(regression.test(testFeatures, testLabels));

plot({
    x: regression.crossentropyHistory.reverse()
});