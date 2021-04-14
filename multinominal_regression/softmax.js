const tf = require("@tensorflow/tfjs");
const _ = require("lodash");

class softmaxRegression {
    constructor(features, labels, options) {
        this.features = this.processFeatures(features);
        this.labels = tf.tensor(labels);
        this.crossentropyHistory = [];

        this.options = Object.assign(
            { learningRate: 0.1, iterations: 1000, decisionBoundary: 0.5 },
            options
        );
        this.weights = tf.zeros([this.features.shape[1], this.labels.shape[1]]);
    }

    gradientDescent(features, labels) {
        const currentGuesses = features.matMul(this.weights).softmax();
        const differences = currentGuesses.sub(labels);

        const slopes = features
            .transpose()
            .matMul(differences)
            .div(features.shape[0]);

        this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
    }

    train() {
        const batchQuantity = Math.floor(
            this.features.shape[0] / this.options.batchSize
        );
        for (let i = 0; i < this.options.iterations; i++) {
            for (let j = 0; j < batchQuantity; j++) {
                const startIndex = j * this.options.batchSize;
                const { batchSize } = this.options;
                const featureSlice = this.features.slice(
                    [startIndex, 0],
                    [batchSize, -1]
                );
                const labelSlice = this.labels.slice(
                    [startIndex, 0],
                    [batchSize, -1]
                );
                this.gradientDescent(featureSlice, labelSlice);
            }
            this.recordCrossentropy();
            this.updateLearningRate();
        }
    }

    predict(observations) {
      return this.processFeatures(observations)
        .matMul(this.weights)
        .softmax()
        .argMax(1);
    }

    test(testFeatures, testLabels) {
        const predictions = this.predict(testFeatures);
        testLabels = tf.tensor(testLabels).argMax(1);

        const incorrect = predictions.notEqual(testLabels).sum().get();

        return (predictions.shape[0] - incorrect) / predictions.shape[0];
    }

    processFeatures(features) {
        features = tf.tensor(features);

        if (this.mean && this.variance) {
            features = features.sub(this.mean).div(this.variance.pow(0.5));
        } else {
            features = this.standardize(features);
        }
        features = tf.ones([features.shape[0], 1]).concat(features, 1); // To get better a standardization first then column of ones

        return features;
    }

    standardize(features) {
        const { mean, variance } = tf.moments(features, 0);
        this.mean = mean;
        this.variance = variance;

        return features.sub(mean).div(variance.pow(0.5));
    }

    recordCrossentropy() {
        const guesses = this.features.matMul(this.weights).sigmoid();

        const termOne = this.labels.transpose().matMul(guesses.log());

        const termTwo = this.labels
            .mul(-1)
            .add(1)
            .transpose()
            .matMul(guesses.mul(-1).add(1).log());

        const cost = termOne
            .add(termTwo)
            .div(this.features.shape[0])
            .mul(-1)
            .get(0, 0);

        this.crossentropyHistory.unshift(cost); // reverse order.
    }

    updateLearningRate() {
        if (this.crossentropyHistory.length < 2) {
            return;
        }

        if (this.crossentropyHistory[0] > this.crossentropyHistory[1]) {
            this.options.learningRate = this.options.learningRate / 2;
        } else {
            this.options.learningRate = this.options.learningRate * 1.05;
        }
    }
}
module.exports = softmaxRegression;