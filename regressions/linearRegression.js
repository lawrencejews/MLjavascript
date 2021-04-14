const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LinearRegression{
  constructor(features, labels, options) {
    this.features = features;
    this.labels = labels;

    this.options = Object.assign({ learningRate: 0.1, iterations: 1000 }, options);
    this.m = 0;
    this.b = 0;
  }

  gradientDescent() {
    const currentGuessForMpg = this.features.map(row => { // mx+b
      return this.m * row[0] + this.b
    });
    const bSlope = _.sum(currentGuessForMpg.map((guess, i) => { 
      return guess - this.labels[i][0]
    })) * 2 / this.features.length;
    const mSlope = _.sum(currentGuessForMpg.map((guess, i) => {
      return -1 * this.features[i][0] * (this.labels[i][0] - guess)
    })) * 2 / this.features.length;
    this.m = this.m - mSlope * this.options.learningRate;
    this.b = this.b - bSlope * this.options.learningRate;
  }

  train() {
    for (let i = 0; i < this.options.iterations; i++){
      this.gradientDescent();
    }
  }
}
module.exports = LinearRegression;